import os
import pickle
import gradio as gr
from dotenv import load_dotenv
from openai import AzureOpenAI
from operator import itemgetter
from typing import Iterable

# LangChain core
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage

# LangChain prompts
from langchain.prompts import PromptTemplate

# LangChain community tools
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# LangChain retrievers
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter


# Load environment variables
load_dotenv()

# LangSmith tracing (optional)
if os.getenv('LANGSMITH_TRACING_ACTIVE', 'false').lower() == 'true':
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

def build_rag_chain():
    # -------------------------
    # Azure Credentials
    # -------------------------
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME")
    CHROMA_PATH = os.getenv("CHROMA_PATH")

    # -------------------------
    # Embedding and Vector Store
    # -------------------------
    embedding = AzureOpenAIEmbeddings(
        model=EMBEDDING_DEPLOYMENT_NAME,          # e.g. "text-embedding-3-small"
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )

    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    # -------------------------
    # Retriever Setup
    # -------------------------
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": 30})

    # Load BM25 retriever
    BM25_INDEX_FILE = "bm25_retriever.pkl"  # Adjust path if needed
    with open(BM25_INDEX_FILE, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 30 

    base_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.75, 0.25],
        k=40,                                   
    )

    # redundant-filter (near-duplicate remover)
    deduper = EmbeddingsRedundantFilter(
        embeddings=embedding,
        similarity_threshold=0.90               # tweak if needed
    )

    # Transform the list *after* RRF and slice to 25 docs
    retriever = base_retriever | RunnableLambda(
        lambda docs: deduper.transform_documents(docs)[:25]
    )


    # -------------------------
    # LLM Setup (gpt-4o chat model)
    # -------------------------
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=CHAT_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.25
    )

    # -------------------------
    # Summary Chain
    # -------------------------

    # Running summary container 
    summary_text = ""  # grows by a sentence or two each turn

    # One-shot summariser chain (GPT-4o)
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    summariser_prompt = PromptTemplate.from_template(
        "Progressively summarize the conversation.\n\n"
        "Current summary:\n{summary}\n\n"
        "New lines:\n{lines}\n\n"
        "Updated summary:"
    )

    summariser_chain = summariser_prompt | llm | StrOutputParser()

    def update_summary(lines: str) -> str:
        """Append the latest human + AI lines to the running summary."""
        global summary_text
        summary_text = summariser_chain.invoke({
            "summary": summary_text,
            "lines": lines
        })
        return summary_text




    # -------------------------
    # Prompt Template
    # -------------------------
    prompt = PromptTemplate.from_template(
    """
    You are an intelligent assistant specializing in financial and economic news.

    Conversation so far:
    {history}

    Guidelines:
    - Your response should be **clear and informative**.
    - Answer in **as much detail as the question requires** — use a few sentences for simple questions and multiple paragraphs for more complex ones.
    - Structure your answer with full sentences and logical flow.
    - Cite retrieved information in-text, such as (Reuters).
    - **Ignore advertisements** or irrelevant content in the documents.
    - Be professional and concise — avoid filler, but don’t cut important details.

    Question: {question}

    Context: {context}

    Answer:
    """
    )


    format_docs = RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs))

    # -------------------------
    # RAG Chain
    # -------------------------
    rag_chain = (
        {
            "context": itemgetter("retrieval_question") | retriever | format_docs,
            "question": itemgetter("prompt_question"),
            "history":  itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# keep a global to avoid re-initialising on each request
RAG_CHAIN = build_rag_chain()

def stream_answer(question: str, history_summary: str = "(no prior context)") -> Iterable[str]:
    """Generator that yields chunks for FastAPI's StreamingResponse."""
    inputs = {
        "retrieval_question": question,
        "prompt_question":    question,
        "history":            history_summary,
    }
    for chunk in RAG_CHAIN.stream(inputs):
        yield chunk if isinstance(chunk, str) else chunk.content
