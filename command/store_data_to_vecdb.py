from langchain_unstructured import UnstructuredLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Batch, VectorParams, Distance
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from uuid import uuid4
from loguru import logger
import os

# Set the environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_vector_store(
    collection_name: str,
    embedding_model: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    vector_dim: int = 768,
    host: str = "localhost",
    port: int = 6333,
    distance: Distance = Distance.DOT,

) -> QdrantVectorStore:
    """
    Create a QdrantVectorStore if it doesn't exist, otherwise return the existing one.

    Args:
        collection_name (str): The name of the Qdrant collection.
        embedding_model (str, optional): The name of the embedding model to use. Defaults to 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'.
        vector_dim (int, optional): The dimensionality of the vectors. Defaults to 768.
        host (str, optional): The host of the Qdrant server. Defaults to "localhost".
        port (int, optional): The port of the Qdrant server. Defaults to 6333.
        distance (Distance, optional): The distance metric to use. Defaults to Distance.DOT.

    Returns:
        QdrantVectorStore: The QdrantVectorStore instance.
    """
    qdr_client = QdrantClient(host=host, port=port)
    llm_embeddings = FastEmbedEmbeddings(model_name=embedding_model)
    try:
        qdr_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=distance),
        )
    except Exception as e:
        logger.warning(e)
        logger.warning("Skip creating collection")

    
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=llm_embeddings,
        distance=distance,
        host=host,
        port=port,
    )
    return vector_store

def process_document(
    document_path: str = None,
    document_url: str = None,
    chunking_strategy: str = "basic",
    strategy: str = "fast"
) -> list[Document]:
    """
    Load and process a document from a local path or URL.

    This function utilizes the UnstructuredLoader to load a document from either
    a specified file path or web URL, applying a chosen chunking strategy.

    Args:
        document_path (str, optional): The path to the document file on the local system.
        document_url (str, optional): The URL of the document to be loaded from the web.
        chunking_strategy (str, optional): The strategy to use for chunking the document.
            Defaults to "by_title".
        strategy (str, optional): The strategy to use for processing the document.
            Defaults to "fast".

    Returns:
        list[Document]: A list of Document objects processed from the input source.

    Raises:
        Exception: If neither document_path nor document_url is provided.
    """
    if document_path is not None:
        logger.info(f"Loading document from local path {document_path}...")
        loader = UnstructuredLoader(
            file_path=document_path,
            chunking_strategy=chunking_strategy,
            strategy=strategy,
        )
    elif document_url is not None:
        logger.info(f"Loading document from web url {document_url}...")
        loader = UnstructuredLoader(
            web_url=document_url,
            chunking_strategy=chunking_strategy,
            strategy=strategy,
        )
    else:
        raise Exception("document_path or document_url is required")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents!")
    return documents

def store_data_to_vector_store(
    vector_store: QdrantVectorStore,
    documents: list[Document],
) -> None:
    """
    Store a list of documents to a Qdrant vector store.

    This function assigns unique UUIDs to each document and adds them
    to the specified Qdrant vector store.

    Args:
        vector_store (QdrantVectorStore): The vector store to which documents should be added.
        documents (list[Document]): A list of Document objects to be stored.

    Returns:
        None
    """
    logger.info(f"Storing {len(documents)} documents...")
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    logger.success(f"Stored {len(documents)} documents!")
    

if __name__ == "__main__":
    document_root = "./data/"
    document_paths = [
        f"{document_root}Iga_Narendra_synthetic_data.pdf",
    ]

    collection_name = "robot_dog_collection"
    chunking_strategy = "by_title"
    strategy = "hi_res"
    
    vector_store = get_vector_store(
        collection_name=collection_name,
        distance=Distance.DOT,
        embedding_model=os.getenv("FASTEMBED_MODEL_NAME"),
        vector_dim=int(os.getenv("FASTEMBED_VECTOR_DIM")),
    )
    for document_path in document_paths:
        logger.info(f"Processing document: {document_path}")
        documents = process_document(
            document_path=document_path,
            chunking_strategy=chunking_strategy,
            strategy=strategy,
        )
        store_data_to_vector_store(
            vector_store=vector_store,
            documents=documents
        )
    logger.success("All documents stored successfully!")
