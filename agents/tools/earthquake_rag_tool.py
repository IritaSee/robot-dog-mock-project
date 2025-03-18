import os
import sys
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client.http.models.models import Distance
from unstructured.staging.base import elements_from_base64_gzipped_json

# Add the project root to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from command.store_data_to_vecdb import get_vector_store
from prompts.rag_earthquake_sysprompt import RAG_EARTHQUAKE_SYSPROMPT

from langchain_core.tools import tool

# Load environment variables
load_dotenv()

@tool
def retrieve_earthquake_knowledge(query: str) -> str:
    """
    Retrieve common information based on the query.
    
    Args:
        query (str): The user's question about the product
        
    Returns:
        str: The response with information about the product based on the retrieved context
    """
    # Vector store configuration
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    top_k = 7
    
    # Initialize vector store
    try:
        vector_store = get_vector_store(
            collection_name=collection_name,
            distance=Distance.DOT,
            embedding_model=os.getenv("FASTEMBED_MODEL_NAME"),
            vector_dim=int(os.getenv("FASTEMBED_VECTOR_DIM")),
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        return "I encountered an error while trying to retrieve earthquake information. Please try again later."
    
    # Retrieve relevant context from vector store
    try:
        context = ""
        results = vector_store.similarity_search(query, k=top_k)
        for idx, result in enumerate(results):
            orig_elements = elements_from_base64_gzipped_json(result.metadata["orig_elements"])
            for oe in orig_elements:
                oe = oe.to_dict()
                context += f"{idx+1}. Type: {oe['type']}\nContent: {oe['text']}\n\n"
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return "I encountered an error while searching for the related information. Please try again later."
    
    # If no context was retrieved
    if not context.strip():
        return "I couldn't find any relevant information about related for your query."
    
    # Initialize LLM
    try:
        llm = ChatGroq(
            model=os.getenv("GROQ_LLM_NAME"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return "I encountered an error while setting up the language model. Please try again later."
    
    # Format user prompt with context and query
    user_prompt = f"""Context:
{context}

Query:
{query}"""
    
    # Create messages for LLM
    messages = [
        SystemMessage(content=RAG_EARTHQUAKE_SYSPROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    # Get response from LLM
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error during LLM invocation: {e}")
        return "I encountered an error while processing your related query. Please try again later."