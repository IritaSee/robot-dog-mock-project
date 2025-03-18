import os
import sys
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client.http.models.models import Distance
from unstructured.staging.base import elements_from_base64_gzipped_json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from store_data_to_vecdb import get_vector_store
from prompts import RAG_EARTHQUAKE_SYSPROMPT

load_dotenv()

if __name__ == "__main__":
    collection_name = "robot_dog_collection"
    with_score = False
    top_k = 7

    logger.info(f"Using collection: {collection_name}")
    vector_store = get_vector_store(
        collection_name=collection_name,
        distance=Distance.DOT,
        embedding_model=os.getenv("FASTEMBED_MODEL_NAME"),
        vector_dim=int(os.getenv("FASTEMBED_VECTOR_DIM")),
    )
    logger.success(f"Vector store loaded successfully!")

    while True:
        query = input("Enter your query: ")

        logger.debug("-------- RETRIEVED CONTEXT --------")
        context = ""
        if with_score:
            results = vector_store.similarity_search_with_score(query, k=top_k)
            for idx, result in enumerate(results):
                orig_elements = elements_from_base64_gzipped_json(result[0].metadata["orig_elements"])
                for oe in orig_elements:
                    oe = oe.to_dict()
                    context += f"{idx+1}. Type: {oe['type']}\nContent: {oe['text']}\n\n"
        else:
            results = vector_store.similarity_search(query, k=top_k)
            for idx, result in enumerate(results):
                orig_elements = elements_from_base64_gzipped_json(result.metadata["orig_elements"])
                for oe in orig_elements:
                    oe = oe.to_dict()
                    context += f"{idx+1}. Type: {oe['type']}\nContent: {oe['text']}\n\n"
        logger.debug(f"Context:\n{context}")
        logger.debug("-------- END RETRIEVED CONTEXT --------")
        logger.debug(f"Current model name: {os.getenv("GROQ_LLM_NAME")}")

        llm = ChatGroq(
            model=os.getenv("GROQ_LLM_NAME"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )

        user_prompt = f"""Context:
    {context}

    Query:
    {query}"""

        messages = [
            SystemMessage(content=RAG_EARTHQUAKE_SYSPROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        logger.info(f"\nRESPONSE:\n{response.content}")
        logger.success("Done!")