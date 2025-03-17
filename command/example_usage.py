"""
Example usage of the LangGraphAgent class.

This script demonstrates how to use the LangGraphAgent class in different scenarios,
including interactive sessions, programmatic usage, and custom configurations.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.tools import tool
from loguru import logger

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.langgraph_agent import LangGraphAgent
from agents.tools import get_current_timestamp, retrieve_earthquake_knowledge, query_earthquake_database
from prompts import REACT_SYSTEM_PROMPT
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

def example_interactive_session():
    """
    Example of an interactive session with the agent.
    """
    logger.info("=== Example: Interactive Session ===")
    
    # Create an agent with default settings
    agent = LangGraphAgent(
        model_name=os.getenv("GROQ_LLM_NAME"),
        system_prompt=REACT_SYSTEM_PROMPT,
    )
    
    # Start an interactive session
    agent.interactive_session()


def example_programmatic_usage():
    """
    Example of programmatic usage of the agent.
    """
    logger.info("=== Example: Programmatic Usage ===")
    
    # Create an agent with default settings
    agent = LangGraphAgent(
        model_name=os.getenv("GROQ_LLM_NAME"),
        system_prompt=REACT_SYSTEM_PROMPT,
    )
    
    # Start a new conversation
    thread_id = agent.start_conversation()
    
    # Process a query
    query = "What is the current time?"
    logger.info(f"QUERY: {query}")
    result = agent.process_query(query, thread_id=thread_id)
    # logger.info(f"RESPONSE: {result}")
    
    # Process another query in the same conversation
    query = "Calculate 123 * 456"
    logger.info(f"QUERY: {query}")
    result = agent.process_query(query, thread_id=thread_id)
    # logger.info(f"RESPONSE: {result}")


def example_custom_configuration():
    """
    Example of creating an agent with custom configuration.
    """
    logger.info("=== Example: Custom Configuration ===")
    
    # Define custom tools
    custom_tools = [
        get_current_timestamp,
        retrieve_earthquake_knowledge,
        query_earthquake_database
    ]
    
    # Define custom system prompt
    custom_system_prompt = """You are a helpful assistant specialized in mathematics. 
You can solve complex mathematical problems and provide clear explanations.
Always show your work step by step."""
    
    # Create an agent with custom configuration
    agent = LangGraphAgent(
        model_name=os.getenv("GROQ_LLM_NAME"),
        system_prompt=custom_system_prompt,
        tools=custom_tools,
        collect_memory=True,
        api_keys={
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        }
    )
    
    # Start a new conversation
    thread_id = agent.start_conversation()
    
    # Process a query
    query = "What is the square root of 144 multiplied by the cube of 3?"
    logger.info(f"Query: {query}")
    result = agent.process_query(query, thread_id=thread_id)
    # logger.info(f"Response: {result}")
    
    # Generate a summary of the conversation
    summary = agent.get_conversation_summary(thread_id=thread_id)
    logger.info(f"Conversation Summary:\n{summary}")


def example_using_tool_methods():
    """
    Example of using the default_tools and get_default_tools methods.
    """
    logger.info("=== Example: Using Tool Methods ===")
    
    # Get default tools without instantiating the class
    class_tools = LangGraphAgent.get_default_tools()
    logger.info(f"Default tools from class method: {[LangGraphAgent.get_tool_name(t) for t in class_tools]}")
    
    # Create an agent and get its default tools
    agent = LangGraphAgent()
    instance_tools = agent.default_tools()
    logger.info(f"Default tools from instance method: {[agent.get_tool_name(t) for t in instance_tools]}")

    @tool
    def show_random_datascience_knowledge(query: str):
        """
        Show a random data science concept or technique.

        The query parameter is not used in this tool.

        Returns:
            A string containing a random data science concept or technique.
        """
        llm = ChatGroq(model=os.getenv("GROQ_LLM_NAME"))
        return llm.invoke(query)
    
    # Get default tools and add new tool
    all_tools = LangGraphAgent.get_default_tools() + [show_random_datascience_knowledge]
    logger.info(f"Extended tools: {[LangGraphAgent.get_tool_name(t) for t in all_tools]}")
    
    # Create agent with extended tools
    agent_with_extended_tools = LangGraphAgent(tools=all_tools)
    logger.info("Created agent with extended tools successfully")

    # Get all tools including the extended tools
    all_tools = agent_with_extended_tools.default_tools()
    logger.info(f"All tools: {[agent_with_extended_tools.get_tool_name(t) for t in all_tools]}")

    # Demonstrate using the static method directly
    logger.info(f"Tool name using static method: {LangGraphAgent.get_tool_name_static(show_random_datascience_knowledge)}")
    


if __name__ == "__main__":
    # Choose which example to run
    example_interactive_session()
    # example_programmatic_usage()
    # example_custom_configuration()
    # example_using_tool_methods()
