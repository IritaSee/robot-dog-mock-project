"""
LangGraph Agent - An efficient and concise implementation of a LangGraph-based agent.

This module provides a class-based implementation of a LangGraph agent with
ReAct capabilities, memory management, and conversation summarization.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import State, BasicToolNode, create_graph, run_chat
from agents.tools import (
    get_current_timestamp, 
    retrieve_earthquake_knowledge,
    query_earthquake_database
)
from prompts import REACT_SYSTEM_PROMPT

# Load environment variables
load_dotenv()


class LangGraphAgent:
    """
    A class-based implementation of a LangGraph agent with ReAct capabilities.
    
    This class provides a more efficient and concise way to create and interact
    with a LangGraph-based agent, with support for memory management and
    conversation summarization.
    """
    
    def __init__(
        self, 
        model_name: str,
        system_prompt: str,
        tools: Optional[List[Any]] = None,
        collect_memory: bool = True,
        api_keys: Optional[Dict[str, str]] = None,
        debug: bool = False
    ):
        """
        Initialize a new LangGraphAgent instance.
        
        Args:
            model_name: The name of the Anthropic model to use
            system_prompt: The system prompt to use for the agent
            tools: A list of tools to make available to the agent
            collect_memory: Whether to collect and store conversation memory
            api_keys: A dictionary of API keys to use for various services
            debug: Whether to output debug information during execution
        """
        # Set up API keys
        self.api_keys = api_keys or {}
        for key, value in self.api_keys.items():
            os.environ[key] = value
            
        # Set up the LLM
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Set up tools
        self.tools = tools or self.default_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = BasicToolNode(tools=self.tools)
        self.debug = debug
        
        # Set up memory management
        self.collect_memory = collect_memory
        self.thread_id = None
        self.checkpointer = None
        
        # Create the graph
        self._create_graph()
    
    def default_tools(self) -> List[Any]:
        """
        Create the default set of tools for the agent.
        
        Returns:
            A list of default tools
        """
        return [
            get_current_timestamp,
            retrieve_earthquake_knowledge,
            query_earthquake_database
        ]
    
    @classmethod
    def get_default_tools(cls) -> List[Any]:
        """
        Class method to get the default set of tools without instantiating the class.
        
        Returns:
            A list of default tools
        """
        return [
            get_current_timestamp,
            retrieve_earthquake_knowledge,
            query_earthquake_database
        ]
    
    def get_tool_name(self, tool: Any) -> str:
        """
        Get the name of a tool regardless of its type.
        
        Args:
            tool: A tool object which could be a function, class instance, or other object
            
        Returns:
            The name of the tool
        """
        return self.__class__.get_tool_name_static(tool)
    
    @classmethod
    def get_tool_name(cls, tool: Any) -> str:
        """
        Class method to get the name of a tool regardless of its type.
        
        Args:
            tool: A tool object which could be a function, class instance, or other object
            
        Returns:
            The name of the tool
        """
        return cls.get_tool_name_static(tool)
    
    @staticmethod
    def get_tool_name_static(tool: Any) -> str:
        """
        Static method to get the name of a tool regardless of its type.
        
        Args:
            tool: A tool object which could be a function, class instance, or other object
            
        Returns:
            The name of the tool
        """
        if callable(tool) and hasattr(tool, "__name__"):
            return tool.__name__
        elif hasattr(tool, "name"):
            return tool.name
        else:
            return str(tool)
    
    def _chatbot_node(self, state: State) -> Dict[str, List[Any]]:
        """
        The chatbot node function for the graph.
        
        Args:
            state: The current state of the conversation
            
        Returns:
            A dictionary with the updated messages
        """
        # Add system message at the beginning if not already present
        messages = state["messages"]
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=self.system_prompt)] + messages
            
        return {
            "messages": [
                self.llm_with_tools.invoke(messages)
            ]
        }
    
    def _create_graph(self) -> None:
        """
        Create and compile the LangGraph state graph.
        """
        if self.collect_memory:
            self.graph, self.checkpointer = create_graph(
                self._chatbot_node, 
                self.tool_node, 
                collect_memory=True
            )
        else:
            self.graph = create_graph(
                self._chatbot_node, 
                self.tool_node, 
                collect_memory=False
            )
    
    def start_conversation(self) -> str:
        """
        Start a new conversation with a unique thread ID.
        
        Returns:
            The thread ID for the new conversation
        """
        self.thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        logger.info(f"Started new conversation with thread ID: {self.thread_id}")
        return self.thread_id
    
    def process_query(
        self, 
        query: str, 
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Process a query using the agent.
        
        Args:
            query: The query to process
            thread_id: The thread ID to use for the conversation (if None, uses the current thread ID)
            
        Returns:
            The responses from the graph execution
        """
        logger.info(f"Processing query with thread ID: {thread_id}")
        return run_chat(self.graph, query, thread_id, debug=self.debug)
        """
        Retrieve a conversation summary from the database.
        
        Args:
            thread_id: The thread ID to retrieve the summary for (if None, uses the current thread ID)
            
        Returns:
            The summary text or None if not found
        """
        thread_id = thread_id or self.thread_id
        if not thread_id:
            logger.warning("Cannot retrieve summary: no thread ID provided")
            return None
        
        db_manager = DatabaseManager()
        summary = db_manager.get_conversation_summary(thread_id)
        
        if summary:
            logger.info(f"Retrieved conversation summary for thread {thread_id} from database")
        else:
            logger.info(f"No stored summary found for thread {thread_id}")
            
        return summary
    
    def interactive_session(self) -> None:
        """
        Start an interactive session with the agent.
        
        This method starts a loop that allows the user to interact with the agent
        via the command line. The session continues until the user interrupts it
        with Ctrl+C.
        """
        # Start a new conversation
        self.start_conversation()
        logger.info("\n--- ReAct Pattern Interactive Session ---")
        logger.info(f"Thread ID: {self.thread_id}")
        
        try:
            while True:
                query = input("User: ")
                if query.lower() in ["exit", "quit", "q"]:
                    break
                    
                # Process the query
                self.process_query(query, thread_id=self.thread_id)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Exiting the session...")
        
        logger.success("Session completed.")


# Example usage
if __name__ == "__main__":
    # Create an agent with default settings
    agent = LangGraphAgent(
        model_name=os.getenv("GROQ_LLM_NAME"),
        system_prompt=REACT_SYSTEM_PROMPT
    )
    
    # Start an interactive session
    agent.interactive_session()
