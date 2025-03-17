"""
This module contains the graph builder and compiler for the LangGraph-based agent.
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from loguru import logger

from agents.utils import State, route_tools
from helpers.messages import check_message_type
from typing import Callable, Any, Union
from agents.utils import BasicToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg_pool import ConnectionPool

import os
from dotenv import load_dotenv

load_dotenv()

def create_graph(
    chatbot_node: Callable, 
    tool_node: BasicToolNode,
    collect_memory: bool = False
) -> Union[StateGraph, tuple[StateGraph, BaseCheckpointSaver]]:
    """
    Create and compile a LangGraph state graph with the provided nodes.
    
    Args:
        chatbot_node: The node function that processes messages with the LLM
        tool_node: The node that executes tools requested by the LLM
        collect_memory: If True, returns both the graph and checkpointer
        
    Returns:
        If collect_memory is False: A compiled LangGraph graph ready for invocation
        If collect_memory is True: A tuple of (compiled graph, checkpointer instance)
    """

    # Initialize memory saver with checkpoint_id
    # checkpointer = MemorySaver()

    POSTGRESQL_DB_URI = os.getenv("POSTGRESQL_DB_URI")
    
    # Default to in-memory checkpointer if PostgreSQL URI is not available
    checkpointer = MemorySaver()
    
    # If PostgreSQL URI is available, use PostgreSQL checkpointer
    if POSTGRESQL_DB_URI:
        try:
            # Connection pool configuration
            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
            }
            
            # Create connection pool
            pool = ConnectionPool(
                conninfo=POSTGRESQL_DB_URI,
                max_size=20,
                kwargs=connection_kwargs,
            )
            
            # Initialize PostgreSQL checkpointer
            checkpointer = PostgresSaver(pool)
            
            # Setup the checkpointer tables if they don't exist
            checkpointer.setup()
            
            logger.info("Using PostgreSQL checkpointer for state persistence")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")
            logger.info("Falling back to in-memory checkpointer")
            checkpointer = MemorySaver()
    else:
        logger.warning("POSTGRESQL_DB_URI not found, using in-memory checkpointer")

    # Initialize the graph builder with our State type
    graph_builder = StateGraph(State)
    
    # Add the chatbot and tools nodes
    graph_builder.add_node("chatbot_w_tools", chatbot_node)
    graph_builder.add_node("tools", tool_node)
    
    # Add conditional edges to route messages based on whether they contain tool calls
    graph_builder.add_conditional_edges(
        "chatbot_w_tools",
        route_tools,
        # Map the condition outputs to specific nodes
        {"tools": "tools", END: END},
    )
    
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot_w_tools")
    
    # Set the entry point to the chatbot node
    graph_builder.add_edge(START, "chatbot_w_tools")
    
    # Compile the graph with debug mode enabled
    graph_compiled = graph_builder.compile(
        checkpointer=checkpointer,
        debug=True
    )

    if collect_memory:
        return graph_compiled, checkpointer
    else:
        return graph_compiled


def run_chat(
    graph: StateGraph, 
    complex_query: str,
    thread_id: str = "default_thread",
    debug: bool = False
) -> Any:
    """
    Run a chat interaction with the graph using the provided query.
    
    Args:
        graph: The compiled LangGraph graph
        complex_query: The query to process
        thread_id: The thread ID for maintaining context across invocations
        debug: Whether to output debug information during execution
        
    Returns:
        The responses from the graph execution
    """
    initial_message = HumanMessage(content=complex_query)
    
    # Run the graph with the query and thread_id for persistence
    config = {"configurable": {"thread_id": thread_id}}
    react_responses = graph.invoke({"messages": [initial_message]}, config)
    
    # Analyze and display the results if debug is enabled
    if debug:
        logger.debug("ReAct Example Query: %s", complex_query)
        for i, msg in enumerate(react_responses['messages']):
            result = check_message_type(msg)
            logger.debug(f"Message {i}: {result['type']}")
            if result['has_tool_calls']:
                logger.debug(f"  Tool Calls: {', '.join(result['tool_names'])}")
            logger.debug(f"  Content: {msg.content}")
            logger.debug("-"*50)
    else:
        logger.info(f"FINAL ANSWER: {react_responses['messages'][-1].content}")
    
    return react_responses
