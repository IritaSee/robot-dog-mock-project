import os
import sys
import sqlite3
from dotenv import load_dotenv
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

# Add the project root to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prompts.text2sql_earthquake_sysprompt import TEXT2SQL_EARTHQUAKE_SYSPROMPT

# Load environment variables
load_dotenv()

# Path to the SQLite database
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH")

def convert_text_to_sql(query: str) -> str:
    """
    Convert a natural language query about conversations into a SQL query.
    
    Args:
        query (str): The natural language query about conversation data
        
    Returns:
        str: The generated SQL query
    """
    try:
        # Initialize LLM
        llm = ChatGroq(
            model=os.getenv("GROQ_LLM_NAME"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return "I encountered an error while setting up the language model. Please try again later."
    
    # Create messages for LLM
    messages = [
        SystemMessage(content=TEXT2SQL_EARTHQUAKE_SYSPROMPT),
        HumanMessage(content=query)
    ]
    
    # Get response from LLM
    try:
        response = llm.invoke(messages)
        # Extract SQL from the response (it might be wrapped in ```sql ... ``` blocks)
        sql_query = response.content
        
        # Clean up the SQL query if it's wrapped in markdown code blocks
        if "```sql" in sql_query and "```" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
            
        return sql_query
    except Exception as e:
        logger.error(f"Error during LLM invocation: {e}")
        return "I encountered an error while generating the SQL query. Please try again later."

def execute_earthquake_sql(sql_query: str) -> str:
    """
    Execute a SQL query against the conversation SQLite database and return the results.
    
    Args:
        sql_query (str): The SQL query to execute
        
    Returns:
        str: The query results formatted as a string
    """
    # Connect to the SQLite database
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return f"Failed to connect to the database: {str(e)}"
    
    try:
        # Execute the query
        cursor.execute(sql_query)
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Format the results
        if not results:
            return "Query executed successfully, but no results were returned."
        
        # Format as a table
        formatted_results = "| " + " | ".join(column_names) + " |\n"
        formatted_results += "| " + " | ".join(["---" for _ in column_names]) + " |\n"
        
        # Add each row
        for row in results:
            formatted_row = []
            for item in row:
                # Format the item based on its type
                if item is None:
                    formatted_row.append("NULL")
                else:
                    formatted_row.append(str(item))
            formatted_results += "| " + " | ".join(formatted_row) + " |\n"
        
        return formatted_results
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return f"Error executing query: {str(e)}"
    finally:
        cursor.close()
        conn.close()

@tool
def query_earthquake_database(natural_language_query: str) -> str:
    """
    Convert a natural language query about conversations to SQL and execute it.
    
    This tool combines text-to-SQL conversion and SQL execution in one step.
    
    Args:
        natural_language_query (str): The natural language query about conversation data
        
    Returns:
        str: The query results formatted as a string
    """
    # First convert the natural language query to SQL
    sql_query = convert_text_to_sql(natural_language_query)
    
    # Check if conversion was successful
    if "error" in sql_query.lower() or "failed" in sql_query.lower():
        return sql_query
    
    # Log the generated SQL query
    logger.info(f"Generated SQL query: {sql_query}")
    
    # Execute the SQL query and return results
    return execute_earthquake_sql(sql_query)
