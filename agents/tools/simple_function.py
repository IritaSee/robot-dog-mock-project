from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_timestamp():
    """
    Returns a string representing the current timestamp in the format
    %Y-%m-%d %H:%M:%S.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


