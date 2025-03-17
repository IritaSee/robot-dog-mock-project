from typing import Any, Optional, List
import re
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def check_message_type(
    message: HumanMessage | AIMessage | ToolMessage
) -> dict[str, Any]:
    """
    Check the type of a message and whether it contains tool calls.
    
    Args:
        message: A LangChain message object
        
    Returns:
        A dictionary with the following keys:
            - type: A string describing the type of message
            - has_tool_calls: A boolean indicating whether the message contains tool calls
            - tool_names: A list of strings containing the names of any tools called in the message
    """
    message_type = "Unknown"
    has_tool_calls = False
    
    if isinstance(message, HumanMessage):
        message_type = "HumanMessage"
    elif isinstance(message, AIMessage):
        message_type = "AIMessage"
        # Check if the message has tool calls
        has_tool_calls = hasattr(message, "tool_calls") and len(message.tool_calls) > 0
    elif isinstance(message, ToolMessage):
        message_type = "ToolMessage"
    
    return {
        "type": message_type,
        "has_tool_calls": has_tool_calls,
        "tool_names": [tc["name"] for tc in message.tool_calls] if has_tool_calls else []
    }

def extract_final_answer(
    content: str,
    final_answer_markers: List[str]
) -> Optional[str]:
    """
    Extract the final answer from the content if it exists.

    This function takes a string content and a list of final answer markers.
    It checks for each marker in the content and if found, extracts the text
    after the marker as the final answer.

    The function also checks for the following patterns:
    - "## Final Answer (Indonesian)" followed by any text
    - "## Observation" followed by "## Final Answer" followed by any text
    If any of these patterns are found, the function extracts the final answer
    from the content.

    Args:
        content (str): The content to extract the final answer from
        final_answer_markers (List[str]): A list of final answer markers to check

    Returns:
        Optional[str]: The final answer if found, otherwise None
    """
    final_answer = content
    
    # Check for each marker and extract the final answer if found
    for marker in final_answer_markers:
        if marker in content:
            # Extract the text after the marker
            final_answer = content.split(marker, 1)[1].strip()
            break
    
    # Also check for "## Final Answer (Indonesian)" pattern
    final_answer_indonesian_pattern = r"## Final Answer \(Indonesian\)([\s\S]*?)(?=##|$)"
    match = re.search(final_answer_indonesian_pattern, content)
    if match:
        final_answer = match.group(1).strip()
    
    # Check for "## Observation" followed by "## Final Answer" pattern
    observation_pattern = r"## Observation([\s\S]*?)## Final Answer"
    final_answer_pattern = r"## Final Answer([\s\S]*?)(?=##|$)"
    
    # If we find both patterns, extract just the Final Answer part
    if re.search(observation_pattern, content) and re.search(final_answer_pattern, content):
        match = re.search(final_answer_pattern, content)
        if match:
            final_answer = match.group(1).strip()
    
    return final_answer