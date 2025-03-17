from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage
from langgraph.graph import END

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {}
        for tool in tools:
            if hasattr(tool, "name"):
                # For LangChain tools that have a name attribute
                self.tools_by_name[tool.name] = tool
            elif callable(tool):
                # For regular Python functions, use the __name__ attribute
                self.tools_by_name[tool.__name__] = tool
            else:
                raise ValueError(f"Tool {tool} is neither a LangChain tool nor a callable function")

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]
            tool = self.tools_by_name[tool_name]
            
            # Handle different types of tools
            if hasattr(tool, "invoke"):
                # LangChain tool
                tool_result = tool.invoke(tool_call["args"])
            elif callable(tool):
                # Regular Python function
                try:
                    # Try to parse args as JSON if it's a string
                    if isinstance(tool_call["args"], str):
                        try:
                            args = json.loads(tool_call["args"])
                        except json.JSONDecodeError:
                            args = tool_call["args"]
                    else:
                        args = tool_call["args"]
                    
                    # Call the function with or without args depending on what it expects
                    if args and isinstance(args, dict):
                        tool_result = tool(**args)
                    else:
                        tool_result = tool()
                except Exception as e:
                    tool_result = f"Error invoking {tool_name}: {str(e)}"
            else:
                tool_result = f"Unknown tool type: {type(tool)}"
                
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    # Check for standard tool_calls attribute
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    
    # Check for function calls in content format: <function=name{...}</function>
    if hasattr(ai_message, "content") and ai_message.content:
        content = ai_message.content
        import re
        function_pattern = r'<function=([a-zA-Z0-9_]+)(\{.*?\})</function>'
        function_matches = re.findall(function_pattern, content)
        
        if function_matches:
            # Convert function calls in content to tool_calls format
            ai_message.tool_calls = []
            for idx, (func_name, func_args) in enumerate(function_matches):
                try:
                    # Clean up the args string and parse as JSON
                    args_str = func_args.strip()
                    args = json.loads(args_str)
                    
                    # Add to tool_calls
                    ai_message.tool_calls.append({
                        "name": func_name,
                        "args": args,
                        "id": f"call_{idx}",
                    })
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the raw string
                    ai_message.tool_calls.append({
                        "name": func_name,
                        "args": func_args,
                        "id": f"call_{idx}",
                    })
            
            if ai_message.tool_calls:
                return "tools"
    
    return END