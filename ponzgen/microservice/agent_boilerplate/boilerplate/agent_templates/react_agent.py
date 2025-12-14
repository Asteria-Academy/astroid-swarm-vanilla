from langgraph.prebuilt import create_react_agent
import sys
import os

# Fix the import path
from ..utils.get_llms import get_llms

def get_react_agent(model_name="custom-vlm", temperature=0, langchain_tools=[], memory=None):
    """
    Create a ReAct agent with the specified model, tools, and memory.
    
    Args:
        model_name: The name of the LLM to use
        temperature: The temperature setting for the model (0-1)
        langchain_tools: List of LangChain tools to provide to the agent
        memory: Memory instance for maintaining conversation state
        
    Returns:
        A configured ReAct agent
    """
    # model_name = "anthropic/claude-3.5-sonnet" # for tools, one of the most general model
    model = get_llms(model_name, temperature)
    
    # Add system prompt to instruct the model to use tools
    system_prompt = """You are a helpful AI assistant with access to tools.

When the user asks you to perform actions like:
- "list", "show", "get", "find", "search" → Use search/list tools
- "create", "add", "make", "new" → Use create tools  
- "update", "edit", "modify", "change" → Use update tools
- "delete", "remove" → Use delete tools

IMPORTANT: When you have tools available and the user's request matches a tool's capability, you MUST use the tool instead of just describing what you would do.

For example:
- "list linear issues" → Use the Linear search/list tool
- "create an issue" → Use the Linear create issue tool
- "hello" → Just respond conversatively (no tool needed)

Always use tools when appropriate to provide accurate, real-time information."""
    
    return create_react_agent(
        model, 
        langchain_tools, 
        checkpointer=memory,
        state_modifier=system_prompt
    )
# Example usage
if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver
    
    # Create memory and agent
    memory = MemorySaver()
    agent = get_react_agent(model_name="custom-vlm", memory=memory)
    
    # First interaction
    query = "My name is kelvin"
    config = {'thread_id': "1"}
    try:
        response = agent.invoke({"messages": query}, {"configurable": config})
        ai_message = response['messages'][-1].content
        print(ai_message)
    except Exception as e:
        print(f"Error: {e}")
    
    # Follow up question (demonstrates memory persistence)
    query = "What is my name?"
    config = {'thread_id': "1"}
    try:
        response = agent.invoke({"messages": query}, {"configurable": config})
        ai_message = response['messages'][-1].content
        print(ai_message)
    except Exception as e:
        print(f"Error: {e}")