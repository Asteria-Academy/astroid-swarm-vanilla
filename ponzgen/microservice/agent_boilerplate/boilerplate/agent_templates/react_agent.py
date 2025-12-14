from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
import sys
import os

# Fix the import path
from ..utils.get_llms import get_llms

def get_react_agent(model_name="custom-vlm", temperature=0, langchain_tools=[], memory=None):
    """
    Create a ReAct agent with the specified model, tools, and memory.
    """
    model = get_llms(model_name, temperature)
    
    # 1. Get tool names and descriptions to feed into the prompt
    # Gemma needs to read these as text since it doesn't support native tool binding well
    
    # Print available tools to debug
    print("Available Tools:", [t.name for t in langchain_tools])
    
    tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in langchain_tools])
    tool_names = ", ".join([t.name for t in langchain_tools])

    # 2. STRICT ReAct System Prompt
    # This forces Gemma to output "Action:" instead of Python code
    system_prompt = f"""You are an intelligent agent interacting with the Linear issue tracking system.
    
You have access to the following tools:
{tool_desc}

To use a tool, you MUST use the following format:

Thought: Do I need to use a tool? Yes
Action: the name of the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool in JSON format
Observation: [Tool output will appear here]

Example 1 (Listing issues):
User: "List my linear issues"
Thought: I need to find the issues assigned to the user.
Action: linear_issues_list
Action Input: {{}}

Example 2 (Creating an issue):
User: "Create a bug report about the login page"
Thought: I need to create a new issue with the title provided.
Action: linear_issue_create
Action Input: {{"title": "Login page bug", "description": "User cannot login"}}

IMPORTANT:
1. Do NOT write Python code (like `print(...)` or `import...`).
2. Do NOT just describe the action.
3. ALWAYS start your turn with "Thought:".
4. If no tool is needed, just respond with the final answer.
"""

    # 3. Create the agent with the enforced prompt
    return create_react_agent(
        model, 
        langchain_tools, 
        checkpointer=memory,
        state_modifier=system_prompt
    )

if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver
    
    # Mocking a tool for testing if you run this file directly
    # In your actual app, 'langchain_tools' comes from the MCP connection
    from langchain_core.tools import tool

    @tool
    def linear_issue_create(title: str):
        """Creates a linear issue."""
        return "Issue created successfully"

    memory = MemorySaver()
    # Pass the mock tool for the test
    agent = get_react_agent(model_name="custom-vlm", langchain_tools=[linear_issue_create], memory=memory)
    
    query = "create linear issue for fixing the login bug"
    config = {'thread_id': "1"}
    
    print(f"User: {query}")
    try:
        # We stream the output to see if it's trying to call the tool
        for chunk in agent.stream({"messages": [("user", query)]}, config):
            print(chunk)
    except Exception as e:
        print(f"Error: {e}")