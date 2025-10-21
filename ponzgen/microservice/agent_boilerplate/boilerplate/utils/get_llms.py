import os

from langchain_openai import ChatOpenAI

def get_llms(model_name: str="gpt-4o-mini", temperature=0):
    """
    Helper function to get OpenAI model instance with necessary API key and endpoint.
    
    Args:
        model_name: The name of the model to use
        
    Returns:
        A configured ChatOpenAI instance
    """
    return ChatOpenAI(
        openai_api_key=os.getenv(
            "OPEN_ROUTER_API_KEY", 
            "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"
        ),
        openai_api_base=os.getenv(
            "OPEN_ROUTER_BASE_URL", 
            "https://openrouter.ai/api/v1/chat/completions"
        ),
        model=model_name,
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
        streaming=True
    )