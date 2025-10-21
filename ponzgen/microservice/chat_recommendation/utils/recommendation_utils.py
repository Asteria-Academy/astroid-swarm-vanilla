from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
import logging
import json
import re
import traceback

from ...agent_boilerplate.boilerplate.utils.get_llms import get_llms
from ...agent_boilerplate.boilerplate.errors import BadRequestError, InternalServerError
from others.prompts.recommendation_prompts import create_recommendation_system_prompt, create_recommendation_human_prompt

def extract_topics(text: str) -> List[str]:
    """
    Extract key topics from the input text.
    
    Args:
        text: Input text to extract topics from
        
    Returns:
        List of extracted topics
    """
    # Simple implementation - split by common separators and take first few words
    words = re.split(r'[,\s]+', text.strip())
    # Take first 3 words as topics
    return words[:3]

def parse_recommendation_response(response: str) -> List[str]:
    """
    Parse the LLM response to extract recommendations.
    
    Args:
        response: LLM response text
        
    Returns:
        List of recommendations
    """
    try:
        # Try to find a JSON array in the response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        # Try to parse as JSON
        recommendations = json.loads(response)
        
        # Handle both list and dict formats
        if isinstance(recommendations, dict):
            if 'recommendations' in recommendations:
                recommendations = recommendations['recommendations']
            else:
                recommendations = list(recommendations.values())
        elif not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
            
        # Clean up recommendations
        recommendations = [
            rec.strip(' "\'') for rec in recommendations 
            if rec and isinstance(rec, str)
        ]
            
        return recommendations
    except json.JSONDecodeError:
        # Fallback to line-based parsing
        lines = response.strip().split('\n')
        recommendations = []
        for line in lines:
            line = line.strip(' "\'[]{}')
            # Skip empty lines and common delimiters
            if line and not line.startswith(('```', '---', '===', '###')):
                recommendations.append(line)
        return recommendations
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return []

def validate_recommendations(recommendations: List[str]) -> List[str]:
    """
    Validate and clean up recommendations.
    
    Args:
        recommendations: List of recommendations
        
    Returns:
        List of validated recommendations
    """
    # Remove empty recommendations
    recommendations = [rec for rec in recommendations if rec]
    
    # Limit to 4 recommendations
    return recommendations[:4]

async def generate_recommendations_impl(
    user_input: str,
    chat_history_messages: List[str],
    model_name: str = "openai/gpt-4o-mini",
    temperature: float = 0,
    streaming: bool = False
) -> List[str]:
    """
    Implementation of chat recommendations generation.

    Args:
        user_input: The current message from the user
        chat_history_messages: List of previous chat messages
        model_name: Name of the LLM model to use
        temperature: Temperature parameter for the LLM
        streaming: Whether to stream the response (not used)

    Returns:
        List of up to 4 chat recommendations

    Raises:
        BadRequestError: If user_input is empty
        InternalServerError: If recommendation generation fails
    """
    if not user_input:
        raise BadRequestError("User input cannot be empty")

    try:
        print(f"Getting LLM with model: {model_name}, temperature: {temperature}")
        # Get the LLM with specified parameters
        llm = get_llms(model_name=model_name, temperature=temperature)
        print("Got LLM instance")

        print("Generating recommendations...")
        # Generate the recommendations with both system and human messages
        messages = [
            SystemMessage(content=create_recommendation_system_prompt()),
            HumanMessage(content=create_recommendation_human_prompt(user_input, chat_history_messages))
        ]
        
        try:
            # Get response from LLM
            response = await llm.ainvoke(messages)
            response_content = response.content
            print(f"Raw LLM response: {response_content}")

            if not response_content:
                raise InternalServerError("Empty response from LLM")

            recommendations = parse_recommendation_response(response_content)
            print(f"Parsed recommendations: {recommendations}")
            
            if not recommendations or len(recommendations) < 2:
                raise InternalServerError(f"Failed to generate valid recommendations. Response: {response_content}")
                
            validated_recommendations = validate_recommendations(recommendations)
            if len(validated_recommendations) < 2:
                raise InternalServerError(f"Generated recommendations were insufficient. Response: {response_content}")
                
            return validated_recommendations
                
        except Exception as e:
            print(f"Error in LLM invocation or parsing: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error args: {e.args}")
            print(f"Traceback: {traceback.format_exc()}")
            raise InternalServerError(f"Failed to generate recommendations: {str(e)}")
            
    except Exception as e:
        print(f"Error in generate_recommendations_impl: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        print(f"Traceback: {traceback.format_exc()}")
        raise InternalServerError(f"Failed to initialize LLM: {str(e)}") 