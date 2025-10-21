"""
Agent Invoke Router

This module provides routes for agent invocation.
"""

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any
from uuid import UUID
from supabase import Client

from ..boilerplate.agent_boilerplate import agent_boilerplate
from ..boilerplate.models import AgentInput
from ..boilerplate.errors import (
    BadRequestError, NotFoundError, ForbiddenError, 
    InternalServerError, ERROR_RESPONSES
)

# Create router
router = APIRouter(
    prefix="/agent-invoke",
    tags=["agent-invoke"],
    responses={**ERROR_RESPONSES}
)

# Dependency to get Supabase client
def get_supabase_client(request: Request):
    return request.app.state.supabase


@router.post("/{agent_id}/invoke", response_model=Dict[str, Any])
async def invoke_agent(
    agent_id: str,
    agent_input: AgentInput,
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Invoke an agent by its agent_id.
    
    This endpoint matches the agent_id to an agent and invokes it with the provided input.
    If agent_config is provided in the request body, it will be used directly.
    Otherwise, the agent configuration will be fetched from the database.
    """
    try:
        # Get user_id from request state (set by middleware)
        user_id = request.state.user_id
        
        # Check if agent_config is provided in the request
        if agent_input.agent_config:
            print("Using agent_config from request")
            agent_config = agent_input.agent_config
            
            # Verify that the agent_id matches
            if agent_config.get("agent_id") != agent_id:
                raise BadRequestError(
                    f"agent_id in URL ({agent_id}) does not match agent_id in agent_config ({agent_config.get('agent_id')})",
                    additional_info={
                        "url_agent_id": agent_id,
                        "config_agent_id": agent_config.get("agent_id")
                    }
                )
            
            # Check if the user has access to the agent
            has_access = False
            
            # Check if the user is the owner of the agent
            if agent_config.get("user_id") == user_id:
                has_access = True
            # Check if the agent belongs to a company the user has access to
            elif agent_config.get("company_id"):
                try:
                    user_company_response = (
                        supabase.table("user_companies")
                        .select("role_id")
                        .eq("user_id", user_id)
                        .eq("company_id", agent_config["company_id"])
                        .execute()
                    )
                    if user_company_response.data:
                        has_access = True
                except Exception as e:
                    raise InternalServerError(f"Error checking company access: {str(e)}")
            
            # If no access yet, check if the user's email has editor access
            if not has_access:
                # Get the user's email
                try:
                    user_response = (
                        supabase.table("users")
                        .select("email")
                        .eq("user_id", user_id)
                        .execute()
                    )
                except Exception as e:
                    raise InternalServerError(f"Error fetching user email: {str(e)}")
                
                if user_response.data:
                    user_email = user_response.data[0].get("email")
                    # Check if the email is in the share_editor_with list
                    if user_email in agent_config.get("share_editor_with", []):
                        has_access = True
            
            # If still no access, deny permission
            if not has_access:
                raise ForbiddenError(
                    "You don't have access to this agent",
                    additional_info={
                        "agent_id": agent_id
                    }
                )
        else:
            print("Fetching agent_config from database")
            # Get agent by agent_id
            try:
                agent_response = (
                    supabase.table("agents")
                    .select("agent_id, agent_name, description, agent_style, on_status, company_id, user_id, tools, share_editor_with")
                    .eq("agent_id", agent_id)
                    .eq("on_status", True)
                    .execute()
                )
            except Exception as e:
                raise InternalServerError(f"Error fetching agent: {str(e)}")
            
            if not agent_response.data:
                raise NotFoundError(
                    f"Agent with ID '{agent_id}' not found",
                    additional_info={
                        "agent_id": agent_id
                    }
                )
            
            agent_config = agent_response.data[0]
            
            # Check if the user has access to the agent
            has_access = False
            
            # Check if the user is the owner of the agent
            if agent_config.get("user_id") == user_id:
                has_access = True
            # Check if the agent belongs to a company the user has access to
            elif agent_config.get("company_id"):
                try:
                    user_company_response = (
                        supabase.table("user_companies")
                        .select("role_id")
                        .eq("user_id", user_id)
                        .eq("company_id", agent_config["company_id"])
                        .execute()
                    )
                    if user_company_response.data:
                        has_access = True
                except Exception as e:
                    raise InternalServerError(f"Error checking company access: {str(e)}")
            
            # If no access yet, check if the user's email has editor access
            if not has_access:
                # Get the user's email
                try:
                    user_response = (
                        supabase.table("users")
                        .select("email")
                        .eq("user_id", user_id)
                        .execute()
                    )
                except Exception as e:
                    raise InternalServerError(f"Error fetching user email: {str(e)}")
                
                if user_response.data:
                    user_email = user_response.data[0].get("email")
                    # Check if the email is in the share_editor_with list
                    if user_email in agent_config.get("share_editor_with", []):
                        has_access = True
            
            # If still no access, deny permission
            if not has_access:
                raise ForbiddenError(
                    "You don't have access to this agent",
                    additional_info={
                        "agent_id": agent_id
                    }
                )
        
        # Invoke the agent
        try:
            response = await agent_boilerplate.invoke_agent(
                agent_id=agent_id,
                agent_input=agent_input,
                agent_config=agent_config
            )
            return response
        except Exception as e:
            raise InternalServerError(f"Error invoking agent: {str(e)}")
    
    except (BadRequestError, NotFoundError, ForbiddenError, InternalServerError) as e:
        # Re-raise known errors
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise InternalServerError(f"Unexpected error: {str(e)}")


@router.post("/{agent_id}/invoke-stream", response_model=Dict[str, Any])
async def invoke_agent_stream(
    agent_id: str,
    agent_input: AgentInput,
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Invoke an agent by its agent_id with streaming response.
    
    This endpoint matches the agent_id to an agent and invokes it with the provided input.
    If agent_config is provided in the request body, it will be used directly.
    Otherwise, the agent configuration will be fetched from the database.
    """
    try:
        # Get user_id from request state (set by middleware)
        user_id = request.state.user_id
        
        # Check if agent_config is provided in the request
        if agent_input.agent_config:
            print("Using agent_config from request")
            agent_config = agent_input.agent_config
            
            # Verify that the agent_id matches
            if agent_config.get("agent_id") != agent_id:
                raise BadRequestError(
                    f"agent_id in URL ({agent_id}) does not match agent_id in agent_config ({agent_config.get('agent_id')})",
                    additional_info={
                        "url_agent_id": agent_id,
                        "config_agent_id": agent_config.get("agent_id")
                    }
                )
            
            # Check if the user has access to the agent
            has_access = False
            
            # Check if the user is the owner of the agent
            if agent_config.get("user_id") == user_id:
                has_access = True
            # Check if the agent belongs to a company the user has access to
            elif agent_config.get("company_id"):
                try:
                    user_company_response = (
                        supabase.table("user_companies")
                        .select("role_id")
                        .eq("user_id", user_id)
                        .eq("company_id", agent_config["company_id"])
                        .execute()
                    )
                    if user_company_response.data:
                        has_access = True
                except Exception as e:
                    raise InternalServerError(f"Error checking company access: {str(e)}")
            
            # If no access yet, check if the user's email has editor access
            if not has_access:
                # Get the user's email
                try:
                    user_response = (
                        supabase.table("users")
                        .select("email")
                        .eq("user_id", user_id)
                        .execute()
                    )
                except Exception as e:
                    raise InternalServerError(f"Error fetching user email: {str(e)}")
                
                if user_response.data:
                    user_email = user_response.data[0].get("email")
                    # Check if the email is in the share_editor_with list
                    if user_email in agent_config.get("share_editor_with", []):
                        has_access = True
            
            # If still no access, deny permission
            if not has_access:
                raise ForbiddenError(
                    "You don't have access to this agent",
                    additional_info={
                        "agent_id": agent_id
                    }
                )
        else:
            print("Fetching agent_config from database")
            # Get agent by agent_id
            try:
                agent_response = (
                    supabase.table("agents")
                    .select("agent_id, agent_name, description, agent_style, on_status, company_id, user_id, tools, share_editor_with")
                    .eq("agent_id", agent_id)
                    .eq("on_status", True)
                    .execute()
                )
            except Exception as e:
                raise InternalServerError(f"Error fetching agent: {str(e)}")
            
            if not agent_response.data:
                raise NotFoundError(
                    f"Agent with ID '{agent_id}' not found",
                    additional_info={
                        "agent_id": agent_id
                    }
                )
            
            agent_config = agent_response.data[0]
            
            # Check if the user has access to the agent
            has_access = False
            
            # Check if the user is the owner of the agent
            if agent_config.get("user_id") == user_id:
                has_access = True
            # Check if the agent belongs to a company the user has access to
            elif agent_config.get("company_id"):
                try:
                    user_company_response = (
                        supabase.table("user_companies")
                        .select("role_id")
                        .eq("user_id", user_id)
                        .eq("company_id", agent_config["company_id"])
                        .execute()
                    )
                    if user_company_response.data:
                        has_access = True
                except Exception as e:
                    raise InternalServerError(f"Error checking company access: {str(e)}")
            
            # If no access yet, check if the user's email has editor access
            if not has_access:
                # Get the user's email
                try:
                    user_response = (
                        supabase.table("users")
                        .select("email")
                        .eq("user_id", user_id)
                        .execute()
                    )
                except Exception as e:
                    raise InternalServerError(f"Error fetching user email: {str(e)}")
                
                if user_response.data:
                    user_email = user_response.data[0].get("email")
                    # Check if the email is in the share_editor_with list
                    if user_email in agent_config.get("share_editor_with", []):
                        has_access = True
            
            # If still no access, deny permission
            if not has_access:
                raise ForbiddenError(
                    "You don't have access to this agent",
                    additional_info={
                        "agent_id": agent_id
                    }
                )
        
        # Invoke the agent
        return StreamingResponse(
            agent_boilerplate.invoke_agent_stream(
                agent_id=agent_id,
                agent_input=agent_input,
                agent_config=agent_config
            ),
            media_type="text/event-stream"
        )
    
    except (BadRequestError, NotFoundError, ForbiddenError, InternalServerError) as e:
        # Re-raise known errors
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise InternalServerError(f"Unexpected error: {str(e)}")
    

@router.get("/{agent_id}/info")
async def get_agent_info(
    agent_id: str,
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Get information about an agent.
    
    This endpoint returns basic information about an agent, such as its name,
    description, and style. It does not include sensitive information.
    """
    try:
        # Get user_id from request state (set by middleware)
        user_id = request.state.user_id
        
        # Get agent by agent_id
        try:
            agent_response = (
                supabase.table("agents")
                .select("agent_id, agent_name, description, agent_style, on_status, company_id, user_id, tools, share_editor_with, share_visitor_with")
                .eq("agent_id", agent_id)
                .eq("on_status", True)
                .execute()
            )
        except Exception as e:
            raise InternalServerError(f"Error fetching agent: {str(e)}")
        
        if not agent_response.data:
            raise NotFoundError(
                f"Agent with ID '{agent_id}' not found",
                additional_info={
                    "agent_id": agent_id
                }
            )
        
        agent_config = agent_response.data[0]
        
        # Check if the user has access to the agent
        has_access = False
        access_level = "none"
        
        # Check if the user is the owner of the agent
        if agent_config.get("user_id") == user_id:
            has_access = True
            access_level = "owner"
        # Check if the agent belongs to a company the user has access to
        elif agent_config.get("company_id"):
            try:
                user_company_response = (
                    supabase.table("user_companies")
                    .select("role_id")
                    .eq("user_id", user_id)
                    .eq("company_id", agent_config["company_id"])
                    .execute()
                )
                if user_company_response.data:
                    has_access = True
                    access_level = "company"
            except Exception as e:
                raise InternalServerError(f"Error checking company access: {str(e)}")
        
        # If no access yet, check if the user's email has editor or visitor access
        if not has_access:
            # Get the user's email
            try:
                user_response = (
                    supabase.table("users")
                    .select("email")
                    .eq("user_id", user_id)
                    .execute()
                )
            except Exception as e:
                raise InternalServerError(f"Error fetching user email: {str(e)}")
            
            if user_response.data:
                user_email = user_response.data[0].get("email")
                # Check if the email is in the share_editor_with list
                if user_email in agent_config.get("share_editor_with", []):
                    has_access = True
                    access_level = "editor"
                # Check if the email is in the share_visitor_with list
                elif user_email in agent_config.get("share_visitor_with", []):
                    has_access = True
                    access_level = "visitor"
        
        # If still no access, deny permission
        if not has_access:
            raise ForbiddenError(
                "You don't have access to this agent",
                additional_info={
                    "agent_id": agent_id
                }
            )
        
        # Return basic information about the agent
        return {
            "agent_name": agent_config["agent_name"],
            "description": agent_config.get("description"),
            "agent_style": agent_config.get("agent_style"),
            "on_status": agent_config.get("on_status", True),
            "tools_count": len(agent_config.get("tools", [])),
            "access_level": access_level
        }
    
    except (BadRequestError, NotFoundError, ForbiddenError, InternalServerError) as e:
        # Re-raise known errors
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise InternalServerError(f"Unexpected error: {str(e)}")


@router.get("/shared-agent/{agent_hash}")
async def get_shared_agent(
    agent_hash: str,
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Get a shared agent by its public hash.
    This endpoint can be accessed without JWT authentication.
    """
    try:
        # Search for the agent with this hash
        try:
            # Look for agents with a matching public_hash
            print(f"Looking up agent with public_hash: {agent_hash}")
            agent_response = (
                supabase.table("agents")
                .select("agent_id, agent_name, description, agent_style, tools, share_editor_with, share_visitor_with, is_public, public_hash")
                .eq("public_hash", agent_hash)
                .eq("is_public", True)
                .execute()
            )
        except Exception as e:
            raise InternalServerError(f"Error fetching shared agent: {str(e)}")
        
        if not agent_response.data:
            print(f"No agent found for public_hash: {agent_hash}. agent_response: {agent_response}")
            # If we didn't find the agent in the agents table, return a not found error
            raise NotFoundError(f"Shared agent with hash '{agent_hash}' not found or not public")
        
        agent = agent_response.data[0]
        
        # Get tool details
        tool_details = []
        for tool_id in agent.get("tools", []):
            print(f"Looking up tool_id: {tool_id}")
            try:
                tool_response = (
                    supabase.table("tools_with_decrypted_keys")
                    .select("tool_id, name, description, versions")
                    .eq("tool_id", tool_id)
                    .execute()
                )
                print(f"tool_response for {tool_id}: {tool_response}")
            except Exception as e:
                print(f"Error fetching tool details for {tool_id}: {str(e)}")
                raise InternalServerError(f"Error fetching tool details: {str(e)}")
            
            if tool_response.data:
                tool_details.append(tool_response.data[0])
        
        # Fetch the latest chat_history from agent_logs
        chat_history = []
        try:
            log_response = (
                supabase.table("agent_logs")
                .select("chat_history")
                .eq("agent_id", agent["agent_id"])
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            if log_response.data and "chat_history" in log_response.data[0]:
                chat_history = log_response.data[0]["chat_history"] or []
        except Exception as e:
            print(f"Error fetching chat history for agent {agent['agent_id']}: {str(e)}")
            chat_history = []

        return {
            "agent_id": agent["agent_id"],
            "agent_name": agent["agent_name"],
            "description": agent.get("description"),
            "agent_style": agent.get("agent_style"),
            "on_status": agent.get("on_status", True),
            "company_id": agent.get("company_id"),
            "tools": tool_details,
            "chat_history": chat_history,
            "is_shared": True,
            "read_only": True
        }
    
    except (BadRequestError, NotFoundError, ForbiddenError, InternalServerError) as e:
        # Re-raise known errors
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise InternalServerError(f"Unexpected error: {str(e)}")

@router.get("/shared-thread/{thread_hash}")
async def get_shared_thread(
    thread_hash: str,
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Get a shared thread by its public hash.
    This endpoint can be accessed without JWT authentication.
    """
    try:
        # Search for the thread with this hash
        try:
            logs_response = (
                supabase.table("agent_logs")
                .select("agent_log_id, date, chat_history, agent_id")
                .eq(f"chat_history->0->share_info->>public_hash", thread_hash)
                .eq(f"chat_history->0->share_info->>is_public", "true") # JSON boolean true becomes text "true" via ->>
                .order("date", desc=True)  # Get the latest if multiple (should be unique by hash)
                .limit(1)
                .execute()
            )
        except Exception as e:
            raise InternalServerError(f"Error fetching shared thread: {str(e)}")
        
        if not logs_response.data:
            raise NotFoundError(f"Shared thread with hash '{thread_hash}' not found or not public")
        
        found_log = logs_response.data[0]
        chat_history_payload = found_log.get("chat_history")
        
        # Get agent info if available
        agent_info = {}
        if found_log.get("agent_id"):
            try:
                agent_response = (
                    supabase.table("agents")
                    .select("agent_id, agent_name, description, agent_style")
                    .eq("agent_id", found_log["agent_id"])
                    .execute()
                )
                
                if agent_response.data:
                    agent_info = {
                        "agent_id": agent_response.data[0]["agent_id"],
                        "agent_name": agent_response.data[0]["agent_name"],
                        "description": agent_response.data[0].get("description"),
                        "agent_style": agent_response.data[0].get("agent_style")
                    }
            except Exception as e:
                # Just log the error, we can still return the thread without agent info
                print(f"Error fetching agent info: {str(e)}")
        
        return {
            "agent_log_id": found_log.get("agent_log_id"),
            "date": found_log.get("date"),
            "chat_history": chat_history_payload,  # Return chat_history payload
            "agent": agent_info,
            "is_shared": True,
            "read_only": True
        }
    
    except (BadRequestError, NotFoundError, ForbiddenError, InternalServerError) as e:
        # Re-raise known errors
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise InternalServerError(f"Unexpected error: {str(e)}")
