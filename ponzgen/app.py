"""
Combined Agent API and Boilerplate Service

This application combines the functionality of both the agent backend and agent boilerplate microservices.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
import logging
import os
import sys
import json
import subprocess
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path
from utils.logflare_utils import logflare_logger

# Import error handling from boilerplate
from microservice.agent_boilerplate.boilerplate.errors import (
    APIError, BadRequestError, NotFoundError, ValidationError, 
    InternalServerError, handle_pydantic_validation_error, ERROR_RESPONSES
)

# Import auth middleware
from auth_middleware import AuthMiddleware

# Import routes from agent_backend microservice
from microservice.mcp_tools.routes.tools import router as tools_router
from microservice.agent_backend.routes.agents import router as agents_router
from microservice.agent_backend.routes.agent_logs import router as agent_logs_router
from microservice.feature_sharing.routes.feature_sharing import router as feature_sharing_router
from microservice.agent_backend.routes.companies import router as companies_router
from microservice.agent_backend.routes.roles import router as roles_router, initialize_roles

# Import routes from agent_boilerplate microservice
from microservice.agent_boilerplate.routes.agent_invoke import router as agent_invoke_router
from microservice.agent_boilerplate.routes.agent_api import router as agent_api_router

# Import routes from mcp_tools microservice
from microservice.mcp_tools.routes.mcp_tools import router as mcp_tools_router

# Import routes from agent_field_autofill microservice
from microservice.agent_field_autofill.routes.autofill import router as agent_field_autofill_router

# Import routes from agent_creator microservice
from microservice.agent_creator.routes.user_input_routes import router as agent_creator_user_input_router
from microservice.agent_creator.routes.autofill import router as agent_creator_autofill_router

# Import routes from chat_recommendation microservice
from microservice.chat_recommendation.routes.chat_recommendation_routes import router as chat_recommendation_router

# Import routes from avatar_bucket microservice
from microservice.avatar_bucket.routes.avatars import router as avatars_router

# Import routes from rag microservice
from microservice.rag.routes.rag import router as rag_router

# Import routes from sendgrid_webhook microservice
from microservice.sendgrid_webhook.routes.webhook import router as sendgrid_webhook_router

# Add microservice directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "microservice"))

# Load environment variables from .env file
load_dotenv()

# Configure logging
import logging
from utils.logflare_utils import logflare_logger

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Logflare handler
logflare_logger.setup_logflare_handler()

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-anon-key")

# Define base directory and default LLM models
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = ["gpt-3.5-turbo", "gpt-4"]

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Create FastAPI app
app = FastAPI(
    title="Combined Agent API",
    description="API for managing agents, tools, companies, logs, and agent invocation",
    version="1.0.0",
    trust_remote_host=True
)


# Store Supabase client in app state
app.state.supabase = supabase

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    # allow_origins=["http://localhost:8008", "http://127.0.0.1:8008"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)

# Add authentication middleware
app.add_middleware(AuthMiddleware, supabase_client=supabase)

# Include all routers
ROUTERS = [
    tools_router,
    agents_router,
    agent_logs_router,
    feature_sharing_router,
    companies_router,
    roles_router,
    agent_invoke_router,
    agent_api_router,
    mcp_tools_router,
    agent_field_autofill_router,
    agent_creator_user_input_router,
    agent_creator_autofill_router,
    chat_recommendation_router,
    avatars_router,
    rag_router,
    sendgrid_webhook_router
]

for router in ROUTERS:
    app.include_router(router)

# Public endpoints
@app.get("/public", tags=["public"])
def public_route():
    return {"message": "This is a public route"}

@app.get("/health", tags=["public"])
def health_check():
    return {"status": "ok"}

@app.get("/mcp-logs", tags=["public"])
def get_mcp_logs():
    """
    Get all MCP log files and their contents.
    This endpoint is public and doesn't require authentication.
    """
    try:
        # Get runner directory from environment variable or use default paths
        runner_dir_env = os.getenv("MCP_RUNNER_DIR")
        if runner_dir_env:
            logs_dir = Path(runner_dir_env) / "logs"
        else:
            # Try different possible locations
            possible_dirs = [
                Path("/home/runner_files/logs"),  # Original path
                Path("./runner_files/logs"),      # Relative to current directory
                Path("/tmp/runner_files/logs"),   # /tmp is usually writable in most environments
                Path(os.path.expanduser("~/runner_files/logs"))  # User's home directory
            ]
            
            # Find first existing directory
            for dir_path in possible_dirs:
                if dir_path.exists():
                    logs_dir = dir_path
                    break
            else:
                # If no directory exists, use the original path
                logs_dir = Path("/home/runner_files/logs")
        
        if not logs_dir.exists():
            return {
                "status": "success",
                "message": f"No MCP logs directory found at {logs_dir}",
                "logs": []
            }
        
        log_files = []
        
        # Get all .log files in the logs directory
        for log_file in logs_dir.glob("*.log"):
            try:
                # Read file content
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get file stats
                stat = log_file.stat()
                
                log_files.append({
                    "filename": log_file.name,
                    "path": str(log_file),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "content": content
                })
            except Exception as e:
                # If we can't read a specific file, include error info
                log_files.append({
                    "filename": log_file.name,
                    "path": str(log_file),
                    "error": f"Could not read file: {str(e)}",
                    "content": None
                })
        
        return {
            "status": "success",
            "total_files": len(log_files),
            "logs": log_files
        }
        
    except Exception as e:
        logger.error(f"Error reading MCP logs: {e}")
        return {
            "status": "error",
            "message": f"Error reading MCP logs: {str(e)}",
            "logs": []
        }

# User info route
@app.get("/user/info", tags=["user"])
def get_user_info(request: Request):
    return request.state.user  # Return user details as JSON

# Endpoint: list available LLM models
@app.get("/get-llms", tags=["LLMs"])
async def list_models():
    """
    Get a list of all available language models.
    """
    try:
        content = (BASE_DIR / "config" / "openrouter_llm_list.json").read_text()
        data = json.loads(content)
    except Exception as e:
        logger.warning("Unable to load models file: %s; using defaults", e)
        data = {"available_models": DEFAULT_MODELS}
    models = sorted(data.get("available_models", DEFAULT_MODELS))
    return {"available_models": models}

# Startup event: initialize roles and refresh tools
@app.on_event("startup")
async def startup_event():
    # from microservice.mcp_tools.routes.mcp_tools import refresh_tools
    
    # logger.info("Initializing roles...")
    # await initialize_roles(supabase)
    
    # logger.info("Refreshing MCP tools...")
    # try:
    #     result = await refresh_tools()
    #     logger.info("MCP tools refreshed successfully: %s tools", result.data['active_tools'])
    # except Exception as e:
    #     logger.error("Error refreshing MCP tools: %s", e)
    
    # logger.info("Application startup complete")
    # Launch tool status checker
    # subprocess.Popen([sys.executable, "./microservice/mcp_tools/utils/_check_tools_status.py"] )

    # Set MCP environment variables if not already set
    # Set MCP_RUNNER_DIR
    if "MCP_RUNNER_DIR" not in os.environ:
        # Use a fixed directory path
        runner_dir = os.path.join(os.getcwd(), "microservice", "mcp_2", "runner_files")
        os.makedirs(runner_dir, exist_ok=True)
        os.makedirs(os.path.join(runner_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(runner_dir, "envs"), exist_ok=True)
        # Ensure directory has proper permissions
        os.chmod(runner_dir, 0o777)
        os.environ["MCP_RUNNER_DIR"] = runner_dir
        logger.info(f"Set MCP_RUNNER_DIR to {runner_dir}")
    # Start MCP auto manager
    subprocess.Popen([sys.executable, "./microservice/mcp_2/mcp_auto_manager.py"])
    logger.info("Started MCP auto manager")

# Exception handlers
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle all custom API exceptions with standardized format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI validation errors"""
    return JSONResponse(
        status_code=422,
        content=handle_pydantic_validation_error(exc).detail
    )

@app.exception_handler(PydanticValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: PydanticValidationError):
    """Handle Pydantic validation errors"""
    return JSONResponse(
        status_code=422,
        content=handle_pydantic_validation_error(exc).detail
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions"""
    # Log the error here
    logger.error("Unhandled exception: %s", exc)
    error = InternalServerError(f"An unexpected error occurred")
    return JSONResponse(
        status_code=error.status_code,
        content=error.detail
    )



if __name__ == "__main__":
    # Test logging
    logflare_logger.log_event("application_startup", {
        "message": "Application starting up",
        "environment": os.getenv("ENVIRONMENT", "development")
    })
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
