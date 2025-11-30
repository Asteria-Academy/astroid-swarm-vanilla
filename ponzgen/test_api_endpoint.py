#!/usr/bin/env python3
"""
Simple FastAPI test to verify get_llms integration with HTTP endpoints.

This creates a minimal FastAPI app to test the LLM endpoints.
"""

import sys
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add microservice to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "microservice"))

from agent_boilerplate.boilerplate.utils.get_llms import get_llms

# Create FastAPI app
app = FastAPI(title="Custom VLM API Test")

# Request/Response models
class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_path: str
    prompt: str

class TextResponse(BaseModel):
    response: str
    model_type: str
    device: str

class ImageResponse(BaseModel):
    caption: str
    model_type: str
    device: str

# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        llm = get_llms()
        return {
            "status": "healthy",
            "model_type": llm._llm_type,
            "device": llm.device,
            "model_class": type(llm).__name__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/text", response_model=TextResponse)
async def process_text(request: TextRequest):
    """Process text with the LLM"""
    try:
        llm = get_llms()
        response = llm._call(request.text)
        return TextResponse(
            response=response,
            model_type=llm._llm_type,
            device=llm.device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@app.post("/image", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    """Process image with the VLM"""
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
        
        llm = get_llms()
        caption = llm.invoke_with_image(request.image_path, request.prompt, max_new_tokens=64)
        
        return ImageResponse(
            caption=caption,
            model_type=llm._llm_type,
            device=llm.device
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    try:
        llm = get_llms()
        return {
            "model_type": llm._llm_type,
            "device": llm.device,
            "model_class": type(llm).__name__,
            "has_model": llm.model is not None,
            "model_path": llm.model_path,
            "base_dir": llm.base_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("CUSTOM VLM API TEST SERVER")
    print("="*60)
    print("\nStarting FastAPI server on http://localhost:8000")
    print("\nAvailable endpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /model-info      - Get model information")
    print("  POST /text            - Process text")
    print("  POST /image           - Process image")
    print("\nExample curl commands:")
    print("  curl http://localhost:8000/health")
    print("  curl http://localhost:8000/model-info")
    print("  curl -X POST http://localhost:8000/text -H 'Content-Type: application/json' -d '{\"text\": \"Hello\"}'")
    print("  curl -X POST http://localhost:8000/image -H 'Content-Type: application/json' -d '{\"image_path\": \"/path/to/image.jpg\", \"prompt\": \"Describe this\"}'")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
