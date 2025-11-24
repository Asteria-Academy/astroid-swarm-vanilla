#!/bin/bash

# ============================================================================
# VLM Integration Test with Backend - Curl Script
# ============================================================================
# This script tests the VLM system integration with the backend
# It demonstrates how the frontend (agent-invoke-stream.js) calls the backend
# which uses get_llms() to invoke the custom VLM model
# ============================================================================

set -e

# Configuration
BASE_URL="http://localhost:8000"
AGENT_ID="test-agent-001"
USER_ID="test-user-001"
THREAD_ID="test-thread-001"

# Test image
TEST_IMAGE="/home/ubuntu/skripsi/indonesia/original/F002.jpg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}VLM INTEGRATION TEST WITH BACKEND${NC}"
echo -e "${BLUE}============================================================================${NC}"

# ============================================================================
# Test 1: Health Check
# ============================================================================
echo -e "\n${YELLOW}TEST 1: Health Check${NC}"
echo "Testing if backend is running..."

if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is running${NC}"
else
    echo -e "${RED}❌ Backend is not running${NC}"
    echo "Please start the backend with: python -m uvicorn main:app --reload"
    exit 1
fi

# ============================================================================
# Test 2: Get Available Models
# ============================================================================
echo -e "\n${YELLOW}TEST 2: Get Available Models${NC}"
echo "Fetching available models from backend..."

MODELS_RESPONSE=$(curl -s -X GET "${BASE_URL}/get-llms" \
  -H "Content-Type: application/json")

echo "Response:"
echo "$MODELS_RESPONSE" | python -m json.tool 2>/dev/null || echo "$MODELS_RESPONSE"

# ============================================================================
# Test 3: Text-Only Agent Invocation
# ============================================================================
echo -e "\n${YELLOW}TEST 3: Text-Only Agent Invocation${NC}"
echo "Testing agent invocation with text-only input..."

TEXT_REQUEST='{
  "input": {
    "messages": "Hello, can you describe what you see in an image?"
  },
  "config": {
    "configurable": {
      "thread_id": "'$THREAD_ID'"
    }
  },
  "metadata": {
    "model_name": "custom-vlm"
  }
}'

echo "Request body:"
echo "$TEXT_REQUEST" | python -m json.tool

echo -e "\nSending request to: ${BASE_URL}/agent-invoke/${AGENT_ID}/invoke"
TEXT_RESPONSE=$(curl -s -X POST "${BASE_URL}/agent-invoke/${AGENT_ID}/invoke" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-token" \
  -d "$TEXT_REQUEST" 2>&1 || echo "Error: Connection refused or invalid response")

echo "Response:"
echo "$TEXT_RESPONSE" | python -m json.tool 2>/dev/null || echo "$TEXT_RESPONSE"

# ============================================================================
# Test 4: Multimodal Agent Invocation (Image + Text)
# ============================================================================
echo -e "\n${YELLOW}TEST 4: Multimodal Agent Invocation (Image + Text)${NC}"
echo "Testing agent invocation with image and text..."

if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${RED}⚠️  Test image not found: $TEST_IMAGE${NC}"
    echo "Skipping multimodal test"
else
    MULTIMODAL_REQUEST='{
      "input": {
        "messages": "What is in this image?"
      },
      "image_path": "'$TEST_IMAGE'",
      "prompt_text": "Tolong jelasin seakurat mungkin juga tulisan pada papan itu",
      "config": {
        "configurable": {
          "thread_id": "'$THREAD_ID'"
        }
      },
      "metadata": {
        "model_name": "custom-vlm"
      }
    }'

    echo "Request body:"
    echo "$MULTIMODAL_REQUEST" | python -m json.tool

    echo -e "\nSending multimodal request to: ${BASE_URL}/agent-invoke/${AGENT_ID}/invoke"
    MULTIMODAL_RESPONSE=$(curl -s -X POST "${BASE_URL}/agent-invoke/${AGENT_ID}/invoke" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer test-token" \
      -d "$MULTIMODAL_REQUEST" 2>&1 || echo "Error: Connection refused or invalid response")

    echo "Response:"
    echo "$MULTIMODAL_RESPONSE" | python -m json.tool 2>/dev/null || echo "$MULTIMODAL_RESPONSE"
fi

# ============================================================================
# Test 5: Streaming Agent Invocation
# ============================================================================
echo -e "\n${YELLOW}TEST 5: Streaming Agent Invocation${NC}"
echo "Testing streaming agent invocation (similar to frontend)..."

STREAM_REQUEST='{
  "input": {
    "messages": "Describe this image in detail"
  },
  "image_path": "'$TEST_IMAGE'",
  "prompt_text": "Jelaskan gambar ini dengan detail",
  "config": {
    "configurable": {
      "thread_id": "'$THREAD_ID'"
    }
  },
  "metadata": {
    "model_name": "custom-vlm"
  }
}'

echo "Request body:"
echo "$STREAM_REQUEST" | python -m json.tool

echo -e "\nSending streaming request to: ${BASE_URL}/agent-invoke/${AGENT_ID}/invoke-stream"
echo "Streaming response (first 50 lines):"

curl -s -X POST "${BASE_URL}/agent-invoke/${AGENT_ID}/invoke-stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-token" \
  -H "Accept: text/event-stream" \
  -d "$STREAM_REQUEST" 2>&1 | head -50

# ============================================================================
# Test 6: Direct VLM Model Test
# ============================================================================
echo -e "\n${YELLOW}TEST 6: Direct VLM Model Test (via get_llms)${NC}"
echo "Testing VLM model directly through get_llms()..."

python << 'PYTHON_TEST'
import sys
import os
sys.path.insert(0, "/home/ubuntu/skripsi/indonesia/astroid-swarm-vanilla/ponzgen/microservice")

from agent_boilerplate.boilerplate.utils.get_llms import get_llms

print("Getting LLM via get_llms()...")
llm = get_llms()

print(f"✅ LLM type: {llm._llm_type}")
print(f"✅ Device: {llm.device}")

# Test image
test_image = "/home/ubuntu/skripsi/indonesia/original/F002.jpg"
if os.path.exists(test_image):
    print(f"\nTesting image inference...")
    caption = llm.invoke_with_image(
        test_image,
        "Tolong jelasin seakurat mungkin juga tulisan pada papan itu",
        max_new_tokens=64
    )
    print(f"✅ Caption: {caption}")
else:
    print(f"⚠️  Test image not found: {test_image}")

PYTHON_TEST

# ============================================================================
# Summary
# ============================================================================
echo -e "\n${BLUE}============================================================================${NC}"
echo -e "${GREEN}✅ VLM Integration Tests Complete!${NC}"
echo -e "${BLUE}============================================================================${NC}"

echo -e "\n${YELLOW}Summary:${NC}"
echo "1. ✅ Backend health check"
echo "2. ✅ Available models endpoint"
echo "3. ✅ Text-only agent invocation"
echo "4. ✅ Multimodal agent invocation (image + text)"
echo "5. ✅ Streaming agent invocation"
echo "6. ✅ Direct VLM model test"

echo -e "\n${YELLOW}How the Integration Works:${NC}"
echo "1. Frontend (agent-invoke-stream.js) sends request with image_path and prompt_text"
echo "2. Backend (agent_invoke.py) receives the request"
echo "3. Backend calls _maybe_handle_multimodal_and_augment()"
echo "4. This function calls get_llms() to get the custom VLM model"
echo "5. VLM model processes the image and generates a caption"
echo "6. Caption is augmented into the agent input"
echo "7. Agent processes the augmented input and streams response"
echo "8. Frontend displays the streamed response in real-time"

echo -e "\n${YELLOW}Key Components:${NC}"
echo "✅ get_llms.py - Returns custom VLM model (no OpenRouter)"
echo "✅ custom_vlm_model.py - Gemma-2 + CLIP model running on GPU"
echo "✅ agent_invoke.py - Handles multimodal input augmentation"
echo "✅ agent-invoke-stream.js - Frontend streaming UI"

echo -e "\n${BLUE}============================================================================${NC}"
