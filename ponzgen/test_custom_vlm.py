# #!/usr/bin/env python3
# """
# Test script for custom VLM integration with ponzgen boilerplate.

# This script tests:
# 1. Model loading
# 2. Image inference
# 3. LangChain wrapper functionality
# 4. Integration with agent pipeline
# """

# import sys
# import os
# import asyncio
# from pathlib import Path

# # Add microservice to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "microservice"))

# def test_model_loading():
#     """Test 1: Model loading"""
#     print("\n" + "="*60)
#     print("TEST 1: Model Loading")
#     print("="*60)
    
#     try:
#         import torch
#         from agent_boilerplate.boilerplate.utils.custom_vlm_model import get_custom_vlm_model
        
#         print("Loading custom VLM model...")
#         model = get_custom_vlm_model()
#         print("✅ Model loaded successfully!")
#         print(f"   Model type: {model._llm_type}")
#         print(f"   Device: {model.device}")
#         print(f"   CUDA available: {torch.cuda.is_available()}")
#         if torch.cuda.is_available():
#             print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
#             print(f"   Model on GPU: {next(model.model.model_language.parameters()).is_cuda}")
#         return True
#     except Exception as e:
#         print(f"❌ Model loading failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False


# def test_image_inference():
#     """Test 2: Image inference"""
#     print("\n" + "="*60)
#     print("TEST 2: Image Inference")
#     print("="*60)
    
#     try:
#         import torch
#         from agent_boilerplate.boilerplate.utils.custom_vlm_model import get_custom_vlm_model
        
#         model = get_custom_vlm_model()
        
#         # Test image path
#         test_image = "/home/ubuntu/skripsi/indonesia/original/F002.jpg"
        
#         if not os.path.exists(test_image):
#             print(f"⚠️  Test image not found: {test_image}")
#             print("   Skipping image inference test")
#             return None
        
#         print(f"Testing with image: {test_image}")
#         print(f"Device: {model.device}")
#         print(f"Model on GPU: {next(model.model.model_language.parameters()).is_cuda}")
        
#         prompt = "Describe this image in detail"
        
#         print(f"Prompt: {prompt}")
#         print("Generating caption...")
        
#         caption = model.invoke_with_image(test_image, prompt, max_new_tokens=64)
        
#         print(f"✅ Caption generated successfully!")
#         print(f"   Caption: {caption}")
#         return True
#     except Exception as e:
#         print(f"❌ Image inference failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False


# def test_langchain_wrapper():
#     """Test 3: LangChain wrapper"""
#     print("\n" + "="*60)
#     print("TEST 3: LangChain Wrapper")
#     print("="*60)
    
#     try:
#         from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        
#         print("Getting custom VLM via get_llms()...")
#         llm = get_llms("custom-vlm", temperature=0.7)
        
#         print(f"✅ LLM retrieved successfully!")
#         print(f"   LLM type: {llm._llm_type}")
        
#         # Test text-only call
#         print("\nTesting text-only invocation...")
#         result = llm._call("Hello, how are you?")
#         print(f"✅ Text invocation successful!")
#         print(f"   Result: {result}")
        
#         return True
#     except Exception as e:
#         print(f"❌ LangChain wrapper test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False


# def test_model_selection():
#     """Test 4: Model selection logic"""
#     print("\n" + "="*60)
#     print("TEST 4: Model Selection Logic")
#     print("="*60)
    
#     try:
#         from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        
#         # Test custom VLM selection
#         print("Testing custom VLM selection...")
#         vlm = get_llms("custom-vlm")
#         print(f"✅ Custom VLM selected: {vlm._llm_type}")
        
#         # Test OpenRouter selection
#         print("\nTesting OpenRouter selection...")
#         openai_llm = get_llms("gpt-3.5-turbo")
#         print(f"✅ OpenRouter selected: {type(openai_llm).__name__}")
        
#         return True
#     except Exception as e:
#         print(f"❌ Model selection test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False


# async def test_multimodal_handling():
#     """Test 5: Multimodal input handling"""
#     print("\n" + "="*60)
#     print("TEST 5: Multimodal Input Handling")
#     print("="*60)
    
#     try:
#         from agent_boilerplate.routes.agent_invoke import _maybe_handle_multimodal_and_augment
#         from agent_boilerplate.boilerplate.models import AgentInput, AgentInputMessage, AgentInputMetadata, AgentInputConfig
        
#         # Create test input
#         test_image = "/home/ubuntu/skripsi/indonesia/original/F002.jpg"
        
#         if not os.path.exists(test_image):
#             print(f"⚠️  Test image not found: {test_image}")
#             print("   Skipping multimodal handling test")
#             return None
        
#         print(f"Creating test agent input with image: {test_image}")
        
#         agent_input = AgentInput(
#             input=AgentInputMessage(messages="What is in this image?"),
#             image_path=test_image,
#             prompt_text="Describe the image",
#             metadata=AgentInputMetadata(model_name="custom-vlm"),
#             config=AgentInputConfig(configurable={})
#         )
        
#         print("Processing multimodal input...")
#         result = await _maybe_handle_multimodal_and_augment(
#             agent_input,
#             max_new_tokens=64,
#             model_name="custom-vlm"
#         )
        
#         print(f"✅ Multimodal handling successful!")
#         print(f"   Augmented input: {result.input.messages[:100]}...")
        
#         return True
#     except Exception as e:
#         print(f"❌ Multimodal handling test failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False


# def print_summary(results):
#     """Print test summary"""
#     print("\n" + "="*60)
#     print("TEST SUMMARY")
#     print("="*60)
    
#     test_names = [
#         "Model Loading",
#         "Image Inference",
#         "LangChain Wrapper",
#         "Model Selection",
#         "Multimodal Handling"
#     ]
    
#     for name, result in zip(test_names, results):
#         status = "✅ PASS" if result is True else ("⚠️  SKIP" if result is None else "❌ FAIL")
#         print(f"{status} - {name}")
    
#     passed = sum(1 for r in results if r is True)
#     skipped = sum(1 for r in results if r is None)
#     failed = sum(1 for r in results if r is False)
    
#     print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")
    
#     if failed == 0:
#         print("\n🎉 All tests passed!")
#         return True
#     else:
#         print(f"\n⚠️  {failed} test(s) failed")
#         return False


# async def main():
#     """Run all tests"""
#     print("\n" + "="*60)
#     print("CUSTOM VLM INTEGRATION TEST SUITE")
#     print("="*60)
    
#     results = []
    
#     # Test 1: Model loading
#     results.append(test_model_loading())
    
#     # Test 2: Image inference
#     results.append(test_image_inference())
    
#     # Test 3: LangChain wrapper
#     results.append(test_langchain_wrapper())
    
#     # Test 4: Model selection
#     results.append(test_model_selection())
    
#     # Test 5: Multimodal handling
#     results.append(await test_multimodal_handling())
    
#     # Print summary
#     success = print_summary(results)
    
#     return 0 if success else 1


# if __name__ == "__main__":
#     exit_code = asyncio.run(main())
#     sys.exit(exit_code)


import sys
from pathlib import Path
from PIL import Image

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from microservice.agent_boilerplate.boilerplate.utils.custom_vlm_model import get_custom_vlm_model

def test_image_captioning(image_path, prompt="Describe this image in detail"):
    try:
        print(f"Loading VLM model...")
        vlm = get_custom_vlm_model()
        
        print(f"\nGenerating caption for: {image_path}")
        caption = vlm.invoke_with_image(image_path, prompt)
        
        print("\n" + "="*50)
        print("IMAGE CAPTIONING RESULT:")
        print("="*50)
        print(f"Prompt: {prompt}")
        print("-"*50)
        print("Generated Caption:")
        print(caption)
        print("="*50)
        
    except Exception as e:
        print(f"Error during image captioning: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    image_path = "/home/ubuntu/skripsi/indonesia/original/F002.jpg"
    test_image_captioning(image_path)