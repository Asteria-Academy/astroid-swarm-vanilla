#!/usr/bin/env python3
"""
Integration test for get_llms.py with microservice modules.

This test verifies that get_llms.py can:
1. Connect to custom_vlm_model.py
2. Integrate with agent_boilerplate modules
3. Work with the agent invoke routes
"""

import sys
import os
import asyncio
from pathlib import Path

# Add microservice to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "microservice"))

def test_get_llms_basic():
    """Test 1: Basic get_llms functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic get_llms() Functionality")
    print("="*60)
    
    try:
        from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        
        print("Calling get_llms()...")
        llm = get_llms()
        
        print(f"✅ get_llms() returned successfully!")
        print(f"   LLM type: {llm._llm_type}")
        print(f"   Device: {llm.device}")
        print(f"   Model class: {type(llm).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ get_llms() test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_llms_with_model_name():
    """Test 2: get_llms with model_name parameter"""
    print("\n" + "="*60)
    print("TEST 2: get_llms() with model_name Parameter")
    print("="*60)
    
    try:
        from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        
        # Test with custom-vlm
        print("Calling get_llms('custom-vlm')...")
        llm1 = get_llms("custom-vlm")
        print(f"✅ get_llms('custom-vlm') returned: {type(llm1).__name__}")
        
        # Test with arbitrary model name (should still return custom VLM)
        print("Calling get_llms('gpt-4')...")
        llm2 = get_llms("gpt-4")
        print(f"✅ get_llms('gpt-4') returned: {type(llm2).__name__}")
        
        # Verify both return the same type
        if type(llm1).__name__ == type(llm2).__name__:
            print(f"✅ Both calls return the same model type: {type(llm1).__name__}")
            return True
        else:
            print(f"❌ Model types don't match: {type(llm1).__name__} vs {type(llm2).__name__}")
            return False
            
    except Exception as e:
        print(f"❌ get_llms with model_name test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_llms_with_agent_boilerplate():
    """Test 3: get_llms integration with agent_boilerplate"""
    print("\n" + "="*60)
    print("TEST 3: get_llms Integration with AgentBoilerplate")
    print("="*60)
    
    try:
        from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        from agent_boilerplate.boilerplate.agent_boilerplate import AgentBoilerplate
        
        print("Creating AgentBoilerplate instance...")
        boilerplate = AgentBoilerplate()
        print(f"✅ AgentBoilerplate created: {type(boilerplate).__name__}")
        
        print("Getting LLM via get_llms()...")
        llm = get_llms()
        print(f"✅ LLM retrieved: {type(llm).__name__}")
        
        print("Testing LLM _call method...")
        result = llm._call("Hello, test message")
        print(f"✅ LLM _call successful!")
        print(f"   Result: {result[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_llms_with_image():
    """Test 4: get_llms with image inference"""
    print("\n" + "="*60)
    print("TEST 4: get_llms with Image Inference")
    print("="*60)
    
    try:
        from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        
        llm = get_llms()
        
        # Test image path
        test_image = "/home/ubuntu/skripsi/indonesia/original/F002.jpg"
        
        if not os.path.exists(test_image):
            print(f"⚠️  Test image not found: {test_image}")
            print("   Skipping image inference test")
            return None
        
        print(f"Testing with image: {test_image}")
        prompt = "Describe this image"
        
        print(f"Calling invoke_with_image()...")
        caption = llm.invoke_with_image(test_image, prompt, max_new_tokens=64)
        
        print(f"✅ Image inference successful!")
        print(f"   Caption: {caption[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Image inference test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_llms_with_agent_invoke():
    """Test 5: get_llms with agent_invoke module"""
    print("\n" + "="*60)
    print("TEST 5: get_llms Integration with agent_invoke")
    print("="*60)
    
    try:
        from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        from agent_boilerplate.routes.agent_invoke import _maybe_handle_multimodal_and_augment
        from agent_boilerplate.boilerplate.models import AgentInput, AgentInputMessage, AgentInputMetadata, AgentInputConfig
        
        print("Creating test agent input...")
        test_image = "/home/ubuntu/skripsi/indonesia/original/F002.jpg"
        
        if not os.path.exists(test_image):
            print(f"⚠️  Test image not found: {test_image}")
            print("   Skipping agent_invoke test")
            return None
        
        agent_input = AgentInput(
            input=AgentInputMessage(messages="What is in this image?"),
            image_path=test_image,
            prompt_text="Describe the image",
            metadata=AgentInputMetadata(model_name="custom-vlm"),
            config=AgentInputConfig(configurable={})
        )
        
        print("Testing multimodal handling with get_llms integration...")
        
        # This should use get_llms internally
        result = asyncio.run(_maybe_handle_multimodal_and_augment(
            agent_input,
            max_new_tokens=64,
            model_name="custom-vlm"
        ))
        
        print(f"✅ Multimodal handling successful!")
        print(f"   Augmented input: {str(result.input.messages)[:100]}...")
        return True
    except Exception as e:
        print(f"❌ agent_invoke integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_llms_gpu_verification():
    """Test 6: Verify GPU usage through get_llms"""
    print("\n" + "="*60)
    print("TEST 6: GPU Verification via get_llms")
    print("="*60)
    
    try:
        import torch
        from agent_boilerplate.boilerplate.utils.get_llms import get_llms
        
        print("Getting LLM via get_llms()...")
        llm = get_llms()
        
        print(f"Device: {llm.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            is_on_gpu = next(llm.model.model_language.parameters()).is_cuda
            print(f"Model on GPU: {is_on_gpu}")
            
            if is_on_gpu:
                print("✅ Model is running on GPU!")
                return True
            else:
                print("❌ Model is NOT on GPU")
                return False
        else:
            print("⚠️  CUDA not available, model running on CPU")
            return None
            
    except Exception as e:
        print(f"❌ GPU verification test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    test_names = [
        "Basic get_llms() Functionality",
        "get_llms() with model_name Parameter",
        "get_llms Integration with AgentBoilerplate",
        "get_llms with Image Inference",
        "get_llms Integration with agent_invoke",
        "GPU Verification via get_llms"
    ]
    
    for name, result in zip(test_names, results):
        status = "✅ PASS" if result is True else ("⚠️  SKIP" if result is None else "❌ FAIL")
        print(f"{status} - {name}")
    
    passed = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r is None)
    failed = sum(1 for r in results if r is False)
    
    print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GET_LLMS INTEGRATION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Basic get_llms
    results.append(test_get_llms_basic())
    
    # Test 2: get_llms with model_name
    results.append(test_get_llms_with_model_name())
    
    # Test 3: Integration with AgentBoilerplate
    results.append(test_get_llms_with_agent_boilerplate())
    
    # Test 4: Image inference
    results.append(test_get_llms_with_image())
    
    # Test 5: Integration with agent_invoke
    results.append(await asyncio.create_task(asyncio.to_thread(test_get_llms_with_agent_invoke)))
    
    # Test 6: GPU verification
    results.append(test_get_llms_gpu_verification())
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
