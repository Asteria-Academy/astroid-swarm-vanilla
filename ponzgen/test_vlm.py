import sys
import os

# Add the project root to the python path
sys.path.append("/home/lyfesan/development/git-repo/astroid-swarm-vanilla/ponzgen")

try:
    from microservice.agent_boilerplate.boilerplate.utils.custom_vlm_model import get_custom_vlm_model
    print("Import successful. Attempting to load model...")
    vlm = get_custom_vlm_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
