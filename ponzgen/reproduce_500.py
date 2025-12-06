
import sys
import os
from pydantic import BaseModel

# Mock classes to simulate models.py
class AgentInputMessage(BaseModel):
    messages: str
    context: str = ""

class AgentInputMetadata(BaseModel):
    model_name: str = "gpt-3.5-turbo"

class AgentInput(BaseModel):
    input: AgentInputMessage
    metadata: AgentInputMetadata

try:
    # Simulate parsing
    data = '{"input": {"messages": "test"}, "metadata": {"model_name": "custom-vlm"}}'
    agent_input = AgentInput.parse_raw(data)
    print("Parsed agent_input:", agent_input)

    # Simulate app.py logic
    file_path = "/tmp/test.png"
    
    print("Attempting setattr...")
    if isinstance(agent_input.input, dict):
        agent_input.input['image_path'] = file_path
    else:
        # This is where we suspect it might fail or behave unexpectedly
        setattr(agent_input.input, 'image_path', file_path)
    
    print("setattr success!")
    print("Has image_path:", getattr(agent_input.input, 'image_path', 'Not Found'))
    
    # Simulate custom_vlm_model.py logic
    print("Checking getattr in helper logic...")
    image_path = getattr(agent_input.input, 'image_path', None)
    print("Retrieved image_path:", image_path)

except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
