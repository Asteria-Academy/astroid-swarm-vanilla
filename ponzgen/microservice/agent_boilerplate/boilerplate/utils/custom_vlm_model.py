"""
Custom VLM Model Wrapper for LangChain Integration

This module wraps the Gemma-2 + CLIP vision model with LangChain's BaseLLM interface,
enabling it to work seamlessly with the agent boilerplate.
"""

import torch
import torch.nn as nn
import warnings
import os
from typing import Any, List, Optional
from PIL import Image
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    AutoProcessor,
)
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


# ===============================================================
# CONFIGURATION
# ===============================================================

BASE_DIR = "/home/ubuntu/skripsi/indonesia"
MODEL_PATH = os.path.join(BASE_DIR, "gemma_adam_consine_model.pt")
IMAGE_DIR = os.path.join(BASE_DIR, "original")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
GEMMA_MODEL_ID = "google/gemma-2-2b-it"
NUM_VIS_TOKEN = 50
DUMMY_IMG_TOKEN = (" ".join(["the"] * NUM_VIS_TOKEN)).strip()
TRIGGER_STR = "<start_image>"


# ===============================================================
# MODEL ARCHITECTURE (From your training script)
# ===============================================================

class MyAdaptor(nn.Module):
    """Adapter to project vision embeddings to language model space."""
    
    def __init__(self, vis_token_embedding_size, word_embedding_size):
        super(MyAdaptor, self).__init__()
        self.vis_token_embedding_size = vis_token_embedding_size
        self.word_embedding_size = word_embedding_size
        self.adapter_mlp = nn.Sequential(
            nn.Linear(self.vis_token_embedding_size, self.word_embedding_size),
            nn.GELU(),
            nn.Linear(self.word_embedding_size, self.word_embedding_size)
        )

    def forward(self, img_output):
        img_embed = self.adapter_mlp(img_output)
        return img_embed


class MyModel(nn.Module):
    """Custom VLM combining Gemma-2 language model with CLIP vision model."""
    
    def __init__(self):
        super(MyModel, self).__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_language = AutoModelForCausalLM.from_pretrained(
                GEMMA_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto" if DEVICE == "cuda" else None
            )
        self.tokenizer_language = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, padding_side='right')
        self.image_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID).image_processor
        self.model_image = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

        self.word_embedding_size = self.model_language.config.hidden_size
        self.num_vocab = self.model_language.config.vocab_size
        self.trigger_str_img = TRIGGER_STR
        self.num_vis_token_summary = NUM_VIS_TOKEN
        self.vis_token_embedding_size = self.model_image.config.hidden_size
        self.adaptor = MyAdaptor(self.vis_token_embedding_size, self.word_embedding_size)
        self.dummy_img_token = (" ".join(["the"] * self.num_vis_token_summary)).strip()

    def search_trigger_idx(self, text_token, trigger_str):
        """Find the position of trigger string in tokenized text."""
        all_token = text_token
        all_string_now = ""
        all_token_now = []
        dummy_start_token = None
        for token_idx in range(len(all_token)):
            token_now = int(all_token[token_idx].detach().cpu().numpy())
            all_token_now.append(token_now)
            token_as_string = self.tokenizer_language.batch_decode(
                [all_token_now],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            if trigger_str in token_as_string:
                dummy_start_token = token_idx + 1
                break
        return dummy_start_token

    def get_image_embed(self, image_input):
        """Extract and adapt image embeddings."""
        image_input_float = image_input.to(DEVICE, dtype=self.model_image.dtype)
        img_output = self.model_image(image_input_float)['last_hidden_state']
        img_output_bfloat16 = img_output.to(torch.bfloat16)
        img_embed = self.adaptor(img_output_bfloat16)
        return img_embed

    def split_and_replace(self, now_input_tokens, replacement_embed, start_loc):
        """Replace tokens at a specific location with embeddings."""
        num_token = len(replacement_embed)
        start_embed = now_input_tokens[0:start_loc]
        end_embed = now_input_tokens[start_loc + num_token:]
        replaced_embed = torch.cat((start_embed, replacement_embed.to(now_input_tokens.dtype), end_embed), 0)
        return replaced_embed

    def generate_answer_image(self, prompt_text: str, pil_image: Image.Image, max_new_tokens=64):
        """Generate answer given an image and prompt text."""
        instruction_now = "<start_of_turn>user\n"
        instruction_now += f"<start_image> {self.dummy_img_token}\n<end_image>\n"
        instruction_now += f"{prompt_text}\n<end_of_turn>\n<start_of_turn>model\n"

        prompt_tokens = self.tokenizer_language([instruction_now], padding=False, return_tensors="pt")
        prompt_tokens = {k: v.to(self.model_language.device) for k, v in prompt_tokens.items()}

        prompt_embeds = self.model_language.model.embed_tokens(prompt_tokens['input_ids'])

        image_input = self.image_processor([pil_image], return_tensors="pt")['pixel_values']
        img_embed = self.get_image_embed(image_input)

        tokens_text_now = prompt_tokens['input_ids'][0].detach().cpu()
        dummy_location = self.search_trigger_idx(tokens_text_now, self.trigger_str_img)

        if dummy_location is None:
            print("WARNING: Could not find trigger string in prompt.")
            return ""

        replaced_embeds = self.split_and_replace(prompt_embeds[0], img_embed[0], dummy_location)
        replaced_embeds = replaced_embeds.unsqueeze(0)

        output_now = self.model_language.generate(
            inputs_embeds=replaced_embeds,
            attention_mask=prompt_tokens['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer_language.eos_token_id
        )

        output_string = self.tokenizer_language.decode(
            output_now[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_string.strip()


# ===============================================================
# LANGCHAIN LLM WRAPPER
# ===============================================================

class CustomVLMLLM(LLM):
    """
    LangChain LLM wrapper for custom Gemma-2 + CLIP VLM model.
    
    This class integrates the custom vision-language model with LangChain's
    agent framework, supporting both text-only and multimodal (image+text) inputs.
    """

    model: MyModel = None
    device: str = DEVICE
    model_path: str = MODEL_PATH
    base_dir: str = BASE_DIR

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom_vlm"

    def __init__(self, **kwargs):
        """Initialize the custom VLM LLM."""
        super().__init__(**kwargs)
        if self.model is None:
            self._load_model()

    def _load_model(self):
        """Load the model and checkpoint."""
        print("Initializing custom VLM model...")
        self.model = MyModel()
        self.model.adaptor.to(self.device, dtype=torch.bfloat16)
        self.model.to(self.device)

        print(f"Loading checkpoint from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.adaptor.load_state_dict(checkpoint['model_state_dict'])

        print(f"Successfully loaded model from Step {checkpoint.get('global_step', 'N/A')}")
        self.model.eval()
        print(f"✅ Custom VLM model ready on device: {self.device}")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the LLM on the given prompt and input.
        
        For text-only prompts, this returns the prompt as-is since the model
        is primarily designed for vision-language tasks.
        """
        # For text-only input, return a placeholder or simple response
        # The actual multimodal handling happens in the agent_invoke route
        return f"[VLM Ready] Received: {prompt[:100]}..."

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of _call."""
        return self._call(prompt, stop, run_manager, **kwargs)

    def invoke_with_image(self, image_path: str, prompt_text: str, max_new_tokens: int = 64) -> str:
        """
        Invoke the model with an image and text prompt.
        
        Args:
            image_path: Path to the image file
            prompt_text: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"

            image_raw = Image.open(image_path).convert("RGB")

            with torch.no_grad():
                caption = self.model.generate_answer_image(
                    prompt_text,
                    image_raw,
                    max_new_tokens=max_new_tokens
                )
            return caption

        except Exception as e:
            return f"Error during inference: {str(e)}"


# ===============================================================
# GLOBAL MODEL INSTANCE
# ===============================================================

_custom_vlm_instance = None


def get_custom_vlm_model() -> CustomVLMLLM:
    """Get or create the global custom VLM model instance."""
    global _custom_vlm_instance
    if _custom_vlm_instance is None:
        _custom_vlm_instance = CustomVLMLLM()
    return _custom_vlm_instance
