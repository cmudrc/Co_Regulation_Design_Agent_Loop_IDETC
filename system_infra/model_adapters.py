"""Model adapters for different LLM backends.

This module implements adapters for OpenAI GPT models and local models (via transformers).
"""
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelResponse:
    """Response from model generation."""
    text: str                                    # Raw text response
    metadata: Optional[Dict[str, Any]] = None    # Model info (tokens, etc.)
    error: Optional[str] = None                  # Error message if failed


# Available model configurations
LLAMA_CONFIGS = {
    "llama-3-8b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3-8B-Instruct",
    "llama-3.1-8b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3.2-3B-Instruct",
    "llama-3.3-70b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3.3-70B-Instruct",
    "llama-4-scout": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-4-Scout-17B-16E-Instruct",
}

QWEN_CONFIGS = {
    "qwen3-14b": "/mnt/c/Users/kovac/Desktop/pre_trained/Qwen3-14B",
}


class LocalLlamaAdapter:
    """Adapter for local Llama models using HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        """Initialize local model adapter.

        Args:
            model_name: Model key from MODEL_CONFIGS (e.g., "llama-3.1-8b")
            load_in_4bit: Use 4-bit quantization (default: True)
            load_in_8bit: Use 8-bit quantization (default: False)
            device_map: Device mapping strategy (default: "auto")
            trust_remote_code: Trust remote code (default: True)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        import gc

        self.model_name = model_name

        # Set environment variables for multi-GPU
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        # Clear GPU memory
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        gc.collect()

        print(f"[INFO] Loading model: {model_name}")

        # Get model path
        if model_name not in LLAMA_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(LLAMA_CONFIGS.keys())}")
        model_id = LLAMA_CONFIGS[model_name]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Quantization config
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_memory={
                0: "24GiB",   # 4090
                1: "24GiB",   # 3090 Ti
                2: "24GiB",   # 3090
                3: "24GiB",   # 3090
            },
        )

        self.device = self.model.device
        print(f"[INFO] Model {model_name} loaded successfully on {self.device}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        top_p: float = 0.95,
        **kwargs
    ) -> ModelResponse:
        """Generate response from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            history: Optional conversation history to prepend
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            ModelResponse with raw text output
        """
        import torch
        try:
            # Combine history and current messages
            full_messages = []
            if history:
                full_messages.extend(history)
            full_messages.extend(messages)

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response (remove prompt)
            # Most chat templates end with an assistant marker, so we split on that
            if "assistant" in generated_text.lower():
                parts = generated_text.lower().split("assistant")
                if len(parts) > 1:
                    # Get everything after the last "assistant" marker
                    response_text = generated_text[generated_text.lower().rfind("assistant") + len("assistant"):].strip()
                    # Remove common prefixes
                    response_text = response_text.lstrip(":").lstrip()
                else:
                    response_text = generated_text
            else:
                response_text = generated_text

            # Clean up
            del inputs, outputs
            torch.cuda.empty_cache()

            return ModelResponse(
                text=response_text,
                metadata={
                    "model": self.model_name,
                    "temperature": temperature,
                }
            )

        except Exception as e:
            return ModelResponse(
                text="",
                error=f"Generation failed: {str(e)}"
            )


import re


class LocalQwenAdapter:
    """Adapter for local Qwen models using HuggingFace Transformers.

    Handles Qwen3's thinking mode: strips <think>...</think> blocks from output.
    """

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        enable_thinking: bool = False,
    ):
        """Initialize Qwen model adapter.

        Args:
            model_name: Model key from QWEN_CONFIGS (e.g., "qwen3-14b")
            load_in_4bit: Use 4-bit quantization (default: True)
            load_in_8bit: Use 8-bit quantization (default: False)
            enable_thinking: Enable Qwen3 thinking mode (default: False)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        import gc

        self.model_name = model_name
        self.enable_thinking = enable_thinking

        # Set environment variables for multi-GPU
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        # Clear GPU memory
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        gc.collect()

        print(f"[INFO] Loading model: {model_name}")

        # Get model path
        if model_name not in QWEN_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(QWEN_CONFIGS.keys())}")
        model_id = QWEN_CONFIGS[model_name]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Quantization config
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_memory={
                0: "24GiB",   # 4090
                1: "24GiB",   # 3090 Ti
                2: "24GiB",   # 3090
                3: "24GiB",   # 3090
            },
        )

        self.device = self.model.device
        print(f"[INFO] Model {model_name} loaded successfully on {self.device}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        top_p: float = 0.95,
        **kwargs
    ) -> ModelResponse:
        """Generate response from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            history: Optional conversation history to prepend
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            ModelResponse with raw text output (thinking content stripped)
        """
        import torch
        try:
            # Combine history and current messages
            full_messages = []
            if history:
                full_messages.extend(history)
            full_messages.extend(messages)

            # Apply chat template with thinking mode control
            prompt = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode only the new tokens (skip the input prompt tokens)
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Strip thinking content if present
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

            # Clean up
            del inputs, outputs
            torch.cuda.empty_cache()

            return ModelResponse(
                text=response_text,
                metadata={
                    "model": self.model_name,
                    "temperature": temperature,
                }
            )

        except Exception as e:
            return ModelResponse(
                text="",
                error=f"Generation failed: {str(e)}"
            )



class LMStudioAdapter:
    """Adapter for models hosted in LM Studio via its OpenAI-compatible local API."""

    def __init__(
        self,
        model_name: str,
        host: Optional[str] = None,
        enable_thinking: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize LM Studio adapter.

        Args:
            model_name: Exact model identifier as shown in LM Studio
                        (e.g., "qwen/qwen3-14b-mlx-4bit")
            host: LM Studio server host and port. If None, reads from LM_STUDIO_HOST env var.
            enable_thinking: Enable Qwen3 thinking mode (default: True).
                             Prepends /think or /no_think to the first user message.
            seed: Random seed for reproducible generation (default: None).
        """
        self.model_name = model_name
        self.host = host or os.getenv("LM_STUDIO_HOST")
        if not self.host:
            raise ValueError("LM Studio host not provided and LM_STUDIO_HOST env var not set")
        
        self.enable_thinking = enable_thinking
        self.thinking_level = "enabled" if enable_thinking else "disabled"  # for metadata consistency
        self.seed = seed

        from openai import OpenAI
        self.client = OpenAI(
            api_key="lm-studio",  # LM Studio ignores the key; any non-empty string works
            base_url=f"http://{self.host}/v1",
        )

        print(f"[INFO] LM Studio adapter initialized with model: {model_name} at http://{self.host}/v1")

    def generate(
        self,
        messages: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 1.0,
        max_tokens: int = 32000,
        top_p: float = 0.95,
        **kwargs
    ) -> ModelResponse:
        """Generate response from messages using LM Studio's local API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            history: Optional conversation history to prepend
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            ModelResponse with raw text output
        """
        try:
            full_messages = []
            if history:
                full_messages.extend(history)
            full_messages.extend(messages)

            # Inject Qwen3 thinking toggle into the first user message
            thinking_prefix = "/think " if self.enable_thinking else "/no_think "
            patched, injected = [], False
            for msg in full_messages:
                if not injected and msg["role"] == "user":
                    patched.append({**msg, "content": thinking_prefix + msg["content"]})
                    injected = True
                else:
                    patched.append(msg)
            full_messages = patched

            api_params = {
                "model": self.model_name,
                "messages": full_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            if self.seed is not None:
                api_params["seed"] = self.seed

            api_params.update(kwargs)

            response = self.client.chat.completions.create(**api_params)

            response_text = response.choices[0].message.content

            return ModelResponse(
                text=response_text,
                metadata={
                    "model": self.model_name,
                    "temperature": temperature,
                    "tokens_used": response.usage.total_tokens if response.usage else None,
                    "finish_reason": response.choices[0].finish_reason,
                }
            )

        except Exception as e:
            return ModelResponse(
                text="",
                error=f"LM Studio API call failed: {str(e)}"
            )



class OpenAIAdapter:
    """Adapter for OpenAI API models."""

    def __init__(
        self,
        model_name: str = "gpt-5",
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize OpenAI adapter.

        Args:
            model_name: OpenAI model name (e.g., "gpt-5", "gpt-5.4")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            reasoning_effort: Reasoning effort for gpt-5 models.
                              One of "low", "medium", "high" (default: None = model default).
            seed: Random seed for reproducible generation (default: None).
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

        print(f"[INFO] OpenAI adapter initialized with model: {model_name}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response from messages using OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            history: Optional conversation history to prepend

        Returns:
            ModelResponse with raw text output
        """
        try:
            # Combine history and current messages
            full_messages = []
            if history:
                full_messages.extend(history)
            full_messages.extend(messages)

            api_params = {
                "model": self.model_name,
                "messages": full_messages,
                "max_completion_tokens": 32000,
            }

            if self.reasoning_effort is not None:
                api_params["reasoning_effort"] = self.reasoning_effort

            if self.seed is not None:
                api_params["seed"] = self.seed

            api_params.update(kwargs)

            response = self.client.chat.completions.create(**api_params)

            response_text = response.choices[0].message.content

            u = response.usage
            metadata = {
                "model": self.model_name,
                "tokens_used":     u.total_tokens      if u else None,
                "input_tokens":    u.prompt_tokens     if u else None,
                "output_tokens":   u.completion_tokens if u else None,
                "thinking_tokens": (
                    u.completion_tokens_details.reasoning_tokens
                    if u and u.completion_tokens_details else None
                ),
                "finish_reason": response.choices[0].finish_reason,
            }

            if self.reasoning_effort is not None:
                metadata["reasoning_effort"] = self.reasoning_effort

            return ModelResponse(
                text=response_text,
                metadata=metadata
            )

        except Exception as e:
            return ModelResponse(
                text="",
                error=f"OpenAI API call failed: {str(e)}"
            )


class GoogleGeminiAdapter:
    """Adapter for Google Gemini API models (via google-genai SDK)."""

    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        api_key: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Initialize Google Gemini adapter.

        Args:
            model_name: Gemini model name (e.g., "gemini-2.0-flash", "gemini-3.1-pro-preview")
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            thinking_level: Thinking effort level for Gemini 3.x models.
                            One of "none", "low", "medium", "high", "auto" (default: None).
            thinking_budget: Thinking token budget for Gemini 2.x models (e.g., Gemini 2.5 Flash/Pro).
                             Integer (0 = disable thinking, max ~24576). Mutually exclusive with thinking_level.
            seed: Random seed for reproducible generation (default: None = no seed).
        """
        self.model_name = model_name
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.seed = seed
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key not provided and GOOGLE_API_KEY env var not set")

        from google import genai
        from google.genai import types as genai_types
        self._client = genai.Client(api_key=self.api_key)
        self._types = genai_types

        print(f"[INFO] Google Gemini adapter initialized with model: {model_name}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 1.0,
        max_tokens: int = 32000,
        top_p: float = 0.95,
        **kwargs
    ) -> ModelResponse:
        """Generate response from messages using Gemini API.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            history: Optional conversation history to prepend
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            ModelResponse with raw text output
        """
        try:
            # Combine history and current messages
            full_messages = []
            if history:
                full_messages.extend(history)
            full_messages.extend(messages)

            # Extract system message — Gemini handles it via system_instruction
            system_instruction = None
            chat_messages = []
            for msg in full_messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    chat_messages.append(msg)

            # Convert to Gemini Content objects
            # Gemini uses "user"/"model" roles (not "assistant")
            contents = []
            for msg in chat_messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    self._types.Content(
                        role=role,
                        parts=[self._types.Part.from_text(text=msg["content"])]
                    )
                )

            if self.thinking_budget is not None:
                thinking_config = self._types.ThinkingConfig(thinking_budget=self.thinking_budget)
            elif self.thinking_level is not None:
                thinking_config = self._types.ThinkingConfig(thinking_level=self.thinking_level)
            else:
                thinking_config = None
            config = self._types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                seed=self.seed,
                system_instruction=system_instruction,
                thinking_config=thinking_config,
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            response_text = response.text
            if response_text is None:
                # response.text is None when non-text parts (e.g. function_call) are present
                # Fall back to concatenating only the text parts manually
                try:
                    parts = response.candidates[0].content.parts
                    response_text = "".join(
                        p.text for p in parts if hasattr(p, "text") and p.text is not None
                    )
                except Exception:
                    response_text = ""

            um = response.usage_metadata
            metadata = {
                "model": self.model_name,
                "temperature": temperature,
                "tokens_used":        um.total_token_count       if um else None,
                "input_tokens":       um.prompt_token_count      if um else None,
                "output_tokens":      um.candidates_token_count  if um else None,
                "thinking_tokens":    um.thoughts_token_count    if um else None,
            }

            return ModelResponse(
                text=response_text,
                metadata=metadata
            )

        except Exception as e:
            return ModelResponse(
                text="",
                error=f"Gemini API call failed: {str(e)}"
            )
