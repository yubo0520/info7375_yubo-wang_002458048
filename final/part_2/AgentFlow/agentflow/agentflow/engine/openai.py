# Reference: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/openai.py

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
import re
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine

import openai

from dotenv import load_dotenv
load_dotenv(override=True)


from pydantic import BaseModel

class DefaultFormat(BaseModel):
    response: str


def validate_structured_output_model(model_string: str):
    """
    TODO: add more models that support structured outputs as follows:
    o3-mini-2025-01-31
    o4-mini-2025-04-16
    o1-2024-12-17
    o3-2025-04-16
    o1-pro-2025-03-19
    """
    # Ref: https://platform.openai.com/docs/guides/structured-outputs
    Structure_Output_Models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
    return any(x in model_string for x in Structure_Output_Models)

def validate_chat_model(model_string: str):
    # Treat everything as chat model unless it's a reasoning/pro model
    return not validate_reasoning_model(model_string) and not validate_pro_reasoning_model(model_string)

def validate_reasoning_model(model_string: str):
    # Ref: https://platform.openai.com/docs/guides/reasoning
    return any(x in model_string for x in ["o1", "o3", "o4"]) and not validate_pro_reasoning_model(model_string)

def validate_pro_reasoning_model(model_string: str):
    # Ref: https://platform.openai.com/docs/guides/reasoning
    return any(x in model_string for x in ["o1-pro", "o3-pro", "o4-pro"])

class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="gpt-4o-mini-2024-07-18",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        use_cache: bool=True, # disable cache for now
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        """

        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        self.support_structured_output = validate_structured_output_model(self.model_string)
        self.is_chat_model = validate_chat_model(self.model_string)
        self.is_reasoning_model = validate_reasoning_model(self.model_string)
        self.is_pro_reasoning_model = validate_pro_reasoning_model(self.model_string)

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_path = os.path.join(root, f"cache_openai_{self.model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)
        
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
        
        _base_url = os.getenv("OPENAI_BASE_URL")
        _extra_headers = {}
        if _base_url and "portkey" in _base_url:
            _extra_headers["x-portkey-provider"] = "openai"
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=_base_url or None,
            default_headers=_extra_headers if _extra_headers else None,
        )


    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    full_text = "\n".join(content)
                    return self._generate_text(full_text, system_prompt=system_prompt, **kwargs)

                elif any(isinstance(item, bytes) for item in content):
                    if not self.is_multimodal:
                        raise NotImplementedError(
                            f"Multimodal generation is only supported for {self.model_string}. "
                            "Consider using a multimodal model like 'gpt-4o'."
                        )
                    return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

                else:
                    raise ValueError("Unsupported content in list: only str or bytes are allowed.")

        except openai.LengthFinishReasonError as e:
            print(f"Token limit exceeded: {str(e)}")
            print(f"Tokens used - Completion: {e.completion.usage.completion_tokens}, Prompt: {e.completion.usage.prompt_tokens}, Total: {e.completion.usage.total_tokens}")
            return {
                "error": "token_limit_exceeded",
                "message": str(e),
                "details": {
                    "completion_tokens": e.completion.usage.completion_tokens,
                    "prompt_tokens": e.completion.usage.prompt_tokens,
                    "total_tokens": e.completion.usage.total_tokens
                }
            }
        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {str(e)}")
            return {
                "error": "rate_limit",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        
    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=3000, top_p=0.99, response_format=None
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
                
        # Chat models given structured output format
        if self.is_chat_model and self.support_structured_output and response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format
            )
            response = response.choices[0].message.parsed

        # Chat models without native structured outputs but response_format requested: use json_object mode + schema injection
        elif self.is_chat_model and not self.support_structured_output and response_format is not None:
            from pydantic import BaseModel as _BaseModel
            import inspect as _inspect
            schema_str = ""
            if _inspect.isclass(response_format) and issubclass(response_format, _BaseModel):
                schema_str = json.dumps(response_format.model_json_schema(), indent=2)
            json_system_prompt = (
                sys_prompt_arg
                + "\n\nYou MUST respond with valid JSON only. No markdown code fences, no explanation, no extra text."
                + (f"\nYour JSON must conform to this schema:\n{schema_str}" if schema_str else "")
            )
            extra = {}
            if "qwen3" in self.model_string.lower():
                extra["extra_body"] = {"enable_thinking": False}
            raw_response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": json_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format={"type": "json_object"},
                **extra,
            )
            raw_text = raw_response.choices[0].message.content
            # Strip markdown fences if model ignored the instruction
            clean = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
            clean = re.sub(r'\s*```$', '', clean.strip())
            try:
                data = json.loads(clean)
                if _inspect.isclass(response_format) and issubclass(response_format, _BaseModel):
                    response = response_format(**data)
                else:
                    response = data
            except Exception as parse_err:
                print(f"[JSON parse fallback] failed to parse JSON for {self.model_string}: {parse_err}")
                response = raw_text

        # Chat models without structured outputs and no response_format
        elif self.is_chat_model and response_format is None:
            extra = {}
            if "qwen3" in self.model_string.lower():
                extra["extra_body"] = {"enable_thinking": False}
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **extra,
            )
            response = response.choices[0].message.content

        # Reasoning models: currently only supports base response
        elif self.is_reasoning_model:
            print(f"Using reasoning model: {self.model_string}")
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort="medium"
            )
            # Workaround for handling length finish reason
            if "finishreason" in response.choices[0] and response.choices[0].finishreason == "length":
                response = "Token limit exceeded"
            else:
                response = response.choices[0].message.content

        # Reasoning models: Pro models using v1/completions
        elif self.is_pro_reasoning_model:
            response = self.client.responses.create(
                model=self.model_string,
                input=prompt,
                reasoning={
                    "effort": "medium"
                },
            )
            response = response.output[1].content[0].text

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                continue
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_multimodal(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=512, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        # Chat models given structured output format
        if self.is_chat_model and self.support_structured_output and response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format
            )
            response_text = response.choices[0].message.parsed

        # Chat models without structured outputs
        elif self.is_chat_model and (not self.support_structured_output or response_format is None):
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response_text = response.choices[0].message.content

        # Reasoning models: currently only supports base response
        elif self.is_reasoning_model:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "user", "content": formatted_content},
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort="medium"
            )
            # Workaround for handling length finish reason
            if "finishreason" in response.choices[0] and response.choices[0].finishreason == "length":
                response_text = "Token limit exceeded"
            else:
                response_text = response.choices[0].message.content

        # Reasoning models: Pro models using v1/completions
        elif self.is_pro_reasoning_model:
            response = self.client.responses.create(
                model=self.model_string,
                input=str(formatted_content), # NOTE: simple string conversion for now
                reasoning={
                    "effort": "medium"
                },
            )
            response_text = response.output[1].content[0].text

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text
