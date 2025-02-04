from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
import requests
from typing import Optional, List, Iterator, Any, Dict


class CustomLLM(LLM):
    """Custom LLM that connects to a FastAPI streaming endpoint.

    Args:
        api_url: The URL of the FastAPI endpoint
        timeout: Timeout in seconds for API calls
    """

    api_url: str
    timeout: int = 30

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous call to the API endpoint."""
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        try:
            response = requests.post(
                self.api_url,
                json={"content": prompt},
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Accumulate the streaming response
            full_response = ""
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    full_response += chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk)

            return full_response

        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call failed: {str(e)}")

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Synchronous streaming implementation."""
        try:
            response = requests.post(
                self.api_url,
                json={"content": prompt},
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    yield GenerationChunk(text=chunk)
                    if run_manager:
                        run_manager.on_llm_new_token(chunk)

        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call failed: {str(e)}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {"model_name": "CustomAPILLM", "api_url": self.api_url}

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom_api"