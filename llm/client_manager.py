import os
import logging
from openai import OpenAI
from openai import AzureOpenAI


class ClientManager:
    """
    Unified OpenAI client that supports both Azure OpenAI and official OpenAI endpoints.
    """

    def __init__(self):
        self.mode = None
        self.client = None

        azure_api_key = os.getenv("AZURE_OPENAI_APIKEY")
        azure_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAIAPI_VERSION")

        openai_key = os.getenv("OPENAI_APIKEY")

        if azure_api_key and azure_api_endpoint and azure_api_version:
            self.mode = "azure"
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_api_endpoint,
                api_version=azure_api_version,
            )
            logging.info("Using Azure OpenAI Client.")
        elif openai_key:
            self.mode = "openai"
            self.client = OpenAI(api_key=openai_key)
            logging.info("Using OpenAI Official Client.")
        else:
            raise ValueError(
                "No valid OpenAI credentials found in environment variables."
            )

    def chat_completion(
        self,
        messages: list,
        model: str,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 120,
    ) -> str | None:
        """
        Unified chat completion method for both Azure and OpenAI.

        Args:
            messages (list): List of message dicts with roles 'system', 'user', 'assistant'.
            model (str): Model name or Azure deployment name.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            timeout (int): Request timeout in seconds.

        Returns:
            str | None: Generated text or None if failed.
        """

        try:

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()

            return None

        except Exception as e:
            logging.error(f"Error calling {self.mode} OpenAI: {e}")
            return None
