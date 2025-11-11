import os
from openai import AzureOpenAI

class AzureOpenAIClient:
    def __init__(self, api_key=None, endpoint=None, api_version=None):
        self._client = self._init_client(api_key, endpoint, api_version)

    def _init_client(self, api_key, endpoint, api_version):
        return AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_APIKEY"),
            azure_endpoint=endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version or os.getenv("AZURE_OPENAIAPI_VERSION")
        )

    @property
    def client(self):
        return self._client
