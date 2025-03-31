"""
This module provides NL-to-SQL and NL-to-NoSQL conversion capabilities for multi-modal applications.
It supports translation of natural language queries into query templates and parameters compatible with
Postgres, Azure Cosmos (document) and Redis (key–value).

Classes:
    NLToSQL:
        Translates natural language queries into SQL query templates for Postgres.
    NLToNoSQL:
        Translates natural language queries into NoSQL query templates for document stores (Azure Cosmos DB)
        and key-value stores (Redis).

Usage:
    Instantiate the desired class and use the convert method to translate natural language queries into
    query templates and parameters.
"""

from azure.cognitiveservices.search.websearch import WebSearchClient
from msrest.authentication import CognitiveServicesCredentials


class BingSearch:
    """
    Provides internet search capabilities using the Bing API with Azure Bing SDK.
    """
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.client = WebSearchClient(
            endpoint=self.endpoint,
            credentials=CognitiveServicesCredentials(self.api_key)
        )

    def search(self, query: str) -> dict:
        # Use Azure Bing SDK to perform the search.
        response = self.client.web.search(query=query)
        if response is None:
            return {}
        try:
            return response.output.as_dict()
        except AttributeError:
            if isinstance(response, dict):
                return response
            return {}

    def ground_content(self, topic: str) -> dict:
        search_results = self.search(topic)
        return {"topic": topic, "grounded_data": search_results}
