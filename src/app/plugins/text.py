"""
This module provides advanced text processing capabilities for multi-modal applications by integrating
Azure Document Intelligence (formerly Form Recognizer) with Azure CosmosDB and Azure AI Search. It supports
document analysis (as a proxy for language extraction, entity recognition, etc.), text embedding generation,
summarization, and question answering.

Classes:
    TextProcessor:
        Handles analysis of input text documents using Azure Document Intelligence to simulate language detection,
        sentiment analysis, and entity extraction.
    TextEmbedder:
        Generates text embeddings and persists results to CosmosDB and Azure AI Search.
    TextAnswer:
        Provides text summarization and question answering capabilities.

Dependencies:
    - azure-ai-documentintelligence
    - azure-cosmos
    - azure-search-documents
    - azure-identity
    - requests
    - numpy
    - asyncio
"""

import os
import io
import time
import logging
import asyncio
from typing import Dict, List, Any

from azure.ai.documentintelligence.models import AnalyzeResult
import requests
import numpy as np

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.cosmos.aio import CosmosClient
from azure.search.documents import SearchClient
from azure.ai.inference import EmbeddingsClient

from semantic_kernel.functions.kernel_function_decorator import kernel_function

logger = logging.getLogger(__name__)

DOC_INTELLIGENCE_KEY = os.environ.get("DOC_INTELLIGENCE_KEY", "")
DOC_INTELLIGENCE_URL = os.environ.get("DOC_INTELLIGENCE_URL", "")
AZURE_MODEL_KEY = os.environ.get("AZURE_MODEL_KEY", "")
AZURE_AUDIO_EMBEDDINGS_URL = os.environ.get("AZURE_AUDIO_EMBEDDINGS_URL", "")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-large")


class TextProcessor:
    """
    Handles text operations using Azure Document Intelligence (prebuilt-document model) 
    to extract textual content from document files (e.g. PDFs or Word documents).
    Since Document Intelligence is not designed specifically for language detection or sentiment analysis, 
    these methods return dummy values (language always 'en', sentiment "neutral") along with any entities extracted.
    """

    def __init__(self):
        """
        Initialize the TextProcessor with Azure Document Intelligence credentials.
        """
        self.client = DocumentIntelligenceClient(
            endpoint=DOC_INTELLIGENCE_URL,
            credential=AzureKeyCredential(DOC_INTELLIGENCE_KEY)
        )

    async def _analyze_document(self, document_path: str) -> AnalyzeResult:
        """
        Analyze the document (PDF, Word, etc.) using the "prebuilt-layout" model.

        Args:
            document_path (str): The file path to the document.

        Returns:
            The analysis result.
        """
        async with self.client as client:
            with open(document_path, "rb") as doc_file:
                document_bytes = doc_file.read()
            response = await client.begin_analyze_document("prebuilt-layout", io.BytesIO(document_bytes))
            result: AnalyzeResult = await response.result()
            if not result:
                raise ValueError("No result returned from Document Intelligence.")
        return result

    @kernel_function(
        name="DetectLanguage",
        description="Detects the language of the provided document (always returns English in this implementation)"
    )
    async def detect_language(self, document_path: str) -> Dict[str, Any]:
        """
        Detect the language from a document by analyzing its content.
        (This implementation always returns 'en' for English.)

        Args:
            document_path (str): The file path to the document.

        Returns:
            Dict[str, Any]: Detected language data.
        """
        try:
            _ = await self._analyze_document(document_path)
            return {"language": "en", "score": 1.0}
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return {"error": str(e)}

    @kernel_function(
        name="AnalyzeSentiment", 
        description="Analyzes sentiment of the document by summing line confidence scores and deriving a simple label"
    )
    async def analyze_sentiment(self, document_path: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the document by examining line-level confidence scores.

        Args:
            document_path (str): The file path to the document.
            
        Returns:
            Dict[str, Any]: A dictionary with 'sentiment', 'positive_score', 'neutral_score', 'negative_score'.
        """
        try:
            result = await self._analyze_document(document_path)
            confidence_values = []

            # Look at each page, then each line
            if result.pages:
                for page in result.pages:
                    if page.lines:
                        for line in page.lines:
                            if hasattr(line, "confidence") and line.content is not None:
                                confidence_values.append(line.content)

            # If no lines or no confidence data, default to 0.5
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.5

            if avg_confidence > 0.7:
                sentiment = "positive"
            elif avg_confidence < 0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "positive_score": max(avg_confidence - 0.5, 0.0),
                "neutral_score": 1.0 - abs(avg_confidence - 0.5),
                "negative_score": max(0.5 - avg_confidence, 0.0)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"error": str(e)}

    @kernel_function(
        name="ExtractLayoutEntities",
        description="Extracts text and polygons from the analyzed document layout"
    )
    async def extract_layout_entities(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Extract text blocks (e.g., lines) from the document analysis result, including their polygons.

        Args:
            document_path (str): The file path to the document.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing text and polygon coordinates.
        """
        try:
            result = await self._analyze_document(document_path)
            entities = []

            # Make sure pages exist
            if result.pages:
                for page in result.pages:
                    # Lines are often the easiest place to get text + polygon
                    if page.lines:
                        for line in page.lines:
                            # 'line.content' is the extracted text
                            # 'line.polygon' are the coordinates (list of floats [x1, y1, x2, y2, ...])
                            entities.append({
                                "text": line.content,
                                "category": "line",  # or "paragraph"/"word" if you iterate those
                                "polygon": line.polygon
                            })

            return entities

        except Exception as e:
            logger.error(f"Error extracting layout entities: {str(e)}")
            raise e




class TextEmbedder:
    """
    Generates text embeddings and persists results to CosmosDB and Azure AI Search.
    """

    def __init__(self):
        """
        Initialize the TextEmbedder with an embedding service key and endpoint.
        (For demonstration, a dummy embedding function is provided.)
        
        Args:
            key (str): The subscription key for the embedding service.
            endpoint (str): The endpoint URL for the embedding service.
        """
        self.key = DOC_INTELLIGENCE_KEY
        self.endpoint = DOC_INTELLIGENCE_URL

    @kernel_function(
        name="EmbedText",
        description="Generates text embeddings and saves them to CosmosDB and Azure AI Search"
    )
    async def embed_text(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any] | List[float]:
        """
        Generate a dummy text embedding from the provided text and save it to CosmosDB and Azure AI Search.
        
        Args:
            text (str): The input text.
            metadata (Dict[str, Any]): Metadata associated with the text.
            
        Returns:
            Dict[str, Any]: The saved embedding data.
        """
        try:
            embedding = self._generate_embedding(text)
            print(f"Generated embedding: {embedding}")

            try:
                cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT", "")
                cosmos_key = os.environ.get("COSMOS_KEY", self.key)
                database_name = os.environ.get("COSMOS_DATABASE", "")
                container_name = os.environ.get("COSMOS_CONTAINER", "")
                
                client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
                database = client.get_database_client(database_name)
                container = database.get_container_client(container_name)
                
                cosmos_item = {
                    "id": metadata.get("id", f"text-{int(time.time())}"),
                    "embedding": embedding,
                    "metadata": metadata,
                    "type": "text",
                    "timestamp": time.time()
                }
                
                await container.upsert_item(cosmos_item)
                
                # Save to Azure AI Search
                search_endpoint = os.environ.get("SEARCH_ENDPOINT", "")
                search_key = os.environ.get("SEARCH_KEY", self.key)
                index_name = os.environ.get("SEARCH_INDEX", "")
                search_client = SearchClient(
                    endpoint=search_endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(search_key)
                )
                
                search_document = {
                    "id": cosmos_item["id"],
                    "embedding": embedding,
                    "metadata_str": str(metadata),
                    "type": "text"
                }
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        search_document[key] = value
                
                search_client.upload_documents(documents=[search_document])
                return cosmos_item
            except Exception as e:
                logger.error(f"Error saving to CosmosDB or Azure Search: {str(e)}")
                pass
        
            return embedding

        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise Exception(f"Failed to embed text: {str(e)}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a dummy embedding vector from the input text.
        
        In production, replace this with a call to a real text embedding service.
        
        Args:
            text (str): The input text.
            
        Returns:
            List[float]: The embedding vector.
        """
        client = EmbeddingsClient(
            endpoint=AZURE_AUDIO_EMBEDDINGS_URL,
            credential=AzureKeyCredential(AZURE_MODEL_KEY)
        )
        response = client.embed(
            input=text.split(".")[:-1],
            model=EMBEDDINGS_MODEL
        )

        return [float(x) for embeddings_data in response.data for x in embeddings_data.embedding]

    @kernel_function(
        name="CalculateSimilarity",
        description="Calculates cosine similarity between a set of text embeddings"
    )
    def calculate_similarity(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Calculates cosine similarity between a set of text embeddings.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            
        Returns:
            List[List[float]]: Similarity matrix.
        """
        try:
            vectors = np.array(embeddings)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = vectors / norms
            similarity_matrix = np.dot(normalized, normalized.T)
            return similarity_matrix.tolist()
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return [[0.0] * len(embeddings) for _ in range(len(embeddings))]



class TextAnswer:
    """
    Provides text summarization and question answering capabilities using an external GPT-based service.
    """

    def __init__(self):
        """
        Initialize the TextAnswer with configuration for the external QA/Summarization service.
        
        Args:
            key (str): The API key for the external service.
            endpoint (str): The endpoint URL for the external service.
        """
        self.key = DOC_INTELLIGENCE_KEY
        self.endpoint = DOC_INTELLIGENCE_URL
        self.gpt_url = os.environ.get("GPT_TEXT_URL", "")
        self.gpt_key = os.environ.get("GPT_TEXT_KEY", "")

    @kernel_function(
        name="SummarizeText",
        description="Summarizes the input text using an external GPT-based service"
    )
    def summarize_text(self, text: str) -> str:
        """
        Summarize the input text using an external GPT-based service.
        
        Args:
            text (str): The input text to summarize.
        
        Returns:
            str: The summary.
        """
        if not self.gpt_url or not self.gpt_key:
            return "GPT service configuration missing. Cannot summarize text."
        try:
            payload = {
                "prompt": f"Summarize the following text:\n{text}",
                "temperature": 0.7,
                "max_tokens": 200
            }
            headers = {
                "Content-Type": "application/json",
                "api-key": self.gpt_key
            }
            response = requests.post(self.gpt_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            summary = result.get("summary", "")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return f"Error summarizing text: {str(e)}"

    @kernel_function(
        name="AnswerQuestion",
        description="Answers a question based on the provided reference text using an external service"
    )
    def answer_question(self, text: str, question: str) -> str:
        """
        Answer a question based on the provided text using an external service.
        
        Args:
            text (str): The reference text.
            question (str): The query.
        
        Returns:
            str: The answer.
        """
        if not self.gpt_url or not self.gpt_key:
            return "GPT service configuration missing. Cannot answer question."
        try:
            payload = {
                "prompt": f"Based on the text below, answer the following question.\n\nText: {text}\n\nQuestion: {question}",
                "temperature": 0.7,
                "max_tokens": 150
            }
            headers = {
                "Content-Type": "application/json",
                "api-key": self.gpt_key
            }
            response = requests.post(self.gpt_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result.get("answer", "")
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error answering question: {str(e)}"
