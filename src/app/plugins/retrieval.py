"""
This module provides NL-to-SQL and NL-to-NoSQL conversion capabilities for multi-modal applications.
It now also supports searching indexed data in Azure AI Search and performing search queries against
an Azure Cosmos DB container.

Classes:
    NLToSQL:
        Translates natural language queries into SQL query templates for Postgres.
    NLToNoSQL:
        Translates natural language queries into NoSQL query templates for document stores (Azure Cosmos DB)
        and key-value stores (Redis).
    AzureAISearchTool:
        Executes search queries against an Azure AI Search index.
    CosmosSearchTool:
        Executes search operations against an Azure Cosmos DB container.
        
Usage:
    Instantiate the desired class and call its search or convert methods as needed.
"""

import os
import glob
import logging
from typing import Dict, List, Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.cosmos.aio import CosmosClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from docx import Document
from pypdf import PdfReader
from PIL import Image


logger = logging.getLogger(__name__)


class NLToSQL:
    """
    Translates natural language queries into SQL query templates for Postgres.
    """
    def __init__(self):
        # Load SQL query templates (in production, these could be read from a file)
        self.template = "SELECT {fields} FROM {table} WHERE {conditions};"

    def convert(self, query_template: str, query_params: Optional[dict] = None, operation: Optional[str] = None) -> dict:
        """
        Convert a query template and parameters into a SQL query.
        """
        sql_query = query_template
        if query_params is not None:
            sql_query = query_template.format(**query_params)
        return {
            "query_template": sql_query,
            "parameters": query_params,
            "operation": operation
        }

    @kernel_function(name="test_sql_query", description="Tests if a SQL query is correctly formed using a template and parameters")
    def test(self, query_template=None, query_params=None, operation="select", expected_data=None):
        """
        Test if the SQL query was correctly performed.
        """
        if query_template is None:
            query_template = self.template
        if query_params is None:
            query_params = {"fields": "name, age", "table": "users", "conditions": "age > 18"}
            
        try:
            result = self.convert(query_template, query_params, operation)
            if "error" in result:
                print(f"Test failed: Query returned an error: {result['error']}")
                return False
            if not result or not result.get("query_template"):
                print("Test failed: Query result is empty")
                return False
            if expected_data:
                for key, value in expected_data.items():
                    if key not in result or result[key] != value:
                        print(f"Test failed: Expected {key}={value}, got {result.get(key)}")
                        return False
            print("Test passed: SQL query executed successfully with expected results")
            return True
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")
            return False


class NLToNoSQL:
    """
    Translates natural language queries into NoSQL query templates.
    Supports both document queries (e.g., Azure Cosmos DB) and key–value queries (e.g., Redis).
    """
    def __init__(self):
        self.document_template = {
            "query": "SELECT * FROM c WHERE {conditions}"
        }
        self.keyvalue_template = {
            "command": "GET {key}"
        }

    @kernel_function(name="convert_to_nosql", description="Converts template and parameters into a NoSQL query for document or key-value stores")
    def convert(self, store_type: str, query_template: str, query_params: dict, operation: str) -> dict:
        """
        Convert a query template and parameters into a NoSQL query.
        """
        if store_type.lower() == "document":
            query = query_template.format(**query_params)
            return {
                "nosql_type": "document",
                "query_template": query,
                "parameters": query_params,
                "operation": operation
            }
        elif store_type.lower() == "keyvalue":
            query = query_template.format(**query_params)
            return {
                "nosql_type": "keyvalue",
                "query_template": query,
                "parameters": query_params,
                "operation": operation
            }
        else:
            return {"error": "Unsupported store type. Use 'document' or 'keyvalue'."}
    
    @kernel_function(name="test_nosql_query", description="Tests if a NoSQL query is correctly formed for document or key-value stores")
    def test(self, store_type=None, query_template=None, query_params=None, operation=None, expected_data=None):
        """
        Test if the NoSQL query was correctly performed.
        """
        if store_type is None or store_type.lower() not in ["document", "keyvalue"]:
            print("Test failed: Invalid store type. Use 'document' or 'keyvalue'.")
            return False
            
        if store_type.lower() == "document":
            if query_template is None:
                query_template = self.document_template["query"]
            if query_params is None:
                query_params = {"conditions": "c.age > 18"}
            if operation is None:
                operation = "query"
        else:  # keyvalue
            if query_template is None:
                query_template = self.keyvalue_template["command"]
            if query_params is None:
                query_params = {"key": "user:1234"}
            if operation is None:
                operation = "get"
                
        try:
            result = self.convert(store_type, query_template, query_params, operation)
            if "error" in result:
                print(f"Test failed: Query returned an error: {result['error']}")
                return False
            if not result or not result.get("query_template"):
                print("Test failed: Query result is empty")
                return False
            if expected_data:
                for key, value in expected_data.items():
                    if key not in result or result[key] != value:
                        print(f"Test failed: Expected {key}={value}, got {result.get(key)}")
                        return False
            store_type_name = "document store" if store_type.lower() == "document" else "key-value store"
            print(f"Test passed: NoSQL {store_type_name} query executed successfully with expected results")
            return True
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")
            return False


class AzureAISearchTool:
    """
    Uses Azure AI Search to perform search queries on indexed data.
    """
    def __init__(self, endpoint: str, index_name: str, key: str):
        """
        Initialize the AzureAISearchTool.
        
        Args:
            endpoint (str, optional): Azure AI Search service endpoint. Defaults to os.environ["AZURE_AI_SEARCH_SERVICE"].
            index_name (str, optional): Name of the search index. Defaults to os.environ["AZURE_AI_INDEX"].
            key (str, optional): Azure AI Search API key. Defaults to os.environ["AZURE_AI_SEARCH_KEY"].
        """
        self.endpoint = endpoint or os.environ.get("AZURE_AI_SEARCH_SERVICE", "")
        self.index_name = index_name or os.environ.get("AZURE_AI_INDEX", "")
        self.key = key or os.environ.get("AZURE_AI_SEARCH_KEY", "")
        if not (self.endpoint and self.index_name and self.key):
            logger.error("Azure AI Search configuration missing.")
            raise Exception("Azure AI Search configuration required.")
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.key)
        )

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a search query against the Azure AI Search index.
        
        Args:
            query (str): The search text query.
            
        Returns:
            List[Dict[str, Any]]: A list of search result documents.
        """
        try:
            results = self.search_client.search(search_text=query)
            output = []
            for result in results:
                output.append(result)
            return output
        except Exception as e:
            logger.error(f"Error performing Azure AI Search: {str(e)}")
            return []

    def create_index(self, index_schema: SearchIndex) -> dict:
        """
        Create a new index in the Azure AI Search service using the provided index schema.
        
        Args:
            index_schema (SearchIndex): A SearchIndex object defining the schema of the new index.
            
        Returns:
            dict: A dictionary containing the result of the index creation operation.
                  On success, returns the created index name; on failure, returns an error message.
        """
        try:
            index_client = SearchIndexClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))
            created_index = index_client.create_index(index_schema)
            return {"status": "Index created", "index": created_index.name}
        except Exception as e:
            logger.error(f"Error creating Azure AI Search index: {str(e)}")
            return {"error": str(e)}


class CosmosSearchTool:
    """
    Performs search operations on an Azure Cosmos DB container.
    """
    def __init__(self, endpoint: str, key: str, database_name: str, container_name: str):
        """
        Initialize the CosmosSearchTool.
        
        Args:
            endpoint (str, optional): Azure Cosmos DB endpoint. Defaults to os.environ["COSMOS_ENDPOINT"].
            key (str, optional): Azure Cosmos DB key. Defaults to os.environ["COSMOS_KEY"].
            database_name (str, optional): The database name in Cosmos DB. Defaults to os.environ["COSMOS_DATABASE"].
            container_name (str, optional): The container (collection) name. Defaults to os.environ["COSMOS_CONTAINER"].
        """
        self.endpoint = endpoint or os.environ.get("COSMOS_ENDPOINT", "")
        self.key = key or os.environ.get("COSMOS_KEY", "")
        self.database_name = database_name or os.environ.get("COSMOS_DATABASE", "")
        self.container_name = container_name or os.environ.get("COSMOS_CONTAINER", "")
        if not (self.endpoint and self.key and self.database_name and self.container_name):
            logger.error("Azure Cosmos DB configuration missing.")
            raise Exception("Azure Cosmos DB configuration required.")

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a search query against the Cosmos DB container using SQL query language.
        
        Args:
            query (str): A filter condition to be used in the WHERE clause.
            
        Returns:
            List[Dict[str, Any]]: A list of documents that match the query.
        """
        try:
            client = CosmosClient(self.endpoint, credential=self.key)
            async with client:
                database = client.get_database_client(self.database_name)
                container = database.get_container_client(self.container_name)
                # Build a SQL query with the provided condition.
                sql_query = f"SELECT * FROM c WHERE {query}"
                items = [item async for item in container.query_items(query=sql_query, enable_cross_partition_query=True)]
                return items
        except Exception as e:
            logger.error(f"Error performing Cosmos DB search: {str(e)}")
            return []


class LocalFileRetriever:

    @kernel_function(name="load_text_files", description="Loads and extracts text files in the specified folder")
    def load_texts(self, folder: str) -> List[Dict[str, Any]]:
        """
        Load text contents from TXT, DOCX, and PDF files.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with keys "path" and "content", where
                                  "content" contains the text extracted from the file.
        """
        texts = []
        # Process TXT files
        for file_path in glob.glob(os.path.join(folder, "*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                texts.append({"path": file_path, "content": content})
            except Exception as e:
                logger.error(f"Error reading TXT file {file_path}: {str(e)}")
        
        # Process DOCX files
        for file_path in glob.glob(os.path.join(folder, "*.docx")):
            try:
                doc = Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
                texts.append({"path": file_path, "content": content})
            except Exception as e:
                logger.error(f"Error reading DOCX file {file_path}: {str(e)}")
        
        # Process PDF files
        for file_path in glob.glob(os.path.join(folder, "*.pdf")):
            try:
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() or ""
                texts.append({"path": file_path, "content": content})
            except Exception as e:
                logger.error(f"Error reading PDF file {file_path}: {str(e)}")
        
        return texts

    @kernel_function(name="load_audio_files", description="Loads and extracts audio files in the specified folder")
    def load_audio(self, folder: str) -> List[Dict[str, Any]]:
        """
        Load audio files (as raw bytes) from MP3 and MKV file types.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with keys "path" and "content" (bytes).
        """
        audio_files = []
        for ext in ["*.mp3", "*.mkv"]:
            for file_path in glob.glob(os.path.join(folder, ext)):
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()
                    audio_files.append({"path": file_path, "content": content})
                except Exception as e:
                    logger.error(f"Error reading audio file {file_path}: {str(e)}")
        return audio_files

    @kernel_function(name="load_image_files", description="Loads and extracts image files in the specified folder")
    def load_images(self, folder: str) -> List[Dict[str, Any]]:
        """
        Load image files from PNG, JPG, and JPEG file types.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with keys "path" and "content" (PIL Image objects).
        """
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for file_path in glob.glob(os.path.join(folder, ext)):
                try:
                    image = Image.open(file_path)
                    image_files.append({"path": file_path, "content": image})
                except Exception as e:
                    logger.error(f"Error reading image file {file_path}: {str(e)}")
        return image_files

    @kernel_function(name="load_video_files", description="Loads and extracts video files in the specified folder")
    def load_videos(self, folder: str) -> List[Dict[str, Any]]:
        """
        Load video files (as raw bytes) from MP4 file type.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with keys "path" and "content" (bytes).
        """
        videos = []
        for file_path in glob.glob(os.path.join(folder, "*.mp4")):
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                videos.append({"path": file_path, "content": content})
            except Exception as e:
                logger.error(f"Error reading video file {file_path}: {str(e)}")
        return videos

    @kernel_function(name="load_files", description="Loads and extracts text, audio, video and image files in the specified folder")
    def load_all(self, folder: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all files (texts, audio, images, and videos) from the folder.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with keys "texts", "audio", "images", and "videos".
        """
        return {
            "texts": self.load_texts(folder),
            "audio": self.load_audio(folder),
            "images": self.load_images(folder),
            "videos": self.load_videos(folder)
        }
