"""
This module provides advanced image processing capabilities for multi-modal applications by integrating
Azure AI Vision with Azure CosmosDB and Azure AI Search. It supports image content detection,
face recognition, OCR, object detection, and embedding generation for images.

Classes:
    ImageProcessor:
        Handles image file encoding and performs content detection, face recognition, OCR, 
        and object detection with Azure AI Vision SDK.
    ImageEmbedder:
        Generates image embeddings from images and persists results to CosmosDB and Azure AI Search.
    ImageAnswer:
        Applies load, save and frame images for better and complementary usage of vision models.

Dependencies:
    - azure-ai-vision
    - azure-cosmos
    - azure-search-documents
    - azure-identity
    - requests
    - numpy
    - PIL
"""

import os
import time
import json
import logging
import base64
import requests

import urllib.request
import numpy as np
from typing import Dict, List, Any, Union
from io import BytesIO
from PIL import Image

from azure.ai.vision.imageanalysis.aio import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.cosmos.aio import CosmosClient
from azure.search.documents import SearchClient

from semantic_kernel.functions.kernel_function_decorator import kernel_function


logger = logging.getLogger(__name__)


AZURE_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT", "")
AZURE_VISION_KEY = os.environ.get("AZURE_VISION_KEY", "")


class ImageProcessor:
    """
    Handles image file encoding and performs content detection as well as face recognition, 
    OCR and object detection with Azure AI Vision SDK.
    """

    def __init__(self):
        """
        Initialize the ImageProcessor with an Azure AI Vision key and endpoint.
        
        Args:
            key (str): The Azure AI Vision subscription key.
            endpoint (str): The endpoint URL for Azure AI Vision service.
        """
        self.vision_client = ImageAnalysisClient(
            endpoint=AZURE_VISION_ENDPOINT,
            credential=AzureKeyCredential(AZURE_VISION_KEY)
        )

    @kernel_function(name="IngestImage", description="Ingests an image from a URL and converts it to bytes")
    @kernel_function(name="IngestImage", description="Ingests an image from a URL and converts it to bytes")
    async def ingest_image(self, image_url: str) -> bytes:
        """
        Ingest an image from a URL and convert it to bytes.
        
        Args:
            image_url (str): URL of the image to ingest.
            
        Returns:
            bytes: The image content as bytes.
            
        Raises:
            Exception: If the image cannot be retrieved or processed.
        """
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error ingesting image from URL: {str(e)}")
            raise Exception(f"Failed to ingest image: {str(e)}")

    @kernel_function(name="LoadImages", description="Loads multiple images from URLs or byte arrays")
    @kernel_function(name="LoadImages", description="Loads multiple images from URLs or byte arrays")
    async def load_images(self, image_sources: List[Union[str, bytes]]) -> List[bytes]:
        """
        Load multiple images from various sources (URLs or bytes) into an array of bytes.
        
        Args:
            image_sources (List[Union[str, bytes]]): List of image URLs or byte arrays.
            
        Returns:
            List[bytes]: List of images as byte arrays.
        """
        images = []
        for source in image_sources:
            try:
                if isinstance(source, str):
                    image_bytes = await self.ingest_image(source)
                else:
                    image_bytes = source
                images.append(image_bytes)
            except Exception as e:
                logger.error(f"Error loading image: {str(e)}")
                images.append(None)
        return images

    @kernel_function(name="SaveImageSet", description="Saves a set of images with metadata to the database")
    @kernel_function(name="SaveImageSet", description="Saves a set of images with metadata to the database")
    async def save_image_set(self, images: List[bytes], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Save a set of images with their metadata to the database.


        Args:
            images (List[bytes]): List of images as bytes.
            metadata (List[Dict[str, Any]]): List of metadata for each image.


        Returns:
            List[Dict[str, Any]]: List of responses from the database.
        """
        try:
            cosmos_endpoint = os.environ.get("COSMOS_VISION_ENDPOINT", "")
            cosmos_key = os.environ.get("COSMOS_KEY", AZURE_VISION_KEY)
            cosmos_endpoint = os.environ.get("COSMOS_VISION_ENDPOINT", "")
            cosmos_key = os.environ.get("COSMOS_KEY", AZURE_VISION_KEY)
            database_name = os.environ.get("COSMOS_DATABASE", "")
            container_name = os.environ.get("COSMOS_CONTAINER", "")

            client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)

            results = []
            for i, (image_bytes, meta) in enumerate(zip(images, metadata)):
                if image_bytes is None:
                    continue

                encoded = base64.b64encode(image_bytes).decode('utf-8')
                item = {
                    "id": meta.get("id", f"image-{int(time.time())}-{i}"),
                    "image_base64": encoded,
                    "metadata": meta,
                    "type": "image",
                    "timestamp": time.time()
                }

                response = await container.upsert_item(item)
                results.append(response)

            return results

        except Exception as e:
            logger.error(f"Error saving image set: {str(e)}")
            raise Exception(f"Failed to save image set: {str(e)}")

    @kernel_function(name="AddCaptions", description="Uses Azure AI Vision to add captions to images")
    @kernel_function(name="AddCaptions", description="Uses Azure AI Vision to add captions to images")
    async def add_captions(self, images: List[bytes]) -> List[Dict[str, str]]:
        """
        Uses Azure AI Vision SDK to add captions to images.
        
        Args:
            images (List[bytes]): List of images as bytes.
            
        Returns:
            List[Dict[str, str]]: List of captions for each image.
        """
        captions = []
        
        async with self.vision_client as client:
            for image in images:
                if image is None:
                    captions.append({"caption": "No image available"})
                    continue
                    
                try:
                    analysis_result= await client.analyze(
                        image_data=image,
                        visual_features=[VisualFeatures.CAPTION]
                    )
                    if hasattr(analysis_result, 'caption') and analysis_result.caption:
                        captions.append({
                            "caption": analysis_result.caption.text,  # Use .content instead of .text
                            "confidence": analysis_result.caption.confidence
                        })
                    else:
                        captions.append({"caption": "No caption generated"})

                except Exception as e:
                    logger.error(f"Error generating caption: {str(e)}")
                    captions.append({"caption": f"Caption error: {str(e)}"})
                
        return captions

    @kernel_function(name="ExtractTags", description="Extracts tags from images using Azure AI Vision")
    @kernel_function(name="ExtractTags", description="Extracts tags from images using Azure AI Vision")
    async def extract_tags(self, images: List[bytes]) -> List[Dict[str, Any]]:
        """
        Uses Azure AI Vision SDK to extract tags from images.
        
        Args:
            images (List[bytes]): List of images as bytes.
            
        Returns:
            List[Dict[str, Any]]: List of tag sets for each image.
        """
        all_tags = []
        async with self.vision_client as client:
            for image in images:
                if image is None:
                    all_tags.append({"tags": []})
                    continue

                try:
                    result= await client.analyze(
                        image_data=image,
                        visual_features=[VisualFeatures.TAGS]
                    )

                    tags = []
                    if result.tags:
                        for tag in result.tags.list:
                            tags.append({
                                "name": tag.get("name"),
                                "confidence": tag.get("confidence")
                            })

                    all_tags.append({"tags": tags})

                except Exception as e:
                    logger.error(f"Error extracting tags: {str(e)}")
                    all_tags.append({"tags": [], "error": str(e)})

        return all_tags

    @kernel_function(name="CropImages", description="Crops images to their region of interest")
    @kernel_function(name="CropImages", description="Crops images to their region of interest")
    async def crop_images(self, images: List[bytes]) -> List[bytes]:
        """
        Uses Azure AI Vision SDK to crop images to the region of interest.
        
        Args:
            images (List[bytes]): List of images as bytes.
            
        Returns:
            List[bytes]: List of cropped images.
        """
        cropped_images = []

        for image in images:
            if image is None:
                cropped_images.append(None)
                continue

            try:
                image_data = Image.open(image)
                original_size = image_data.size
                result = await self.vision_client.analyze(
                    image_data=image,
                    visual_features=[VisualFeatures.SMART_CROPS, VisualFeatures.OBJECTS]
                )

                if result.objects and len(result.objects) > 0:
                    valid_objects = [obj for obj in result.objects if hasattr(obj, 'bounding_box')]
                    largest_object = max(valid_objects, key=lambda x: (max(p.x for p in x.bounding_box) - min(p.x for p in x.bounding_box)) * (max(p.y for p in x.bounding_box) - min(p.y for p in x.bounding_box)))
                    bbox = largest_object.bounding_box
                    x_coords = [p.x for p in bbox]
                    y_coords = [p.y for p in bbox]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)

                    crop_box = (
                        int(min_x * original_size[0]),
                        int(min_y * original_size[1]),
                        int(max_x * original_size[0]),
                        int(max_y * original_size[1])
                    )
                    cropped = image_data.crop(crop_box)

                    buffer = BytesIO()
                    cropped.save(buffer, format="PNG")
                    cropped_images.append(buffer.getvalue())
                else:
                    cropped_images.append(image)
                    
            except Exception as e:
                logger.error(f"Error cropping image: {str(e)}")
                cropped_images.append(image)

        return cropped_images


class ImageEmbedder:
    """
    Generates image embeddings from images and persists results to CosmosDB and Azure AI Search.
    """

    def __init__(self):
        """
        Initialize the ImageEmbedder with an Azure AI Vision key and endpoint.
        
        Args:
            key (str): The Azure AI Vision subscription key.
            endpoint (str): The endpoint URL for Azure AI Vision service.
        """
        self.vision_client = ImageAnalysisClient(
            endpoint=AZURE_VISION_ENDPOINT,
            credential=AzureKeyCredential(AZURE_VISION_KEY)
        )

    @kernel_function(name="EmbedImage", description="Generates and stores embeddings for an image")
    async def embed_image(self, image: bytes, metadata: Dict[str, Any]) -> Dict[str, Any] | List[float]:
        """
        Embeds an image and saves the embeddings into a database.
        
        Args:
            image (bytes): The image content as bytes.
            metadata (Dict[str, Any]): Metadata associated with the image.
            
        Returns:
            Dict[str, Any]: The database response containing the saved embedding.
        """
        try:
            embedding = self._generate_embedding(image, metadata.get("name", "image"))


            cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT", "")
            cosmos_key = os.environ.get("COSMOS_KEY", AZURE_VISION_KEY)
            cosmos_key = os.environ.get("COSMOS_KEY", AZURE_VISION_KEY)
            database_name = os.environ.get("COSMOS_DATABASE", "")
            container_name = os.environ.get("COSMOS_CONTAINER", "")
            
            try:
                client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
                database = client.get_database_client(database_name)
                container = database.get_container_client(container_name)

                item = {
                    "id": metadata.get("id", f"embedding-{int(time.time())}"),
                    "embedding": embedding,
                    "metadata": metadata,
                    "type": "image_embedding",
                    "timestamp": time.time()
                }

                response = await container.upsert_item(item)

                search_endpoint = os.environ.get("SEARCH_ENDPOINT", "")
                search_key = os.environ.get("SEARCH_KEY", AZURE_VISION_KEY)
                index_name = os.environ.get("SEARCH_INDEX", "")

                search_client = SearchClient(
                    endpoint=search_endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(search_key)
                )

                search_document = {
                    "id": item["id"],
                    "embedding": embedding,
                    "metadata_str": str(metadata),
                    "type": "image_embedding"
                }

                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        search_document[key] = value

                search_client.upload_documents(documents=[search_document])
                return response
            except Exception as e:
                logger.error(f"Error saving embedding to database: {str(e)}")
                pass

            return embedding

        except Exception as e:
            logger.error(f"Error embedding image: {str(e)}")
            raise Exception(f"Failed to embed image: {str(e)}")

    def _generate_embedding(self, image: bytes, image_name: str) -> List[float]:
        """
        Generate an embedding vector from the image using Azure AI Vision.
        
        Args:
            image (bytes): The image content as bytes.
            
        Returns:
            List[float]: The embedding vector.
        """
        data = {
        "input_data": {
            "columns": ["image"],
            "index": [image_name],
            "data": [
                [image.decode("utf-8")]
            ],
        },
        "params": {}
        }

        body = str.encode(json.dumps(data))
        url = os.getenv('AZURE_IMAGE_EMBEDDINGS_URL', '')

        api_key = os.getenv('AZURE_IMAGE_EMBEDDINGS_KEY', None)
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
        req = urllib.request.Request(url, body, headers)
        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read().decode("utf8", 'ignore'))[0]['image_features']
            return result
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 1024

    @kernel_function(name="CalculateSimilarity", description="Calculates cosine similarity between image embeddings")
    @kernel_function(name="CalculateSimilarity", description="Calculates cosine similarity between image embeddings")
    def calculate_similarity(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Calculates cosine similarity between a set of image embeddings.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            
        Returns:
            List[List[float]]: Similarity matrix where each element [i][j] is the 
                              similarity between embeddings[i] and embeddings[j].
        """
        try:
            vectors = np.array(embeddings)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = vectors / norms
            similarity_matrix = np.dot(normalized, normalized.T)
            
            return similarity_matrix.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            # Return empty similarity matrix on error
            return [[0.0] * len(embeddings) for _ in range(len(embeddings))]


class ImageAnswer:
    """
    Applies load, save, and frame images for better and complementary usage of vision models.
    """

    def __init__(self):
        """
        Initialize the ImageAnswer with an Azure AI Vision key and endpoint.
        
        Args:
            key (str): The Azure AI Vision subscription key.
            endpoint (str): The endpoint URL for Azure AI Vision service.
        """
        self.vision_client = ImageAnalysisClient(
            endpoint=AZURE_VISION_ENDPOINT,
            credential=AzureKeyCredential(AZURE_VISION_KEY)
        )

    @kernel_function(name="ExtractText", description="Extracts text from images using OCR")
    @kernel_function(name="ExtractText", description="Extracts text from images using OCR")
    async def extract_text(self, image: bytes) -> Dict[str, Any]:
        """
        Uses Azure AI Vision SDK to detect image text with OCR and adds it to the image data.
        
        Args:
            image (bytes): The image content as bytes.
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted text and related metadata.
        """
        try:
            result = await self.vision_client.analyze(
                image_data=image,
                visual_features=[VisualFeatures.READ]
            )

            extracted_text = ""
            text_regions = []

            if hasattr(result, 'read') and result.read:
                if hasattr(result.read, 'blocks') and result.read.blocks:
                    for block in result.read.blocks:
                        if hasattr(block, 'lines') and block.lines:
                            for line in block.lines:
                                line_text = line.text if hasattr(line, 'text') else ""
                                if line_text:
                                    extracted_text += line_text + " "
                                if hasattr(line, 'bounding_polygon') and line.bounding_polygon:
                                    bbox = line.bounding_polygon
                                    if isinstance(bbox, list) and all(hasattr(point, 'x') for point in bbox):
                                        x_coords = [point.x for point in bbox]
                                        y_coords = [point.y for point in bbox]
                                        min_x, max_x = min(x_coords), max(x_coords)
                                        min_y, max_y = min(y_coords), max(y_coords)
                                        text_regions.append({
                                            "text": line_text,
                                            "bounding_box": {
                                                "x": min_x,
                                                "y": min_y,
                                                "width": max_x - min_x,
                                                "height": max_y - min_y
                                            },
                                            "confidence": 0.0
                                        })

                language = "unknown"
                
                return {
                    "text": extracted_text.strip(),
                    "regions": text_regions,
                    "language": language
                }
            else:
                return {
                    "text": "",
                    "regions": [],
                    "language": "unknown"
                }

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {"text": "", "regions": [], "error": str(e)}

    @kernel_function(name="DetectObjects", description="Detects and identifies objects in images")
    @kernel_function(name="DetectObjects", description="Detects and identifies objects in images")
    async def detect_objects(self, image: bytes) -> Dict[str, Any]:
        """
        Uses Azure AI Vision SDK to detect objects in an image.
        
        Args:
            image (bytes): The image content as bytes.
            
        Returns:
            Dict[str, Any]: Dictionary containing detected objects and related metadata.
        """
        try:
            result = await self.vision_client.analyze(
                image_data=image,
                visual_features=[VisualFeatures.OBJECTS]
            )
            
            # Extract objects
            objects = []
            
            if result.objects:
                for obj in result.objects:
                    x_coords = [p.x for p in obj.bounding_box]
                    y_coords = [p.y for p in obj.bounding_box]
                    x_val = min(x_coords)
                    y_val = min(y_coords)
                    width_val = max(x_coords) - x_val
                    height_val = max(y_coords) - y_val
                    objects.append({
                        "name": obj.name,
                        "confidence": obj.confidence,
                        "bounding_box": {
                            "x": x_val,
                            "y": y_val,
                            "width": width_val,
                            "height": height_val
                        }
                    })
            
            return {
                "objects": objects,
                "count": len(objects)
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return {"objects": [], "count": 0, "error": str(e)}
