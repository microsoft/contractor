"""
This module provides advanced video processing capabilities for multi-modal applications by integrating
Azure AI Vision and Speech services with Azure CosmosDB and Azure AI Search. It supports video frame
analysis, audio extraction and transcription, and embedding generation for video content.

Classes:
    VideoProcessor:
        Handles video file processing, frame extraction, and audio extraction using MoviePy and Azure services.
    VideoEmbedder:
        Generates embeddings from video frames and audio, and persists results to storage.
    VideoAnswer:
        Analyzes video content and provides summaries and insights based on frames and audio.

Dependencies:
    - azure-cognitiveservices-speech
    - azure-ai-vision
    - azure-storage-blob
    - azure-search-documents
    - azure-identity
    - moviepy
    - PIL
    - numpy

Usage:
    Import the desired class and instantiate it with the necessary configuration parameters (e.g., Azure keys,
    endpoints, etc.). Replace the placeholder values with your actual Azure service details before deployment.
"""

import os
import time
import json
import logging
import base64
import io
from typing import Dict, List, Any, Optional

import numpy as np
import requests
from PIL import Image

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from .audio import AudioProcessor
from .image import ImageProcessor, ImageAnswer
import cv2


logger = logging.getLogger(__name__)



class VideoProcessor:
    """
    Handles video file processing, frame extraction, and audio extraction.

    Attributes:
        vision_key (str): Azure Vision subscription key.
        vision_endpoint (str): Azure Vision endpoint.
        speech_key (str): Azure Speech subscription key.
        speech_region (str): Azure Speech region.
    """

    def __init__(self):
        """
        Initialize the VideoProcessor with Azure Vision and Speech credentials.

        Args:
            vision_key (str): Azure Vision subscription key.
            vision_endpoint (str): Azure Vision endpoint.
            speech_key (str): Azure Speech subscription key.
            speech_region (str): Azure Speech region.
        """
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.blob_storage_connection = os.environ.get("BLOB_STORAGE_CONNECTION_STRING", "")
        self.container_name = os.environ.get("BLOB_STORAGE_STORAGE_ACCOUNT", "")

    @kernel_function(
        name="process_video",
        description="Process a video file by extracting frames and audio"
    )
    async def process_video(self, video_path: str, frame_count: int = 10) -> Dict[str, Any]:
        """
        Process a video file by extracting frames and audio.
        
        Args:
            video_path (str): Path to the video file.
            frame_count (int, optional): Number of frames to extract. Defaults to 10.
            
        Returns:
            Dict[str, Any]: Processing results including frames, audio transcription, and metadata.
        """
        path, file_name = os.path.split(video_path) 

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps if fps > 0 else 0

            frames = []
            frame_times = []
            frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
            
            for frame_idx in frame_indices:
                # Set position to the exact frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame at index {frame_idx}")
                    continue
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)

                buffered = io.BytesIO()
                frame_image.save(buffered, format="PNG")
                frames.append(buffered.getvalue())

                frame_time = frame_idx / fps
                frame_times.append(frame_time)

            cap.release()

            audio_transcription = None
            try:
                temp_audio_file = f"temp_audio_{int(time.time())}.wav"
                os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {temp_audio_file} -y")
                
                if os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                    audio_transcription = self.audio_processor.process_transcription(temp_audio_file)
                    os.remove(temp_audio_file)
            except Exception as audio_error:
                logger.warning(f"Audio extraction failed: {str(audio_error)}")
            
            return {
                "video_name": file_name,
                "frames": frames,
                "frame_times": frame_times,
                "audio_transcription": audio_transcription,
                "duration": total_duration,
                "fps": fps,
                "total_frames": total_frames,
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {"error": str(e)}

    @kernel_function(
        name="analyze_frames",
        description="Analyze extracted frames using Azure AI Vision to get captions and tags"
    )
    async def analyze_frames(self, frames: List[bytes]) -> List[Dict[str, Any]]:
        """
        Analyze extracted frames using Azure AI Vision.
        
        Args:
            frames (List[bytes]): List of frame images as bytes.
            
        Returns:
            List[Dict[str, Any]]: Analysis results for each frame.
        """
        try:
            captions = await self.image_processor.add_captions(frames)

            tags = await self.image_processor.extract_tags(frames)

            frame_analyses = []
            for i, (frame, caption, tag_set) in enumerate(zip(frames, captions, tags)):
                frame_analyses.append({
                    "frame_index": i,
                    "caption": caption.get("caption", "No caption"),
                    "tags": tag_set.get("tags", [])
                })
            
            return frame_analyses
        
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            return [{"error": str(e)}]

    @kernel_function(
        name="save_processed_video",
        description="Save processed video frames, metadata and audio to blob storage"
    )
    async def save_processed_video(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save processed video information to blob storage.
        
        Args:
            video_info (Dict[str, Any]): Video processing results.
            
        Returns:
            Dict[str, Any]: Blob storage response.
        """
        try:
            video_name = video_info.get("video_name", f"video-{int(time.time())}")
            frames = video_info.get("frames", [])
            frame_times = video_info.get("frame_times", [])
            
            if not self.blob_storage_connection:
                return {"error": "Blob storage connection string not configured"}

            blob_service_client = BlobServiceClient.from_connection_string(self.blob_storage_connection)
            
            saved_frames = []
            for i, (frame, frame_time) in enumerate(zip(frames, frame_times)):
                frame_base64 = base64.b64encode(frame).decode('utf-8')

                frame_metadata = {
                    "video_name": video_name,
                    "frame_index": i,
                    "timestamp": frame_time,
                    "total_frames": len(frames)
                }

                blob_client = blob_service_client.get_blob_client(
                    container=f"{self.container_name}/streaming/video/frames",
                    blob=f"{video_name}_frame_{i}.png"
                )
                blob_client.upload_blob(frame, overwrite=True)

                blob_client = blob_service_client.get_blob_client(
                    container=f"{self.container_name}/streaming/video/metadata",
                    blob=f"{video_name}_frame_{i}.json"
                )
                blob_client.upload_blob(json.dumps(frame_metadata), overwrite=True)
                
                saved_frames.append({
                    "frame_index": i,
                    "frame_time": frame_time,
                    "blob_url": blob_client.url
                })

            if video_info.get("audio_transcription"):
                blob_client = blob_service_client.get_blob_client(
                    container=f"{self.container_name}/streaming/video/audio",
                    blob=f"{video_name}_audio.json"
                )
                audio_data = {
                    "video_name": video_name,
                    "transcription": video_info["audio_transcription"]
                }
                blob_client.upload_blob(json.dumps(audio_data), overwrite=True)
            
            return {
                "video_name": video_name,
                "saved_frames": saved_frames,
                "audio_saved": bool(video_info.get("audio_transcription")),
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error(f"Error saving processed video: {str(e)}")
            return {"error": str(e)}



class VideoEmbedder:
    """
    Generates embeddings from video frames and audio, and persists results to storage.
    
    Attributes:
        vision_key (str): Azure Vision subscription key.
        vision_endpoint (str): Azure Vision endpoint.
        speech_key (str): Azure Speech subscription key.
        speech_region (str): Azure Speech region.
    """
    
    def __init__(self):
        """
        Initialize the VideoEmbedder with Azure Vision and Speech credentials.

        Args:
            vision_key (str): Azure Vision subscription key.
            vision_endpoint (str): Azure Vision endpoint.
            speech_key (str): Azure Speech subscription key.
            speech_region (str): Azure Speech region.
        """
        self.image_embedder = ImageProcessor()
        self.azureml_api_key = os.environ.get('AZUREML_API', '')

    @kernel_function(
        name="generate_frame_embedding",
        description="Generate an embedding vector for a single video frame"
    )
    async def _generate_frame_embedding(self, frame: bytes, frame_name: str) -> List[float]:
        """
        Generate an embedding for a single video frame.

        Args:
            frame (bytes): Frame image as bytes.
            frame_name (str): Name/identifier for the frame.

        Returns:
            List[float]: The embedding vector.
        """
        try:
            data = {
                "input_data": {
                    "columns": ["image"],
                    "index": [frame_name],
                    "data": [
                        [base64.b64encode(frame).decode("utf-8")]
                    ],
                },
                "params": {}
            }

            body = str.encode(json.dumps(data))
            url = 'https://image-embeddings-ibscq.eastus2.inference.ml.azure.com/score'

            if not self.azureml_api_key:
                raise Exception("AzureML API key not provided for embedding generation")

            headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + self.azureml_api_key)}
            response = requests.post(url, headers=headers, data=body)
            response.raise_for_status()
            
            result = json.loads(response.text)[0]['image_features']
            return result
                
        except Exception as e:
            logger.error(f"Error generating frame embedding: {str(e)}")
            return [0.0] * 1024

    @kernel_function(
        name="embed_video",
        description="Generate embeddings for video frames and audio, storing results in search index"
    )
    async def embed_video(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for video frames and audio.
        
        Args:
            video_info (Dict[str, Any]): Video processing results.
            
        Returns:
            Dict[str, Any]: Embedding results for the video.
        """
        try:
            video_name = video_info.get("video_name", f"video-{int(time.time())}")
            frames = video_info.get("frames", [])
            frame_times = video_info.get("frame_times", [])
            audio_transcription = video_info.get("audio_transcription")

            frame_embeddings = []
            for i, (frame, frame_time) in enumerate(zip(frames, frame_times)):
                frame_embedding = await self._generate_frame_embedding(frame, f"{video_name}_frame_{i}")
                frame_embeddings.append({
                    "frame_index": i,
                    "frame_time": frame_time,
                    "embedding": frame_embedding
                })

            audio_embedding = None
            if audio_transcription:
                audio_embedding = [float(ord(c))/1000 for c in audio_transcription[:100]]

            search_endpoint = os.environ.get("SEARCH_ENDPOINT", "")
            search_key = os.environ.get("SEARCH_KEY", "")
            index_name = os.environ.get("SEARCH_INDEX", "")
            
            if search_endpoint and search_key and index_name:
                search_client = SearchClient(
                    endpoint=search_endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(search_key)
                )
                
                search_document = {
                    "id": f"video-{video_name}-{int(time.time())}",
                    "video_name": video_name,
                    "frame_count": len(frames),
                    "duration": video_info.get("duration", 0),
                    "has_audio": audio_transcription is not None,
                    "timestamp": time.time(),
                    "type": "video"
                }
                
                search_client.upload_documents(documents=[search_document])
            
            return {
                "video_name": video_name,
                "frame_embeddings": frame_embeddings,
                "audio_embedding": audio_embedding,
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error(f"Error embedding video: {str(e)}")
            return {"error": str(e)}

    @kernel_function(
        name="calculate_video_similarity",
        description="Calculate similarity between multiple videos based on their frame embeddings"
    )
    def calculate_video_similarity(self, video_embeddings_list: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Calculate similarity between multiple videos based on their frame embeddings.
        
        Args:
            video_embeddings_list (List[Dict[str, Any]]): List of video embedding results.
            
        Returns:
            List[List[float]]: Similarity matrix where each element [i][j] is the 
                              similarity between video_embeddings_list[i] and video_embeddings_list[j].
        """
        try:
            # Average frame embeddings for each video
            avg_embeddings = []
            for video_embeddings in video_embeddings_list:
                frame_embeddings = video_embeddings.get("frame_embeddings", [])
                if not frame_embeddings:
                    avg_embeddings.append(None)
                    continue
                
                # Extract the actual embedding vectors
                embedding_vectors = [frame["embedding"] for frame in frame_embeddings if "embedding" in frame]
                if not embedding_vectors:
                    avg_embeddings.append(None)
                    continue
                
                # Calculate average embedding
                embedding_array = np.array(embedding_vectors)
                avg_embedding = np.mean(embedding_array, axis=0).tolist()
                avg_embeddings.append(avg_embedding)
            
            # Filter out None values
            valid_embeddings = [emb for emb in avg_embeddings if emb is not None]
            if not valid_embeddings:
                return []
            
            # Calculate cosine similarity
            vectors = np.array(valid_embeddings)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = vectors / norms
            similarity_matrix = np.dot(normalized, normalized.T)
            
            return similarity_matrix.tolist()
        
        except Exception as e:
            logger.error(f"Error calculating video similarity: {str(e)}")
            return []



class VideoAnswer:
    """
    Analyzes video content and provides summaries and insights based on frames and audio.
    
    Attributes:
        vision_key (str): Azure Vision subscription key.
        vision_endpoint (str): Azure Vision endpoint.
        speech_key (str): Azure Speech subscription key.
        speech_region (str): Azure Speech region.
    """
    
    def __init__(self):
        """
        Initialize the VideoAnswer with Azure Vision and Speech credentials.
        
        Args:
            vision_key (str): Azure Vision subscription key.
            vision_endpoint (str): Azure Vision endpoint.
            speech_key (str): Azure Speech subscription key.
            speech_region (str): Azure Speech region.
        """
        self.image_answer = ImageAnswer()
        self.gpt4v_url = os.environ.get("GPT4V_URL", "")
        self.gpt4v_key = os.environ.get("GPT4V_KEY", "")

    @kernel_function
    async def extract_objects_from_frames(self, frames: List[bytes]) -> List[Dict[str, Any]]:
        """
        Extract objects from video frames.
        
        Args:
            frames (List[bytes]): List of frame images as bytes.
            
        Returns:
            List[Dict[str, Any]]: Object detection results for each frame.
        """
        try:
            frame_objects = []
            for i, frame in enumerate(frames):
                objects = await self.image_answer.detect_objects(frame)
                frame_objects.append({
                    "frame_index": i,
                    "objects": objects.get("objects", []),
                    "count": objects.get("count", 0)
                })
            
            return frame_objects
        
        except Exception as e:
            logger.error(f"Error extracting objects from frames: {str(e)}")
            return [{"error": str(e)}]

    @kernel_function
    async def extract_text_from_frames(self, frames: List[bytes]) -> List[Dict[str, Any]]:
        """
        Extract text from video frames using OCR.
        
        Args:
            frames (List[bytes]): List of frame images as bytes.
            
        Returns:
            List[Dict[str, Any]]: Text extraction results for each frame.
        """
        try:
            frame_texts = []
            for i, frame in enumerate(frames):
                text_data = await self.image_answer.extract_text(frame)
                frame_texts.append({
                    "frame_index": i,
                    "text": text_data.get("text", ""),
                    "regions": text_data.get("regions", []),
                    "language": text_data.get("language", "unknown")
                })
            
            return frame_texts
        
        except Exception as e:
            logger.error(f"Error extracting text from frames: {str(e)}")
            return [{"error": str(e)}]

    @kernel_function
    async def summarize_video(self, video_info: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        """
        Generate a summary of the video based on frame captions and audio transcription.
        
        Args:
            video_info (Dict[str, Any]): Video processing results.
            user_profile (Dict[str, Any]): User profile information.
            
        Returns:
            str: Video summary text.
        """
        try:
            video_name = video_info.get("video_name", "")
            frames = video_info.get("frames", [])
            
            # Get captions for frames
            frame_captions = []
            image_processor = ImageProcessor()
            captions = await image_processor.add_captions(frames)
            for caption in captions:
                frame_captions.append(caption.get("caption", ""))
            
            if not self.gpt4v_url or not self.gpt4v_key:
                return "GPT-4V configuration missing. Cannot summarize video."
            
            # Create payload for GPT-4V
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                You are a video producer assistant.
                                In your prompts, you will receive description for movie scenes and will provide a summary for the movie.
                                Pay attention to the details of the descriptions and summarize based on the profile.
                                Your role is to summarize the scenes descriptions according with the profile of the user requesting the summary.
                                You will always return a string with no escape characters.
                                You should always start your response with greetings to the user.
                                You should always translate your answer to mexican spanish.
                                """
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                    Movie Name: {video_name}.\n
                                    User Profile:\n
                                    {json.dumps(user_profile)}\n
                                    VIDEO SCENE DESCRIPTIONS:\n
                                    {json.dumps(frame_captions)}
                                """
                            }
                        ]
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 2000
            }
            
            # Call GPT-4V
            headers = {
                "Content-Type": "application/json",
                "api-key": self.gpt4v_key
            }
            response = requests.post(self.gpt4v_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            summary = result.get("choices", [])[0].get("message", {}).get("content", "")
            
            return summary
        
        except Exception as e:
            logger.error(f"Error summarizing video: {str(e)}")
            return f"Error summarizing video: {str(e)}"

    @kernel_function
    async def search_videos(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for videos in the Azure Search index.
        
        Args:
            query (str): Search query.
            
        Returns:
            List[Dict[str, Any]]: Search results.
        """
        try:
            endpoint = os.environ.get("AZURE_AI_SEARCH_SERVICE", "")
            index_name = os.environ.get("AZURE_AI_INDEX", "")
            search_key = os.environ.get("AZURE_AI_SEARCH_KEY", "")

            if not endpoint or not index_name or not search_key:
                return [{"error": "Azure Search configuration missing"}]

            search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(search_key)
            )

            results = []
            search_results = search_client.search(
                search_text=query,
                query_type="semantic",
                search_fields=["video_name", "description"],
                semantic_configuration_name="movie_retrieval"
            )

            for result in search_results:
                if result.get("@search.score", 0) > 1:
                    video_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "videos")
                    for filename in os.listdir(video_folder):
                        if filename == result["video_name"]:
                            result["video_address"] = filename
                            break
                    results.append(result)

            return results
        
        except Exception as e:
            logger.error(f"Error searching videos: {str(e)}")
            return [{"error": str(e)}]

    @kernel_function
    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a user profile from the profiles data file.
        
        Args:
            profile_name (str): Name of the profile to retrieve.
            
        Returns:
            Optional[Dict[str, Any]]: User profile information or None if not found.
        """
        try:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(current_dir, "data/profile.json"), "r") as f:
                profiles = json.load(f)
            
            for profile in profiles:
                if profile.get("name", "") == profile_name:
                    return profile
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting profile: {str(e)}")
            return None
