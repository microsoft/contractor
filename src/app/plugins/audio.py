"""
This module provides advanced audio processing capabilities for multi-modal applications by integrating
Azure Cognitive Services with Azure CosmosDB and Azure AI Search. It supports transcription, speaker
diarization, and embedding generation for audio files and real-time audio streams.

Classes:
    AudioProcessor:
        Handles audio file transcription and performs speaker diarization using the Azure Speech SDK.
    AudioEmbedder:
        Generates textual embeddings from audio transcriptions and persists results to CosmosDB and Azure AI Search.
    AudioAnswer:
        Processes real-time audio streams for transcription using the Azure Speech SDK.
    AudioStreamer:
        Streams and processes audio data to obtain transcriptions and diarization, then saves the processed content 
        to a specified backend.

Dependencies:
    - azure-cognitiveservices-speech
    - azure-cosmos
    - azure-search-documents>=11.4.0
    - azure-identity>=1.12.0
    - azure-core>=1.26.4
    - time

Usage:
    Import the desired class and instantiate it with the necessary configuration parameters (e.g., Azure subscription
    key, region, etc.). Replace the placeholder values (like "your-region", "your-cosmos-endpoint") with your actual
    Azure service details before deployment.
"""

import os
import time
import logging

from typing import Dict, List, Tuple, Any

import azure.cognitiveservices.speech as speechsdk
from azure.cosmos.aio import CosmosClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import EmbeddingsClient

from semantic_kernel.functions.kernel_function_decorator import kernel_function


logger = logging.getLogger(__name__)


AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")
AZURE_MODEL_KEY = os.environ.get("AZURE_MODEL_KEY", "")
AZURE_AUDIO_EMBEDDINGS_URL = os.environ.get("AZURE_AUDIO_EMBEDDINGS_URL", "")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-large")


class AudioProcessor:
    """
    Processa arquivos de áudio usando o Azure Speech SDK para transcrição e diarização.

    Atributos:
        speech_config (speechsdk.SpeechConfig): Configuração do serviço de fala.
    """

    def __init__(self):
        """
        Inicializa o AudioProcessor com uma chave de assinatura da Azure e região.

        Args:
            key (str): A chave de assinatura da Azure.
            region (str): A região da Azure para o serviço de fala.
        """
        self.speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        self.speech_config.request_word_level_timestamps()
        self.speech_config.enable_audio_logging()
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000"
        )

    @kernel_function(
        name="process_transcription",
        description="Transcribes an audio file to text using Azure Speech services"
    )
    def process_transcription(self, audio_file: str) -> str:
        """
        Transcreve o arquivo de áudio fornecido para texto.

        Args:
            audio_file (str): O caminho para o arquivo de áudio.

        Returns:
            str: O texto transcrito se reconhecido; caso contrário, uma mensagem de erro.
        """
        try:
            audio_input = speechsdk.AudioConfig(filename=audio_file)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, audio_config=audio_input
            )
            result = speech_recognizer.recognize_once_async().get()
            if result is not None:
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    return result.text
                return f"Transcrição falhou: {result.reason}"
            else:
                return "Erro: resultado da transcrição é None"
        except Exception as e:
            logger.error(f"Erro durante a transcrição de áudio: {str(e)}")
            return f"Erro durante a transcrição: {str(e)}"


    @kernel_function(
        name="perform_diarization",
        description="Identifies different speakers in an audio file and returns their transcribed speech"
    )
    def perform_diarization(self, audio_file: str) -> List[Dict[str, Any]]:
        """
        Realiza a diarização de falantes no arquivo de áudio fornecido.

        Args:
            audio_file (str): O caminho para o arquivo de áudio.

        Returns:
            list: Uma lista de dicionários contendo identificadores de falantes e seus textos correspondentes.
        """
        try:
            audio_input = speechsdk.AudioConfig(filename=audio_file)
            transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=self.speech_config, audio_config=audio_input
            )

            diarization_results = []

            def recognized_handler(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    speaker_id = evt.result.speaker_id if hasattr(evt.result, 'speaker_id') else 'unknown'
                    diarization_results.append({
                        'speaker': speaker_id,
                        'text': evt.result.text,
                        'timestamp': evt.result.offset / 10000000 if hasattr(evt.result, 'offset') else 0
                    })

            transcriber.transcribed.connect(recognized_handler)

            transcriber.start_transcribing_async().get()

            done = False
            start_time = time.time()
            while not done and time.time() - start_time < 60:  # Timeout de 60 segundos
                time.sleep(1)
                if len(diarization_results) > 0 and time.time() - start_time > 5:
                    done = True

            transcriber.stop_transcribing_async().get()

            return diarization_results
        except Exception as e:
            logger.error(f"Erro durante a diarização: {str(e)}")
            return [{"speaker": "error", "text": f"Diarização falhou: {str(e)}"}]


class AudioEmbedder:
    """
    Handles transcription and embedding generation from audio files, 
    and persists results to both CosmosDB and Azure AI Search.

    Attributes:
        azure_key (str): Azure subscription key.
        region (str): Azure region.
        embedding_service (str): The target backend for saving embeddings ("cosmosdb" or other).
    """

    def __init__(self, embedding_service: str = 'cosmosdb'):
        """
        Initialize the AudioEmbedder with an Azure subscription key and a specified backend service.

        Args:
            azure_key (str): The Azure subscription key.
            region (str): The Azure region.
            embedding_service (str, optional): The target embedding service backend. Defaults to 'cosmosdb'.
        """
        self.embedding_service = embedding_service

    @kernel_function(
        name="embed_audio",
        description="Generates text embeddings from transcribed audio content"
    )
    async def embed_audio(self, audio_file: str) -> Tuple[List[float], str]:
        """
        Transcribe the audio file and generate a textual embedding from the transcribed text.

        Args:
            audio_file (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the generated embedding (list) and the transcription (str).
        """
        processor = AudioProcessor()
        transcription = processor.process_transcription(audio_file)
        embedding = self._generate_embedding(transcription)
        return embedding, transcription

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a dummy embedding by converting each character in the text to its ordinal value.
        
        In a production environment, this should use a proper embedding service like Azure AI Language.

        Args:
            text (str): The input text.

        Returns:
            List[float]: A list of float values representing the embedding.
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
        name="save_embedding",
        description="Saves audio embeddings and metadata to CosmosDB and Azure Search"
    )
    def save_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Persist the embedding and its metadata to both CosmosDB and Azure AI Search.

        Args:
            embedding (List[float]): The embedding vector.
            metadata (Dict[str, Any]): Metadata related to the embedding (must include an "id" field if available).

        Returns:
            str: A message confirming that the embedding has been saved.
        """
        try:
            cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT", "")
            cosmos_key = os.environ.get("COSMOS_KEY", "")
            database_name = os.environ.get("COSMOS_DATABASE", "")
            container_name = os.environ.get("COSMOS_CONTAINER", "")

            cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
            cosmos_container = (
                cosmos_client.get_database_client(database_name)
                .get_container_client(container_name)
            )

            cosmos_item = {
                "id": metadata.get("id", f"audio-{int(time.time())}"),
                "embedding": embedding,
                "metadata": metadata,
                "type": "audio"
            }
            cosmos_container.upsert_item(cosmos_item)

            # Save the embedding to Azure AI Search
            search_endpoint = os.environ.get("SEARCH_ENDPOINT", "")
            search_key = os.environ.get("SEARCH_KEY", "")
            index_name = os.environ.get("SEARCH_INDEX", "")
            search_endpoint = os.environ.get("SEARCH_ENDPOINT", "")
            search_key = os.environ.get("SEARCH_KEY", "")
            index_name = os.environ.get("SEARCH_INDEX", "")
            
            search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(search_key)
            )
            
            # Convert embedding to the format expected by the index
            search_document = {
                "id": metadata.get("id", f"audio-{int(time.time())}"),
                "embedding": embedding,
                "content": metadata.get("transcription", ""),
                "metadata_str": str(metadata),
                "type": "audio"
            }
            
            # Add any direct fields from metadata
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    search_document[key] = value
            
            search_client.upload_documents(documents=[search_document])
            return "Embedding saved to both CosmosDB and Azure AI Search"
        
        except Exception as e:
            logger.error(f"Error saving embedding: {str(e)}")
            return f"Error saving embedding: {str(e)}"


class AudioAnswer:
    """
    Processes real-time audio streams for transcription using Azure Speech SDK.
    
    Attributes:
        azure_key (str): Azure subscription key.
        region (str): Azure region.
    """

    @kernel_function(
        name="process_audio",
        description="Processes real-time audio streams for transcription"
    )
    def process_audio(self, audio_stream) -> str:
        """
        Process an audio stream in real time and return the concatenated transcription.

        Args:
            audio_stream: A valid audio stream input.

        Returns:
            str: The concatenated transcription of the audio input.
        """
        try:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            audio_config = speechsdk.AudioConfig(stream=audio_stream)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

            transcriptions = []
            done = False

            def recognized_handler(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    transcriptions.append(evt.result.text)

            def session_stopped_handler(evt):
                nonlocal done
                done = True

            recognizer.recognized.connect(recognized_handler)
            recognizer.session_stopped.connect(session_stopped_handler)

            recognizer.start_continuous_recognition()

            timeout = 5  # seconds
            start_time = time.time()
            while not done and time.time() - start_time < timeout:
                time.sleep(0.1)

            recognizer.stop_continuous_recognition()
            return " ".join(transcriptions)
        
        except Exception as e:
            logger.error(f"Error processing audio stream: {str(e)}")
            return f"Error processing audio: {str(e)}"


class AudioStreamer:
    """
    Streams audio data for transcription and diarization, then persists processed results.

    Attributes:
        azure_key (str): Azure subscription key.
        region (str): Azure region.
        destination (str): The backend destination for processed content ("cosmosdb" or other).
    """

    def __init__(self, destination: str = 'cosmosdb'):
        """
        Initialize the AudioStreamer with an Azure subscription key and a destination backend.

        Args:
            azure_key (str): The Azure subscription key.
            region (str): The Azure region.
            destination (str, optional): The destination to save content. Defaults to 'cosmosdb'.
        """
        self.azure_key = AZURE_SPEECH_KEY
        self.region = AZURE_SPEECH_REGION
        self.azure_key = AZURE_SPEECH_KEY
        self.region = AZURE_SPEECH_REGION
        self.destination = destination

    @kernel_function(
        name="stream_audio",
        description="Streams audio data for transcription and diarization"
    )
    def stream_audio(self, audio_input_stream, duration: int = 10) -> Dict[str, Any]:
        """
        Stream the given audio input for a specified duration and perform both transcription and diarization.

        Args:
            audio_input_stream: A valid audio stream input.
            duration (int, optional): Duration in seconds to process the stream. Defaults to 10.

        Returns:
            Dict[str, Any]: A dictionary containing 'transcription' and 'diarization' results.
        """
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.azure_key, region=self.region)
            audio_config = speechsdk.AudioConfig(stream=audio_input_stream)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            transcriptions = []
            diarization_results = []
            done = False

            def recognized_handler(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text
                    transcriptions.append(text)
                    diarization_results.append({
                        'speaker': getattr(evt.result, 'speaker_id', 'unknown'),
                        'text': text,
                        'timestamp': getattr(evt.result, 'offset', 0) / 10000000  # Convert to seconds
                    })

            def session_stopped_handler(evt):
                nonlocal done
                done = True

            recognizer.recognized.connect(recognized_handler)
            recognizer.session_stopped.connect(session_stopped_handler)
            
            recognizer.start_continuous_recognition()
            
            start_time = time.time()
            while not done and time.time() - start_time < duration:
                time.sleep(0.1)
                
            recognizer.stop_continuous_recognition()
            
            return {
                'transcription': " ".join(transcriptions),
                'diarization': diarization_results
            }
            
        except Exception as e:
            logger.error(f"Error streaming audio: {str(e)}")
            return {
                'transcription': f"Error streaming audio: {str(e)}",
                'diarization': []
            }

    @kernel_function(
        name="save_processed_content",
        description="Saves processed transcription and diarization content to the specified backend"
    )
    def save_processed_content(self, content: Dict[str, Any]) -> str:
        """
        Save processed transcription and diarization content to the specified backend.

        Args:
            content (Dict[str, Any]): A dictionary containing content with keys 'id', 'transcription', and 'diarization'.

        Returns:
            str: A message indicating where the content has been saved.
        """
        try:
            if self.destination.lower() == 'cosmosdb':
                cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT", "your-cosmos-endpoint")
                cosmos_key = os.environ.get("COSMOS_KEY", self.azure_key)
                database_name = os.environ.get("COSMOS_DATABASE", "your-database")
                container_name = os.environ.get("COSMOS_CONTAINER", "your-container")
                
                cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
                cosmos_container = (
                    cosmos_client.get_database_client(database_name)
                    .get_container_client(container_name)
                )
                
                cosmos_item = {
                    "id": content.get("id", f"stream-{int(time.time())}"),
                    "transcription": content.get("transcription", ""),
                    "diarization": content.get("diarization", []),
                    "timestamp": time.time(),
                    "type": "audio_stream"
                }
                
                cosmos_container.upsert_item(cosmos_item)
                return "Content saved to CosmosDB"
            else:
                search_endpoint = os.environ.get("SEARCH_ENDPOINT", "your-search-endpoint")
                search_key = os.environ.get("SEARCH_KEY", self.azure_key)
                index_name = os.environ.get("SEARCH_INDEX", "your-index")
                
                search_client = SearchClient(
                    endpoint=search_endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(search_key)
                )
                
                search_document = {
                    "id": content.get("id", f"stream-{int(time.time())}"),
                    "content": content.get("transcription", ""),
                    "diarization_str": str(content.get("diarization", [])),
                    "timestamp": time.time(),
                    "type": "audio_stream"
                }
                
                search_client.upload_documents(documents=[search_document])
                return "Content saved to Azure AI Search"
                
        except Exception as e:
            logger.error(f"Error saving processed content: {str(e)}")
            return f"Error saving processed content: {str(e)}"
