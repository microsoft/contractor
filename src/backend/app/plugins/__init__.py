import os

# Existing plugins
from .audio import AudioProcessor, AudioEmbedder, AudioAnswer
from .image import ImageProcessor, ImageEmbedder, ImageAnswer
from .text import TextProcessor, TextEmbedder, TextAnswer
from .video import VideoProcessor, VideoEmbedder, VideoAnswer
from .retrieval import NLToSQL, NLToNoSQL, LocalFileRetriever, PDFLoader, AzureAISearchTool
from .statistical import StatisticalAnalysisPlugin
from .compliance import TestContext, SendingPromptsStrategy, CrescendoStrategy


_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "") or os.getenv("AZURE_AI_SEARCH_SERVICE", "")
_search_index = os.getenv("PDF_INDEX_NAME", "pdf-index-two") or os.getenv("AZURE_AI_INDEX", "")
_search_key = os.getenv("AZURE_SEARCH_API_KEY", os.getenv("AZURE_SEARCH_KEY", os.getenv("AZURE_AI_SEARCH_KEY", "")))
search_plugin = []
try:
	if _search_endpoint and _search_index and _search_key:
		search_plugin = [AzureAISearchTool(_search_endpoint, _search_index, _search_key)]
except Exception:  # pragma: no cover
	search_plugin = []

__BASE_PLUGINS = [NLToSQL(), NLToNoSQL(), LocalFileRetriever(), StatisticalAnalysisPlugin(), PDFLoader(), *search_plugin]
COMPLIANCE_PLUGINS = [TestContext(), SendingPromptsStrategy(), CrescendoStrategy()]

AUDIO_PLUGINS = [AudioProcessor(), AudioEmbedder(), AudioAnswer(), *__BASE_PLUGINS]
IMAGE_PLUGINS = [ImageProcessor(), ImageEmbedder(), ImageAnswer(), *__BASE_PLUGINS]
TEXT_PLUGINS = [TextProcessor(), TextEmbedder(), TextAnswer(), *__BASE_PLUGINS]
VIDEO_PLUGINS = [VideoProcessor(), VideoEmbedder(), VideoAnswer(), *__BASE_PLUGINS]
