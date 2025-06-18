import os

# Existing plugins
from .audio import AudioProcessor, AudioEmbedder, AudioAnswer
from .image import ImageProcessor, ImageEmbedder, ImageAnswer
from .text import TextProcessor, TextEmbedder, TextAnswer
from .video import VideoProcessor, VideoEmbedder, VideoAnswer
from .retrieval import NLToSQL, NLToNoSQL, LocalFileRetriever, PDFLoader
from .statistical import StatisticalAnalysisPlugin
from .compliance import TestContext, SendingPromptsStrategy, CrescendoStrategy


__BASE_PLUGINS = [NLToSQL(), NLToNoSQL(), LocalFileRetriever(), StatisticalAnalysisPlugin(), PDFLoader()]
COMPLIANCE_PLUGINS = [TestContext(), SendingPromptsStrategy(), CrescendoStrategy()]

AUDIO_PLUGINS = [AudioProcessor(), AudioEmbedder(), AudioAnswer(), *__BASE_PLUGINS]
IMAGE_PLUGINS = [ImageProcessor(), ImageEmbedder(), ImageAnswer(), *__BASE_PLUGINS]
TEXT_PLUGINS = [TextProcessor(), TextEmbedder(), TextAnswer(), *__BASE_PLUGINS]
VIDEO_PLUGINS = [VideoProcessor(), VideoEmbedder(), VideoAnswer(), *__BASE_PLUGINS]
