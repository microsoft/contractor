"""
A package that holds response schemas and models.
"""

__all__ = [
    "RESPONSES",
    "ErrorMessage",
    "SuccessMessage",
    "Agent",
    "Assembly",
    "JobResponse",
    "Tool",
    "TextData",
    "ImageData",
    "AudioData",
    "VideoData",
    "database_schema",
]

__author__ = "AI GBBS"

from .models import Assembly, Agent, JobResponse, Tool, TextData, ImageData, AudioData, VideoData
from .responses import RESPONSES, ErrorMessage, SuccessMessage

database_schema = {
    "Agent Table": Agent.model_json_schema(),
    "Assembly Table": Assembly.model_json_schema(),
}
