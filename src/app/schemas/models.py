"""
This module defines the data models for the application using Pydantic.

These models represent the core entities of the augmented RAG system including agents,
assemblies, tools, and various data types (text, image, audio, video) used for
retrieval-augmented generation tasks.

Classes:
    Agent: Represents an AI agent with specific capabilities and configuration.
    Assembly: Represents a collection of agents working together for a specific objective.
    Tool: Represents a utility function available to agents.
    TextData: Represents textual data with metadata and embeddings.
    ImageData: Represents image data with metadata and embeddings.
    AudioData: Represents audio data with metadata and embeddings.
    VideoData: Represents video data with metadata and embeddings.
    JobResponse: Represents the response from a processing job.
"""

from typing import Callable, List, Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator


class Agent(BaseModel):
    """
    Represents an AI agent with its configuration and capabilities.

    Attributes:
        id (str): The unique identifier for the agent.
        name (str): The human-readable name of the agent.
        model_id (str): The identifier of the model used by this agent.
        metaprompt (str): The system prompt that defines the agent's behavior.
        objective (Literal["image", "text", "audio", "video"]): The data type this agent specializes in.
    """

    id: int = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent Name")
    model_id: str = Field(..., description="Model ID")
    metaprompt: str = Field(..., description="Agent System Prompt")
    objective: Literal["image", "text", "audio", "video"] = Field(default='text', description="Agent Objective")

    @classmethod
    @field_validator("model_id")
    def model_must_be_small(cls, v):
        if len(v) > 32:
            raise ValueError("model ID shouldn't have more than 32 characters")
        return v

    @classmethod
    @field_validator("objective")
    def objective_must_be_small(cls, v):
        if len(v) > 32:
            raise ValueError("objective shouldn't have more than 32 characters")
        return v


class Assembly(BaseModel):
    """
    Represents a collection of agents working together toward a specific objective.

    Attributes:
        id (str): The unique identifier for the assembly.
        objective (str): The goal or task this assembly is designed to achieve.
        agents (List[Agent]): The collection of agents that form this assembly.
        roles (List[str]): The defined roles for agents within this assembly.
    """

    id: int = Field(..., description="Agent Assembly ID")
    objective: str = Field(..., description="The Agent Assembly Object to operate on")
    agents: List[Agent] = Field(..., description="Agents Assemblies")
    roles: List[str] = Field(..., description="Agent Roles ID")
    order: Optional[List[int]] = Field(default=None, description="Agent Order of Execution")

    @classmethod
    @field_validator("roles")
    def roles_must_not_exceed_length(cls, v):
        for role in v:
            if len(role) > 360:
                raise ValueError("each role must have at most 360 characters")
        return v

    @classmethod
    @field_validator("order")
    def orders_must_contain_ids(cls, v):
        for order in v:
            if order not in [agent.id for agent in v.agents]:
                raise ValueError("each role must have at most 360 characters")
        return v


class Tool(BaseModel):
    """
    Represents a tool that can be used by agents to perform specific functions.

    Attributes:
        id (str): The unique identifier for the tool.
        name (str): The name of the tool.
        description (str): A description of what the tool does.
        func (Callable[..., Any]): The executable function that implements the tool's functionality.
    """
    id: str = Field(..., description="Tool ID")
    name: str = Field(..., description="Tool Name")
    description: str = Field(..., description="Tool Description")
    func: Callable[..., Any] = Field(..., description="Tool Function")


class TextData(BaseModel):
    """
    Represents textual data with associated metadata and vector embeddings.

    Attributes:
        source (str): Source location of the text data.
        value (str): The actual text content.
        objective (str): The purpose or context of this text.
        encoding (str): The text encoding format.
        tags (List[str]): Keywords or categories associated with this text.
        original_document (Optional[str]): Reference to source document if applicable.
        embeddings (Optional[List[float]]): Vector embeddings for semantic search.
    """
    source: str = Field(..., description="Image Source address")
    objective: str = Field(..., description="Text Objective")
    tags: List[str] = Field(..., description="Text Tags")
    encoding: Optional[str] = Field(default=None, description="Video Encoding")
    embeddings: Optional[List[float]] = Field(default=None, description="Video Embeddings")


class ImageData(BaseModel):
    """
    Represents image data with associated metadata and vector embeddings.

    Attributes:
        source (str): The location or path to the image file.
        objective (str): The purpose or context of this image.
        encoding (str): The image encoding format.
        tags (List[str]): Keywords or categories associated with this image.
        embeddings (Optional[List[float]]): Vector embeddings for semantic search.
    """

    source: str = Field(..., description="Image Source address")
    objective: str = Field(..., description="Text Objective")
    tags: List[str] = Field(..., description="Image Tags")
    encoding: Optional[str] = Field(default=None, description="Video Encoding")
    embeddings: Optional[List[float]] = Field(default=None, description="Video Embeddings")


class AudioData(BaseModel):
    """
    Represents audio data with associated metadata and vector embeddings.

    Attributes:
        source (str): The location or path to the audio file.
        objective (str): The purpose or context of this audio.
        encoding (str): The audio encoding format.
        tags (List[str]): Keywords or categories associated with this audio.
        embeddings (Optional[List[float]]): Vector embeddings for semantic search.
    """

    source: str = Field(..., description="Audio Source address")
    objective: str = Field(..., description="Text Objective")
    tags: List[str] = Field(..., description="Audio Tags")
    encoding: Optional[str] = Field(default=None, description="Video Encoding")
    embeddings: Optional[List[float]] = Field(default=None, description="Video Embeddings")


class VideoData(BaseModel):
    """
    Represents video data with associated metadata and vector embeddings.

    Attributes:
        source (str): The location or path to the video file.
        objective (str): The purpose or context of this video.
        encoding (str): The video encoding format.
        tags (List[str]): Keywords or categories associated with this video.
        embeddings (Optional[List[float]]): Vector embeddings for semantic search.
    """

    source: str = Field(..., description="Video Source address")
    objective: str = Field(..., description="Video Objective")
    tags: List[str] = Field(..., description="Video Tags")
    encoding: Optional[str] = Field(default=None, description="Video Encoding")
    embeddings: Optional[List[float]] = Field(default=None, description="Video Embeddings")


class JobResponse(BaseModel):
    """
    Represents the response from a processing job executed by an assembly.

    Attributes:
        assembly_id (str): The identifier of the assembly that processed the job.
        prompt (str): The input prompt or query that initiated the job.
    """

    assembly_id: str = Field(..., description="Assembly ID")
    prompt: str = Field(..., description="Job Status")
