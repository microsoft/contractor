"""
Module: questions

This module implements the orchestration of multiple avatar agents (graders) in parallel 
using Semantic Kernel and Azure OpenAI services. It provides the core components needed 
to configure, execute, and aggregate responses from multiple grader agents. 

Components:
  - Mediator: An abstract interface for mediator implementations that handle notifications 
    from sub-avatar agents.
  - GraderBase: An abstract base class defining the interface for grader agents capable of 
    interacting with prompts asynchronously.
  - AnswerGrader: A concrete grader that uses a ChatCompletionAgent to process prompts and 
    generate responses. It loads its configuration via Jinja2 templates.
  - GraderFactory: A factory for building a shared Semantic Kernel and creating AnswerGrader 
    instances from an Assembly document.
  - AnswerOrchestrator: A high-level orchestrator that fetches an Assembly document from 
    Cosmos DB, creates grader instances, and executes their interactions concurrently or 
    sequentially.

Usage:
    1. Ensure environment variables (e.g. COSMOS_QNA_NAME, COSMOS_ENDPOINT, COSMOS_ASSEMBLY_TABLE, 
       AZURE_MODEL_KEY, AZURE_MODEL_URL) are defined in a .env file placed at the root (two levels 
       above this module).
    2. Prepare an Assembly document in Cosmos DB which contains grader (avatar) configurations.
    3. Instantiate an AnswerOrchestrator and call its run_interaction method with an assembly_id, 
       question, and answer to obtain aggregated responses from all configured grader agents.
       
Dependencies:
    - semantic_kernel: Provides ChatCompletionAgent, ChatHistory, KernelArguments, etc.
    - jinja2: Used for templating of prompt instructions.
    - azure.cosmos and azure.identity: For Cosmos DB access and authentication.
    - dotenv: For environment variable loading.
    - asyncio: For asynchronous execution.

"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Union, Self
import uuid

from pydantic import BaseModel
from dotenv import load_dotenv
import semantic_kernel as sk
import jinja2

from semantic_kernel.agents import ChatCompletionAgent, GroupChatManager  # pylint: disable=no-name-in-module

from semantic_kernel.agents.runtime import InProcessRuntime

from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationBase
from semantic_kernel.agents.orchestration.concurrent import ConcurrentOrchestration
from semantic_kernel.agents.orchestration.sequential import SequentialOrchestration
from semantic_kernel.agents.orchestration.group_chat import GroupChatOrchestration, RoundRobinGroupChatManager

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import logger
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.exceptions import ServiceResponseException, KernelFunctionAlreadyExistsError

from azure.cosmos import exceptions
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential

from app.schemas import Assembly, Agent
from app.schemas.models import TextData, ImageData, AudioData, VideoData
from .operators import Observer

from app.plugins import AUDIO_PLUGINS, IMAGE_PLUGINS, TEXT_PLUGINS, VIDEO_PLUGINS
from app.plugins.compliance import TestContext, SendingPromptsStrategy


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_FILE)

COSMOS_DB_NAME = os.getenv("COSMOS_QNA_NAME", "mydb")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://myendpoint.documents.azure.com:443/")
COSMOS_ASSEMBLY_TABLE = os.getenv("COSMOS_ASSEMBLY_TABLE", "assembly")
AZURE_FOUNDRY_KEY = os.getenv("AZURE_FOUNDRY_KEY", "")
AZURE_FOUNDRY_URL = os.getenv("AZURE_FOUNDRY_URL", "")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "prompts")
JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_DIR))


AVAILABLE_MODELS: list[AzureChatCompletion] = [
    AzureChatCompletion(
        service_id="default",
        api_key=AZURE_FOUNDRY_KEY,
        deployment_name="contractor-4o",
        endpoint=AZURE_FOUNDRY_URL
    ),
    AzureChatCompletion(
        service_id="quatroum",
        api_key=AZURE_FOUNDRY_KEY,
        deployment_name="gpt-4.1",
        endpoint=AZURE_FOUNDRY_URL
    ),
    AzureChatCompletion(
        service_id="mini",
        api_key=AZURE_FOUNDRY_KEY,
        deployment_name="gpt-4o-mini",
        endpoint=AZURE_FOUNDRY_URL
    ),
    AzureChatCompletion(
        service_id="reasoning",
        api_key=AZURE_FOUNDRY_KEY,
        deployment_name="o3-mini",
        endpoint=AZURE_FOUNDRY_URL,
        api_version="2024-12-01-preview"
    )
]

PROMPT_TYPE = Union[TextData, ImageData, AudioData, VideoData]


class BaseOutput(BaseModel):
    output: str


class ToolerBase(ABC):
    """
    Abstract base for an avatar (grader) that can interact with a prompt.

    This class defines the core interface for graders, including an asynchronous
    interaction method.
    """
    def __init__(self, tooler: Agent, kernel: sk.Kernel) -> None:
        """
        Initialize an AnswerGrader.

        :param grader: The Grader configuration data (including id, name, metaprompt, model_id).
        :param kernel: A shared Semantic Kernel instance configured with services.
        """
        self._observer: Optional[Observer] = None
        self.kernel = kernel
        self.tooler = tooler
        self.agent: ChatCompletionAgent
        self.__prepare()

    async def _compliance_validation(self, prompt: str) -> bool:
        """
        Validate the prompt for compliance issues.

        Uses the compliance tools to check for governance issues in the prompt.

        :param prompt: The prompt to be validated.
        :return: True if the prompt is compliant, False otherwise.
        """
        ctx = TestContext()
        compliance_strategy = SendingPromptsStrategy()
        compliance_params = {
            "direct_prompts": [{"value": prompt, "data_type": "text"}],
            "print_results": False
        }
        compliance_results = await compliance_strategy(ctx, compliance_params)
        logger.info("Compliance check results: %s", compliance_results)
        return compliance_results

    def __prepare(self):
        """
        Prepare the grader for interaction by configuring the ChatCompletionAgent.

        Renders the instruction settings from a Jinja2 template and retrieves the settings based on
        the grader's model_id. Then creates the ChatCompletionAgent instance.
        """
        instruction_template = JINJA_ENV.get_template("instruction.jinja")
        rendered_settings = instruction_template.render(prompt=self.tooler.metaprompt)
        settings = self.kernel.get_prompt_execution_settings_from_service_id(service_id=self.tooler.model_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        self.agent = ChatCompletionAgent(
            kernel=self.kernel,
            name=self.tooler.name,
            description=self.tooler.description,
            instructions=rendered_settings,
            arguments=KernelArguments(settings=settings),
        )

    @property
    def observer(self) -> Optional[Observer]:
        return self._observer

    @observer.setter
    def observer(self, observer: Observer) -> None:
        """
        Set the mediator for the grader.

        :param mediator: The mediator instance to be set.
        """
        self._observer = observer

    @abstractmethod
    def add_tools(self) -> Self:  # pylint: disable=arguments-differ
        """
        Interact with a provided question and answer using the ChatCompletionAgent.

        Renders the prompt using a Jinja2 template, sends it to the agent, accumulates the responses,
        adds messages to the provided ChatHistory, and notifies the mediator upon completion.

        :param question: The question object for the interaction.
        :param answer: The answer object associated with the question.
        :param chat: A ChatHistory instance for logging the conversation.
        :return: The aggregated response generated by the agent.
        """

    async def _execute(self, chat: ChatHistory, rendered_prompt: str) -> str:
        await self._compliance_validation(rendered_prompt)
        chat.add_message(ChatMessageContent(role=AuthorRole.USER, content=rendered_prompt))
        response = ""
        while True:
            try:
                async for message in self.agent.invoke(messages=chat.messages):  # type: ignore[assignment]
                    if message.content.content == "":
                        break
                    response += message.content.content
                    chat.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content=message.content.content))
                break
            except ServiceResponseException as e:
                logger.error("Service response error: %s", e)
                await asyncio.sleep(60)

        if self.observer:
            self.observer.notify(
                sender=self,
                event="interaction_done",
                data={
                    "tooler_id": self.tooler.id,
                    "tooler_name": self.tooler.name,
                }
            )
        return response


class TextTooler(ToolerBase):
    """
    A grader that interacts with a prompt using a ChatCompletionAgent.

    Loads its personality and instructions from its profile, renders the prompt via a
    Jinja2 template, and generates a response. Notifies a mediator when interaction is complete.
    """

    def add_tools(self):
        """
        Always add TEXT_PLUGINS (which includes compliance plugins).

        :param tools: Optional list of additional tools to add.
        """
        plugins = set(TEXT_PLUGINS)
        for tool in plugins:
            self.kernel.add_plugin(tool, tool.__class__.__name__)
        return self


class ImageTooler(ToolerBase):
    """
    A grader that interacts with a prompt using a ChatCompletionAgent.

    Loads its personality and instructions from its profile, renders the prompt via a
    Jinja2 template, and generates a response. Notifies a mediator when interaction is complete.
    """

    def add_tools(self):
        """
        Always add TEXT_PLUGINS (which includes compliance plugins).

        :param tools: Optional list of additional tools to add.
        """
        plugins = set(IMAGE_PLUGINS)
        for tool in plugins:
            self.kernel.add_plugin(tool, tool.__class__.__name__)
        return self


class AudioTooler(ToolerBase):
    """
    A grader that interacts with a prompt using a ChatCompletionAgent.

    Loads its personality and instructions from its profile, renders the prompt via a
    Jinja2 template, and generates a response. Notifies a mediator when interaction is complete.
    """

    def add_tools(self):
        """
        Always add TEXT_PLUGINS (which includes compliance plugins).

        :param tools: Optional list of additional tools to add.
        """
        plugins = set(AUDIO_PLUGINS)
        for tool in plugins:
            self.kernel.add_plugin(tool, tool.__class__.__name__)
        return self


class VideoTooler(ToolerBase):
    """
    A grader that interacts with a prompt using a ChatCompletionAgent.

    Loads its personality and instructions from its profile, renders the prompt via a
    Jinja2 template, and generates a response. Notifies a mediator when interaction is complete.
    """

    def add_tools(self):
        """
        Always add TEXT_PLUGINS (which includes compliance plugins).

        :param tools: Optional list of additional tools to add.
        """
        plugins = set(VIDEO_PLUGINS)
        for tool in plugins:
            self.kernel.add_plugin(tool, tool.__class__.__name__)
        return self


class ToolerFactory:
    """
    Factory for building a shared Semantic Kernel and creating AnswerGrader instances.

    Uses the provided Assembly object to create graders for each agent defined in the assembly.
    """
    @staticmethod
    def __build_kernel() -> sk.Kernel:
        """
        Build and return a shared Semantic Kernel.
        Adds each service in AVAILABLE_MODELS to the kernel.
        :return: A configured Semantic Kernel instance.
        """
        kernel = sk.Kernel()
        for service in AVAILABLE_MODELS:
            try:
                kernel.add_service(service)
            except KernelFunctionAlreadyExistsError as e:
                logger.error("Kernel Function already exist: %s", e)
        return kernel

    def create_toolers(self, assembly: Assembly) -> List[ToolerBase]:
        """
        Create a list of AnswerGrader instances from an Assembly.

        Assumes that assembly.agents is a list of Grader configuration models.

        :param assembly: The Assembly containing agent configurations.
        :return: A list of instantiated AnswerGrader objects.
        """
        common_kernel = self.__build_kernel()
        toolers = []
        for agent in assembly.agents:
            match agent.objective.lower():
                case "text":
                    toolers.append(TextTooler(agent, common_kernel))
                case "image":
                    toolers.append(ImageTooler(agent, common_kernel))
                case "audio":
                    toolers.append(AudioTooler(agent, common_kernel))
                case "video":
                    toolers.append(VideoTooler(agent, common_kernel))
                case _:
                    raise ValueError(f"Unsupported objective: {agent.objective}")
        return toolers


class ToolerOrchestrator:
    """
    High-level orchestrator that merges grader management and interaction execution.

    Fetches assemblies from Cosmos DB, creates grader instances via GraderFactory, and executes
    interactions in either parallel or sequential mode.
    """

    def __init__(self) -> None:
        self.toolers: List[ToolerBase] = []
        self.runtime = InProcessRuntime()
        self.orchestrator: OrchestrationBase

    async def __aenter__(self):
        """
        Asynchronous context manager entry point.

        Initializes the orchestrator and prepares it for use.
        """
        self.runtime.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Asynchronous context manager exit point.

        Cleans up resources and stops the runtime.
        """
        await self.runtime.stop_when_idle()

    def _response_callback(self, message: ChatMessageContent) -> None:
        """
        Callback function to handle responses from graders.

        This function can be used to process or log responses as they are received.
        """
        log_id = str(uuid.uuid4())
        # Ensure the .log directory exists at the repo root
        log_dir = os.path.join(BASE_DIR, ".log")
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"agent_message_{log_id}_{message.name}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Agent: {message.name}\n")
            f.write(f"Message:\n{message.content}\n")
        print(f"# {message.name}\n{message.content}")

    async def _concurrent_processing(
            self,
            task: str
        ):
        """
        Execute the 'interact' method of all graders concurrently.

        :param question: The question to be processed.
        :param answer: The associated answer object.
        :return: A list of responses from all graders executed in parallel.
        """
        prompt_template = JINJA_ENV.get_template("instruction.jinja")
        self.orchestrator = ConcurrentOrchestration[str, ChatMessageContent](
            members=[tooler.add_tools().agent for tooler in self.toolers],
            input_transform=lambda task: ChatMessageContent(
                role=AuthorRole.USER,
                content=prompt_template.render(prompt=task)
            ),
            agent_response_callback=self._response_callback,  # type: ignore
        )
        result = await self.orchestrator.invoke(runtime=self.runtime, task=task)  # type: ignore
        return await result.get(timeout=240)

    async def _sequential_processing(
            self,
            task: str
        ):
        """
        Execute the 'interact' method of all graders sequentially.

        :param question: The question to be processed.
        :param answer: The corresponding answer object.
        :return: A list of dictionaries mapping grader identifiers to their responses.
        """
        prompt_template = JINJA_ENV.get_template("instruction.jinja")
        self.orchestrator = SequentialOrchestration[str, ChatMessageContent](
            members=[tooler.add_tools().agent for tooler in self.toolers],
            input_transform=lambda task: ChatMessageContent(
                role=AuthorRole.USER,
                content=prompt_template.render(prompt=task)
            ),
            agent_response_callback=self._response_callback,  # type: ignore
        )
        result = await self.orchestrator.invoke(runtime=self.runtime, task=task)  # type: ignore
        return await result.get(timeout=240)

    async def _group_processing(
            self,
            task: str,
            max_rounds: int = 6,
            manager: Optional[GroupChatManager] = None
        ):
        """
        Execute the 'interact' method of all graders sequentially.

        :param question: The question to be processed.
        :param answer: The corresponding answer object.
        :param manager: Custom GroupChatManager instance (optional).
        :return: A list of dictionaries mapping grader identifiers to their responses.
        """
        prompt_template = JINJA_ENV.get_template("instruction.jinja")
        members = [tooler.add_tools().agent for tooler in self.toolers]
        if manager is None:
            manager = RoundRobinGroupChatManager(max_rounds=max_rounds)
        self.orchestrator = GroupChatOrchestration[str, ChatMessageContent](
            members=members,  #type: ignore
            manager=manager,
            input_transform=lambda task: ChatMessageContent(
                role=AuthorRole.USER,
                content=prompt_template.render(prompt=task)
            ),
            agent_response_callback=self._response_callback,  # type: ignore
        )
        result = await self.orchestrator.invoke(runtime=self.runtime, task=task)  # type: ignore
        return await result.get(timeout=1200)

    async def invoke(
            self,
            assembly: str | Assembly,
            prompt: str,
            *args,
            strategy: Literal["concurrent", "sequential", "group"] = "concurrent",
            **kwargs
        ):
        """
        Orchestrate the grader interactions for a given assembly.

        This method performs the following steps:
          1) Builds a shared kernel.
          2) Creates AnswerGrader instances using GraderFactory.
          3) Fetches the Assembly document from Cosmos DB.
          4) Executes grader interactions using the specified strategy (parallel or sequential).
          5) Returns the aggregated responses.

        :param assembly_id: The ID of the Assembly in Cosmos DB.
        :param question: The question object to send to graders.
        :param answer: The answer object associated with the question.
        :param strategy: The processing strategy ("concurrent" or "sequential").
        :return: The aggregated responses from all grader interactions.
        """
        factory = ToolerFactory()
        if isinstance(assembly, str):
            assembly = await self.fetch_assembly(assembly)
        self.toolers = factory.create_toolers(assembly)
        answers = await getattr(self, f"_{strategy}_processing")(prompt, *args, **kwargs)
        return answers

    async def fetch_assembly(self, assembly_id: str) -> Assembly:
        """
        Fetch an Assembly document from Cosmos DB using its ID.

        Raises a ValueError if the database or assembly is not found.

        :param assembly_id: The ID of the assembly to fetch.
        :return: An Assembly object constructed from the retrieved document.
        """
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as client:
            try:
                database = client.get_database_client(COSMOS_DB_NAME)
                await database.read()
            except exceptions.CosmosResourceNotFoundError as exc:
                raise ValueError(f"Database not found: {COSMOS_DB_NAME}") from exc

            container = database.get_container_client(COSMOS_ASSEMBLY_TABLE)
            try:
                item = await container.read_item(item=assembly_id, partition_key=assembly_id)
            except exceptions.CosmosResourceNotFoundError as exc:
                raise ValueError(f"Assembly not found: {assembly_id}") from exc
            return Assembly(**{"id": item["id"], "agents": item["avatars"], "topic_name": item["topic_name"]})
