"""
agents.py

This module implements:
  - A SuperAgent that is also a Agent (extends AgentBase), orchestrating sub-agents.
  - A Mediator-like pattern: SuperAgent collects notifications from sub-agents.
  - A Plan used by the SuperAgent to evaluate each sub-agent's output in a structured way.
  - A Factory (AgentFactory) that builds sub-agents from a Pydantic Assembly.
  - An Orchestrator (AgentOrchestrator) that merges the SuperAgent + AgentFactory logic into a
    single evaluation procedure.

Classes:
    AgentBase: Abstract base for a agent that can evaluate a prompt.
    ConcreteAgent: A agent that is a ChatCompletionAgent for LLM interactions.
    SuperAgent: Extends AgentBase, orchestrates multiple sub-agents, collects outputs, and
                can itself be considered a "agent" with a final verdict.
    AgentEvaluationPlan: A Plan describing how the SuperAgent calls each sub-agent's evaluate().
    AgentFactory: Builds sub-agents from an Assembly. Optionally also builds or configures a Kernel.
    AgentOrchestrator: High-level class that uses the Factory + SuperAgent to produce a final verdict.

Usage:
    # 1) Create or load an Assembly (with .agents).
    # 2) Call AgentOrchestrator.run_evaluation(assembly, prompt).
    #    -> Returns the final verdict from the SuperAgent after orchestrating sub-agents.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Callable, Dict

import semantic_kernel as sk
from azure.cosmos import exceptions
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.planners.plan import Plan

from app.schemas import Assembly, Agent

COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "mydb")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://myendpoint.documents.azure.com:443/")
COSMOS_ASSEMBLY_TABLE = os.getenv("COSMOS_ASSEMBLY_TABLE", "assemblies")


class Listener(ABC):
    """
    The Mediator interface declares the notify method used by agents (agents)
    to report events or results. Any concrete mediator (such as SuperAgent)
    must implement this method.
    """

    def __init__(self, operator: Callable) -> None:
        self.operator = operator

    @abstractmethod
    def execute(self, sender: object, event: str, data: dict) -> None:
        """
        Notifies the mediator of an event that has occurred.

        :param sender: The agent sending the notification.
        :param event: A string describing the event type (e.g., "evaluation_done").
        :param data: A dictionary containing additional data (e.g., agent_id, result).
        """
        pass


class Mediator(ABC):
    """ 
    The Mediator interface declares the notify method used by agents (agents)
    to report events or results. Any concrete mediator (such as SuperAgent)
    must implement this method.
    """

    def __init__(self, operator: Callable) -> None:
        self.operator = operator
        self.results: List[Dict[str, Any]] = []

    @abstractmethod
    def execute(self, sender: object, event: str, data: dict) -> None:
        """
        Notifies the mediator of an event that has occurred.

        :param sender: The agent sending the notification.
        :param event: A string describing the event type (e.g., "evaluation_done").
        :param data: A dictionary containing additional data (e.g., agent_id, result).
        """
        pass

    @abstractmethod
    def notify(self, sender: object, event: str, data: dict) -> None:
        """
        Notifies the mediator of an event that has occurred.

        :param sender: The agent sending the notification.
        :param event: A string describing the event type (e.g., "evaluation_done").
        :param data: A dictionary containing additional data (e.g., agent_id, result).
        """
        self.results.append({'snder': data, 'event': event, 'data': data})


class Observer(ABC):
    """
    The Mediator interface declares the notify method used by agents (agents)
    to report events or results. Any concrete mediator (such as SuperAgent)
    must implement this method.
    """

    def __init__(self, listeners: List[Listener]) -> None:
        self.listeners = listeners

    @abstractmethod
    def notify(self, sender: object, event: str, data: dict) -> None:
        """
        Notifies the mediator of an event that has occurred.

        :param sender: The agent sending the notification.
        :param event: A string describing the event type (e.g., "evaluation_done").
        :param data: A dictionary containing additional data (e.g., agent_id, result).
        """
        pass

    @abstractmethod
    def act(self, sender: object, event: str, data: dict) -> None:
        """
        Notifies the mediator of an event that has occurred.

        :param sender: The agent sending the notification.
        :param event: A string describing the event type (e.g., "evaluation_done").
        :param data: A dictionary containing additional data (e.g., agent_id, result).
        """
        pass
