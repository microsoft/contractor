import os
import pathlib
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, cast

from pyrit.common.initialization import initialize_pyrit
from pyrit.common.path import DATASETS_PATH
from pyrit.memory.central_memory import CentralMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import AzureContentFilterScorer, SelfAskRefusalScorer, LikertScalePaths, SelfAskLikertScorer
from pyrit.models.seed_prompt import SeedPromptDataset, SeedPrompt, SeedPromptGroup, PromptDataType

from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_converter.charswap_attack_converter import CharSwapGenerator
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration
from pyrit.orchestrator import PromptSendingOrchestrator, CrescendoOrchestrator
from pyrit.prompt_converter import TenseConverter, TranslationConverter
from pyrit.models.filter_criteria import PromptFilterCriteria

from semantic_kernel.functions.kernel_function_decorator import kernel_function


class TestContext:
    """
    Owns all shared configuration: memory init, dataset paths,
    default labels, endpoints, converters, scorers, etc.
    """

    def __init__(self,
                 memory_db_type: str = "InMemory",
                 azure_endpoint: Optional[str] = None,
                 azure_key: Optional[str] = None,
                 azure_gpt4o_endpoint: Optional[str] = None,
                 azure_gpt4o_key: Optional[str] = None,
                 default_labels: Optional[dict[str, Any]] = None):

        initialize_pyrit(memory_db_type=memory_db_type)
        self.memory = CentralMemory.get_memory_instance()

        self.labels = default_labels or {}

        self.target = OpenAIChatTarget(
            endpoint=azure_endpoint or os.getenv("AZURE_MODEL_URL"),
            api_key=azure_key or os.getenv("AZURE_FOUNDRY_KEY")
        )

        self.gpt4o_target = OpenAIChatTarget(
            endpoint=azure_gpt4o_endpoint or os.getenv("AZURE_MODEL_URL"),
            api_key=azure_gpt4o_key or os.getenv("AZURE_FOUNDRY_KEY")
        )

        self.scorers = [
            AzureContentFilterScorer(api_key=os.getenv("AZURE_FOUNDRY_KEY", ""), endpoint=os.getenv("AZURE_AI_SERVICES_URL", "")),
            SelfAskRefusalScorer(chat_target=self.target)
        ]

        self.likert_scorer = SelfAskLikertScorer(
            likert_scale_path=LikertScalePaths.HARM_SCALE.value, 
            chat_target=self.target
        )

    @kernel_function(
        name="load_seed_prompts",
        description="Load a YAML dataset into memory as seed prompts."
    )
    def load_seed_prompts(self, dataset_name: str, added_by: str = "automated"):
        """
        Load a YAML dataset into memory (once).
        """
        ds_path = pathlib.Path(DATASETS_PATH) / "seed_prompts" / f"{dataset_name}.prompt"
        dataset = SeedPromptDataset.from_yaml_file(ds_path)
        asyncio.run(
            self.memory.add_seed_prompts_to_memory_async(
                prompts=dataset.prompts,
                added_by=added_by
            )
        )

    @kernel_function(
        name="get_prompt_groups",
        description="Retrieve seed prompt groups, optionally filtered by dataset_name."
    )
    def get_prompt_groups(self, dataset_name: Optional[str] = None):
        """
        Retrieve seed prompt groups, optionally filtered by dataset_name.
        """
        return self.memory.get_seed_prompt_groups(dataset_name=dataset_name)

    @kernel_function(
        name="default_labels",
        description="Return a copy of the default labels for this context."
    )
    def default_labels(self) -> dict[str, Any]:
        """
        Return a copy of the default labels for this context.
        This is useful for ensuring that the labels are not modified outside of this class.
        """

        return dict(self.labels)  # defensive copy

    @kernel_function(
        name="create_prompt_group",
        description="Create a SeedPromptGroup from a list of prompt dictionaries."
    )
    def create_prompt_group(self, prompts: list[dict[str, str]]) -> SeedPromptGroup:
        """
        Create a SeedPromptGroup from a list of prompt dictionaries.
        Each dictionary should have 'value' and 'data_type' keys.
        """
        # Only allow valid data_type literals
        allowed_types = {"text", "image_path", "audio_path", "video_path", "url", "error"}
        def get_data_type(val) -> PromptDataType:
            if val in allowed_types:
                return cast(PromptDataType, val)
            return cast(PromptDataType, "text")
        return SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=p["value"],
                    data_type=get_data_type(p.get("data_type"))
                )
                for p in prompts
            ]
        )
    
    @kernel_function(
        name="get_interesting_prompts",
        description="Return prompts with interesting scores (e.g., high harm scores or refusal failures)."
    )
    def get_interesting_prompts(self, filter_labels: Optional[dict[str, Any]] = None) -> list[Any]:
        """
        Return prompts with interesting scores (e.g., high harm scores or refusal failures).
        """
        labels = filter_labels or self.default_labels()
        result_pieces = self.memory.get_prompt_request_pieces(labels=labels)
        
        interesting_prompts = []
        for piece in result_pieces:
            for score in piece.scores:
                if ((score.score_type == "float_scale" and score.get_value() > 0) or
                    (score.scorer_class_identifier["__type__"] == "SelfAskRefusalScorer" and 
                     score.get_value() == False)):
                    interesting_prompts.append(piece)
                    break
                    
        return interesting_prompts
    
    @kernel_function(
        name="rescore_prompts",
        description="Apply additional scoring to a list of prompts."
    )
    async def rescore_prompts(self, prompts: list[Any], scorer=None) -> list[Any]:
        """
        Apply additional scoring to a list of prompts.
        """
        if scorer is None:
            scorer = self.likert_scorer
            
        return await scorer.score_responses_inferring_tasks_batch_async(
            request_responses=prompts
        )


class TestStrategy(ABC):

    @abstractmethod
    async def __call__(self, ctx: TestContext, params: dict[str, Any]) -> Any:
        """
        Run the test given a shared TestContext and strategy-specific params.
        """

    @kernel_function(
        name="print_results",
        description="Print the conversation results."
    )
    async def print_results(self, results: list[Any]) -> None:
        """Print the conversation results."""
        for result in results:
            await result.print_conversation_async()


class SendingPromptsStrategy(TestStrategy):
    @kernel_function(
        name="send_prompts",
        description="Send prompts using the configured orchestrator and return results."
    )
    async def __call__(self, ctx: TestContext, params: dict[str, Any]) -> list:
        dataset = params.get("dataset")
        if dataset:
            ctx.load_seed_prompts(dataset_name=dataset, added_by=params.get("user", "auto"))

        groups = ctx.get_prompt_groups(dataset_name=dataset)

        orchestrator = PromptSendingOrchestrator(
            objective_target=ctx.target,
            scorers=ctx.scorers
        )

        if params.get("skip_criteria"):
            skip_criteria = PromptFilterCriteria(**params.get("skip_criteria", {}))
            orchestrator.set_skip_criteria(
                skip_criteria=skip_criteria, 
                skip_value_type=params.get("skip_value_type", "original")
            )

        sys_text = params.get("system_prompt")
        if sys_text:
            orchestrator.set_prepended_conversation(prepended_conversation=[
                PromptRequestResponse(
                    request_pieces=[PromptRequestPiece(original_value=sys_text, role="system")]
                )
            ])

        # 5) create normalizer requests (with CharSwap attack example)
        requests = []

        if params.get("direct_prompts"):
            for prompt_data in params["direct_prompts"]:
                prompt_group = ctx.create_prompt_group([prompt_data])
                requests.append(
                    NormalizerRequest(
                        seed_prompt_group=prompt_group,
                        request_converter_configurations=params.get("converter_configs", [])
                    )
                )

        for group in groups:
            requests.append(
                NormalizerRequest(
                    seed_prompt_group=group,
                    request_converter_configurations=[
                        PromptConverterConfiguration(
                            converters=[CharSwapGenerator()],
                            prompt_data_types_to_apply=["text"]
                        )
                    ],
                    response_converter_configurations=[]
                )
            )

        # 6) send them
        results = await orchestrator.send_normalizer_requests_async(
            prompt_request_list=requests,
            memory_labels=ctx.default_labels()
        )
        
        # 7) print results if requested
        if params.get("print_results", False):
            await orchestrator.print_conversations_async()
            
        return results

    @kernel_function(
        name="analyze_results",
        description="Analyze results and optionally rescore interesting prompts."
    )
    async def analyze_results(
            self,
            ctx: TestContext,
            params: dict[str, Any],
            _results: list[Any]
        ) -> list[Any]:
        """
        Analyze results and optionally rescore interesting prompts.
        """
        interesting_prompts = ctx.get_interesting_prompts(params.get("filter_labels"))
        
        if params.get("rescore", False) and interesting_prompts:
            return await ctx.rescore_prompts(interesting_prompts)
        
        return interesting_prompts


class CrescendoStrategy(TestStrategy):
    @kernel_function(
        name="run_crescendo",
        description="Run the Crescendo attack/objectives and return results."
    )
    async def __call__(self, ctx: TestContext, params: dict[str, Any]) -> list:
        # Use ctx.target as default for converter_target
        converter_target = params.get("converter_target", ctx.target)
        
        converters = []
        if params.get("use_tense_converter", True):
            converters.append(TenseConverter(
                converter_target=converter_target, 
                tense=params.get("tense", "past")
            ))
            
        if params.get("use_translation_converter", True):
            converters.append(TranslationConverter(
                converter_target=converter_target, 
                language=params.get("language", "spanish")
            ))
        
        # Allow custom converters
        if params.get("custom_converters"):
            converters.extend(params["custom_converters"])

        # 2) configure orchestrator with more flexible target options
        target = params.get("target", ctx.target)
        adv_target = params.get("adversarial_chat", ctx.gpt4o_target)
        scoring_target = params.get("scoring_target", ctx.target)
        
        orchestrator = CrescendoOrchestrator(
            objective_target=target,
            adversarial_chat=adv_target,
            max_turns=params.get("max_turns", 10),
            max_backtracks=params.get("max_backtracks", 5),
            scoring_target=scoring_target,
            prompt_converters=converters
        )

        # 3) run the attack/objectives
        results = await orchestrator.run_attacks_async(
            objectives=params.get("objectives", []),
            memory_labels=ctx.default_labels()
        )
        
        # 4) print results if requested
        if params.get("print_results", True):
            await self.print_results(results)
            
        return results
