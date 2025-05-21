from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class TestBase(BaseModel):
    test_name: str = Field(..., description="Name of the test")
    user_name: str = Field(..., description="Name of the user running the test")
    memory_db_type: str = Field("InMemory", description="Type of memory database")
    print_results: bool = Field(False, description="Whether to print results")


class SendingPromptsRequest(TestBase):
    dataset: Optional[str] = Field(None, description="Dataset name to load")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    direct_prompts: Optional[List[Dict[str, str]]] = Field(None, description="Direct prompts to send")
    skip_criteria: Optional[Dict[str, Any]] = Field(None, description="Criteria for skipping prompts")
    skip_value_type: Optional[str] = Field("original", description="Type of value to skip")
    converter_configs: Optional[List[Dict[str, Any]]] = Field(None, description="Converter configurations")
    filter_labels: Optional[Dict[str, Any]] = Field(None, description="Labels for filtering results")
    rescore: Optional[bool] = Field(False, description="Whether to rescore results")


class CrescendoRequest(TestBase):
    objectives: List[str] = Field(..., description="Objectives for conversation")
    use_tense_converter: bool = Field(True, description="Whether to use tense converter")
    use_translation_converter: bool = Field(True, description="Whether to use translation converter")
    tense: Optional[str] = Field("past", description="Tense to convert to")
    language: Optional[str] = Field("spanish", description="Language to translate to")
    max_turns: int = Field(10, description="Maximum turns")
    max_backtracks: int = Field(5, description="Maximum backtracks")


class TestResponse(BaseModel):
    test_id: str = Field(..., description="ID of the test")
    status: str = Field(..., description="Status of the test")
    message: str = Field(..., description="Message about the test")


class TestResultResponse(BaseModel):
    test_id: str = Field(..., description="ID of the test")
    status: str = Field(..., description="Status of the test")
    results: List[Dict[str, Any]] = Field(..., description="Results of the test")
    interesting_count: int = Field(0, description="Count of interesting prompts")
