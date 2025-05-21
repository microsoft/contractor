import os
import sys
import uuid
import asyncio
from dotenv import load_dotenv
from app.schemas.models import Agent, Assembly, TextData
from app.agents.main import ToolerOrchestrator
from app.plugins.compliance import TestContext, SendingPromptsStrategy

if __name__ == "__main__":
    src_path = os.path.abspath(os.path.join(os.getcwd(), "src"))
    print("Adding src folder to path:", src_path)
    sys.path.insert(0, src_path)
    load_dotenv(f"{src_path}/.env")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    conversation_dir = os.path.join(data_dir, "conversation")
    profile_dir = os.path.join(data_dir, "profile")

    conversation_path = os.path.join(conversation_dir, "customer_bianca_rodriguez_conversations.json")
    profile_path = os.path.join(profile_dir, "customer_bianca_rodriguez.txt")

    text_data_conversation = TextData(
        source=conversation_path,
        objective="Customer conversation history for compliance review",
        tags=["conversation", "compliance", "json"]
    )
    text_data_profile = TextData(
        source=profile_path,
        objective="Customer profile and history for compliance review",
        tags=["profile", "compliance", "txt"]
    )

    # Define agents
    conversation_agent = Agent(
        id=int(uuid.uuid4()),
        name="ConversationComplianceAgent",
        model_id="default",
        metaprompt=(
            "You are a compliance specialist. Your job is to analyze customer conversations "
            "and identify any compliance issues, such as admissions of fraud, violence, or other crimes. "
            "Use the compliance tools to evaluate the structure and content of the conversation. "
            "Highlight any compliance risks or issues found."
        ),
        objective="text"
    )

    history_agent = Agent(
        id=int(uuid.uuid4()),
        name="HistoryComplianceAgent",
        model_id="default",
        metaprompt=(
            "You are a compliance auditor. Your job is to review customer history documents, "
            "including purchase, debt, and prosecution data, and identify any compliance issues. "
            "Use the compliance tools to check for criminal records, financial risks, or any other compliance concerns. "
            "Summarize your findings and flag any issues."
        ),
        objective="text"
    )

    # Build Assembly for those agents
    assembly = Assembly(
        id=int(uuid.uuid4()),
        objective="Compliance evaluation for customer conversation and profile.",
        agents=[conversation_agent, history_agent],
        roles=["text", "compliance"]
    )

    # Compose a prompt that includes the data source (not content)
    prompt = (
        "You are tasked with evaluating compliance for a customer. "
        "You have access to two data sources: a conversation log and a customer profile. "
        "The conversation log is located at: {conversation_path} "
        "and the customer profile is located at: {profile_path}. "
        "Analyze these sources for any compliance issues, such as admissions of fraud, violence, criminal records, or financial risks. "
        "Summarize your findings and flag any issues."
    ).format(
        conversation_path=text_data_conversation.source,
        profile_path=text_data_profile.source
    )

    ctx = TestContext()
    compliance_strategy = SendingPromptsStrategy()
    compliance_params = {
        "direct_prompts": [{"value": prompt, "data_type": "text"}],
        "print_results": False
    }
    compliance_results = asyncio.run(compliance_strategy(ctx, compliance_params))

    # Orchestrate using ToolerOrchestrator with the new assembly and prompt
    orchestrator = ToolerOrchestrator()
    result = asyncio.run(
        orchestrator.run_interaction(
            assembly=assembly,
            prompt=prompt,
            strategy="llm"
        )
    )
    print("\n--- Compliance Orchestration Result (Bianca Rodriguez, Data Source Only) ---")
    for r in result:
        print(r)
