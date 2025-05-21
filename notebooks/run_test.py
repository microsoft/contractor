import os
import sys
import uuid
import asyncio

from dotenv import load_dotenv

from app.agents.main import ToolerOrchestrator
from app.schemas.models import AudioData, TextData, Agent, Assembly


if __name__ == "__main__":
    src_path = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
    print("Adding src folder to path:", src_path)
    sys.path.insert(0, src_path)

    load_dotenv(f"{src_path}/.env")

    audio_data_1 = AudioData(
        source="./data/audios/azure-podcast-2.mp3",
        objective="Explains Azure ACA and its features",
        tags=["azure", "aca", "containers"]
    )

    audio_data_2 = AudioData(
        source="./data/audios/azure-podcast-2.mp3",
        objective="Explains Azure in general",
        tags=["azure", "definitions"]
    )

    print("AudioData instance:", audio_data_1)
    print("AudioData instance:", audio_data_2)

    text_data_1 = TextData(
        source="./data/documents/aoai-assistants.pdf",
        objective="Explains Azure OpenAI and its features",
        tags=["azure", "aoai", "assistants"]
    )

    text_data_2 = TextData(
        source="./data/documents/aoai-prompting.pdf",
        objective="Explains Prompting in Azure OpenAI",
        tags=["azure", "aoai", "prompting"]
    )

    print("TextData instance:", text_data_1)
    print("TextData instance:", text_data_2)

    audio_agent = Agent(
        id=int(uuid.uuid4()),
        name="AudioAgent",
        model_id="default",
        metaprompt="This agent handles audio processing and extraction tasks.",
        objective="audio"
    )

    text_agent = Agent(
        id=int(uuid.uuid4()),
        name="TextAgent",
        model_id="default",
        metaprompt="This agent analyzes textual information for semantic understanding.",
        objective="text"
    )

    assembly = Assembly(
        id=int(uuid.uuid4()),
        objective="multimodal processing using local data for architecture review on azure",
        agents=[audio_agent, text_agent],
        roles=["audio", "text"]
    )

    print("Assembly created with agents:", assembly)

    orchestrator = ToolerOrchestrator()
    response = asyncio.run(
        orchestrator.run_interaction(
            assembly=assembly,
            prompt="Which files you have locally available that contains best practices for building multi-agent systems?",
            strategy="llm"
        )
    )
    flattened = [str(item) for sublist in response for item in (sublist if isinstance(sublist, list) else [sublist])]
    print("Orchestrator response:", "\n".join(flattened))
