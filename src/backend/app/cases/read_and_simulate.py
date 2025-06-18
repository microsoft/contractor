import io
import sys
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from semantic_kernel.agents import GroupChatManager, BooleanResult, MessageResult, StringResult  # pylint: disable=no-name-in-module
from semantic_kernel.contents import ChatMessageContent, ChatHistory, AuthorRole

from app.schemas.models import Agent, Assembly
from app.agents.main import ToolerOrchestrator
from .prompts import BASE_PROMPT, DIALOGUE_AGENT_PROMPT, TRANSLATOR_AGENT_PROMPT, REVIEWER_AGENT_PROMPT

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'src/backend'
sys.path.append(str(SRC))


class TransliterationChatManager(GroupChatManager):
    """
    Manages the group chat flow for the transliteration and review process.
    This custom manager coordinates the interaction between translation and review agents,
    determines when to terminate the chat, and selects the next agent to act based on the chat history.
    """
    async def filter_results(self, chat_history: ChatHistory) -> MessageResult:
        """
        Extracts the latest transliteration and review results from the chat history and composes a combined response.

        Args:
            chat_history (ChatHistory): The chat history containing all messages exchanged.
        Returns:
            MessageResult: A result object containing the composed transliteration and review as JSON.
        """
        # Extrai transliteração e avaliação do histórico
        translit = None
        review = None
        for msg in reversed(chat_history.messages):
            content = getattr(msg, 'content', '')
            last_agent = getattr(msg, 'name', '')
            if last_agent and "TranslationReviewerAgent" in last_agent and review is None:
                try:
                    review = json.loads(content)
                except Exception:
                    continue
            elif last_agent and "TermExtractorTranslatorAgent" in last_agent and translit is None:
                try:
                    translit = json.loads(content)
                except Exception:
                    continue
            if translit and review:
                break
        response = {
            "transliteration": translit,
            "review": review
        }
        return MessageResult(
            result=ChatMessageContent(role=AuthorRole.ASSISTANT, content=json.dumps(response, ensure_ascii=False, indent=2)),
            reason="Composed transliteration and review."
        )

    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """
        Determines whether the group chat should terminate based on the latest review score or approval status.

        Args:
            chat_history (ChatHistory): The chat history containing all messages exchanged.
        Returns:
            BooleanResult: True if the process should terminate, False otherwise.
        """
        # Procura score_global e acao_recomendada na última avaliação
        for msg in reversed(chat_history.messages):
            content = getattr(msg, 'content', '')
            last_agent = getattr(msg, 'name', 'User')
            if last_agent and "TranslationReviewerAgent" in last_agent:
                try:
                    review = json.loads(content)
                    score = review.get("score_global", 0)
                    acao = review.get("acao_recomendada", "")
                    if score >= 80 or acao == "aprovado":
                        return BooleanResult(result=True, reason=f"Score {score} e aprovado.")
                except Exception:
                    continue
        return BooleanResult(result=False, reason="Score ou aprovação ainda não atingidos.")

    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """
        Indicates whether user input is required during the automatic cycle (always False for this pipeline).

        Args:
            chat_history (ChatHistory): The chat history containing all messages exchanged.
        Returns:
            BooleanResult: Always False for this implementation.
        """
        # Nunca solicita input do usuário durante o ciclo automático
        return BooleanResult(result=False, reason="Ciclo automático, sem input do usuário.")

    async def select_next_agent(self, chat_history: ChatHistory, participant_descriptions: dict[str, str]) -> StringResult:
        """
        Selects the next agent to act in the group chat based on the last message and review status.

        Args:
            chat_history (ChatHistory): The chat history containing all messages exchanged.
            participant_descriptions (dict): Mapping of agent names to their descriptions.
        Returns:
            StringResult: The name of the next agent to act.
        """
        agents = list(participant_descriptions.keys())
        # Se não há mensagens de agentes ainda (apenas user), começa com transliteração
        if not chat_history.messages or (
            len(chat_history.messages) == 1 and chat_history.messages[0].role == AuthorRole.USER
        ):
            return StringResult(result=agents[0], reason="Primeira rodada: transliteração.")
        # Determina se o último respondente foi o reviewer pelo conteúdo
        last_msg = chat_history.messages[-1]
        last_agent = getattr(last_msg, 'name', '')
        is_reviewer = False
        if "TranslationReviewerAgent" in last_agent:
            is_reviewer = True
        # Busca pelo score na última avaliação
        score = None
        aprovacao = "reexecutar"
        if is_reviewer:
            try:
                review = json.loads(last_msg.content)
                aprovacao = review.get("acao_recomendada", "reexecutar")
                score = review.get("score_global", 0)
            except Exception:
                pass
            if score is not None and score < 90 or aprovacao == "reexecutar":
                return StringResult(result=agents[0], reason=last_msg.content)
            return StringResult(result=agents[0], reason="Score suficiente, transliteração final.")
        # Se não foi reviewer, alterna para o avaliador
        return StringResult(result=agents[1], reason="Alterna para avaliador.")

class Orchestrator(ToolerOrchestrator):
    """
    Orchestrates the dialogue generation and transliteration pipeline for indigenous language translation.
    Inherits from ToolerOrchestrator and encapsulates all orchestration logic, agent creation, and helper utilities.
    """
    def clean_json_output(self, text):
        """
        Removes markdown code block markers from model output, if present.

        Args:
            text (str): The text output from the model.
        Returns:
            str: The cleaned text without markdown code block markers.
        """
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()

    def extract_json(self, text):
        """
        Extracts the largest valid JSON block from the given text.
        Tries to find and parse a JSON object or array, falling back to the whole text if needed.

        Args:
            text (str): The text to extract JSON from.
        Returns:
            dict or list: The parsed JSON object or array.
        Raises:
            ValueError: If no valid JSON is found in the text.
        """
        text = self.clean_json_output(text)
        # Regex para encontrar o maior bloco JSON
        matches = re.findall(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
        for match in matches:
            try:
                return json.loads(match)
            except Exception:
                continue
        # fallback: tenta o texto inteiro
        try:
            return json.loads(text)
        except Exception:
            pass
        raise ValueError("Nenhum JSON válido encontrado no output.")

    def build_translator_prompt(self, dialogo, feedback=None, translit_anterior=None):
        """
        Builds the prompt for the translator agent, optionally including feedback and previous transliteration.

        Args:
            dialogo (dict): The dialogue entry to be translated.
            feedback (dict, optional): Feedback from the reviewer to be incorporated.
            translit_anterior (dict, optional): Previous transliteration result.
        Returns:
            str: The constructed prompt as a JSON string.
        """
        base_prompt = {
            "json_object": True,
            "prompt": f"""
            Você é um tradutor especializado em línguas indígenas. Você receberá diálogos em português e deve traduzi-los para Katukina e Pano, utilizando os pdfs disponíveis para referência. Os PDFs são conjuntos de texto que oferecem uma indicação da fonética da língua, mas não indicam a gramática. Você deve citar as fontes de cada tradução, incluindo a página do PDF consultado.\n\nFormate a resposta como JSON puro.\n\nSe houver feedback do avaliador, implemente as recomendações explicitamente, não repita a resposta anterior.\n\n{BASE_PROMPT}"""
        }
        if feedback:
            base_prompt["feedback_anterior"] = feedback
        if translit_anterior:
            base_prompt["transliteracao_anterior"] = translit_anterior
        base_prompt["dialogo"] = dialogo
        base_prompt["instrucao"] = "Ajuste a transliteração conforme o feedback acima, mantendo as referências aos PDFs e justificativas."
        return json.dumps(base_prompt, ensure_ascii=False)

    def create_agents(self):
        """
        Creates and returns the dialogue, translator, and reviewer agent objects for the pipeline.

        Returns:
            tuple: (dialogue_agent, translator_agent, reviewer_agent)
        """
        dialogue_agent = Agent(
            id=1,
            name="DialogueCreatorAgent",
            model_id="default",
            objective="text",
            metaprompt=json.dumps({
                "json_object": True,
                "prompt": DIALOGUE_AGENT_PROMPT
            }),
            description="Gera as falas em português a partir dos cases/jornada."
        )
        translator_agent = Agent(
            id=2,
            name="TermExtractorTranslatorAgent",
            model_id="quatroum",
            objective="text",
            metaprompt=json.dumps({
                "json_object": True,
                "prompt": TRANSLATOR_AGENT_PROMPT
            }),
            description="Traduz as falas PT-BR usando os PDFs e cita as fontes."
        )
        reviewer_agent = Agent(
            id=3,
            name="TranslationReviewerAgent",
            model_id="reasoning",
            objective="text",
            metaprompt=json.dumps({
                "json_object": True,
                "prompt": REVIEWER_AGENT_PROMPT
            }),
            description="Revê traduções e validacoes"
        )
        return dialogue_agent, translator_agent, reviewer_agent

    async def generate_dialogues(self, dialogue_agent):
        """
        Generates dialogues using the provided dialogue agent and saves them to a file.

        Args:
            dialogue_agent (Agent): The agent responsible for generating dialogues.
        Returns:
            list: The list of generated dialogue entries (parsed from JSON).
        """
        assembly1 = Assembly(
            id=1,
            objective="dialogue_generation",
            agents=[dialogue_agent],
            roles=["DialogueCreator"]
        )
        dialogues = await self.invoke(
            assembly=assembly1,
            prompt="generate dialogues",
            strategy="sequential"
        )
        dialogues_path = Path(__file__).parent / "dialogues.json"
        dialogues_json = self.extract_json(dialogues.content)
        with open(dialogues_path, "w", encoding="utf-8") as f:
            json.dump(dialogues_json, f, ensure_ascii=False, indent=2)
        print(f"Saved dialogues to {dialogues_path}")
        return dialogues_json

    async def run_transliteration_pipeline(self, dialogues_json, translator_agent, reviewer_agent):
        """
        Runs the transliteration and review pipeline for each dialogue entry, saving results to files.

        Args:
            dialogues_json (list): The list of dialogue entries to process.
            translator_agent (Agent): The agent responsible for transliteration.
            reviewer_agent (Agent): The agent responsible for reviewing translations.
        """
        translit_dir = Path(__file__).parent / "transliteracoes"
        translit_dir.mkdir(exist_ok=True)
        for entry in dialogues_json:
            assembly2 = Assembly(
                id=2,
                objective="transliteration",
                agents=[translator_agent, reviewer_agent],
                roles=["Translation", "Review"]
            )
            max_attempts = 3
            last_feedback = None
            last_translit = None
            for attempt in range(max_attempts):
                if attempt > 0 and last_feedback and last_translit:
                    translator_prompt = self.build_translator_prompt(entry, feedback=last_feedback, translit_anterior=last_translit)
                else:
                    translator_prompt = self.build_translator_prompt(entry)
                result = await self.invoke(
                    assembly=assembly2,
                    prompt=translator_prompt,
                    strategy="group",
                    max_rounds=7,
                    manager=TransliterationChatManager(max_rounds=7)
                )
                convo_id = entry.get("id", "unknown")
                out_file = translit_dir / f"transliteration_{convo_id}.txt"
                try:
                    result_json = self.extract_json(result.content)
                    if "review" in result_json and result_json["review"]:
                        last_feedback = result_json["review"]
                    if "transliteration" in result_json and result_json["transliteration"]:
                        last_translit = result_json["transliteration"]
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(result_json, f, ensure_ascii=False, indent=2)
                    print(f"Saved transliteration for {convo_id} to {out_file}")
                    break
                except Exception as e:
                    print(f"Erro ao serializar JSON para {convo_id} (tentativa {attempt+1}/{max_attempts}): {e}")
                    if attempt == max_attempts - 1:
                        print(f"Falha definitiva ao salvar transliteração para {convo_id}.")

async def main():
    """
    Entry point for the orchestration pipeline.
    Loads environment variables, sets up agents, and runs the dialogue and transliteration pipeline using the Orchestrator class.
    """
    load_dotenv(SRC / '.env')
    orchestrator = Orchestrator()
    async with orchestrator:
        dialogue_agent, translator_agent, reviewer_agent = orchestrator.create_agents()
        dialogues_json = await orchestrator.generate_dialogues(dialogue_agent)
        await orchestrator.run_transliteration_pipeline(dialogues_json, translator_agent, reviewer_agent)
    return

if __name__ == "__main__":
    asyncio.run(main())
