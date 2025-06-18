import io
import sys
import asyncio
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from semantic_kernel.agents import GroupChatManager, BooleanResult, MessageResult, StringResult  # pylint: disable=no-name-in-module
from semantic_kernel.contents import ChatMessageContent, ChatHistory, AuthorRole

from app.schemas.models import Agent, Assembly
from app.agents.main import ToolerOrchestrator

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'src/essays'
sys.path.append(str(SRC))

base = Path(__file__).parent / "data" / "translation"
BASE_PROMPT = f"""
──────────────────────────────────────────────────────────────────────────────
Pasta de dados: {base}

• Você tem à sua disposição os seguintes PDFs linguísticos: katukina_01.pdf, katukina_02.pdf, pano_01.pdf, pano_02.pdf
• Translitere cada fala para **Katukina** e **Pano**, citando páginas dos PDFs. Use 10 páginas por vez em suas consultas.

Macro-tarefa do sistema  
───────────────────────
1. Gerar falas em português para cada “história” ou “etapa”.  

3. Validar fluidez, sentido, gramática e apontar ajustes.

⚠︎ Todas as saídas DEVEM ser JSON puro (nenhum texto fora do JSON).  
⚠︎ Marque termos inventados com “★” e justifique no campo *fontes*.  
⚠︎ Nunca repita estas instruções no output.
──────────────────────────────────────────────────────────────────────────────
"""

class DialogueOutputItem(BaseModel):
    id: str
    ator: str
    fala: str


class DialogueOutput(BaseModel):
    dialogues: list[DialogueOutputItem] = Field(..., description="Lista de diálogos gerados com base nos casos e jornadas.")


class TransliterationOutput(BaseModel):
    id: str
    ator: str
    portugues: str
    katukina: str = Field(..., description="Transliteração para Katukina")
    fontes_katukina: list[str] = Field(..., description="Fontes consultadas para Katukina")
    pano: str = Field(..., description="Transliteração para Pano")
    fontes_pano: list[str] = Field(..., description="Fontes consultadas para Pano")


class TransliterationChatManager(GroupChatManager):
    async def filter_results(self, chat_history: ChatHistory) -> MessageResult:
        # Extrai transliteração e avaliação do histórico
        translit = None
        review = None
        for msg in reversed(chat_history.messages):
            content = getattr(msg, 'content', '')
            last_agent = getattr(msg, 'name', 'User')
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
        # Nunca solicita input do usuário durante o ciclo automático
        return BooleanResult(result=False, reason="Ciclo automático, sem input do usuário.")

    async def select_next_agent(self, chat_history: ChatHistory, participant_descriptions: dict[str, str]) -> StringResult:
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
            else:
                return StringResult(result=agents[0], reason="Score suficiente, transliteração final.")
        # Se não foi reviewer, alterna para o avaliador
        return StringResult(result=agents[1], reason="Alterna para avaliador.")


def clean_json_output(text):
    """Remove markdown code block markers from model output."""
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


def extract_json(text):
    """Tenta extrair o maior bloco JSON válido de um texto."""
    text = clean_json_output(text)
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


def build_translator_prompt(dialogo, feedback=None, translit_anterior=None):
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


async def main():
    load_dotenv(SRC / '.env')
    orchestrator = ToolerOrchestrator()
    dialogue_agent = Agent(
        id=1,
        name="DialogueCreatorAgent",
        model_id="default",
        objective="text",
        metaprompt=json.dumps({
            "json_object": True,
            "prompt": f"""
            Você é um especialista em criar diálogos a partir de casos e jornadas.
            Você receberá os casos e jornadas em JSON e deve gerar as falas em português.
            seja claro, conciso e use linguagem natural, apenas em português, para gerar falas com base no cenário.
            ───────────────────────────────────────────────────────────────────────────────
            {BASE_PROMPT}
            ───────────────────────────────────────────────────────────────────────────────
            • Abra TODOS os JSON de cenários: case_01.json, case_02.json, case_03.json, jornada.json

            Estrutura dos JSON  
            ──────────────────
            case_XX.json  →  slide.historias[]  (id, ator, nome, descricao…)  
            jornada.json →  slide.etapas[]      (id, nome, descricao…)
            ───────────────────────────────────────────────────────────────────────────────

            Formato de saída (DialogueOutput):
            {{
                \"id\":   \"1.1\",
                \"ator\": \"Cliente\",
                \"fala\": \"Olá, quero consultar meu saldo.\"
            }}

            ⚠︎ Entregue **somente** este objeto JSON. **JAMAIS inclua markdown na saída**.
            """
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
            "prompt": f"""
            Você é um tradutor especializado em línguas indígenas.
            Você receberá diálogos em português e deve traduzi-los para Katukina e Pano,
            utilizando os pdfs disponíveis para referência.
            Os PDFs são conjuntos de texto que oferecem uma indicação da fonética da língua, mas não indicam a gramática.
            Você deve citar as fontes de cada tradução, incluindo a página do PDF consultado.
            ───────────────────────────────────────────────────────────────────────────────
            {BASE_PROMPT}
            ───────────────────────────────────────────────────────────────────────────────
            • Você tem à sua disposição os seguintes PDFs linguísticos: katukina_01.pdf, katukina_02.pdf, pano_01.pdf, pano_02.pdf
            • Translitere cada fala para **Katukina** e **Pano**, citando páginas dos PDFs. Use 10 páginas por vez em suas consultas. 
            ───────────────────────────────────────────────────────────────────────────────

            Formato de ENTRADA (lista):
            [{{
                \"id\":   \"1.1\",
                \"ator\": \"Cliente\",
                \"portugues\": \"Olá, quero consultar meu saldo.\"
            }}, ...
            ]

            Formato de SAÍDA (lista JSON):
            [{{
                \"id\":        \"1.1\",
                \"ator\":      \"Cliente\",
                \"portugues\": \"Olá, quero consultar meu saldo.\",
                \"katukina\":  \"[TRANSLITERAÇÃO PARA KATUKINA]\",
                \"fontes_katukina\": [\"katukina_01.pdf#p42\"],
                \"pano\":      \"[TRANSLITERAÇÃO PARA PANO]\",
                \"fontes_pano\": [\"pano_01.pdf#p17\", \"★\"]
            }},
            [...OUTROS ITENS DA AVALIAÇÃO]
            ]

            ⚠︎ Entregue **somente** este objeto JSON. **JAMAIS inclua marcadores de markdown na saída**.
            """
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
            "prompt": f"""
            Você é um expert em avaliar traduções.
            Você receberá um conjunto de diálogos traduzidos e deve consultar as fontes disponíveis para validar a qualidade.
            Sempre inclua comentários claros e objetivos sobre cada avaliação.
            Você receberá pdfs que informam a fonética das línguas Katukina e Pano.
            Você deve verificar a precisão das traduções, a fluência do texto e se as fontes foram corretamente citadas.
            Se a tradução estiver correta, retorne \"OK\". Se precisar de ajustes, retorne \"Ajustar\" e forneça uma justificativa clara.
            Utilize os PDFs para verificar se a composição de palavras nos textos traduzidos segue a estrutura fonética apresentada nos PDFs.
            ───────────────────────────────────────────────────────────────────────────────
            {BASE_PROMPT}
            ───────────────────────────────────────────────────────────────────────────────
            • Você tem à sua disposição os seguintes PDFs linguísticos: katukina_01.pdf, katukina_02.pdf, pano_01.pdf, pano_02.pdf
            • Translitere cada fala para **Katukina** e **Pano**, citando páginas dos PDFs. Use 10 páginas por vez em suas consultas. 
            ───────────────────────────────────────────────────────────────────────────────
            Saída (objeto JSON):
            {{
            \"avaliacoes\": [
                {{
                \"id\": \"1.1\",
                \"status\": \"OK\" (CASO ESTRUTURA FONÉTICA ESTEJA COMPATÍVEL) | \"Ajustar\" (CASO CONTRÁRIO),
                \"comentario\": \"Razão ou sugestão clara dos pontos de ajuste fonéticos\"
                }},
                [...OUTROS ITENS DA AVALIAÇÃO]
            ],
            \"acao_recomendada\": \"aprovado\" | \"reexecutar\",
            \"score_global\": 0-100 (0=baixa aderência à composição fonética, 100=perfeita aderência à composição fonética da língua)
            }}

            ⚠︎ Entregue **somente** este objeto JSON. **JAMAIS inclua marcadores de markdown na saída**.
            """
        }),
        description="Revê traduções e validacoes"
    )

    async with orchestrator:

        assembly1 = Assembly(
            id=1,
            objective="dialogue_generation",
            agents=[dialogue_agent],
            roles=["DialogueCreator"]
        )
        dialogues = await orchestrator.invoke(
            assembly=assembly1,
            prompt="generate dialogues",
            strategy="sequential"
        )

        dialogues_path = Path(__file__).parent / "dialogues.json"
        dialogues_json = extract_json(dialogues.content)
        with open(dialogues_path, "w", encoding="utf-8") as f:
            json.dump(dialogues_json, f, ensure_ascii=False, indent=2)
        print(f"Saved dialogues to {dialogues_path}")

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
                # Se não for a primeira tentativa, inclua feedback e transliteração anterior
                if attempt > 0 and last_feedback and last_translit:
                    translator_prompt = build_translator_prompt(entry, feedback=last_feedback, translit_anterior=last_translit)
                else:
                    translator_prompt = build_translator_prompt(entry)
                result = await orchestrator.invoke(
                    assembly=assembly2,
                    prompt=translator_prompt,
                    strategy="group",
                    max_rounds=7,
                    manager=TransliterationChatManager(max_rounds=7)
                )
                convo_id = entry.get("id", "unknown")
                out_file = translit_dir / f"transliteration_{convo_id}.txt"
                try:
                    # Clean and parse the result content
                    result_json = extract_json(result.content)
                    # Atualiza feedback e transliteração para próxima iteração, se necessário
                    if "review" in result_json and result_json["review"]:
                        last_feedback = result_json["review"]
                    if "transliteration" in result_json and result_json["transliteration"]:
                        last_translit = result_json["transliteration"]
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(result_json, f, ensure_ascii=False, indent=2)
                    print(f"Saved transliteration for {convo_id} to {out_file}")
                    break  # Sucesso, sai do loop
                except Exception as e:
                    print(f"Erro ao serializar JSON para {convo_id} (tentativa {attempt+1}/{max_attempts}): {e}")
                    if attempt == max_attempts - 1:
                        print(f"Falha definitiva ao salvar transliteração para {convo_id}.")
    return


if __name__ == "__main__":
    asyncio.run(main())
