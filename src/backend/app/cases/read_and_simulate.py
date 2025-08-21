"""Transliteration / generation pipeline (clean implementation).

Features:
- Dialogue generation (Portuguese) via Dialogue agent.
- Translator & Reviewer agents (groupchat or manual loop).
- Evidence prefetch through AzureAISearchTool (heuristic multi-query).
- Multi-language aggregation when TARGET_IDIOMAS provided.
- Always uses groupchat when ORCHESTRATION_MODE=groupchat (single or multi).
"""

from __future__ import annotations

import os, re, json, asyncio, logging, random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent  # pylint: disable=no-name-in-module
from semantic_kernel.agents.orchestration.group_chat import (
    GroupChatOrchestration, RoundRobinGroupChatManager, BooleanResult, StringResult, MessageResult
)  # type: ignore
from semantic_kernel.agents.runtime.in_process.in_process_runtime import InProcessRuntime  # type: ignore
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.exceptions.content_filter_ai_exception import ContentFilterAIException

from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments

from app.plugins import PDFLoader, search_plugin
from app.cases.prompts import (
    BASE_PROMPT_TEMPLATE, DIALOGUE_AGENT_PROMPT_TEMPLATE, TRANSLATOR_AGENT_PROMPT_TEMPLATE, REVIEWER_AGENT_PROMPT_TEMPLATE, render_prompt
)

load_dotenv()
ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src" / "backend"

LOG_LEVEL = getattr(logging, os.getenv("TRANSLIT_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("transliteration")


def build_kernel() -> sk.Kernel:
    api_key = os.getenv("AZURE_FOUNDRY_KEY", "") or os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_FOUNDRY_URL", "") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    kernel = sk.Kernel()
    for sid, dep in (
        ("default", os.getenv("AZURE_DIALOGUE_DEPLOYMENT", "contractor-4o")),
        ("gpt-5-mini", os.getenv("AZURE_TRANSLATOR_DEPLOYMENT", "gpt-5-mini")),
        ("reasoning", os.getenv("AZURE_REVIEW_DEPLOYMENT", "o3-mini")),
    ):
        kernel.add_service(
            AzureChatCompletion(
                service_id=sid,
                api_key=api_key,
                deployment_name=dep,
                endpoint=endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
        )
    for plugin in [PDFLoader(), *search_plugin]:
        kernel.add_plugin(plugin, plugin.__class__.__name__)
    return kernel


def create_agent(kernel: sk.Kernel, name: str, description: str, model_id: str, prompt: str) -> ChatCompletionAgent:
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=model_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    return ChatCompletionAgent(
        kernel=kernel,
        name=name,
        description=description,
        instructions=prompt,
        arguments=KernelArguments(settings=settings),
    )


JSON_BLOCK_RE = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Texto vazio")
    if text[0] in "[{" and text[-1] in "]}":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    for m in JSON_BLOCK_RE.finditer(text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError("JSON não encontrado")


def extract_query_terms(portuguese_text: str, max_terms: int = 8) -> List[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{4,}", (portuguese_text or "").lower())
    stop = {
        "para",
        "como",
        "onde",
        "quando",
        "sobre",
        "entre",
        "isso",
        "dessa",
        "deste",
        "qual",
        "cada",
        "mais",
        "menos",
        "muito",
        "pouco",
        "preciso",
        "acessar",
        "conta",
        "usando",
        "biometria"
    }
    out: List[str] = []
    for t in tokens:
        if t in stop:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


async def fetch_references(kernel: sk.Kernel, dialogue_entry: Dict[str, Any], idioma: str) -> List[Dict[str, Any]]:
    fala = dialogue_entry.get("fala") or dialogue_entry.get("portugues") or ""
    if not fala:
        return []
    terms = extract_query_terms(fala)
    queries: List[str] = []
    if terms:
        queries.append(" ".join(terms[:5]))
    for t in terms[:4]:
        queries.append(f"{idioma} {t}")
    for t in terms[:3]:
        queries.append(t)

    seen: set[str] = set()
    ordered: List[str] = []
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        ordered.append(q)
    refs: List[Dict[str, Any]] = []
    keyed = set()
    for q in ordered:
        try:
            result = await kernel.invoke(
                function_name="search_index",
                plugin_name="AzureAISearchTool",
                arguments=KernelArguments(query=q),
            )
        except (ValueError, RuntimeError) as e:  # narrowed expected invocation failures
            logger.debug("search falhou query=%s err=%s", q, e)
            continue
        items = getattr(result, "value", result)
        if isinstance(items, list):
            for doc in items:
                key = (doc.get("file_name"), doc.get("page_number"))
                if key not in keyed:
                    keyed.add(key)
                    refs.append(doc)
        if len(refs) >= 15:
            break
    return refs[:15]


async def generate_dialogues(kernel: sk.Kernel, idioma: str) -> List[Dict[str, Any]]:
    scenarios_dir = ROOT / "notebooks" / "data" / "translation"
    case_files = ["case_01.json", "case_02.json", "case_03.json"]
    cases_content: Dict[str, Any] = {}
    for fname in case_files:
        fp = scenarios_dir / fname
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    cases_content[fname] = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Falha lendo %s: %s", fname, e)
    jornada_content = None
    jf = scenarios_dir / "jornada.json"
    if jf.exists():
        try:
            with open(jf, "r", encoding="utf-8") as f:
                jornada_content = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Falha lendo jornada.json: %s", e)
    prompt = render_prompt(
        DIALOGUE_AGENT_PROMPT_TEMPLATE,
        base_prompt=render_prompt(BASE_PROMPT_TEMPLATE, base=str(scenarios_dir), idioma=idioma),
        idioma=idioma,
    )
    agent = create_agent(kernel, "DialogueCreatorAgent", "Gera falas", "default", prompt)
    payload = {
        "json_object": True,
        "instrucao": "Gerar falas",
        "cases": cases_content,
        "jornada": jornada_content,
    }
    hist = ChatHistory()
    hist.add_message(ChatMessageContent(role=AuthorRole.USER, content=json.dumps(payload, ensure_ascii=False)))
    resp = ""
    async for m in agent.invoke(messages=hist.messages):  # type: ignore
        if m.role == AuthorRole.ASSISTANT:
            resp += m.content.content
    try:
        dialogues = extract_json(resp)
        if not isinstance(dialogues, list):
            raise ValueError
    except (ValueError, json.JSONDecodeError) as e:
        dialogues = [{"id": "1.1", "ator": "Cliente", "fala": "Olá"}]
        logger.error("Fallback dialogues used (parse failure): %s", e)
    out_path = SRC / "dialogues.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)
    return dialogues


def build_translator_and_reviewer(kernel: sk.Kernel, idioma: str) -> Tuple[ChatCompletionAgent, ChatCompletionAgent]:
    base_rendered = render_prompt(
        BASE_PROMPT_TEMPLATE,
        base=str(ROOT / "notebooks" / "data" / "translation"),
        idioma=idioma,
    )
    translator_prompt = render_prompt(
        TRANSLATOR_AGENT_PROMPT_TEMPLATE, base_prompt=base_rendered, idioma=idioma
    )
    reviewer_prompt = render_prompt(
        REVIEWER_AGENT_PROMPT_TEMPLATE, base_prompt=base_rendered, idioma=idioma
    )
    t = create_agent(
        kernel, "TranslatorAgent", "Gera enunciado", "gpt-5-mini", translator_prompt
    )
    r = create_agent(
        kernel, "TranslationReviewerAgent", "Revisa enunciado", "reasoning", reviewer_prompt
    )
    return t, r


class TranslatorReviewerManager(RoundRobinGroupChatManager):
    def __init__(self, max_rounds: int = 20):
        super().__init__()
        self.max_rounds = max_rounds
        self._last_reviewer_action = None
        self._turn_index = 0  # translator (even) -> reviewer (odd)

    async def should_request_user_input(self, chat_history):  # type: ignore
        return BooleanResult(result=False, reason="auto")

    async def should_terminate(self, chat_history):  # type: ignore
        for msg in reversed(chat_history.messages):  # newest first
            if (
                msg.name
                and msg.name.lower().startswith("translationreviewer")
                and msg.role == AuthorRole.ASSISTANT
            ):
                c = msg.content or ""
                if '"acao_recomendada"' in c:
                    if "aprovado" in c:
                        return BooleanResult(result=True, reason="aprovado")
                    if "reexecutar" in c:
                        self._last_reviewer_action = "reexecutar"
                        break
        reviewer_msgs = [
            m for m in chat_history.messages if m.name and m.name.lower().startswith("translationreviewer")
        ]
        if self.max_rounds is not None and len(reviewer_msgs) >= self.max_rounds:
            return BooleanResult(result=True, reason="limite_rounds")
        return BooleanResult(result=False, reason="continuar")

    async def select_next_agent(self, chat_history, participant_descriptions):  # type: ignore
        agents = list(participant_descriptions.keys())
        if len(agents) != 2:
            return StringResult(result=agents[0], reason="fallback")
        a, b = agents
        if self._last_reviewer_action == "reexecutar":
            self._last_reviewer_action = None
            sel = a
            reason = "retry"
        else:
            sel = a if self._turn_index % 2 == 0 else b
            reason = "alternancia"
        logger.debug("[manager] turn=%d selecting=%s reason=%s", self._turn_index, sel, reason)
        self._turn_index += 1
        return StringResult(result=sel, reason=reason)

    async def filter_results(self, chat_history):  # type: ignore
        for msg in reversed(chat_history.messages):
            if msg.name and msg.name.lower().startswith("translationreviewer"):
                return MessageResult(result=msg, reason="reviewer final")
        return MessageResult(result=chat_history.messages[-1], reason="ultimo")


###############################
# Resilience / retry helpers  #
###############################

TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default

def _get_config() -> Dict[str, Any]:
    # Simples (sem cache global para evitar warning de global); custo desprezível.
    return {
        "retry_max_attempts": _get_env_int("RETRY_MAX_ATTEMPTS", 5),
        "retry_base_delay": _get_env_float("RETRY_BASE_DELAY", 0.8),
        "retry_max_delay": _get_env_float("RETRY_MAX_DELAY", 12.0),
        "call_timeout": _get_env_float("CALL_TIMEOUT_SECONDS", 90.0),
        "circuit_breaker_threshold": _get_env_int("CIRCUIT_BREAKER_THRESHOLD", 8),
        "max_concurrency": max(1, _get_env_int("TRANSLIT_MAX_CONCURRENCY", 1)),
        "partial_flush_interval": _get_env_int("PARTIAL_FLUSH_INTERVAL", 3),
        "progress_file": os.getenv("TRANSLIT_PROGRESS_FILE"),
    }

def _is_transient(exc: Exception) -> bool:  # heurística leve
    code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status", None)
    if isinstance(code, int) and code in TRANSIENT_STATUS_CODES:
        return True
    msg = str(exc).lower()
    transient_markers = [
        "rate limit",
        "timed out",
        "timeout",
        "temporarily",
        "connection",
        "retry-after",
        "did not return any response",
        "empty_response",
    ]
    return any(m in msg for m in transient_markers)

def _calc_backoff(exc: Exception, attempt: int, base: float, max_delay: float) -> float:
    # Respeita retry-after quando presente
    retry_after = None
    for attr in ("retry_after", "retry_after_ms"):
        ra = getattr(exc, attr, None)
        if ra:
            retry_after = ra
            break
    # Header style
    resp = getattr(exc, "response", None)
    if retry_after is None and resp is not None:
        headers = getattr(resp, "headers", {}) or {}
        for k in ("retry-after-ms", "retry-after"):
            if k in headers:
                retry_after = headers[k]
                break
    if retry_after is not None:
        try:
            return min(float(retry_after) / (1000.0 if "ms" in str(retry_after).lower() else 1.0), max_delay)
        except ValueError:
            pass
    # Exponential com jitter
    raw = base * (2 ** (attempt - 1))
    return min(raw + random.uniform(0, 0.5), max_delay)

async def _call_with_retry(op_name: str, coro_factory, *, timeout: Optional[float] = None):
    cfg = _get_config()
    max_attempts = cfg["retry_max_attempts"]
    base = cfg["retry_base_delay"]
    max_delay = cfg["retry_max_delay"]
    attempt = 0
    while True:
        attempt += 1
        try:
            if timeout is not None:
                return await asyncio.wait_for(coro_factory(), timeout=timeout)
            return await coro_factory()
        except (OSError, RuntimeError, ValueError, asyncio.TimeoutError, ContentFilterAIException) as exc:
            if isinstance(exc, ContentFilterAIException):
                # Não retentar content filter: registrar e propagar.
                logger.error(
                    "op=%s content_filter err=%r", op_name, exc
                )
                raise
            transient = _is_transient(exc)
            if not transient or attempt >= max_attempts:
                logger.error(
                    "op=%s attempt=%d abort transient=%s err=%r", op_name, attempt, transient, exc
                )
                raise
            delay = _calc_backoff(exc, attempt, base, max_delay)
            logger.warning(
                "retry op=%s attempt=%d/%d delay=%.2fs transient=%s err=%s", op_name, attempt, max_attempts, delay, transient, exc
            )
            await asyncio.sleep(delay)

async def _process_single_dialog_groupchat(
    translator: ChatCompletionAgent,
    reviewer: ChatCompletionAgent,
    d: Dict[str, Any],
    idioma: str,
    max_rounds: int,
) -> Dict[str, Any]:
    """Processa um único diálogo com isolamento, retry e timeout.

    Retorna sempre um objeto de resultado (nunca lança exc. exceto KeyboardInterrupt).
    """
    cfg = _get_config()
    dialogo = dict(d)
    if "fala" in dialogo and "portugues" not in dialogo:
        dialogo["portugues"] = dialogo["fala"]
    try:
        refs = await fetch_references(translator.kernel, dialogo, idioma)  # type: ignore[attr-defined]
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        logger.warning("refs_fail id=%s err=%r", dialogo.get("id"), e)
        refs = []
    seed = {
        "json_object": True,
        "dialogo": dialogo,
        "idioma_alvo": idioma,
        "referencias_indexadas": refs,
    }
    manager = TranslatorReviewerManager(max_rounds=max_rounds)

    async def _agent_cb(message):  # type: ignore
        try:
            name = getattr(message, "name", None) or "?"
            role = getattr(message, "role", None)
            content = (getattr(message, "content", "") or "")
            logger.debug(
                "[gc-cb] %s role=%s len=%d preview=%s", name, role, len(content), content[:160].replace("\n", " ")
            )
        except RuntimeError as e:  # pragma: no cover
            logger.warning("[gc-cb] logging error: %s", e)

    orch = GroupChatOrchestration(members=[translator, reviewer], manager=manager, agent_response_callback=_agent_cb)
    runtime = InProcessRuntime()
    runtime.start()
    try:
        # Retry envolve orch.invoke + handle.get, não cada token individual.
        async def _invoke_and_get():
            handle = await orch.invoke(
                task=ChatMessageContent(
                    role=AuthorRole.USER, content=json.dumps(seed, ensure_ascii=False)
                ),
                runtime=runtime,
            )
            result_msg = await handle.get()
            # Checa resposta vazia (problema observado de agente sem retorno) -> força retry.
            content_attr = getattr(result_msg, "content", None)
            if content_attr is None or (isinstance(content_attr, str) and not content_attr.strip()):
                raise RuntimeError("empty_response")
            # Alguns objetos podem ter .content como ChatMessageContent; tratar.
            if not isinstance(content_attr, str):
                inner = getattr(content_attr, "content", "")
                if isinstance(inner, str) and not inner.strip():
                    raise RuntimeError("empty_response_inner")
            return result_msg

        final = await _call_with_retry(
            op_name=f"groupchat_{dialogo.get('id','?')}",
            coro_factory=_invoke_and_get,
            timeout=cfg["call_timeout"],
        )
        content = getattr(final, "content", "") or ""
        try:
            parsed = extract_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            parsed = {"raw": content, "erro_parse": str(e)}
        translit = None
        if isinstance(parsed, dict) and "transliteracao" in parsed:
            translit = parsed["transliteracao"]
        if translit is None and isinstance(parsed, list):
            translit = parsed
        if translit is None:
            translit = [
                {
                    "id": dialogo.get("id"),
                    "ator": dialogo.get("ator"),
                    "portugues": dialogo.get("portugues"),
                    idioma: "[TRADUÇÃO_INDEFINIDA]",
                    f"fontes_{idioma}": [
                        f"{r.get('file_name')}#p{r.get('page_number')}"
                        for r in refs[:1]
                        if r.get("file_name")
                    ],
                    "sem_fonte": not refs,
                    "observacao": "fallback",
                }
            ]
        return {"dialogo": d, "transliteracao": translit, "review": parsed}
    except ContentFilterAIException as cf_err:
        logger.warning(
            "[gc] content_filter id=%s detalhe=%s", dialogo.get("id"), cf_err
        )
        translit = [
            {
                "id": dialogo.get("id"),
                "ator": dialogo.get("ator"),
                "portugues": dialogo.get("portugues"),
                idioma: "[BLOQUEADO_POR_CONTENT_FILTER]",
                f"fontes_{idioma}": [
                    f"{r.get('file_name')}#p{r.get('page_number')}" for r in refs[:1] if r.get("file_name")
                ],
                "sem_fonte": not refs,
                "observacao": "Fallback após content filter",
            }
        ]
        return {"dialogo": d, "transliteracao": translit, "review": {"erro": "content_filter"}}
    except asyncio.TimeoutError:
        logger.error("timeout groupchat id=%s", dialogo.get("id"))
        translit = [
            {
                "id": dialogo.get("id"),
                "ator": dialogo.get("ator"),
                "portugues": dialogo.get("portugues"),
                idioma: "[TIMEOUT]",
                f"fontes_{idioma}": [
                    f"{r.get('file_name')}#p{r.get('page_number')}" for r in refs[:1] if r.get("file_name")
                ],
                "sem_fonte": not refs,
                "observacao": "Timeout na orquestração",
            }
        ]
        return {"dialogo": d, "transliteracao": translit, "review": {"erro": "timeout"}}
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("falha_irrecuperavel id=%s err=%r", dialogo.get("id"), e)
        translit = [
            {
                "id": dialogo.get("id"),
                "ator": dialogo.get("ator"),
                "portugues": dialogo.get("portugues"),
                idioma: "[ERRO]",
                f"fontes_{idioma}": [
                    f"{r.get('file_name')}#p{r.get('page_number')}" for r in refs[:1] if r.get("file_name")
                ],
                "sem_fonte": not refs,
                "observacao": f"erro: {type(e).__name__}",
            }
        ]
        return {"dialogo": d, "transliteracao": translit, "review": {"erro": str(e)}}
    finally:
        await runtime.stop_when_idle()


async def run_translation_cycle_groupchat(
    translator: ChatCompletionAgent,
    reviewer: ChatCompletionAgent,
    dialogues: List[Dict[str, Any]],
    idioma: str,
    max_rounds: int = 8,
) -> List[Dict[str, Any]]:
    """Executa ciclos de tradução em groupchat com concorrência opcional e resiliência.

    - Isola exceções por diálogo.
    - Aplica retry / timeout via helpers.
    - Opcionalmente grava progresso parcial.
    """
    cfg = _get_config()
    sem = asyncio.Semaphore(cfg["max_concurrency"])  # controle de explosões de chamadas
    partial_file = cfg.get("progress_file")
    flush_interval = cfg["partial_flush_interval"]
    results_map: Dict[str, Dict[str, Any]] = {}
    fail_streak = 0
    circuit_limit = cfg["circuit_breaker_threshold"]

    async def _wrapped(d):
        nonlocal fail_streak
        async with sem:
            try:
                res = await _process_single_dialog_groupchat(translator, reviewer, d, idioma, max_rounds)
                fail_streak = 0
                return res
            except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as e:
                fail_streak += 1
                logger.error("dialog_global_fail id=%s streak=%d err=%r", d.get("id"), fail_streak, e)
                return {"dialogo": d, "transliteracao": None, "review": {"erro": str(e)}}

    tasks = [asyncio.create_task(_wrapped(d)) for d in dialogues]
    completed = 0
    for coro in asyncio.as_completed(tasks):
        res = await coro
        dlg_id = res.get("dialogo", {}).get("id") or f"idx_{completed}"
        results_map[dlg_id] = res
        completed += 1
        # Persistência incremental
        if partial_file and completed % flush_interval == 0:
            try:
                with open(partial_file, "w", encoding="utf-8") as f:
                    json.dump(list(results_map.values()), f, ensure_ascii=False, indent=2)
            except OSError as e:  # noqa: BLE001
                logger.warning("falha_salvar_progresso err=%r", e)
        if fail_streak >= circuit_limit:
            logger.error(
                "circuit_breaker acionado streak=%d limit=%d restante=%d", fail_streak, circuit_limit, len(tasks) - completed
            )
            break

    # Ordena de volta conforme entrada
    ordered: List[Dict[str, Any]] = []
    for idx, d in enumerate(dialogues):
        id_val = d.get("id")
        if isinstance(id_val, str) and id_val:
            key = id_val
        else:
            key = f"idx_{idx}"
        if key in results_map:
            entry = results_map[key]
        else:
            entry = {"dialogo": d, "transliteracao": None, "review": {"erro": "missing"}}
        ordered.append(entry)
    return ordered


async def run_translation_cycle_manual(
    translator: ChatCompletionAgent,
    reviewer: ChatCompletionAgent,
    dialogues: List[Dict[str, Any]],
    idioma: str,
    max_attempts: int = 3,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for d in dialogues:
        attempt = 0
        feedback = None
        prev = None
        final = None
        while attempt < max_attempts:
            attempt += 1
            dialogo = dict(d)
            if "fala" in dialogo and "portugues" not in dialogo:
                dialogo["portugues"] = dialogo["fala"]
            refs = await fetch_references(translator.kernel, dialogo, idioma)  # type: ignore[attr-defined]
            payload = {
                "json_object": True,
                "dialogo": dialogo,
                "idioma_alvo": idioma,
                "referencias_indexadas": refs,
                "feedback_previo": feedback,
                "transliteracao_anterior": prev,
            }
            hist = ChatHistory()
            hist.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER, content=json.dumps(payload, ensure_ascii=False)
                )
            )
            resp = ""
            try:
                async for m in translator.invoke(messages=hist.messages):  # type: ignore
                    if m.role == AuthorRole.ASSISTANT:
                        resp += m.content.content
            except ContentFilterAIException as cf_err:
                logger.warning(
                    "[manual] Content filter bloqueou tradução id=%s tentativa=%d detalhe=%s", dialogo.get("id"), attempt, cf_err
                )
                trans = [
                    {
                        "id": dialogo.get("id"),
                        "ator": dialogo.get("ator"),
                        "portugues": dialogo.get("portugues"),
                        idioma: "[BLOQUEADO_POR_CONTENT_FILTER]",
                        f"fontes_{idioma}": [
                            f"{r.get('file_name')}#p{r.get('page_number')}" for r in refs[:1] if r.get("file_name")
                        ],
                        "sem_fonte": not refs,
                        "observacao": "Fallback gerado após bloqueio de conteúdo",
                    }
                ]
                final = {"dialogo": d, "transliteracao": trans, "review": {"erro": "content_filter"}}
                break
            try:
                trans = extract_json(resp)
            except (ValueError, json.JSONDecodeError):
                trans = []
            if (
                isinstance(trans, list)
                and not trans
                and attempt == max_attempts
            ):
                trans = [
                    {
                        "id": dialogo.get("id"),
                        "ator": dialogo.get("ator"),
                        "portugues": dialogo.get("portugues"),
                        idioma: "[TRADUÇÃO_PENDENTE]",
                        f"fontes_{idioma}": [
                            f"{r.get('file_name')}#p{r.get('page_number')}"
                            for r in refs[:1]
                            if r.get("file_name")
                        ],
                        "sem_fonte": not refs,
                        "observacao": "fallback",
                    }
                ]
            prev = trans
            r_hist = ChatHistory()
            r_hist.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=json.dumps(
                        {"json_object": True, "transliteracao": trans, "referencias_indexadas": refs},
                        ensure_ascii=False,
                    ),
                )
            )
            r_resp = ""
            async for m in reviewer.invoke(messages=r_hist.messages):  # type: ignore
                if m.role == AuthorRole.ASSISTANT:
                    r_resp += m.content.content
            try:
                review = extract_json(r_resp)
            except (ValueError, json.JSONDecodeError):
                review = {"acao_recomendada": "reexecutar", "raw": r_resp}
            if (
                isinstance(review, dict)
                and review.get("acao_recomendada") == "aprovado"
            ):
                final = {"dialogo": d, "transliteracao": trans, "review": review}
                break
            feedback = review
        if final is None:
            final = {"dialogo": d, "transliteracao": prev, "review": feedback, "status": "forcado"}
        results.append(final)
    return results


async def process_language(
    kernel: sk.Kernel, dialogues: List[Dict[str, Any]], idioma: str, use_groupchat: bool
) -> List[Dict[str, Any]]:
    """Processa um idioma inteiro com construção de agentes e tratamento resiliente.

    Reduz complexidade movendo lógica de retry para funções auxiliares; expõe apenas
    seleção de estratégia (groupchat/manual). Isola construção de agentes e evita
    propagação de exceções não tratadas.
    """
    translator, reviewer = build_translator_and_reviewer(kernel, idioma)
    try:
        fn = run_translation_cycle_groupchat if use_groupchat else run_translation_cycle_manual
        return await fn(translator, reviewer, dialogues, idioma)
    except (OSError, RuntimeError, ValueError, asyncio.TimeoutError, ContentFilterAIException) as e:
        logger.error("process_language erro idioma=%s err=%r", idioma, e)
        # Retorna estrutura fallback para cada diálogo para não quebrar fluxo multi-idioma
        out: List[Dict[str, Any]] = []
        for d in dialogues:
            out.append({
                "dialogo": d,
                "transliteracao": None,
                "review": {"erro": str(e)},
                "status": "falha_global",
            })
        return out


async def main():  # pragma: no cover
    multi_raw = os.getenv("TARGET_IDIOMAS")
    idiomas = (
        [i.strip().lower() for i in multi_raw.split(",") if i.strip()]
        if multi_raw
        else [os.getenv("TARGET_IDIOMA", "matses").lower()]
    )
    use_groupchat = os.getenv("ORCHESTRATION_MODE", "groupchat").lower() == "groupchat"
    kernel = build_kernel()
    dialogues = await generate_dialogues(kernel, idioma=idiomas[0])
    out_dir = Path(__file__).parent / "transliteracoes"
    out_dir.mkdir(exist_ok=True)
    if len(idiomas) == 1:
        idioma = idiomas[0]
        results = await process_language(kernel, dialogues, idioma, use_groupchat)
        for r in results:
            with open(
                out_dir / f"transliteration_{r['dialogo'].get('id','unknown')}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
        with open(out_dir / "transliterations_all.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(
            "Concluído idioma=%s mode=%s",
            idioma,
            "groupchat" if use_groupchat else "manual",
        )
        return
    # Multi-language
    aggregate: Dict[str, Dict[str, Any]] = {
        d.get("id", f"dlg_{i}"): {"dialogo": d, "tentativas": []}
        for i, d in enumerate(dialogues)
    }
    for idioma in idiomas:
        results = await process_language(kernel, dialogues, idioma, use_groupchat)
        for r in results:
            dlg_id = r.get("dialogo", {}).get("id")
            if not dlg_id:
                continue
            aggregate[dlg_id]["tentativas"].append(
                {
                    "idioma": idioma,
                    "transliteracao": r.get("transliteracao"),
                    "review": r.get("review"),
                }
            )
    for dlg_id, obj in aggregate.items():
        with open(out_dir / f"transliteration_{dlg_id}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    with open(out_dir / "transliterations_all.json", "w", encoding="utf-8") as f:
        json.dump(list(aggregate.values()), f, ensure_ascii=False, indent=2)
    logger.info(
        "Concluído multi-línguas=%s mode=%s",
        idiomas,
        "groupchat" if use_groupchat else "manual",
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
