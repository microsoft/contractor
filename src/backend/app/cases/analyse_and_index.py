"""PDF analysis and Azure AI Search indexing module.

This refactored version encapsulates responsibilities into cohesive classes, removes
module-level global state, and applies SOLID principles (notably SRP, DIP and ISP).

High-level flow:
  1. Read and chunk PDF pages ("window" sliding grouping)
  2. Optionally annotate chunks via an orchestrator agent (classification + cleaning)
  3. Generate embeddings for each (cleaned) chunk
  4. Upload documents into an Azure AI Search index (creating it if required)

All services are injected through the main coordinator (`DocumentIndexer`) to allow
easier testability and substitution (dependency inversion).
"""

from __future__ import annotations

import os
import re
import asyncio
import json
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from pypdf import PdfReader

from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
    SearchIndex,
)
from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.models import CodeInterpreterTool

try:  # Lazy optional import for embeddings
    from azure.ai.inference import EmbeddingsClient  # type: ignore
except ImportError:  # pragma: no cover
    EmbeddingsClient = None  # type: ignore

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration container loaded from environment variables.

    Centralizes all parameters needed by the services; simplifies dependency
    injection and test overrides.
    """

    search_service_endpoint: str
    search_service_key: str
    azure_foundry_endpoint: str
    azure_openai_embedding_endpoint: str  # Direct embedding endpoint (if provided)
    azure_openai_resource_url: str  # Base Azure OpenAI resource endpoint (https://<name>.openai.azure.com)
    azure_openai_embedding_deployment: str
    azure_openai_model_name: str
    azure_openai_key: str
    embedding_dimensions: int = 1024
    model_deployment_name: str = "gpt-4.1"
    pdf_index_name: str = "pdf-index-two"
    pdf_folder: str = ""

    @staticmethod
    def from_env() -> "AppConfig":  # pragma: no cover - thin wrapper
        """Factory method to load configuration from environment variables.

        Returns
        -------
        AppConfig
            Fully populated immutable configuration instance.
        """
        cwd_pdf_folder = os.path.join(os.getcwd(), "notebooks", "data", "translation")
        return AppConfig(
            search_service_endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", ""),
            search_service_key=os.getenv("AZURE_SEARCH_SERVICE_KEY", ""),
            azure_foundry_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT", ""),
            azure_openai_embedding_endpoint=(
                # Prefer dedicated embedding endpoint variable if present; fall back to deprecated names.
                os.getenv("AZURE_FOUNDRY_EMBEDDING_URL")
                or os.getenv("AZURE_OPENAI_EMBEDDING_URL", "")
            ),
            azure_openai_resource_url=os.getenv("AZURE_OPENAI_RESOURCE_URL", ""),
            azure_openai_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
            azure_openai_model_name=os.getenv("AZURE_OPENAI_MODEL_NAME", "text-embedding-3-small"),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY", ""),
            embedding_dimensions=int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1024")),
            model_deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
            pdf_index_name=os.getenv("PDF_INDEX_NAME", "pdf-index-two"),
            pdf_folder=os.getenv("PDF_FOLDER", cwd_pdf_folder),
        )


# --------------------------------------------------------------------------------------
# Embedding Service
# --------------------------------------------------------------------------------------

class EmbeddingService:
    """Generates vector embeddings for text using Azure AI Inference EmbeddingsClient.

    This service wraps the (synchronous) embeddings client so code can call
    asynchronously without blocking the event loop. When the underlying client is
    not available, a zero-vector fallback is returned and an error is logged.
    """

    def __init__(self, config: AppConfig):
        """Initialize the embedding service with immutable configuration.

        Parameters
        ----------
        config : AppConfig
            Application configuration providing model name, endpoint and dimensions.
        """
        self._config = config
        self._client = self._build_client()

    def _build_client(self):  # pragma: no cover - minimal logic
        """Instantiate and return an EmbeddingsClient if dependencies exist.

        Returns
        -------
        Optional[EmbeddingsClient]
            A ready embeddings client or None when instantiation fails.
        """
        # Determine best endpoint: explicit embedding endpoint if provided, else foundry endpoint
        endpoint_root = (self._config.azure_openai_embedding_endpoint or self._config.azure_foundry_endpoint).rstrip("/")
        if EmbeddingsClient is None or not endpoint_root:
            return None
        endpoint = endpoint_root + "/models"
        try:
            return EmbeddingsClient(
                endpoint=endpoint,
                credential=SyncDefaultAzureCredential(),
                model=self._config.azure_openai_model_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao inicializar EmbeddingsClient: %s", exc)
            return None

    def is_ready(self) -> bool:
        """Return True if an embeddings client is initialized and usable."""
        return self._client is not None

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for supplied text.

        Parameters
        ----------
        text : str
            Input text to embed.

        Returns
        -------
        List[float]
            Vector representation with configured dimensionality; a zero vector on error.
        """
        if not text:
            return [0.0] * self._config.embedding_dimensions
        if self._client is None:
            # Throttle log spam by only logging first few occurrences
            counter = getattr(self, "_missing_client_log_count", 0)
            if counter < 3:
                logger.error("EmbeddingsClient não inicializado (endpoint ausente?)")
            self._missing_client_log_count = counter + 1  # type: ignore[attr-defined]
            return [0.0] * self._config.embedding_dimensions

        def _call_sync() -> List[float]:
            try:
                client = self._client
                if client is None:  # defensive double-check
                    return [0.0] * self._config.embedding_dimensions
                resp = client.embed(
                    input=[text],
                    dimensions=self._config.embedding_dimensions,
                )
                emb = resp.data[0].embedding  # type: ignore[index]
                if len(emb) != self._config.embedding_dimensions:
                    logger.warning(
                        "Dimensão retornada %s difere da configurada %s.",
                        len(emb), self._config.embedding_dimensions,
                    )
                return list(emb)
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha no embed: %s", exc)
                return [0.0] * self._config.embedding_dimensions

        return await asyncio.to_thread(_call_sync)


# --------------------------------------------------------------------------------------
# PDF Reading & Chunking
# --------------------------------------------------------------------------------------

class PDFReaderService:
    """Encapsulates PDF page extraction logic."""

    def __init__(self) -> None:
        """Create a new PDF reader service (stateless)."""
        # Stateless: no initialization required; method exists for interface symmetry.
        return

    def read_pages(self, pdf_path: str) -> List[str]:
        """Read all pages from a PDF file.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF document.

        Returns
        -------
        List[str]
            A list of raw page texts; missing pages yield empty strings.
        """
        pages: List[str] = []
        with open(pdf_path, "rb") as handler:
            reader = PdfReader(handler)
            for page in reader.pages:
                try:  # type: ignore[attr-defined]
                    pages.append(page.extract_text() or "")  # type: ignore
                except Exception as exc:  # noqa: BLE001
                    logger.error("Erro ao extrair página: %s", exc)
                    pages.append("")
        return pages

    def extract_full_text(self, pdf_path: str) -> str:
        """Concatenate all pages' text into a single string for preview/testing.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        str
            Full text with page separations as newlines.
        """
        return "\n".join(self.read_pages(pdf_path))


class SlidingWindowChunker:
    """Create overlapping window chunks from a sequence of page texts."""

    def __init__(self, window_size: int = 3):
        """Store sliding window size (minimum 1).

        Parameters
        ----------
        window_size : int, optional
            Number of pages per chunk window, by default 3.
        """
        self.window_size = max(window_size, 1)

    def build_chunks(self, pages: Sequence[str]) -> List[Dict[str, Any]]:
        """Produce chunk metadata dictionaries with joined page text.

        Parameters
        ----------
        pages : Sequence[str]
            Ordered sequence of individual page texts.

        Returns
        -------
        List[Dict[str, Any]]
            Each item holds chunk_text, page_start, page_end and page_range_str.
        """
        n = len(pages)
        if n == 0:
            return []
        result: List[Dict[str, Any]] = []
        w = self.window_size
        for start in range(0, n - w + 1):
            end = start + w
            group = pages[start:end]
            result.append(
                {
                    "chunk_text": "\n".join(group),
                    "page_start": start + 1,
                    "page_end": end,
                    "page_range_str": f"{start + 1}-{end}",
                }
            )
        return result

    @staticmethod
    def prepare_batch(file_name: str, page_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign stable identifiers and static metadata to raw window chunks.

        Parameters
        ----------
        file_name : str
            Original PDF filename.
        page_chunks : List[Dict[str, Any]]
            Raw chunk info produced by `build_chunks`.

        Returns
        -------
        List[Dict[str, Any]]
            Structured chunk entries ready for annotation or indexing.
        """
        safe_stem = re.sub(r"[^A-Za-z0-9_\-=]", "_", os.path.splitext(file_name)[0])
        batch: List[Dict[str, Any]] = []
        for idx, info in enumerate(page_chunks, start=1):
            batch.append(
                {
                    "chunk_id": f"{safe_stem}-{idx:04d}",
                    "file_name": file_name,
                    "page_number": info["page_range_str"],
                    "chunk": info["chunk_text"],
                }
            )
        return batch


# --------------------------------------------------------------------------------------
# Orchestrator Agent Service
# --------------------------------------------------------------------------------------

class OrchestratorAgentService:
    """Creates and invokes an Azure AI Agent to annotate (clean/classify) chunk batches."""

    def __init__(self, config: AppConfig):
        """Initialize the orchestrator agent service.

        Parameters
        ----------
        config : AppConfig
            Configuration providing endpoint and model deployment info.
        """
        self._config = config
        self._client: Optional[AIProjectClient] = None
        self._agent_id: Optional[str] = None

    async def _ensure_client(self) -> AIProjectClient:
        """Lazily create and return the underlying AIProjectClient instance.

        Returns
        -------
        AIProjectClient
            Active project client for agent operations.
        """
        if self._client is None:
            self._client = AIProjectClient(
                endpoint=self._config.azure_foundry_endpoint,
                credential=DefaultAzureCredential(),
            )
        return self._client

    async def ensure_agent(self) -> Optional[str]:
        """Create the orchestrator agent once and return its identifier.

        Returns
        -------
        Optional[str]
            Agent ID if creation succeeded, otherwise None.
        """
        if self._agent_id:
            return self._agent_id
        try:
            client = await self._ensure_client()
            code_tool = CodeInterpreterTool()
            instructions = (
                "Você orquestra a normalização e classificação de chunks de PDF. Receberá um JSON com 'chunks'. Para cada item: "
                "1) Limpe o texto removendo cabeçalhos/rodapés óbvios repetidos, espaços duplos e quebras excessivas; 2) Gere 'language_concept' dentre: estrutura gramatical, regra morfológica, regra morfosintática, fonologia, ortografia, semântica, pragmática, léxico, outro; 3) Gere 'topic' no formato '<macrotema> da língua <nome ou família>' ou 'tópico não identificado'; 4) NÃO invente conteúdo. Responda somente com JSON contendo lista em 'chunks' mantendo chunk_id original e adicionando 'chunk_clean'."
            )
            agent = await client.agents.create_agent(  # type: ignore[attr-defined]
                model=self._config.model_deployment_name,
                name="pdf-orchestrator",
                instructions=instructions,
                tools=code_tool.definitions,  # type: ignore[attr-defined]
            )
            self._agent_id = agent.id  # type: ignore[attr-defined]
            logger.info("Agente orquestrador criado: %s", self._agent_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Erro ao criar agente orquestrador: %s", exc)
        return self._agent_id

    async def annotate_batch(self, chunk_batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Send a batch of chunk dictionaries for cleaning/classification.

        Parameters
        ----------
        chunk_batch : List[Dict[str, Any]]
            List of raw chunk dictionaries produced by `SlidingWindowChunker.prepare_batch`.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping from chunk_id to annotated data (language_concept/topic/chunk_clean).
        """
        annotated: Dict[str, Dict[str, Any]] = {}
        agent_id = await self.ensure_agent()
        if not agent_id or not chunk_batch:
            return annotated
        try:
            client = await self._ensure_client()
            thread = await getattr(client.agents, "create_thread")()  # type: ignore[attr-defined]
            payload = json.dumps({"chunks": chunk_batch}, ensure_ascii=False)
            await getattr(client.agents, "create_message")(thread_id=thread.id, role="user", content=payload)  # type: ignore[attr-defined]
            run = await getattr(client.agents, "create_and_process_run")(thread_id=thread.id, agent_id=agent_id)  # type: ignore[attr-defined]
            if getattr(run, "last_error", None):  # type: ignore[arg-type]
                logger.warning("Run reportou erro: %s", run.last_error)  # type: ignore[attr-defined]
            msg_list = await getattr(client.agents, "list_messages")(thread_id=thread.id)  # type: ignore[attr-defined]
            for m in reversed(msg_list.data):  # type: ignore[attr-defined]
                if getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
                    for c in m.content:  # type: ignore
                        if hasattr(c, "text") and getattr(c.text, "value", None):  # type: ignore[attr-defined]
                            candidate = c.text.value  # type: ignore[attr-defined]
                            try:
                                parsed = json.loads(candidate)
                                out_list = parsed.get("chunks") if isinstance(parsed, dict) else None
                                if isinstance(out_list, list):
                                    for item in out_list:
                                        cid = item.get("chunk_id")
                                        if cid:
                                            annotated[cid] = item
                                    return annotated
                            except json.JSONDecodeError:  # pragma: no cover - tolerant parsing
                                continue
            return annotated
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao anotar lote via agente: %s", exc)
            return annotated


# --------------------------------------------------------------------------------------
# Search Index Management
# --------------------------------------------------------------------------------------

class SearchIndexService:
    """Handles creation of Azure AI Search indexes and document uploads."""

    def __init__(self, config: AppConfig):
        """Store configuration for subsequent index/search client creation.

        Parameters
        ----------
        config : AppConfig
            Application configuration with search service credentials and embedding dims.
        """
        self._config = config

    def _index_client(self) -> SearchIndexClient:
        """Internal helper to build a `SearchIndexClient`.

        Returns
        -------
        SearchIndexClient
            Client instance for index administration operations.
        """
        return SearchIndexClient(
            endpoint=self._config.search_service_endpoint,
            credential=AzureKeyCredential(self._config.search_service_key),
        )

    def _search_client(self, index_name: str) -> SearchClient:
        """Internal helper to build a `SearchClient` for data operations.

        Parameters
        ----------
        index_name : str
            Target index name.

        Returns
        -------
        SearchClient
            Data plane client bound to the index.
        """
        return SearchClient(
            endpoint=self._config.search_service_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self._config.search_service_key),
        )

    def create_or_update_index(self, index_name: str) -> None:
        """Create (or update) a vector-enabled search index with semantic config.

        Parameters
        ----------
        index_name : str
            Desired index name.
        """
        # Validate resource URL for Azure OpenAI vectorizer (must end with .openai.azure.com)
        resource_url = self._config.azure_openai_resource_url or self._config.azure_openai_embedding_endpoint
        if resource_url and not resource_url.endswith(".openai.azure.com"):
            msg = (
                "resource_url inválido para AzureOpenAIVectorizer: '%s'. Forneça o endpoint base do recurso Azure OpenAI, ex: https://meu-recurso.openai.azure.com"
                % resource_url
            )
            logger.error(msg)
            raise ValueError(msg)

        fields = [
            SearchField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True, facetable=True),
            SearchField(name="file_name", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchField(name="topic", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchField(name="language_concept", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="chunk", type=SearchFieldDataType.String, sortable=False, filterable=False, facetable=False),
            SearchField(name="page_number", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=self._config.embedding_dimensions,
                vector_search_profile_name="underlyingHnswProfile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="underlyingHnsw")],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="myOpenAI",
                    kind="azureOpenAI",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=resource_url,
                        deployment_name=self._config.azure_openai_embedding_deployment,
                        model_name=self._config.azure_openai_model_name,
                        api_key=self._config.azure_openai_key,
                    ),
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name="underlyingHnswProfile",
                    algorithm_configuration_name="underlyingHnsw",
                    vectorizer_name="myOpenAI",
                )
            ],
        )

        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="chunk")],
                title_field=SemanticField(field_name="file_name"),
            ),
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
        result = self._index_client().create_or_update_index(index)
        logger.info("Index '%s' created/updated", result.name)

    def upload_documents(self, index_name: str, documents: List[Dict[str, Any]]) -> List[bool]:
        """Upload a batch of documents to the specified index.

        Parameters
        ----------
        index_name : str
            Target index name.
        documents : List[Dict[str, Any]]
            Prepared documents each containing vector + metadata fields.

        Returns
        -------
        List[bool]
            For each document an indicator of upload success.
        """
        client = self._search_client(index_name)
        result = client.upload_documents(documents)
        # Upload batch returns a list of IndexingResult; guard if SDK changes
        return [getattr(r, "succeeded", False) for r in result]


# --------------------------------------------------------------------------------------
# Document Indexer (Coordinator)
# --------------------------------------------------------------------------------------

class DocumentIndexer:
    """Coordinates PDF ingestion, annotation, embedding generation and search upload."""

    def __init__(
        self,
        config: AppConfig,
        pdf_reader: PDFReaderService,
        chunker: SlidingWindowChunker,
        embedding_service: EmbeddingService,
        search_service: SearchIndexService,
        agent_service: Optional[OrchestratorAgentService] = None,
    ):
        """Aggregate collaborating services required for indexing PDFs.

        Parameters
        ----------
        config : AppConfig
            Global immutable configuration.
        pdf_reader : PDFReaderService
            Service for extracting page texts.
        chunker : SlidingWindowChunker
            Strategy for chunking pages into overlapping groups.
        embedding_service : EmbeddingService
            Service generating embeddings for cleaned chunks.
        search_service : SearchIndexService
            Service responsible for index creation and document upload.
        agent_service : Optional[OrchestratorAgentService], optional
            Optional service to annotate and clean chunks, by default None.
        """
        self._config = config
        self._pdf_reader = pdf_reader
        self._chunker = chunker
        self._embedding = embedding_service
        self._search = search_service
        self._agent = agent_service

    async def index_pdf(self, pdf_path: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Process a single PDF file and index its chunks.

        Steps: read pages -> window chunks -> (optional) annotate -> embed -> upload.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.
        index_name : Optional[str]
            Target index name (defaults to configured pdf_index_name).

        Returns
        -------
        Dict[str, Any]
            Summary of operation (status, counts, timings, success flags).
        """
        index_name = index_name or self._config.pdf_index_name
        start_time = time.time()
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
        file_name = os.path.basename(pdf_path)
        pages = self._pdf_reader.read_pages(pdf_path)
        page_chunks = self._chunker.build_chunks(pages)
        if not page_chunks:
            return {"status": "empty", "file": file_name, "chunks": 0}
        chunk_batch = self._chunker.prepare_batch(file_name, page_chunks)
        annotated_map: Dict[str, Dict[str, Any]] = {}
        if self._agent:
            annotated_map = await self._agent.annotate_batch(chunk_batch)
        documents: List[Dict[str, Any]] = []
        for original in chunk_batch:
            ann = annotated_map.get(original["chunk_id"], {})
            clean_text = ann.get("chunk_clean", original["chunk"])
            vector = await self._embedding.embed(clean_text)
            documents.append(
                {
                    "chunk_id": original["chunk_id"],
                    "file_name": original["file_name"],
                    "page_number": original["page_number"],
                    "chunk": clean_text,
                    "language_concept": ann.get("language_concept", "desconhecido"),
                    "topic": ann.get("topic", "tópico não identificado"),
                    "vector": vector,
                }
            )
        try:
            upload_result = self._search.upload_documents(index_name, documents)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao indexar documentos: %s", exc)
            return {"status": "error", "error": str(exc)}
        duration = time.time() - start_time
        return {
            "status": "ok",
            "file": file_name,
            "chunks": len(documents),
            "index_name": index_name,
            "duration_sec": round(duration, 2),
            "upload_result": upload_result,
        }

    async def index_folder(self, folder: str, pattern: str = "*.pdf") -> List[Dict[str, Any]]:
        """Index all PDF documents in a folder using a glob pattern.

        Parameters
        ----------
        folder : str
            Target directory path.
        pattern : str, optional
            Glob expression for file selection, by default "*.pdf".

        Returns
        -------
        List[Dict[str, Any]]
            List of per-file result dictionaries.
        """
        import glob

        pdf_files = sorted(glob.glob(os.path.join(folder, pattern)))
        results: List[Dict[str, Any]] = []
        for path in pdf_files:
            try:
                logger.info("Indexando: %s", os.path.basename(path))
                res = await self.index_pdf(path)
                results.append(res)
            except Exception as exc:  # noqa: BLE001
                logger.error("Erro ao indexar %s: %s", path, exc)
        return results


# --------------------------------------------------------------------------------------
# Script Entrypoint
# --------------------------------------------------------------------------------------

async def _main() -> None:  # pragma: no cover
    """Executable entrypoint used when running the module as a script.

    Creates required services, ensures index existence, processes PDFs and performs
    a sample orchestrator query (if agent enabled).
    """
    config = AppConfig.from_env()
    pdf_reader = PDFReaderService()
    chunker = SlidingWindowChunker(window_size=3)
    embedding_service = EmbeddingService(config)
    search_service = SearchIndexService(config)
    agent_service = OrchestratorAgentService(config)

    search_service.create_or_update_index(config.pdf_index_name)
    indexer = DocumentIndexer(
        config=config,
        pdf_reader=pdf_reader,
        chunker=chunker,
        embedding_service=embedding_service,
        search_service=search_service,
        agent_service=agent_service,
    )

    logger.info("Usando pasta de PDFs: %s", config.pdf_folder)
    logger.info("Índice alvo: %s", config.pdf_index_name)

    # Preview first file (if any) for extraction validation
    import glob

    pdf_files = sorted(glob.glob(os.path.join(config.pdf_folder, "*.pdf")))
    if not pdf_files:
        logger.warning("Nenhum PDF encontrado na pasta '%s'", config.pdf_folder)
        return
    first_file = pdf_files[0]
    preview_text = pdf_reader.extract_full_text(first_file)[:500].replace("\n", " ")
    logger.info(
        "Preview de extração (%s) primeiros 500 chars: %s", os.path.basename(first_file), preview_text
    )

    results = await indexer.index_folder(config.pdf_folder)
    ok = sum(1 for r in results if r.get("status") == "ok")
    total_chunks = sum(r.get("chunks", 0) for r in results if r.get("status") == "ok")
    logger.info("PDFs processados com sucesso: %s/%s | Chunks criados: %s", ok, len(results), total_chunks)

    # Optional demonstration query using agent
    agent_id = await agent_service.ensure_agent()
    if agent_id:
        try:
            client = await agent_service._ensure_client()  # Access internal for demo only
            thread = await getattr(client.agents, "create_thread")()  # type: ignore[attr-defined]
            question = "Cuales cursos tienes en cibersecurity?"
            await getattr(client.agents, "create_message")(thread_id=thread.id, role="user", content=question)  # type: ignore[attr-defined]
            run = await getattr(client.agents, "create_and_process_run")(thread_id=thread.id, agent_id=agent_id)  # type: ignore[attr-defined]
            logger.info("Agent run status: %s", getattr(run, "status", "unknown"))
            msg_list = await getattr(client.agents, "list_messages")(thread_id=thread.id)  # type: ignore[attr-defined]
            for m in reversed(msg_list.data):  # type: ignore[attr-defined]
                if getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
                    for c in m.content:  # type: ignore
                        if hasattr(c, "text") and getattr(c.text, "value", None):  # type: ignore[attr-defined]
                            logger.info("Assistant: %s", c.text.value)  # type: ignore[attr-defined]
                    break
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao executar consulta demonstrativa ao agente: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
