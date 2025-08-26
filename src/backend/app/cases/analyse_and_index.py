"""Ingestão e indexação de PDFs usando Semantic Kernel para anotação de chunks.

Fluxo:
  1. Ler páginas do PDF
  2. Agrupar em janelas de 3 páginas (sem overlap)
  3. Agente (Semantic Kernel) limpa e classifica cada chunk retornando JSON
  4. Gerar embedding via Azure AI Inference
  5. Indexar no Azure AI Search (texto + vetor)
"""

from __future__ import annotations

import os
import re
import json
import glob
import uuid
import time
import asyncio
import logging
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from pypdf import PdfReader

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent  # pylint: disable=no-name-in-module
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.ai.inference import EmbeddingsClient
from azure.core.exceptions import HttpResponseError

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------------------
# Util
# --------------------------------------------------------------------------------------

def _short_id() -> str:
    return uuid.uuid4().hex[:8]


@contextmanager
def _timed(stage: str, collector: Dict[str, float]):
    t0 = time.time()
    try:
        yield
    finally:
        collector[stage] = collector.get(stage, 0.0) + (time.time() - t0)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    search_service_endpoint: str
    search_service_key: str
    azure_foundry_key: str
    azure_foundry_url: str
    embedding_model: str
    chat_deployment: str
    pdf_index_name: str = "pdf-index"
    pdf_folder: str = ""
    embedding_url: str = ""
    embedding_dimensions: int = 1024

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig(
            search_service_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            search_service_key=os.getenv("AZURE_SEARCH_API_KEY", ""),
            azure_foundry_key=os.getenv("AZURE_FOUNDRY_KEY", ""),
            azure_foundry_url=os.getenv("AZURE_FOUNDRY_URL", ""),
            embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            pdf_index_name=os.getenv("PDF_INDEX_NAME", "pdf-index"),
            pdf_folder=os.getenv("PDF_FOLDER", "notebooks/data/translation"),
            embedding_dimensions=int(os.getenv("EMBED_DIM", "1024")),
            embedding_url=os.getenv("AZURE_FOUNDRY_EMBEDDING_URL", ""),
        )


# --------------------------------------------------------------------------------------
# Embedding Service
# --------------------------------------------------------------------------------------

class EmbeddingService:
    def __init__(self, config: AppConfig):
        self._config = config
        self._client: Optional[EmbeddingsClient] = None
        try:
            if config.embedding_url and config.azure_foundry_key:
                self._client = EmbeddingsClient(
                    endpoint=config.embedding_url,
                    credential=AzureKeyCredential(config.azure_foundry_key),
                )
        except (HttpResponseError, ValueError, OSError) as exc:  # pragma: no cover
            logger.warning("Falha inicializando EmbeddingsClient: %s", exc)
            self._client = None

    async def embed(self, text: str) -> List[float]:
        if not text or not self._client:
            return [0.0] * self._config.embedding_dimensions

        def _call_sync() -> List[float]:
            try:
                resp = self._client.embed(model=self._config.embedding_model, input=[text])  # type: ignore[attr-defined]
                raw_vec = list(resp.data[0].embedding)  # type: ignore[index]
                vec: List[float] = [float(v) for v in raw_vec]
                dims = self._config.embedding_dimensions
                if len(vec) < dims:
                    vec.extend([0.0] * (dims - len(vec)))
                elif len(vec) > dims:
                    vec = vec[:dims]
                return vec
            except (HttpResponseError, ValueError, OSError) as exc:  # pragma: no cover
                logger.error("Erro gerando embedding: %s", exc)
                return [0.0] * self._config.embedding_dimensions

        return await asyncio.to_thread(_call_sync)


# --------------------------------------------------------------------------------------
# PDF Reading & Chunking
# --------------------------------------------------------------------------------------

class PDFReaderService:
    def read_pages(self, pdf_path: str) -> List[str]:
        pages: List[str] = []
        with open(pdf_path, "rb") as handler:
            reader = PdfReader(handler)
            if reader.pages:
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except (ValueError, RuntimeError, OSError) as exc:  # pragma: no cover
                        logger.warning("Falha extraindo página PDF %s: %s", pdf_path, exc)
                        pages.append("")
        return pages

    def extract_full_text(self, pdf_path: str) -> str:
        return "\n".join(self.read_pages(pdf_path))


class SlidingWindowChunker:
    def __init__(self, window_size: int = 3, overlapping: bool = False):
        self.window_size = max(1, window_size)
        self.overlapping = overlapping

    def build_chunks(self, pages: Sequence[str]) -> List[Dict[str, Any]]:
        n = len(pages)
        if n == 0:
            return []
        res: List[Dict[str, Any]] = []
        w = self.window_size
        if self.overlapping:
            for i in range(0, n - w + 1):
                group = pages[i : i + w]
                res.append({
                    "chunk_id": _short_id(),
                    "start_page": i + 1,
                    "end_page": i + w,
                    "raw": "\n".join(group),
                })
        else:
            page_index = 0
            while page_index < n:
                group = pages[page_index : page_index + w]
                start = page_index + 1
                end = start + len(group) - 1
                res.append({
                    "chunk_id": _short_id(),
                    "start_page": start,
                    "end_page": end,
                    "raw": "\n".join(group),
                })
                page_index += w
        return res

    @staticmethod
    def prepare_batch(file_name: str, page_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for c in page_chunks:
            c["file_name"] = file_name
            c["page_number"] = f"{c['start_page']}-{c['end_page']}"
        return page_chunks


# --------------------------------------------------------------------------------------
# Annotator (Semantic Kernel Agent)
# --------------------------------------------------------------------------------------

ANNOTATION_SYSTEM_PROMPT = (
    "Você receberá um trecho de até 3 páginas concatenadas extraído de um PDF linguístico.\n"
    "Tarefas:\n"
    "1. Limpe o texto removendo cabeçalhos/rodapés repetidos, sequências /uniXXXX, tokens /palavra, múltiplos espaços.\n"
    "2. NÃO invente conteúdo.\n"
    "3. Classifique em 'language_concept' (enum): fonologia, morfologia, sintaxe, semantica, lexico, ortografia, pragmatica, gramatica, outro.\n"
    "4. Determine 'topic' curto (até 6 palavras).\n"
    "5. Responda JSON puro: {chunk_clean, language_concept, topic}.\n"
    "Se nada puder ser classificado use language_concept=outro e topic='indefinido'."
)


class ChunkAnnotator:
    def __init__(self, config: AppConfig):
        self._kernel = sk.Kernel()
        if config.azure_foundry_url and config.azure_foundry_key:
            service = AzureChatCompletion(
                service_id="annotator",
                api_key=config.azure_foundry_key,
                deployment_name=config.chat_deployment,
                endpoint=config.azure_foundry_url,
            )
            self._kernel.add_service(service)
        settings = self._kernel.get_prompt_execution_settings_from_service_id("annotator")
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        self._agent = ChatCompletionAgent(
            kernel=self._kernel,
            name="PDFAnnotatorAgent",
            instructions=ANNOTATION_SYSTEM_PROMPT,
            description="Limpa e classifica chunks",
            arguments=KernelArguments(settings=settings),
        )

    async def annotate(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user_prompt = json.dumps({"chunk": chunk.get("raw", "")}, ensure_ascii=False)
        out_text = ""
        try:
            async for msg in self._agent.invoke(messages=[sk.contents.ChatMessageContent(role=sk.contents.AuthorRole.USER, content=user_prompt)]):  # type: ignore[attr-defined]
                if msg.content.content:
                    out_text += msg.content.content
        except (ValueError, RuntimeError, OSError) as exc:  # pragma: no cover
            logger.error("Erro invocando agente: %s", exc)
            return None
        cleaned = out_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        try:
            data = json.loads(cleaned)
            if not all(k in data for k in ("chunk_clean", "language_concept", "topic")):
                raise ValueError("Campos obrigatórios ausentes")
            data["chunk_id"] = chunk["chunk_id"]
            data["file_name"] = chunk.get("file_name")
            data["page_number"] = chunk.get("page_number")
            return data
        except (ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
            logger.warning("Falha parse anotação %s: %s", chunk.get("chunk_id"), exc)
            return None


# --------------------------------------------------------------------------------------
# Azure Search
# --------------------------------------------------------------------------------------

class SearchIndexService:
    def __init__(self, config: AppConfig):
        self._config = config
        self._cred = AzureKeyCredential(config.search_service_key) if config.search_service_key else None

    def _index_client(self) -> SearchIndexClient:
        if not self._cred:
            raise RuntimeError("Credencial de busca ausente")
        return SearchIndexClient(endpoint=self._config.search_service_endpoint, credential=self._cred)

    def _search_client(self, index_name: str) -> SearchClient:
        if not self._cred:
            raise RuntimeError("Credencial de busca ausente")
        return SearchClient(endpoint=self._config.search_service_endpoint, index_name=index_name, credential=self._cred)

    def create_or_update_index(self, index_name: str) -> None:
        client = self._index_client()
        fields = [
            SearchField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SearchField(name="file_name", type=SearchFieldDataType.String, filterable=True, searchable=True, sortable=True, facetable=True, retrievable=True),
            SearchField(name="page_number", type=SearchFieldDataType.String, filterable=True, sortable=True, retrievable=True, facetable=False),
            SearchField(name="chunk", type=SearchFieldDataType.String, searchable=True, retrievable=True, filterable=False, sortable=False, facetable=False),
            SearchField(name="language_concept", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True, retrievable=True),
            SearchField(name="topic", type=SearchFieldDataType.String, searchable=True, filterable=True, sortable=True, facetable=True, retrievable=True),
            # Campo vetorial: Azure Search exige searchable=True para campos de vetor
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                retrievable=True,
                dimensions=self._config.embedding_dimensions,
                vector_search_dimensions=self._config.embedding_dimensions,
                vector_search_profile_name="underlyingHnswProfile",
            ),
        ]
        # Configuração de busca vetorial (perfil HNSW)
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="underlyingHnsw")],
            profiles=[VectorSearchProfile(name="underlyingHnswProfile", algorithm_configuration_name="underlyingHnsw")],
        )
        semantic = SemanticSearch(configurations=[
            SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="chunk")], title_field=None, keywords_fields=[]),
            )
        ])
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic)
        try:
            client.create_or_update_index(index)
            logger.info("Índice '%s' criado/atualizado", index_name)
        except (ValueError, RuntimeError, OSError, HttpResponseError) as exc:  # pragma: no cover
            logger.error("Falha criando índice %s: %s", index_name, exc)

    def upload_documents(self, index_name: str, documents: List[Dict[str, Any]]) -> List[bool]:
        if not documents:
            return []
        client = self._search_client(index_name)
        try:
            resp = client.upload_documents(documents)
            results: List[bool] = []
            for r in resp:
                succeeded = getattr(r, "succeeded", False)
                results.append(bool(succeeded))
            return results
        except (ValueError, RuntimeError, OSError, HttpResponseError) as exc:  # pragma: no cover
            logger.error("Falha upload docs: %s", exc)
            return [False] * len(documents)


# --------------------------------------------------------------------------------------
# Document Indexer
# --------------------------------------------------------------------------------------

class DocumentIndexer:
    def __init__(
        self,
        config: AppConfig,
        pdf_reader: PDFReaderService,
        chunker: SlidingWindowChunker,
        embedding_service: EmbeddingService,
        search_service: SearchIndexService,
        annotator: ChunkAnnotator,
    ):
        self._config = config
        self._pdf_reader = pdf_reader
        self._chunker = chunker
        self._embedding = embedding_service
        self._search = search_service
        self._annotator = annotator

    async def _annotate_chunks(self, chunks: List[Dict[str, Any]], concurrency: int = 5) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)

        async def worker(i: int, ch: Dict[str, Any]):
            async with sem:
                ann = await self._annotator.annotate(ch)
                results[i] = ann

        await asyncio.gather(*(worker(i, c) for i, c in enumerate(chunks)))
        return [r for r in results if r]

    async def _embed_chunks(self, annotations: List[Dict[str, Any]], concurrency: int = 8) -> None:
        sem = asyncio.Semaphore(concurrency)

        async def worker(ann: Dict[str, Any]):
            async with sem:
                vec = await self._embedding.embed(ann.get("chunk_clean", ""))
                ann["vector"] = vec

        await asyncio.gather(*(worker(a) for a in annotations))

    def _prepare_docs(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for a in annotations:
            docs.append({
                "chunk_id": a["chunk_id"],
                "file_name": a.get("file_name"),
                "page_number": a.get("page_number"),
                "chunk": a.get("chunk_clean"),
                "language_concept": a.get("language_concept"),
                "topic": a.get("topic"),
                "vector": a.get("vector", []),
            })
        return docs

    async def index_pdf(self, pdf_path: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        index_name = index_name or self._config.pdf_index_name
        pages = self._pdf_reader.read_pages(pdf_path)
        file_name = os.path.basename(pdf_path)
        raw_chunks = self._chunker.prepare_batch(file_name, self._chunker.build_chunks(pages))
        ann = await self._annotate_chunks(raw_chunks)
        await self._embed_chunks(ann)
        docs = self._prepare_docs(ann)
        flags = self._search.upload_documents(index_name, docs)
        succeeded = sum(1 for f in flags if f)
        return {
            "file": file_name,
            "status": "ok" if succeeded == len(flags) else "partial" if succeeded > 0 else "fail",
            "chunks": len(raw_chunks),
            "annotated": len(ann),
            "indexed": succeeded,
        }

    async def index_folder(self, folder: str, pattern: str = "*.pdf", max_pdf_concurrency: int = 2) -> List[Dict[str, Any]]:
        pdf_files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not pdf_files:
            logger.warning("Nenhum PDF encontrado em %s", folder)
            return []
        sem = asyncio.Semaphore(max_pdf_concurrency)
        results: List[Dict[str, Any]] = []

        async def worker(path: str):
            async with sem:
                try:
                    res = await self.index_pdf(path)
                except (ValueError, RuntimeError, OSError) as exc:  # pragma: no cover
                    res = {"file": os.path.basename(path), "status": "error", "error": str(exc)}
                results.append(res)

        await asyncio.gather(*(worker(p) for p in pdf_files))
        return results


# --------------------------------------------------------------------------------------
# Script Entrypoint
# --------------------------------------------------------------------------------------

async def _main() -> None:  # pragma: no cover
    config = AppConfig.from_env()
    if not config.pdf_folder:
        raise SystemExit("Defina PDF_FOLDER no .env para executar.")
    pdf_reader = PDFReaderService()
    chunker = SlidingWindowChunker(window_size=3, overlapping=False)
    embedding = EmbeddingService(config)
    search = SearchIndexService(config)
    annotator = ChunkAnnotator(config)
    search.create_or_update_index(config.pdf_index_name)
    indexer = DocumentIndexer(config, pdf_reader, chunker, embedding, search, annotator)
    results = await indexer.index_folder(config.pdf_folder)
    logger.info("Resumo: %s", results)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
