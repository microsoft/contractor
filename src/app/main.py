"""
The configuration for the mailing service.
"""

import logging
import os

from dotenv import find_dotenv, load_dotenv

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __app__, __version__
from app.agents import ToolerOrchestrator
from app.cosmos_crud import CosmosCRUD
from app.schemas import *


load_dotenv(find_dotenv())

BLOB_CONN = os.getenv("BLOB_CONNECTION_STRING", "")
MODEL_URL: str = os.environ.get("GPT4_URL", "")
MODEL_KEY: str = os.environ.get("GPT4_KEY", "")
MONITOR: str = os.environ.get("AZ_CONNECTION_LOG", "")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "")

COSMOS_ASSEMBLY_TABLE = os.getenv("COSMOS_ASSEMBLY_TABLE", "assembly")
COSMOS_TOOL_TABLE = os.getenv("COSMOS_TOOL_TABLE", "tool")
COSMOS_TEXTDATA_TABLE = os.getenv("COSMOS_TEXTDATA_TABLE", "textdata")
COSMOS_IMAGEDATA_TABLE = os.getenv("COSMOS_IMAGEDATA_TABLE", "imagedata")
COSMOS_AUDIODATA_TABLE = os.getenv("COSMOS_AUDIODATA_TABLE", "audiodata")
COSMOS_VIDEODATA_TABLE = os.getenv("COSMOS_VIDEODATA_TABLE", "videodata")


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


tags_metadata: list[dict] = [
    {
        "name": "Inference",
        "description": """
        Use agents to process multi-modal data for RAG.
        """,
    },
    {
        "name": "CRUD - Assemblies",
        "description": "CRUD endpoints for Assembly model.",
    },
    {
        "name": "CRUD - Tools",
        "description": "CRUD endpoints for Tool model.",
    },
    {
        "name": "CRUD - TextData",
        "description": "CRUD endpoints for TextData model.",
    },
    {
        "name": "CRUD - ImageData",
        "description": "CRUD endpoints for ImageData model.",
    },
    {
        "name": "CRUD - AudioData",
        "description": "CRUD endpoints for AudioData model.",
    },
    {
        "name": "CRUD - VideoData",
        "description": "CRUD endpoints for VideoData model.",
    },
]

description: str = """
    .
"""


app: FastAPI = FastAPI(
    title=__app__,
    version=__version__,
    description=description,
    openapi_tags=tags_metadata,
    openapi_url="/api/v1/openapi.json",
    responses=RESPONSES,  # type: ignore
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    validation_exception_handler Exception handler for validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Validation Error",
        title="Your request parameters didn't validate.",
        detail={"invalid-params": list(exc.errors())},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.exception_handler(ResponseValidationError)
async def response_exception_handler(
    request: Request, exc: ResponseValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    response_exception_handler Exception handler for response validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Response Error",
        title="Found Errors on processing your requests.",
        detail={"invalid-params": list(exc.errors())},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.post("/evaluate", tags=["Agents", "Assembly"])
async def evaluate_judgment(job: JobResponse) -> JSONResponse:
    """
    Endpoint that evaluates a prompt using a Agent Assembly.
    """
    try:
        final_verdict = await ToolerOrchestrator().run_interaction(assembly=job.assembly_id, prompt=job.prompt)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    response_body = SuccessMessage(
        title="Evaluation Complete",
        message="Judging completed successfully.",
        content={"assembly_id": job.assembly_id, "result": final_verdict},
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/assemblies", tags=["CRUD - Assemblies"])
async def list_assemblies_endpoint() -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    items = await crud.list_items()
    response_body = SuccessMessage(
        title=f"{len(items) if items else 0} Assemblies Retrieved",
        message="Successfully retrieved assembly data.",
        content=items,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/assemblies", tags=["CRUD - Assemblies"])
async def create_assembly(assembly: Assembly) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    created = await crud.create_item(assembly.model_dump())
    response_body = SuccessMessage(
        title="Assembly Created",
        message="Assembly created successfully.",
        content=created,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/assemblies/{assembly_id}", tags=["CRUD - Assemblies"])
async def update_assembly(assembly_id: str, assembly: Assembly) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    try:
        existing = await crud.read_item(assembly_id)
    except Exception as exc:
        logger.error("Error reading assembly: %s", exc)
        raise HTTPException(status_code=404, detail="Assembly not found.") from exc
    updated = {**existing, **assembly.model_dump()}
    await crud.update_item(assembly_id, updated)
    response_body = SuccessMessage(
        title="Assembly Updated",
        message="Assembly updated successfully.",
        content=updated,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/assemblies/{assembly_id}", tags=["CRUD - Assemblies"])
async def delete_assembly(assembly_id: str) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    try:
        await crud.delete_item(assembly_id)
    except Exception as exc:
        logger.error("Error deleting assembly: %s", exc)
        raise HTTPException(status_code=404, detail="Assembly not found.") from exc
    response_body = SuccessMessage(
        title="Assembly Deleted",
        message="Assembly deleted successfully.",
        content={"assembly_id": assembly_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/tools", tags=["CRUD - Tools"])
async def list_tools_endpoint() -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TOOL_TABLE")
    items = await crud.list_items()
    response_body = SuccessMessage(
        title=f"{len(items) if items else 0} Tools Retrieved",
        message="Successfully retrieved tool data.",
        content=items,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/tools", tags=["CRUD - Tools"])
async def create_tool(tool: Tool) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TOOL_TABLE")
    created = await crud.create_item(tool.model_dump())
    response_body = SuccessMessage(
        title="Tool Created",
        message="Tool created successfully.",
        content=created,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/tools/{tool_id}", tags=["CRUD - Tools"])
async def update_tool(tool_id: str, tool: Tool) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TOOL_TABLE")
    try:
        existing = await crud.read_item(tool_id)
    except Exception as exc:
        logger.error("Error reading tool: %s", exc)
        raise HTTPException(status_code=404, detail="Tool not found.") from exc
    updated = {**existing, **tool.model_dump()}
    await crud.update_item(tool_id, updated)
    response_body = SuccessMessage(
        title="Tool Updated",
        message="Tool updated successfully.",
        content=updated,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/tools/{tool_id}", tags=["CRUD - Tools"])
async def delete_tool(tool_id: str) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TOOL_TABLE")
    try:
        await crud.delete_item(tool_id)
    except Exception as exc:
        logger.error("Error deleting tool: %s", exc)
        raise HTTPException(status_code=404, detail="Tool not found.") from exc
    response_body = SuccessMessage(
        title="Tool Deleted",
        message="Tool deleted successfully.",
        content={"tool_id": tool_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/text-data", tags=["CRUD - TextData"])
async def list_textdata_endpoint() -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TEXTDATA_TABLE")
    items = await crud.list_items()
    response_body = SuccessMessage(
        title=f"{len(items) if items else 0} TextData Retrieved",
        message="Successfully retrieved text data.",
        content=items,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/text-data", tags=["CRUD - TextData"])
async def create_textdata(textdata: TextData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TEXTDATA_TABLE")
    created = await crud.create_item(textdata.model_dump())
    response_body = SuccessMessage(
        title="TextData Created",
        message="TextData created successfully.",
        content=created,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/text-data/{textdata_id}", tags=["CRUD - TextData"])
async def update_textdata(textdata_id: str, textdata: TextData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TEXTDATA_TABLE")
    try:
        existing = await crud.read_item(textdata_id)
    except Exception as exc:
        logger.error("Error reading text data: %s", exc)
        raise HTTPException(status_code=404, detail="TextData not found.") from exc
    updated = {**existing, **textdata.model_dump()}
    await crud.update_item(textdata_id, updated)
    response_body = SuccessMessage(
        title="TextData Updated",
        message="TextData updated successfully.",
        content=updated,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/text-data/{textdata_id}", tags=["CRUD - TextData"])
async def delete_textdata(textdata_id: str) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_TEXTDATA_TABLE")
    try:
        await crud.delete_item(textdata_id)
    except Exception as exc:
        logger.error("Error deleting text data: %s", exc)
        raise HTTPException(status_code=404, detail="TextData not found.") from exc
    response_body = SuccessMessage(
        title="TextData Deleted",
        message="TextData deleted successfully.",
        content={"textdata_id": textdata_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/image-data", tags=["CRUD - ImageData"])
async def list_imagedata_endpoint() -> JSONResponse:
    crud = CosmosCRUD("COSMOS_IMAGEDATA_TABLE")
    items = await crud.list_items()
    response_body = SuccessMessage(
        title=f"{len(items) if items else 0} ImageData Retrieved",
        message="Successfully retrieved image data.",
        content=items,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/image-data", tags=["CRUD - ImageData"])
async def create_imagedata(imagedata: ImageData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_IMAGEDATA_TABLE")
    created = await crud.create_item(imagedata.model_dump())
    response_body = SuccessMessage(
        title="ImageData Created",
        message="ImageData created successfully.",
        content=created,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/image-data/{imagedata_id}", tags=["CRUD - ImageData"])
async def update_imagedata(imagedata_id: str, imagedata: ImageData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_IMAGEDATA_TABLE")
    try:
        existing = await crud.read_item(imagedata_id)
    except Exception as exc:
        logger.error("Error reading image data: %s", exc)
        raise HTTPException(status_code=404, detail="ImageData not found.") from exc
    updated = {**existing, **imagedata.model_dump()}
    await crud.update_item(imagedata_id, updated)
    response_body = SuccessMessage(
        title="ImageData Updated",
        message="ImageData updated successfully.",
        content=updated,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/image-data/{imagedata_id}", tags=["CRUD - ImageData"])
async def delete_imagedata(imagedata_id: str) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_IMAGEDATA_TABLE")
    try:
        await crud.delete_item(imagedata_id)
    except Exception as exc:
        logger.error("Error deleting image data: %s", exc)
        raise HTTPException(status_code=404, detail="ImageData not found.") from exc
    response_body = SuccessMessage(
        title="ImageData Deleted",
        message="ImageData deleted successfully.",
        content={"imagedata_id": imagedata_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/audio-data", tags=["CRUD - AudioData"])
async def list_audiodata_endpoint() -> JSONResponse:
    crud = CosmosCRUD("COSMOS_AUDIODATA_TABLE")
    items = await crud.list_items()
    response_body = SuccessMessage(
        title=f"{len(items) if items else 0} AudioData Retrieved",
        message="Successfully retrieved audio data.",
        content=items,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/audio-data", tags=["CRUD - AudioData"])
async def create_audiodata(audiodata: AudioData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_AUDIODATA_TABLE")
    created = await crud.create_item(audiodata.model_dump())
    response_body = SuccessMessage(
        title="AudioData Created",
        message="AudioData created successfully.",
        content=created,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/audio-data/{audiodata_id}", tags=["CRUD - AudioData"])
async def update_audiodata(audiodata_id: str, audiodata: AudioData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_AUDIODATA_TABLE")
    try:
        existing = await crud.read_item(audiodata_id)
    except Exception as exc:
        logger.error("Error reading audio data: %s", exc)
        raise HTTPException(status_code=404, detail="AudioData not found.") from exc
    updated = {**existing, **audiodata.model_dump()}
    await crud.update_item(audiodata_id, updated)
    response_body = SuccessMessage(
        title="AudioData Updated",
        message="AudioData updated successfully.",
        content=updated,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/audio-data/{audiodata_id}", tags=["CRUD - AudioData"])
async def delete_audiodata(audiodata_id: str) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_AUDIODATA_TABLE")
    try:
        await crud.delete_item(audiodata_id)
    except Exception as exc:
        logger.error("Error deleting audio data: %s", exc)
        raise HTTPException(status_code=404, detail="AudioData not found.") from exc
    response_body = SuccessMessage(
        title="AudioData Deleted",
        message="AudioData deleted successfully.",
        content={"audiodata_id": audiodata_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/video-data", tags=["CRUD - VideoData"])
async def list_videodata_endpoint() -> JSONResponse:
    crud = CosmosCRUD("COSMOS_VIDEODATA_TABLE")
    items = await crud.list_items()
    response_body = SuccessMessage(
        title=f"{len(items) if items else 0} VideoData Retrieved",
        message="Successfully retrieved video data.",
        content=items,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/video-data", tags=["CRUD - VideoData"])
async def create_videodata(videodata: VideoData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_VIDEODATA_TABLE")
    created = await crud.create_item(videodata.model_dump())
    response_body = SuccessMessage(
        title="VideoData Created",
        message="VideoData created successfully.",
        content=created,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/video-data/{videodata_id}", tags=["CRUD - VideoData"])
async def update_videodata(videodata_id: str, videodata: VideoData) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_VIDEODATA_TABLE")
    try:
        existing = await crud.read_item(videodata_id)
    except Exception as exc:
        logger.error("Error reading video data: %s", exc)
        raise HTTPException(status_code=404, detail="VideoData not found.") from exc
    updated = {**existing, **videodata.model_dump()}
    await crud.update_item(videodata_id, updated)
    response_body = SuccessMessage(
        title="VideoData Updated",
        message="VideoData updated successfully.",
        content=updated,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/video-data/{videodata_id}", tags=["CRUD - VideoData"])
async def delete_videodata(videodata_id: str) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_VIDEODATA_TABLE")
    try:
        await crud.delete_item(videodata_id)
    except Exception as exc:
        logger.error("Error deleting video data: %s", exc)
        raise HTTPException(status_code=404, detail="VideoData not found.") from exc
    response_body = SuccessMessage(
        title="VideoData Deleted",
        message="VideoData deleted successfully.",
        content={"videodata_id": videodata_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))
