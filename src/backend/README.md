# Multi-Modal Agentic RAG App - Technical Documentation Review

This application implements a sophisticated multi-modal Retrieval-Augmented Generation (RAG) system using an agentic approach with Semantic Kernel. Below is an analysis of the key components and features.

## Core Architecture Components

### Agent System (`app/agents/main.py`)

- **ToolerBase**: Abstract base class that defines the interface for all specialized agents
- **Specialized Toolers**:
  - `TextTooler`: Processes and analyzes text content
  - `ImageTooler`: Handles image analysis and interpretation
  - `AudioTooler`: Works with audio files and transcriptions
  - `VideoTooler`: Processes video content
- **ToolerFactory**: Creates appropriate toolers based on agent specifications
- **ToolerOrchestrator**: Manages execution with three strategies:
  - `parallel_processing`: Executes all agents concurrently
  - `sequential_processing`: Runs agents in sequence, allowing for context sharing
  - `llm_processing`: Uses a meta-agent approach where LLM decides which tools to use

### Plugin System (`app/plugins/__init__.py`)

The application features modular plugins organized by media type:

- **Text Plugins**: `TextProcessor`, `TextEmbedder`, `TextAnswer`
- **Image Plugins**: `ImageProcessor`, `ImageEmbedder`, `ImageAnswer`
- **Audio Plugins**: `AudioProcessor`, `AudioEmbedder`, `AudioAnswer`
- **Video Plugins**: `VideoProcessor`, `VideoEmbedder`, `VideoAnswer`
- **Cross-Modal Plugins**:
  - `NLToSQL`: Converts natural language to SQL queries
  - `NLToNoSQL`: Converts natural language to NoSQL queries
  - `LocalFileRetriever`: Retrieves content from local files
  - `StatisticalAnalysisPlugin`: Performs statistical analysis on data

### API Layer (`app/main.py`)

The FastAPI application provides comprehensive endpoints:

- **Data Management**:
  - CRUD operations for TextData, ImageData, AudioData, and VideoData
  - Each data type has list, create, update, and delete operations

- **Inference Endpoints**:
  - Endpoints for performing multi-modal RAG operations
  - Routes inference requests to appropriate agent assemblies

- **Assembly Management**:
  - Create, retrieve, update and delete agent assemblies
  - Assemblies define which agents work together

- **Tool Management**:
  - CRUD operations for tools available to agents
  - Configuration of tool parameters and capabilities

## Advanced Features

### Multi-Modal Processing

Each media type has a complete processing pipeline:

- **Processors**: Extract information from raw media
- **Embedders**: Generate vector representations for similarity search
- **Answerers**: Generate responses using the processed information

### Agentic Architecture

- Uses Semantic Kernel to implement intelligent agents
- Agents can leverage tools (plugins) depending on their specialization
- Supports complex reasoning through templated instructions
- Error-handling with retry logic for service interruptions

### Integration with Azure Services

- Azure OpenAI Service integration (`gpt-4o`, `gpt-4o-mini`, `o3-mini`)
- Azure Cosmos DB for structured data storage
- Azure Blob Storage for media file storage
- Azure AI Speech services for audio processing

### Templating System

- Jinja2 templates for consistent prompt engineering
- Specialized templates for each media type (`text.jinja`, `image.jinja`, etc.)
- Meta-prompt system for agent orchestration (`reason.jinja`)

## Technical Implementation

The code uses modern Python practices:

- Async/await for non-blocking operations
- Strong typing with type annotations
- Abstract base classes defining clean interfaces
- Factory pattern for object creation
- Strategy pattern for execution approaches
- Exception handling with retry mechanisms

This documentation review highlights the sophisticated multi-modal, agentic RAG capabilities of the application, showing how it integrates with Azure services to provide comprehensive media processing and AI-driven analysis.

## Getting Started

### Prerequisites

- Windows 10 or later / Ubuntu 23.04 or later
- Python 3.12 or later
- Check [pyproject.toml](pyproject.toml) for Python dependencies

### Installation

1. Clone the repository:

    ```sh
    git clone [repository clone url]
    cd [repository name]
    ```

2. Install Python dependencies:

    ```sh
    pip install poetry
    poetry install
    ```

3. Set up environment variables:

Create a [.env](http://_vscodecontentref_/0) file in the [src](http://_vscodecontentref_/1) directory with the following content:

```env
BLOB_CONNECTION_STRING="your_blob_connection_string"
BLOB_SERVICE_CLIENT="your_blob_service_client"
AI_SPEECH_URL="your_ai_speech_url"
AI_SPEECH_KEY="your_ai_speech_key"
COSMOS_ENDPOINT="your_cosmos_endpoint"
COSMOS_KEY="your_cosmos_key"
GPT4_KEY="your_gpt4_key"
GPT4_URL="your_gpt4_url"
```

### Running the Application

1. Start the FastAPI application:

    ```sh
    poetry run uvicorn app.main:app --reload
    ```

2. Access the API documentation at `http://localhost:8000/docs`.

### Project Structure

- [app/](http://_vscodecontentref_/2): Contains the main application code.
  - [agents.py](http://_vscodecontentref_/3): Implements the SuperJudge, ConcreteJudge, and related classes.
  - [main.py](http://_vscodecontentref_/4): Configures the FastAPI application and defines the API endpoints.
  - [plugins.py](http://_vscodecontentref_/5): Defines example plugins for function-calling usage.
  - [schemas/](http://_vscodecontentref_/6): Contains Pydantic models for the application.
    - [__init__.py](http://_vscodecontentref_/7): Initializes the schemas package.
    - [endpoints.py](http://_vscodecontentref_/8): Defines models for the endpoints.
    - [models.py](http://_vscodecontentref_/9): Defines the data models for the application.
    - [responses.py](http://_vscodecontentref_/10): Manages the response bodies.

- `tests/`: Contains unit tests for the application.
  - [test_agents.py](http://_vscodecontentref_/11): Tests for the agents module.

### Running Tests

To run the tests, use the following command:

```sh
poetry run pytest
```
