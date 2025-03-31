# Multi-Modal Agentic RAG App

A multi-modal, agentic RAG implementation with Semantic Kernel that manages multi-modal information for chat applications.

## Features

This project framework provides the following features:

* Agentic implementation using Semantic Kernel
* Plugins connecting AI Search for indexing and retrieving multi-modal information
* FastAPI endpoints for managing judges and assemblies
* Integration with Azure Cosmos DB for data storage

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
    - Create a [.env](http://_vscodecontentref_/0) file in the [src](http://_vscodecontentref_/1) directory with the following content:
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