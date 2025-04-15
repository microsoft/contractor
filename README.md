# The Contractor

A multi-modal, agentic RAG implementation with Semantic Kernel that manages multi-modal information for chat applications. The objective of The Contractor is to evaluate loan and security contracts by retrieving and verifying contractor information—both from internal data sources and the web—to provide risk, compliance, and governance feedback.

## Objective

The Contractor is an application that implements a multi-agent system capable of evaluating the security and quotization of contracts based on a wide range of multi-modal data sources. It provides risk, compliance, and governance feedback and assesses a contractor’s ability to fulfill its contractual obligations using analytical tools and strategies. Specifically, it:

- **Evaluates loan and security contracts:** Analyzes contract clauses and financial terms.
- **Retrieves and verifies contractor information:** Gathers data from internal repositories and external public records.
- **Searches the web for external information:** Utilizes the Bing API to obtain legal, compliance, and public data (e.g., court records, social network information).

**Repository:** [microsoft/contractor](https://github.com/microsoft/contractor) – A repository that implements an agentic system capable of evaluating security and quotization contracts based on a wide range of data sources.

**Technology:**

- Semantic Kernel (Python)
- FastAPI
- Azure OpenAI (AOAI with Reasoning)
- AI Services (Speech, Vision, and Document)
- Bing API for external web search
- AI Search
- Cosmos DB for data storage
- Azure Container Apps for deployment

**Legal Benefits:**

- **Multi-Agentic System:** Uses an Agent Swarm Pattern to deploy specialized agents orchestrated by a reasoning model that plans the analysis based on contract content and contractor data sources.
- **Web Groundness:** Leverages the Bing API to retrieve legal and compliance information on the contractor, ensuring that provided details are substantiated by public records (e.g., court documents and social network data).
- **Multi-Modal Data Processing:** Seamlessly processes documents, images, audio, and video to gather contextual information on contracts and contractors, evaluating both written clauses and unwritten promises.
- **Risk Assessment:** Analyzes internal data to evaluate fiduciary and non-fiduciary risks, providing actionable insights for analysts when signing contracts.

## Features

This project framework provides the following features:

* Agentic implementation using Semantic Kernel
* Modular plugins for multi-modal processing (text, image, audio, and video)
* Integration with comprehensive AI services (e.g., Azure OpenAI Service, AI Speech)
* Connection to Azure Cosmos DB for structured data storage and contract evaluation
* Advanced data ingestion and processing via a versatile plugin architecture

## Getting Started

### Prerequisites

- Windows 10 or later / Ubuntu 23.04 or later
- Check [pyproject.toml](src/pyproject.toml) for Python dependencies

### Installation

- Install [Python](https://www.python.org/downloads/)
- Install [Node.js](https://nodejs.org/) (if using TypeScript components)
- Install [Poetry](https://python-poetry.org/) for dependency management
- Run [the configuration file for your system](.configure/conf-env.ps1)

### Quickstart

1. Clone the repository:

   ```bash
   git clone https://github.com/microsoft/contractor.git
   ```

2. Change to the repository directory:

   ```bash
   cd contractor
   ```

3. Install Python dependencies using Poetry:

   ```bash
   poetry install
   ```

4. Set up your environment by running the configuration script:

   ```bash
   ./.configure/conf-env.ps1
   ```

5. Start the FastAPI application:

   ```bash
   poetry run uvicorn app.main:app --reload
   ```

## Demo

A demo app is included to show how to use the project.

To run the demo, follow these steps:

1. Ensure the application is running by following the quickstart steps.
2. Open your web browser and navigate to `http://localhost:8000/docs` to access the interactive API documentation.
3. Use the available endpoints to interact with the multi-modal, agentic RAG system and explore its integrations with AI services and Cosmos DB connectivity.

## Resources

You'll find more information on this project based on the proper documentation on each architectural component:

- Read the [FastAPI App Documentation](src/README.md)
- Run the [tests](src/tests/)
