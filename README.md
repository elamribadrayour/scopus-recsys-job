# Scopus RecSys - Research Paper Classification and Recommendation System

A sophisticated system for analyzing, classifying, and finding similarities between research papers from Scopus, with a focus on recommender systems papers.

## 🌟 Features

- **Paper Classification**: Automatically classifies research papers based on their abstracts using Large Language Models.
- **Algorithm Similarity**: Calculates similarities between different algorithms mentioned in papers
- **Application Similarity**: Identifies similar applications across different papers
- **Algorithm-Application Linking**: Creates connections between algorithms and their applications
- **Database Management**: Efficient SQLite database handling for storing and retrieving paper information

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Docker

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd scopus-recsys-job
```

2. Set up environment variables in `.env`:
```bash
DATA_PATH=<path-to-data>
OLLAMA_HHOST=<your-ollama-host>
LLM=<the-required-llm-for-predictions>
```

3. Install dependencies:
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Docker Setup (Alternative)

```bash
docker-compose up -d
```

## 🔧 Usage

The system provides several commands for different functionalities:

1. Initialize the database:
```bash
python -m src.job.main init
```

2. Classify papers:
```bash
python -m src.job.main classify --batch-size 10
```

3. Calculate algorithm similarities:
```bash
python -m src.job.main similarity-algorithm
```

4. Calculate application similarities:
```bash
python -m src.job.main similarity-application
```

5. Link algorithms with applications:
```bash
python -m src.job.main algorithm-application-link
```

## 🏗️ Project Structure

```
├── data/ # Data storage
├── src/
│ └── job/
│ ├── helpers/ # Helper functions
│ ├── models/ # Data models
│ ├── sql/ # SQL queries
│ └── main.py # Main application
├── docker-compose.yml # Docker configuration
├── Dockerfile # Docker build file
└── pyproject.toml # Project configuration
```

## 🛠️ Development

This project uses:
- `typer` for CLI interface
- `loguru` for logging
- `numpy` for data processing
- Ollama's API for classification
- SQLite for data storage

## 📝 License

This project is licensed under the terms of the LICENSE file included in the repository.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📊 Data Sources

The system processes research papers from Scopus, focusing on recommender systems literature.
