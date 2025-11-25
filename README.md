# Workflow-CI: MLflow Customer Churn Prediction Pipeline

A CI/CD pipeline that trains a Random Forest classifier for customer churn prediction using MLflow, containerizes the model with Docker, and automatically deploys it to Docker Hub.

## Overview

This project implements an automated machine learning workflow that:

1. **Trains a model** using scikit-learn's Random Forest Classifier on the Telco Customer Churn dataset
2. **Tracks experiments** with MLflow for reproducibility and model versioning
3. **Builds a Docker image** containing the trained model
4. **Pushes the container** to Docker Hub for deployment

## Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── main.yml          # GitHub Actions CI/CD workflow
├── MLProject/
│   ├── MLProject             # MLflow project configuration
│   ├── conda.yaml            # Conda environment specification
│   ├── modelling.py          # Model training script
│   ├── DockerHub.txt         # Docker Hub repository URL
│   └── WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv  # Preprocessed dataset
├── mlruns/                   # MLflow tracking directory
└── README.md
```

## Dataset

The project uses the **Telco Customer Churn** dataset, which has been preprocessed with:
- Numerical feature scaling (standardization)
- Categorical feature encoding (one-hot encoding)
- Tenure binning

**Target variable**: `Churn` (binary classification - 0 or 1)

## Model Configuration

The Random Forest classifier is configured with the following default parameters:

| Parameter | Default Value |
|-----------|---------------|
| n_estimators | 505 |
| max_depth | 37 |

These parameters can be customized when running the MLflow project.

## Dependencies

The project uses the following main dependencies:

- Python 3.13.9
- MLflow 2.18.0rc0 (as specified in `conda.yaml`) and MLflow 2.22.0 (as used in CI/CD workflow)
- scikit-learn 1.7.2
- pandas 2.3.3
- numpy 2.3.5

See `MLProject/conda.yaml` for the complete dependency list.

## Running Locally

### Prerequisites

- Python 3.12+
- MLflow installed (`pip install mlflow`)
- Conda (optional, for environment management)

### Run the MLflow Project

```bash
# Using local environment
mlflow run ./MLProject --env-manager=local

# Using conda environment
mlflow run ./MLProject

# With custom parameters
mlflow run ./MLProject -P n_estimators=100 -P max_depth=20 --env-manager=local
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/main.yml`) automates the entire pipeline:

### Triggers
- Push to `main` branch
- Pull requests to `main` branch
- Manual trigger via `workflow_dispatch`

### Pipeline Steps

1. **Checkout** - Clone the repository
2. **Setup Python** - Install Python 3.12.7 (CI/CD) or Python 3.13.9 (local conda environment)
3. **Install Dependencies** - Install MLflow
4. **Run Modelling** - Execute the MLflow project
5. **Build Docker Image** - Create container with `mlflow models build-docker`
6. **Push to Docker Hub** - Deploy the containerized model

### Required Secrets

Configure these secrets in your GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `DOCKER_HUB_USERNAME` | Docker Hub username |
| `DOCKER_HUB_ACCESS_TOKEN` | Docker Hub access token |
| `USERNAME` | Git username for commits |
| `EMAIL` | Git email for commits |

## Docker Deployment

The trained model is available as a Docker image:

```bash
# Pull the latest model
docker pull pahlawandocker/churn:latest

# Run the model server
docker run -p 5000:8080 pahlawandocker/churn:latest
```

The model server exposes a REST API for predictions at `http://localhost:5000/invocations`.

## MLflow Tracking

Experiment runs are tracked in the `mlruns/` directory. Each run includes:

- Model artifacts
- Accuracy metrics
- Input examples
- Model signature

## License

This project is open source and available for educational and research purposes.
