# Smart Real Estate Predictor (End to End MLOps)

A production grade Supervised Machine Learning system that predicts property prices using the Ames Housing dataset. This project demonstrates a full MLOps lifecycle from data ingestion to containerized deployment.

## Tech Stack
* **Orchestration:** [ZenML](https://zenml.io/)
* **Experiment Tracking:** [MLflow](https://mlflow.org/)
* **Model:** Random Forest Regressor (Scikit-Learn)
* **API:** FastAPI
* **Deployment:** Docker

## System Architecture
1. **Data Pipeline:** Automated ingestion and preprocessing via ZenML.
2. **Experimentation:** Metrics and model artifacts tracked in MLflow.
3. **Serving:** REST API built with FastAPI, providing a Swagger UI for testing.
4. **Containerization:** Fully Dockerized for platform-independent execution.

## How to Run

### 1. Training the Model
```bash
python run_pipeline.py
THE LINK : https://smart-real-estate-predictor.onrender.com/ TO THE DEPLOYED VERSION SWAGGER UI
