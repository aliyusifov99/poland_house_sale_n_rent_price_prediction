# 🇵🇱 End-to-End Housing Price Prediction in Poland

## 📌 Project Overview
This project is a Master's Thesis deliverable focusing on the **End-to-End MLOps lifecycle**. It predicts real estate prices (Sale and Rent) in major Polish cities using Machine Learning.

Unlike standard data science projects that stop at a Jupyter Notebook, this project implements a full production pipeline including:
* **Data Engineering**: Automated processing of raw monthly snapshots.
* **Model Training**: Scikit-Learn pipelines tracked with **MLflow**.
* **API**: A RESTful backend using **FastAPI**.
* **Frontend**: An interactive web dashboard using **Streamlit**.
* **Deployment**: Fully containerized microservices architecture using **Docker**.

## 🏗 Architecture
The system follows a microservices pattern:
1.  **Backend Service (`/api`)**: Loads the trained Random Forest models and serves predictions via JSON endpoints.
2.  **Frontend Service (`/frontend`)**: A user-facing dashboard that sanitizes inputs and communicates with the backend via internal Docker networking.

## 📂 Project Structure
```text
├── .dvc/               # Data Version Control config
├── api/                # FastAPI application code
├── data/               # Raw and Processed datasets
├── frontend/           # Streamlit application code
├── models/             # Serialized ML models (.pkl)
├── src/                # Scripts for processing and training
├── Dockerfile.api      # Docker build for Backend
├── Dockerfile.frontend # Docker build for Frontend
└── docker-compose.yaml # Orchestration