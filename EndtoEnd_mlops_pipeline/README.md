Iris Classification using KNNeighbor Classification (with MLOps)

This project performs classification of Iris flowers using machine learning techniques and implements a full MLOps pipeline for training, tracking, and deploying the model.

Project Structure

EndtoEnd_mlops_pipeline/
│
├── data/                       # Raw and processed data
├── notebooks/                 # EDA or experimentation notebooks
├── src/
│   ├── train.py               # Training script with MLflow logging
│   ├── inference.py           # Model loading and prediction
│   └── utils.py               # Preprocessing, helpers
├── app/
│   └── main.py                # FastAPI serving code
├── Dockerfile
├── requirements.txt
├── mlruns/                    # MLflow tracking folder (auto-created)
└── README.md
