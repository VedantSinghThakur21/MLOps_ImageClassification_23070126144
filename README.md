# Image Classification DVC

**Author**: Tanniru Hari Sri Shanmukha Sai  
**PRN**: 23070126135

## Project Overview

This repository implements a modular image classification system using Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn, and DVC. The application follows a clean stage-based architecture and uses MobileNetV2 transfer learning to classify images from the Intel Image Dataset.

The machine learning workflow is fully automated with DVC and is organized into five reproducible stages:

1. Data collection
2. Data processing
3. Feature extraction
4. Model training
5. Model evaluation

## Repository Structure

```text
Image-Classification-DVC/
├── artifacts/
│   ├── features/
│   ├── model/
│   ├── metrics/
│   └── processed/
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── feature_selection.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── config.py
│   └── utils.py
├── .github/workflows/ci-cd.yaml
├── samples/
├── app.py
├── Dockerfile
├── requirements-app.txt
├── dvc.yaml
├── dvc.lock
├── main.py
├── params.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

## Environment Setup

The project targets Python 3.11.

Create a virtual environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need to rerun the pipeline on this macOS machine, DVC should use the local site cache path that was validated during execution:

```bash
DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache" dvc repro
```

## Configuration

All tunable parameters are stored in [params.yaml](params.yaml). This includes dataset location, image size, batch size, learning rate, optimizer, dropout rate, dense units, validation and test splits, augmentation settings, seed, loss function, and model metadata.

## DVC Pipeline

The pipeline is fully automated through [dvc.yaml](dvc.yaml). The stages are executed in order and produce versioned outputs and metrics.

```mermaid
flowchart TD
    A[Data Collection] --> B[Data Processing]
    B --> C[Feature Selection]
    C --> D[Model Training]
    D --> E[Model Evaluation]
```

## Running the Project

Execute the full machine learning pipeline with:

```bash
source .venv/bin/activate
DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache" dvc repro
```

For manual stage-level execution during development, the following module entrypoints are available:

```bash
python -m src.data_collection
python -m src.data_processing
python -m src.feature_selection
python -m src.model_training
python -m src.model_evaluation
```

## Expected Outputs

After a successful run, the following artifacts are generated:

- `data/raw/` for the downloaded or loaded dataset
- `data/processed/` for the train, validation, and test splits
- `artifacts/features/` for the MobileNetV2 feature extractor
- `artifacts/model/` for the trained classification model and history files
- `artifacts/metrics/metrics.json` for accuracy, precision, recall, and F1 score
- `artifacts/metrics/confusion_matrix.png` for the confusion matrix

## Submission Checklist

Before zipping the project for submission, keep the source and pipeline files, and exclude local runtime noise such as the virtual environment and DVC site cache.

Recommended files to include:

- `src/`
- `data/`
- `artifacts/`
- `params.yaml`
- `dvc.yaml`
- `dvc.lock`
- `requirements.txt`
- `README.md`
- `main.py`
- `.gitignore`

Recommended folders to exclude from the zip:

- `.venv/`
- `.dvc/site-cache/`
- `__pycache__/`

## Dataset Note

The `data_collection` stage downloads the public Kaggle dataset through Kaggle's download API, so no Kaggle credentials are required. If the download fails, place the dataset manually in `data/raw/` before running the pipeline.

## DVC Remote (AWS S3)

Pipeline artifacts are versioned in the S3 bucket `mlops-imgcls-23070126144` (region `ap-south-1`):

```bash
dvc remote add -d storage s3://mlops-imgcls-23070126144/dvcstore
dvc remote modify storage region ap-south-1
dvc push feature_selection model_training model_evaluation
dvc pull model_training model_evaluation   # fetch the trained model and metrics
```

## Web Application

[app.py](app.py) is a Flask app that serves the trained model:

- `GET /` upload page that shows the predicted class and per-class probabilities
- `POST /predict` JSON API, e.g. `curl -F file=@samples/forest.jpg http://localhost:8080/predict`
- `GET /health` health check

Run it in Docker after `dvc pull`:

```bash
docker build -t image-classifier .
docker run -p 8080:8080 image-classifier
```

## CI/CD and Deployment (GitHub Actions + AWS EC2)

[.github/workflows/ci-cd.yaml](.github/workflows/ci-cd.yaml) runs on every push to `main`:

1. `dvc pull` the trained model and metrics from S3, then print `dvc dag` and `dvc metrics show`
2. Build the Docker image and smoke test it (`samples/forest.jpg` must be classified as `forest`)
3. Push the image to GitHub Container Registry (`ghcr.io`)
4. SSH into the EC2 instance, pull the image and run it on port 80, then check `/health` on the public IP

Required repository secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (read access to the bucket), `EC2_HOST` (public IP) and `EC2_SSH_KEY` (contents of the EC2 key pair `.pem`).

The EC2 instance (Ubuntu, t2.micro) is bootstrapped with this user-data script, which installs Docker and adds swap so TensorFlow fits in 1 GB of RAM:

```bash
#!/bin/bash
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
apt-get update -y && apt-get install -y docker.io
systemctl enable --now docker
usermod -aG docker ubuntu
```
