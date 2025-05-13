# Kidney-Disease-Classification-Tensorflow

A deep learning pipeline to classify kidney disease from CT scans, using TensorFlow, MLflow, and DVC for an efficient machine learning workflow and deployment.

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

## Getting Started

### Step 1: Clone the Repository

```bash
https://github.com/ommallick02/Kidney-Disease-Classification-Tensorflow
```

### Step 2: Set Up and Activate the Conda Environment

```bash
conda create -n kidney-tensorflow python=3.10 -y
conda activate kidney-tensorflow
```

### Step 3: Install Requirements

Install CUDA (for GPU support)

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Run Setup

```bash
pip install pip<25.0.0
pip install -r requirements.txt
```

Verify Tensorflow GPU is working

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## MLflow Integration

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Dagshub Integration

- [Dagshub Website](https://dagshub.com/)

Access your MLflow server on DagsHub for experiment tracking.

### Set Environment Variables for Linux/macOS

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/om.mallick02/Kidney-Disease-Classification-Tensorflow.mlflow
export MLFLOW_TRACKING_USERNAME=om.mallick02
export MLFLOW_TRACKING_PASSWORD=
```

### Set Environment Variables for Windows

```bash
$env:MLFLOW_TRACKING_URI="https://dagshub.com/om.mallick02/Kidney-Disease-Classification-Tensorflow.mlflow"
$env:MLFLOW_TRACKING_USERNAME="om.mallick02"
$env:MLFLOW_TRACKING_PASSWORD=""
```

## Data Version Control (DVC)

Initialize and manage pipelines with DVC:

```bash
dvc init
dvc repro
dvc dag
```

## About MLflow & DVC

### MLflow

- Production-grade tool for tracking and managing experiments.
- Logs and tags models for easy comparison.
- Provides model versioning and experiment tracking.

### DVC 

- Lightweight tool for experiment tracking in Proof-of-Concept (POC) stages.
- Supports orchestration and pipeline creation for ML workflows.

## Running the Application

Run The Flask App

```bash
python app.py
```

Access the app at:

```bash
http://localhost:8080/
```

## AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console.

### 2. Create IAM user for deployment

Provide the following access permissions:

1. EC2 access : It is virtual machine
2. ECR: Elastic Container registry to save your docker image in aws

#### Deployment Steps

1. Build the Docker image from the source code.
2. Push the Docker image to ECR.
3. Launch an EC2 instance.
4. Pull the image from ECR to EC2.
5. Run the Docker image on EC2.

##### Required IAM Policies:

1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess

### 3. Create an ECR Repository

- Example URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken
	
### 4. Launch an EC2 Instance (Ubuntu)

### 5. Install Docker on EC2

Update and install Docker:

```bash
sudo apt-get update -y
sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```
	
### 6. Configure EC2 as a Self-Hosted GitHub Runner

Navigate to: Settings > Actions > Runners > New Self-Hosted Runner and follow the instructions for setting up a runner on your EC2 instance.

### 7. Configure GitHub Secrets for Deployment

Add the following secrets in GitHub Actions:

```bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
AWS_ECR_LOGIN_URI=
ECR_REPOSITORY_NAME=
```
