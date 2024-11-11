# Kidney-Disease-Classification-Tensorflow

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

## How to run?

### STEP 01 - Clone the repo

```bash
https://github.com/ommallick02/Kidney-Disease-Classification-Tensorflow
```
### STEP 02 - Create and activate a conda environment after opening the repository

```bash
conda create -n kidney-tensorflow python=3.10 -y
```

```bash
conda activate kidney-tensorflow
```

### STEP 03 - Install the requirements

Install CUDA

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Run Setup

```bash
pip3 install pip<25.0.0
```

```bash
pip3 install -r requirements.txt
```

Verify Tensorflow GPU is working

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

```bash
mlflow ui
```

## Dagshub

- [Dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/om.mallick02/Kidney-Disease-Classification-Tensorflow.mlflow \
MLFLOW_TRACKING_USERNAME=om.mallick02 \
MLFLOW_TRACKING_PASSWORD= \
python script.py

Run this to export as env variables(Linux/ macOS):

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/om.mallick02/Kidney-Disease-Classification-Tensorflow.mlflow
```

```bash
export MLFLOW_TRACKING_USERNAME=om.mallick02 
```

```bash
export MLFLOW_TRACKING_PASSWORD=
```

Run this to export as env variables(Windows):

```bash
$env:MLFLOW_TRACKING_URI="https://dagshub.com/om.mallick02/Kidney-Disease-Classification-Tensorflow.mlflow"
```

```bash
$env:MLFLOW_TRACKING_USERNAME="om.mallick02"
```

```bash
$env:MLFLOW_TRACKING_PASSWORD=""
```

## DVC

```bash
dvc init
```

```bash
dvc repro
```

```bash
dvc dag
```

## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model

DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)

## Run The App

Run The Flask App

```bash
python app.py
```

Now, open up you local host and port

```bash
http://localhost:8080/
```

## AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console.

### 2. Create IAM user for deployment

#### With specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws

#### Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#### Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess

### 3. Create ECR repo to store/save docker image

- Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken
	
### 4. Create EC2 Cachine (Ubuntu) 

### 5. Open EC2 and Install Docker in EC2 Machine:
		
Optional

```bash
sudo apt-get update -y
sudo apt-get upgrade
```

Required

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```
	
### 6. Configure EC2 as self-hosted runner:

```bash
setting>actions>runner>new self hosted runner> choose os> then run command one by one
```

### 7. Setup github secrets:

```bash
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app
```
