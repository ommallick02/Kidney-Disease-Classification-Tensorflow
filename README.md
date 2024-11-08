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