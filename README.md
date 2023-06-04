# BBCS-June-Intermediate
Group H2

## Setup

### Notebook

A notebook was initially written to play around with model code. Scripts were later used to do actual training.

1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/)
1. Create Conda environment:
    ```
    conda create --name buildingblocs --file conda-list.txt
    ```
1. Activate environment:
    ```
    conda activate buildingblocs
    ```
1. Run Jupyter Notebook:
    ```
    jupyter notebook
    ```
1. Make sure to download Prannay's Edibles dataset and place it (the 0,1,2,3... folders) in a folder called prannays_edibles in datasets/

### Scripts

Scripts were used to prepare data and subsequently train models based on different parameters. There are also scripts for validating the model.

1. Create a virtual env
    ```
    python -m venv venv
    ```
1. Activate environment
1. Install requirements
    ```
    pip install -r requirements.txt
    ```
1. Download datasets: https://drive.google.com/drive/folders/14dO-arrgI8CARxrY0LvchHT0QHB7DuHG?usp=sharing
    - Extract the zip files in datasets/ in their own folders that have the same name as the zip file (e.g. prannays_edibles.zip should be extracted to datasets/prannays_edibles/)
1. Run scripts with working directory as this repo's folder

### App

There is a simple app that exposes an API that runs inference on images to output food classification.

1. Download the model (food_classification.pt): https://drive.google.com/drive/folders/14dO-arrgI8CARxrY0LvchHT0QHB7DuHG?usp=sharing
1. Place the model (food_classification.pt) in app/models/
1. Ensure you have an active python virtual environment with requirements.txt dependencies installed (see [scripts setup instructions](#scripts))
1. Run the app by running app/api.py
1. Open app/website/index.html and upload your image
