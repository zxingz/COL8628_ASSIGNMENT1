# Zero/Few-Shot Learning with Prompt-Based Models

This project explores and implements several prompt-based learning methods for adapting large pre-trained vision-language models (like CLIP) to new downstream tasks with limited data. It includes implementations for CoOp, Co-CoOp, and MaPLe, along with scripts for both initial training and incremental learning on new classes.

## Folder Structure

- **`.venv/`**: Contains the Python virtual environment and its dependencies.
- **`data/`**: Holds the datasets used for training and evaluation. This project uses the `orthonet` and `pacemakers` datasets.
- **`report_data/`**: Contains images and plots generated during experiments, which are used in the final report.
- **`results/`**: Stores the output of training scripts, typically as JSON files containing training history and evaluation metrics.
- **`weights/`**: Saves the trained model weights (`.pth` files) after training.

## Python Files

### Task 1: Orthonet Dataset

- **`Task-1_1.py`**: Script for training and evaluating baseline models (e.g., ImageNet-21k pre-trained, DINO V2) on the Orthonet dataset.
- **`Task-1_2.py`**: Implements a standard CLIP model evaluation on the Orthonet dataset.
- **`Task-1_3_coop_train.py` / `Task-1_3_coop_infer.py`**: Training and inference scripts for the CoOp model on Orthonet.
- **`Task-1_3_cocoop_train.py` / `Task-1_3_cocoop_infer.py`**: Training and inference scripts for the Co-CoOp model on Orthonet.
- **`Task-1_3_maple_train.py` / `Task-1_3_maple_infer.py`**: Training and inference scripts for the MaPLe model on Orthonet.

### Task 2: Pacemakers Dataset

- **`Task-2_1.py`**: Script for training and evaluating baseline models on the Pacemakers dataset.
- **`Task-2_2.py`**: Implements a standard CLIP model evaluation on the Pacemakers dataset.

#### Stage 1: Initial Training
- **`Task-2_3_st1_coop_train.py` / `Task-2_3_st1_coop_infer.py`**: Stage 1 training and inference for the CoOp model on a subset of Pacemaker classes.
- **`Task-2_3_st1_cocoop_train.py` / `Task-2_3_st1_cocoop_infer.py`**: Stage 1 training and inference for the Co-CoOp model.
- **`Task-2_3_st1_maple_train.py` / `Task-2_3_st1_maple_infer.py`**: Stage 1 training and inference for the MaPLe model.

#### Stage 2: Incremental Learning
- **`Task-2_3_st2_coop_train_infer.py`**: Stage 2 script to load a pre-trained CoOp model and continue training on an expanded set of classes.
- **`Task-2_3_st2_cocoop_train_infer.py`**: Stage 2 script for incremental learning with the Co-CoOp model.
- **`Task-2_3_st2_maple_train.py`**: Stage 2 script for incremental learning with the MaPLe model.

## Jupyter Notebooks

- **`EDA.ipynb`**: Contains Exploratory Data Analysis of the datasets.
- **`Reporting.ipynb`**: Used for generating plots, tables, and visualizations from the results for the final report.

## Other Files

- **`requirements.txt`**: Lists all the Python dependencies required to run the project.
- **`pyproject.toml`**: Project configuration file.
- **`report.pdf`**: The final project report.
- **`COL8628_COL828_Assignment_1.pdf`**: The assignment description document.

## Setup and Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Data**: Download the `orthonet` and `pacemakers` datasets and place them in the `data/` directory.
3.  **Run Training**: Execute the desired training scripts. For example:
    ```bash
    python Task-1_3_coop_train.py
    ```
4.  **Run Inference**: After training, run the corresponding inference script to evaluate the model.
    ```bash
    python Task-1_3_coop_infer.py
    ```

## Model Weights and Code Repo

-   **Model Weights**: [Google Drive Link](https://drive.google.com/drive/folders/16qX0yVtcCD7pxJkuAeCVqG6JYmiELYWw?usp=sharing)
-   **Code Repo**: [GitHub Link](https://github.com/zxingz/COL8628_ASSIGNMENT1)
