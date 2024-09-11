# AttenhERG

  Regarding the hERG prediction model, we are currently sharing the inference models, and the trained models are readily available.

## Setup and dependencies 

  To set up the environment, you can utilize the `environment.yaml` file, representing the conda environment for this project. Alternatively, you have the option to deploy the environment using the `requirement.txt` file.

## Dataset
  The dataset used for training and validating the model is located in the `./data/` folder.
  
## Trained models

### Training the model

We were training the model using the `AttenhERG.py` script on a `Tesla P40 GPU`. To train the model, please use the following command in the terminal: `python AttenhERG.py`.

### Hyperparameter search 

We performed hyperparameter optimization by running `python AttenhERG_Hyperparameter.py`. Relevant search details are annotated in the code. Specifically, the selection range for the hidden units was [50, 100, 150, 200, 250, 300]; you can choose one of these values to modify `fingerprint_dim`. The selection range for dropout was [0.1, 0.3, 0.5]; you can select one to adjust `p_dropout`. The L2 regularization rate had a selection range of [10<sup>-3.5</sup>, 10<sup>-4</sup>, 10<sup>-4.5</sup>]; you can pick one of these values to change `weight_decay`.

### Best trained model

Meanwhile, the trained models are stored in `./saved_models`, specifically under the name `model_2023_Aug_10_100_1_200_3_2_4.5_3.5_83.pt`.

## Inference

### Concerning inference, the code details are as follows:

After running `python AttenhERG_Prediction.py`, the calculated metrics in order are Accuracy, MCC, BAC, F1 score, AUROC, AUPRC, Precision, and Recall.

### User application of the AttenhERG model

1. The inference process is conducted through `AttenhERG_Userup.py`, using compound SMILES as input.
2. User-inputted files should be placed in the `./userup` folder.
3. The molecular file provided by the user in SMILES format is located at `./userup/user_input.csv`.
4. After running `python AttenhERG_Userup.py`, the resulting file, `./userup/result_user_input.xlsx`, is generated. This file includes Index_ID, Structure, Predict_Score, Entropy_Uncertainty, MCDropout_Uncertainty, and Smiles. 
5. Molecular image SVG files are temporarily stored in the `./userup/svg_data` folder and are cleared at the end of the script.

## Case

In our case study, running `python AttenhERG_Visualization.py` provides a visualization of the model-derived weights, illustrating how molecular features are captured by the model.


