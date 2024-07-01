# AttenhERG

  Regarding the hERG prediction model, we are currently sharing the inference models, and the trained models are readily available.

## Setup and dependencies 

  To set up the environment, you can utilize the `environment.yaml` file, representing the conda environment for this project. Alternatively, you have the option to deploy the environment using the `requirement.txt` file.

## Trained models

  The trained models are stored in `./saved_models`, specifically under the name `model_2023_Aug_10_100_1_200_3_2_4.5_3.5_83.pt`.


## Inference

### Concerning inference, the code details are as follows:

1. The inference process is conducted through `AttenhERG_Userup.py`, using compound SDF as input.
2. User-inputted files should be placed in the `./userup` folder.
3. The molecular file provided by the user in SDF format is located at `./userup/user_input.sdf`.
4. After running `python AttenhERG_Userup.py`, the resulting file, `./userup/result_user_input.xlsx`, is generated. This file includes Index_ID, Structure, Predict_Score, Entropy_Uncertainty, MCDropout_Uncertainty, and Smiles. This information can assist in creating the initial display page for the GUI.
5. Molecular image SVG files are temporarily stored in the `./userup/svg_data` folder and are cleared at the end of the script.
