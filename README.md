# Uni-Mol-MeCAP
Uni-Mol-backbone Methyl Cation/Anion Affinity Predictor

## Environment

All requirements are written in `envs/environments.yml`.

Please execute the following command to create conda environment.

```bash
conda env create -f ./envs/environment.yml -n mecap
```

## Training
### Data preparation
Dataset can be downloaded and preprocessed by `data/references/data_preparation.ipynb`.

Please execute whole cells from top to bottom in order.

### Generating conformers
Please execute `shell/generate_conformer.sh`.

This script executes `src/optimize_conf.py`.

### Building models
#### v1 models
Please execute `shell/unimol1/layer_{n}/train_ref_maa.sh` (to build the MAA prediction model) or `shell/unimol1/layer_{n}/train_ref_mca.sh` (to build the MCA prediction model) with reference train-val-test split.

#### v2 models
Please execute `shell/unimol2/layer_{n}/train_ref_maa.sh` (to build the MAA prediction model) or `shell/unimol2/layer_{n}/train_ref_mca.sh` (to build the MCA prediction model) with reference train-val-test split.

#### Note
This script executes `src/train.py`.

### Predict using built model
#### v1 models
Please execute `shell/unimol1/predict_ref_maa.sh` (to predict MAA) or `shell/unimol1/predict_ref_mca.sh` (to predict MCA) with reference train-val-test split.

#### v2 models
Please execute `shell/unimol2/predict_ref_maa.sh` (to predict MAA) or `shell/unimol2/predict_ref_mca.sh` (to predict MCA) with reference train-val-test split.

#### Note
This script executes `src/predict.py`.

Please note that the prediction shell script contains prediction by some models with different number of hidden layer. Please modify the number of `for` loop in the script if you want to predict with the specific model.
