# Uni-Mol-MeCAP
Uni-Mol-backbone Methyl Cation/Anion Affinity Predictor

<p align="left">
  <img src="fig/toc.jpg"/>
</p>

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

### Generate conformers
Please execute `shell/generate_conformer.sh`.

This script executes `src/optimize_conf.py`.

### Build models and predict MCA/MAA by built models with reference-split
#### v1 models
Please execute `shell/unimol1/layer_0/train_and_predict_ref_mca.sh` (the MCA prediction model) or `shell/unimol1/layer_0/train_and_predict_ref_maa.sh` (the MAA prediction model).

#### v2 models
Please execute `shell/unimol2/layer_0/train_and_predict_ref_mca.sh` (the MCA prediction model) or `shell/unimol2/layer_0/train_and_predict_ref_maa.sh` (the MAA prediction model).

These scripts execute `src/train.py` and `src/predict.py`.

## Additional models
### Compound-based split
#### v1 models
Please execute `shell/unimol1/layer_0/train_and_predict_cpbased_mca.sh` (the MCA prediction model) or `shell/unimol1/layer_0/train_and_predict_cpbased_maa.sh` (the MAA prediction model). 

#### v2 models
Please execute `shell/unimol2/layer_0/train_and_predict_cpbased_mca.sh` (the MCA prediction model) or `shell/unimol2/layer_0/train_and_predict_cpbased_maa.sh` (the MAA prediction model).

### Train with different conformer types (only v1 model)
#### Generate (prepare) conformers
Please execute `shell/diverse_conformer.sh`.

#### Build models and predict MCA/MAA by built models
 - xTB-optimized: Please execute `shell/different_conformation/train_and_predict_ref_mca_gth.sh` (the MCA prediction model) or `shell/different_conformation/train_and_predict_ref_maa_gth.sh` (the MAA prediction model).
 - Force-field-optimized: Please execute `shell/different_conformation/train_and_predict_ref_mca_mmff.sh` (the MCA prediction model) or `shell/different_conformation/train_and_predict_ref_maa_mmff.sh` (the MAA prediction model).
 - RMSD-max: Please execute `shell/different_conformation/train_and_predict_ref_mca_rmsd.sh` (the MCA prediction model) or `shell/different_conformation/train_and_predict_ref_maa_rmsd.sh` (the MAA prediction model).

### Train with different number of hidden layers in FFN
We evaluated the effect of the number of hidden layers in the FFN over the range $n \in \{0,1,2\}$.

#### v1 models
Please execute `shell/unimol1/layer_{n}/train_and_predict_ref_mca.sh` (the MCA prediction model) or `shell/unimol1/layer_{n}/train_and_predict_ref_maa.sh` (the MAA prediction model).

#### v2 models
Please execute `shell/unimol2/layer_{n}/train_and_predict_ref_mca.sh` (the MCA prediction model) or `shell/unimol2/layer_{n}/train_and_predict_ref_maa.sh` (the MAA prediction model).

## Analysis
All analyses (including reprecation of results in published paper) can be executed using `data/results_analysis.ipynb`.

## Paper information
If you are find our work is useful, kindly cite our paper in your work.

```
(Under preparation)
```
