#!/bin/bash

CONDA_EV=~/miniconda3
WORK_ROT=~/work/UniMea_dev

cd ${WORK_ROT} || exit

RUN_NAME=mecap_cpbased_mca_layer_0
SRCP_DIR=src
RUN_MODE=predict

BASE_DIR=${WORK_ROT}/data/references
RESL_DIR=${WORK_ROT}/data/results
SAVE_DIR=${RESL_DIR}/${RUN_NAME}

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}

cd ${SRCP_DIR} || exit

${EXEC_PAT} -m ${RUN_MODE} \
  --data ${BASE_DIR}/QMdata4ML/df_nuc_x_with_name_fold.csv \
  --checkpoint ${SAVE_DIR}/best_model.pt \
  --atom_index_col nuc_sites \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${RESL_DIR}/confs_from_smiles_rdkit \
  --sdf_ext .sdf \
  --save_path ${SAVE_DIR} \
  --feature_workers 5 \
  --output_csv ${SAVE_DIR}/predictions.csv
