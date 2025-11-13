#!/bin/bash

CONDA_EV=~/miniconda3
WORK_ROT=~/work/UniMea_dev

cd ${WORK_ROT} || exit

LAYER=1
RUN_NAME=mecap_ref_maa_layer_${LAYER}
SRCP_DIR=src
RUN_MODE=train

BASE_DIR=${WORK_ROT}/data/references
RESL_DIR=${WORK_ROT}/data/results
SAVE_DIR=${RESL_DIR}/${RUN_NAME}

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}

cd ${SRCP_DIR} || exit

${EXEC_PAT} -m ${RUN_MODE} \
  --data ${BASE_DIR}/QMdata4ML/df_elec_x_with_name_fold.csv \
  --atom_index_col elec_sites \
  --target_cols MAA_values \
  --split_col Set_fold1 \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${RESL_DIR}/confs_from_smiles_rdkit \
  --sdf_ext .sdf \
  --batch_size 50 --epochs 50 --lr 1e-4 \
  --save_path ${SAVE_DIR} \
  --model_name unimolv1 \
  --atom_head_hidden_dim 512 \
  --scale \
  --feature_workers 5 \
