#!/bin/bash

CONDA_EV=~/miniconda3

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
)"

WORK_ROT="$(
  cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd
)"

cd ${WORK_ROT} || exit

LAYER=0
RUN_NAME=mecap_maa_mmff_layer_${LAYER}
SRCP_DIR=src
RUN_MODE=train

BASE_DIR=${WORK_ROT}/data/references
RESL_DIR=${WORK_ROT}/data/results
SAVE_DIR=${RESL_DIR}/different_metrics/${RUN_NAME}

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_LAUNCH_BLOCKING=1

cd ${SRCP_DIR} || exit

${EXEC_PAT} -m ${RUN_MODE} \
  --data ${BASE_DIR}/df_elec_sample_10000_split.csv \
  --atom_index_col elec_sites \
  --target_cols MAA_values \
  --split_col split \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${RESL_DIR}/confs_from_smiles_rdkit \
  --sdf_ext .sdf \
  --batch_size 50 --epochs 300 --lr 1e-4 \
  --save_path ${SAVE_DIR} \
  --model_name unimolv1 \
  --scale \
  --feature_workers 5 \

wait

RUN_MODE=predict

${EXEC_PAT} -m ${RUN_MODE} \
  --data ${BASE_DIR}/df_elec_sample_10000_split.csv \
  --checkpoint ${SAVE_DIR}/best_model.pt \
  --atom_index_col elec_sites \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${RESL_DIR}/confs_from_smiles_rdkit \
  --sdf_ext .sdf \
  --save_path ${SAVE_DIR} \
  --feature_workers 5 \
  --output_csv ${SAVE_DIR}/predictions.csv
