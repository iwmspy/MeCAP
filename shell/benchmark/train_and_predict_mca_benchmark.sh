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
RUN_NAME=mecap_mca_benchmark_finetune_layer_${LAYER}
SRCP_DIR=src

BASE_DIR=${WORK_ROT}/data/references/benchmark
RESL_DIR=${WORK_ROT}/data/results
SAVE_DIR=${RESL_DIR}/benchmark

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_LAUNCH_BLOCKING=1

cd ${SRCP_DIR} || exit

# for FOLD_NUM in '0' '1' '2' '3' '4'
# do

# RUN_MODE=train

# ${EXEC_PAT} -m ${RUN_MODE} \
#   --data ${BASE_DIR}/nucleophilicity_fold${FOLD_NUM}.csv \
#   --atom_index_col atomIdx \
#   --target_cols 'MCA_values' \
#   --split_col split \
#   --sdf_name_col name \
#   --sdf_mode per_row \
#   --sdf_dir ${SAVE_DIR}/confs_from_smiles_rdkit \
#   --sdf_ext .sdf \
#   --batch_size 50 --epochs 300 --lr 1e-4 \
#   --resume_checkpoint ${RESL_DIR}/mecap_ref_mca_v2_layer_0/best_model.pt \
#   --save_path ${SAVE_DIR}/${RUN_NAME}/fold${FOLD_NUM} \
#   --model_name unimolv2 \
#   --scale \
#   --feature_workers 5 \

# wait

# RUN_MODE=predict

# ${EXEC_PAT} -m ${RUN_MODE} \
#   --data ${BASE_DIR}/nucleophilicity_fold${FOLD_NUM}.csv \
#   --checkpoint ${SAVE_DIR}/${RUN_NAME}/fold${FOLD_NUM}/best_model.pt \
#   --atom_index_col atomIdx \
#   --sdf_name_col name \
#   --sdf_mode per_row \
#   --sdf_dir ${SAVE_DIR}/confs_from_smiles_rdkit \
#   --sdf_ext .sdf \
#   --save_path ${SAVE_DIR} \
#   --feature_workers 5 \
#   --output_csv ${SAVE_DIR}/${RUN_NAME}/fold${FOLD_NUM}/predictions.csv

# wait

# done

RUN_MODE=train

${EXEC_PAT} -m ${RUN_MODE} \
  --data ${BASE_DIR}/nucleophilicity_whole.csv \
  --atom_index_col atomIdx \
  --target_cols 'MCA_values' \
  --split_col split \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${SAVE_DIR}/confs_from_smiles_rdkit \
  --sdf_ext .sdf \
  --batch_size 50 --epochs 300 --lr 1e-4 \
  --resume_checkpoint ${RESL_DIR}/mecap_ref_mca_v2_layer_0/best_model.pt \
  --save_path ${SAVE_DIR}/${RUN_NAME}/whole \
  --model_name unimolv2 \
  --scale \
  --feature_workers 5 \

wait

RUN_MODE=predict

${EXEC_PAT} -m ${RUN_MODE} \
  --data ${BASE_DIR}/nucleophilicity_whole.csv \
  --checkpoint ${SAVE_DIR}/${RUN_NAME}/whole/best_model.pt \
  --atom_index_col atomIdx \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${SAVE_DIR}/confs_from_smiles_rdkit \
  --sdf_ext .sdf \
  --save_path ${SAVE_DIR} \
  --feature_workers 5 \
  --output_csv ${SAVE_DIR}/${RUN_NAME}/whole/predictions.csv
