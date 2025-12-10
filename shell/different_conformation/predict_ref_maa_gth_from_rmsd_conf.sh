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
RUN_NAME=mecap_ref_maa_gth_predict_by_rmsd_layer_${LAYER}
SRCP_DIR=src
RUN_MODE=predict

BASE_DIR=${WORK_ROT}/data/references
RESL_DIR=${WORK_ROT}/data/results
SAVE_DIR=${RESL_DIR}/different_conformation/${RUN_NAME}

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_LAUNCH_BLOCKING=1

cd ${SRCP_DIR} || exit

${EXEC_PAT} -m ${RUN_MODE} \
  --data ${BASE_DIR}/QMdata4ML/df_elec_x_with_name_fold_extracted.csv \
  --checkpoint ${RESL_DIR}/different_conformation/mecap_ref_maa_gth_layer_0/best_model.pt \
  --atom_index_col elec_sites \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${RESL_DIR}/rmsd_max \
  --sdf_ext .sdf \
  --save_path ${SAVE_DIR} \
  --feature_workers 5 \
  --output_csv ${SAVE_DIR}/predictions.csv
