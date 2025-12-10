#!/bin/bash

CONDA_EV=~/miniconda3

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
)"

WORK_ROT="$(
  cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd
)"

cd ${WORK_ROT} || exit

RUN_NAME=optimizing
SRCP_DIR=src
RUN_MODE=further_optimize_conf_by_xTB

BASE_DIR=${WORK_ROT}/data/references
RESL_DIR=${WORK_ROT}/data/results

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}

cd ${SRCP_DIR} || exit

${EXEC_PAT} -m ${RUN_MODE} \
  --input-dir ${RESL_DIR}/confs_from_smiles_rdkit \
  --out-dir ${RESL_DIR}/confs_from_smiles_xtb \
  --max-workers 12 \
  --out-csv ${RESL_DIR}/${RUN_NAME}/df_ChEMBL50K_xtb_opt.csv \
  --verbose \
