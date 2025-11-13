#!/bin/bash

CONDA_EV=~/miniconda3
WORK_ROT=~/work/UniMea_dev

cd ${WORK_ROT} || exit

SRCP_DIR=src
RUN_MODE=diverse_conf

RESL_DIR=${WORK_ROT}/data/results

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}

cd ${SRCP_DIR} || exit

${EXEC_PAT} -m ${RUN_MODE} \
  --input-dir ${RESL_DIR}/confs_from_smiles_rdkit \
  --output-dir ${RESL_DIR}/diversified_confs \
  --summary-csv ${RESL_DIR}/optimizing/diversified_confs_result.csv \
  --workers 20
  