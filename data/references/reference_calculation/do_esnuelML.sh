#!/bin/bash

CONDA_EV=~/miniconda3

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
)"

WORK_ROT="$(
  cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd
)"

cd ${WORK_ROT} || exit

ENV_NAME=esnuelML
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}

cd ${SRCP_DIR} || exit

for OPT in '' '_mopac' '_orca'
do

  ${EXEC_PAT}  ${SCRIPT_DIR}/esnuelML_trainer.py \
      --input-dataframe ${WORK_ROT}/references/df_nuc_sample_10000${OPT}_split.csv \
      --descriptor-column cm5 \
      --target-column MCA_values \
      --set-column split \
      --num-cpu 10 \
      --output-dir ${WORK_ROT}/results/different_metrics/esnuelML_mca${OPT} \
  
  ${EXEC_PAT}  ${SCRIPT_DIR}/esnuelML_trainer.py \
      --input-dataframe ${WORK_ROT}/references/df_elec_sample_10000${OPT}_split.csv \
      --descriptor-column cm5 \
      --target-column MAA_values \
      --set-column split \
      --num-cpu 10 \
      --output-dir ${WORK_ROT}/results/different_metrics/esnuelML_maa${OPT} \

done
