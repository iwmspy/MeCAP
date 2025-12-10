#!/bin/bash

CONDA_EV=~/miniconda3

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
)"

WORK_ROT="$(
  cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd
)"

cd ${WORK_ROT} || exit

SRCP_DIR=src
RUN_MODE_1=diverse_conf
RUN_MODE_2=sanity_check_for_raw_data
RUN_MODE_3=extract_by_index
RUN_MODE_4=compute_rmsd_across_dirs

BASE_DIR=${WORK_ROT}/data/references
RESL_DIR=${WORK_ROT}/data/results

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}

cd ${SRCP_DIR} || exit

${EXEC_PAT} -m ${RUN_MODE_1} \
  --input-dir ${BASE_DIR}/confs_orca \
  --output-dir ${RESL_DIR}/rmsd_max \
  --summary-csv ${RESL_DIR}/optimizing/rmsd_max_result.csv \
  --min-rmsd 2.0 \
  --num-confs 200 \
  --workers 10

${EXEC_PAT} -m ${RUN_MODE_2} \
  --input-csv ${BASE_DIR}/QMdata4ML/df_ChEMBL50K.csv.gz \
  --input-dirs ${BASE_DIR}/confs_orca \
  --input-dirs-no-determine ${RESL_DIR}/rmsd_max \
  --out-csv ${RESL_DIR}/optimizing/sanity_check_smiles.csv

for CSV_PAT in 'elec' 'nuc'
do
${EXEC_PAT} -m ${RUN_MODE_3} \
  --input-csv ${BASE_DIR}/QMdata4ML/df_${CSV_PAT}_x_with_name_fold.csv \
  --index-file ${RESL_DIR}/optimizing/sanity_check_smiles.index \
  --out-csv ${BASE_DIR}/QMdata4ML/df_${CSV_PAT}_x_with_name_fold_extracted.csv
done

${EXEC_PAT} -m ${RUN_MODE_4} \
  ${BASE_DIR}/confs_orca \
  ${RESL_DIR}/confs_from_smiles_rdkit \
  ${RESL_DIR}/rmsd_max \
  -o ${RESL_DIR}/optimizing/conformation_rmsd.csv \
  -j 10 \
  --index-file ${RESL_DIR}/optimizing/sanity_check_smiles.index \
