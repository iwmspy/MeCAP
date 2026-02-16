#!/bin/bash

CONDA_EV=~/miniconda3

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
)"

WORK_ROT="$(
  cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd
)"

cd ${WORK_ROT} || exit

SRCP_DIR=src

BASE_DIR=${WORK_ROT}/data/references
RESL_DIR=${WORK_ROT}/data/results
SAVE_DIR=${RESL_DIR}/casestudy/100_rxn_mechanisms

ENV_NAME=mecap
EXEC_PAT=${CONDA_EV}/envs/${ENV_NAME}/bin/python

source ${CONDA_EV}/etc/profile.d/conda.sh || exit
conda activate ${ENV_NAME}

cd ${SRCP_DIR} || exit


# ${EXEC_PAT} -m identify_sites \
#     --input ${BASE_DIR}/100_rxn_mechanisms/100_rxn_mechanisms_no_error_reactants.csv \
#     --smiles-col reactant_smi --name-col reactant_id_no_underbar \
#     --output-elec ${SAVE_DIR}/maa/100_rxn_mechanisms_no_error_reactants_elec.csv \
#     --output-nuc ${SAVE_DIR}/mca/100_rxn_mechanisms_no_error_reactants_nuc.csv \
#     --output-unique ${SAVE_DIR}/unique/100_rxn_mechanisms_no_error_reactants_unique.csv \


# ${EXEC_PAT} -m optimize_conf \
#   --input-csv ${SAVE_DIR}/unique/100_rxn_mechanisms_no_error_reactants_unique.csv \
#   --name-col name \
#   --smiles-col smiles \
#   --init-mode unimol \
#   --final-mode rdkit \
#   --max-workers 8 \
#   --out-dir ${SAVE_DIR}/confs \
#   --out-csv ${SAVE_DIR}/unique/100_rxn_mechanisms_no_error_reactants_unique_optimized.csv \


${EXEC_PAT} -m predict \
  --data ${SAVE_DIR}/maa/100_rxn_mechanisms_no_error_reactants_elec.csv \
  --checkpoint ${RESL_DIR}/mecap_ref_maa_v2_layer_0/best_model.pt \
  --atom_index_col elec_sites \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${SAVE_DIR}/confs \
  --sdf_ext .sdf \
  --save_path ${SAVE_DIR}/maa_v2 \
  --output_csv ${SAVE_DIR}/maa_v2/predictions.csv


${EXEC_PAT} -m predict \
  --data ${SAVE_DIR}/mca/100_rxn_mechanisms_no_error_reactants_nuc.csv \
  --checkpoint ${RESL_DIR}/mecap_ref_mca_v2_layer_0/best_model.pt \
  --atom_index_col nuc_sites \
  --sdf_name_col name \
  --sdf_mode per_row \
  --sdf_dir ${SAVE_DIR}/confs \
  --sdf_ext .sdf \
  --save_path ${SAVE_DIR}/mca_v2 \
  --output_csv ${SAVE_DIR}/mca_v2/predictions.csv

