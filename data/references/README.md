# Uni-Mol-MeCAP `data/references`

This directory contains reference datasets, intermediate files, and helper scripts used to prepare, validate, split, and analyze the benchmark data used in the MeCAP workflow.

The repository already indicates that the main entry point in this directory is `data_preparation.ipynb`, which is used to download and preprocess the dataset. In practice, this is the first file you should open and run from top to bottom.

## Directory structure

```text
references/
├── 100_rxn_mechanisms/
│   ├── 100_rxn_mechanisms.csv
│   ├── 100_rxn_mechanisms_no_error.csv
│   ├── 100_rxn_mechanisms_no_error_reactants.csv
│   ├── ... calculated/predicted SDF and CSV files
│   └── ... auxiliary text files
├── gaussian_sp/
│   └── extracted_energies/
│       ├── df_elec_wb97xd_tzvp.csv
│       └── df_nuc_wb97xd_tzvp.csv
├── mopac_optimized_confs/
├── orca_optimized_confs/
├── reference_calculation/
│   ├── do_esnuelML.sh
│   ├── esnuelML_predictor.py
│   ├── esnuelML_trainer.py
│   ├── esnuel_calculator.py
│   ├── predictor_output_df.py
│   ├── reference_calculation.ipynb
│   └── run_orca.py
├── _ConvertOrcaToSDF.py
├── _ParseMopacOut.py
├── _ScaffoldSplit.py
├── _rdkitAtomWiseFP.py
├── data_analysis.ipynb
├── data_preparation.ipynb
├── df_elec_sample_10000.csv.gz
├── df_elec_sample_10000_mopac.csv
├── df_elec_sample_10000_mopac_split.csv
├── df_elec_sample_10000_orca.csv
├── df_elec_sample_10000_orca_split.csv
├── df_elec_sample_10000_sanity_mopac.csv
├── df_elec_sample_10000_sanity_orca.csv
├── df_elec_sample_10000_split.csv
├── df_nuc_sample_10000.csv.gz
├── df_nuc_sample_10000_mopac.csv
├── df_nuc_sample_10000_mopac_split.csv
├── df_nuc_sample_10000_orca.csv
├── df_nuc_sample_10000_orca_split.csv
├── df_nuc_sample_10000_sanity_mopac.csv
├── df_nuc_sample_10000_sanity_orca.csv
├── df_nuc_sample_10000_split.csv
├── df_sample_10000.csv.gz
├── df_sample_10000_sanity_mopac.csv
├── df_sample_10000_sanity_orca.csv
├── esnuelML_elec_predicted_values.csv
├── esnuelML_nuc_predicted_values.csv
├── files_to_analyze.txt
├── scaffold_map.csv
├── split.csv
├── structure_file_10000.txt
└── README.md
```

## What is in this directory?

### Core notebooks

- `data_preparation.ipynb`  
  Main starting point for downloading and preprocessing the reference dataset.

- `data_analysis.ipynb`  
  Notebook for inspecting the processed data and analyzing the prepared feature tables.

### Reference calculation workflow

- `reference_calculation/`  
  Scripts and a notebook related to the reference calculation workflow, including eSNUELML-related utilities and ORCA execution helpers.

### Geometry and energy-related resources

- `mopac_optimized_confs/`  
  Files related to MOPAC-optimized conformers.

- `orca_optimized_confs/`  
  Files related to ORCA-optimized conformers.

- `gaussian_sp/extracted_energies/`  
  Extracted Gaussian single-point energy tables for electrophilic and nucleophilic datasets.

### Prepared datasets

The `df_*` CSV files are prepared tables for different subsets and processing stages, including:

- electrophilic (`df_elec_*`)
- nucleophilic (`df_nuc_*`)
- combined or shared sample tables (`df_sample_*`)
- MOPAC-based variants (`*_mopac*`)
- ORCA-based variants (`*_orca*`)
- split-assigned variants (`*_split*`)
- sanity-check variants (`*_sanity_*`)

### Splitting and feature utilities

- `_ScaffoldSplit.py`  
  Helper script for scaffold-based splitting.

- `_rdkitAtomWiseFP.py`  
  Helper script for RDKit atom-wise fingerprint generation.

- `_ConvertOrcaToSDF.py`  
  Utility for converting ORCA outputs into SDF-like structures.

- `_ParseMopacOut.py`  
  Utility for parsing MOPAC output files.

### Additional reference and benchmark files

- `100_rxn_mechanisms/`  
  Benchmark-style reaction mechanism files, including filtered reactant lists and calculated/predicted outputs.

- `esnuelML_elec_predicted_values.csv` / `esnuelML_nuc_predicted_values.csv`  
  Stored prediction outputs from the reference eSNUELML workflow.

- `split.csv`, `scaffold_map.csv`  
  Split definitions and scaffold mapping tables.

- `files_to_analyze.txt`, `structure_file_10000.txt`  
  Input or bookkeeping files used during preprocessing and analysis.

## What should you do first?

If you are using this directory for the first time, start here:

1. Open `data_preparation.ipynb`.
2. Run all cells from top to bottom.
3. Confirm that the expected processed CSV files are generated or updated.
4. After preprocessing is complete, move to `data_analysis.ipynb` if you want to inspect the dataset or reproduce downstream analysis.
