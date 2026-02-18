#!/usr/bin/env python3
"""
ESNUEL Calculator Subprocess Wrapper

This module provides functionality to run ESNUEL/src/esnuel/calculator.py
as a subprocess with command-line arguments.

Adds a per-molecule wall-clock timeout so that if an ESNUEL run exceeds
the specified duration, it will be marked as failed and the wrapper will
move on to the next SMILES.
"""

import argparse
import subprocess
import sys
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import tempfile
from tqdm import tqdm
import pandas as pd
from rdkit import Chem

base_dir = os.path.dirname(os.path.realpath(__file__))
es_dir   = os.path.join(base_dir, 'ESNUEL_ML')   # Original
base_pred_dir = os.path.join(es_dir,'desc_calcs')

import tarfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

props = {
    'MAA': 'elec',
    'MCA': 'nuc',
    }

def save_sdf(mols,save_name):
    with Chem.SDWriter(save_name if save_name.endswith('.sdf') else save_name + '.sdf') as writer:
        for mol in mols:            
            writer.write(mol)

def SetAtomPropsToMol(mol: Chem.Mol):
    atom_props_set = sorted(list(set(sum([list(atom.GetPropNames()) for atom in mol.GetAtoms()],[]))))
    for atom_prop in atom_props_set:
        try:
            Chem.CreateAtomDoublePropertyList(mol, atom_prop)
        except:
            try:
                Chem.CreateAtomBoolPropertyList(mol, atom_prop)
            except:
                try:
                    Chem.CreateAtomIntPropertyList(mol, atom_prop)
                except:
                    # Replace spaces with underscores in string property values
                    for atom in mol.GetAtoms():
                        if atom.HasProp(atom_prop):
                            prop_value = atom.GetProp(atom_prop)
                            if isinstance(prop_value, str) and ' ' in prop_value:
                                atom.SetProp(atom_prop, prop_value.replace(' ', '_'))
                    Chem.CreateAtomStringPropertyList(mol, atom_prop)

def parse_args():
    """
    Argument parser so this can be run from the command line
    """
    parser = argparse.ArgumentParser(description='Run ESNUEL from the command line')
    parser.add_argument('-s', '--smiles', default='C[C+:20](C)CC(C)(C)C1=C(C=CC(=C1)Br)[OH:10]',
                        help='SMILES input to ESNUEL')
    parser.add_argument('-n', '--name', default='testmol', help='The name of the molecule or job depending on CLI or batch mode. Only names without "_" are allowed.')
    parser.add_argument('-b', '--batch', default=None, help='Path to .csv file for running batched calculations e.g. --> python src/esnuel/calculator.py -b example/testmols.csv')
    parser.add_argument('-c', '--batch_smiles_col', default='smiles', help='Name of SMILES column of batch file.')
    parser.add_argument('-i', '--batch_index_col', default='name', help='Name of index column of batch file.')
    parser.add_argument('-a', '--save_name', default=None, help='Path to .txt file for saving results.')
    parser.add_argument('--parallel_calcs', default=2, help='The number of parallel molecule calculations (the total number of CPU cores requested for each SLURM job = parallel_calcs*cpus_per_calc)')
    # Wrapper-only: per-molecule wall-clock timeout for the ESNUEL subprocess (in seconds)
    parser.add_argument('--per_mol_timeout_sec', type=int, default=None,
                        help='Wall-clock timeout in seconds for each SMILES calculation in this wrapper. If exceeded, skip to the next SMILES.')
    return parser.parse_args()

def strip_links(html_text):
    # Capture <a> and return the inner text
    return re.sub(r'<a\b[^>]*>(.*?)</a>', r'\1', html_text, flags=re.IGNORECASE|re.DOTALL)

def build_esnuelML_command(args: argparse.Namespace) -> List[str]:
    """
    Build command line for ESNUEL calculator.

    Args:
        args: Parsed arguments

    Returns:
        List[str]: Command line arguments
    """
    cmd = [os.path.join(sys.prefix,'bin','python'), os.path.join(es_dir, 'src', 'esnuelML', 'predictor_output_df.py')]

    # Do not forward wrapper-only arguments to ESNUEL CLI
    esnuelml_input = {'name','smiles'}
    for key, arg in args.__dict__.items():
        if key in esnuelml_input:
            if arg is not None:
                cmd.extend([f'--{key}', str(arg)])

    return cmd

def run_esnuel_subprocess(cmd: List[str],
                          timeout: Optional[int] = None,
                          verbose: bool = False) -> subprocess.CompletedProcess:
    """
    Run ESNUEL calculator as subprocess.

    Args:
        cmd: Command line arguments
        timeout: Optional wall-clock timeout in seconds for the subprocess
        verbose: Enable verbose output

    Returns:
        subprocess.CompletedProcess: Result of subprocess execution
    """
    if verbose:
        print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Do not raise on non-zero return code
            timeout=timeout
        )

        if verbose:
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")

        return result

    except subprocess.TimeoutExpired as e:
        # Normalize possibly-bytes outputs from TimeoutExpired to text
        def _to_text(x) -> str:
            if x is None:
                return ""
            return x.decode(errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)
        # Note: TimeoutExpired provides .output (stdout) and .stderr
        t_stdout = _to_text(getattr(e, "output", None))
        t_stderr = _to_text(getattr(e, "stderr", None))
        if verbose:
            print(f"Subprocess timed out after {timeout} seconds")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,  # Common convention for timeout
            stdout=t_stdout,
            stderr=t_stderr + (f"\nTimeout after {timeout} seconds" if timeout else "\nTimeout")
        )
    except Exception as e:
        print(f"ERROR: Failed to run subprocess: {e}")
        raise


def process_results(result: subprocess.CompletedProcess,
                   args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process and format subprocess results.

    Args:
        result: Subprocess result
        args: Original arguments

    Returns:
        Dict[str, Any]: Processed results
    """
    output_data = {
        'success': result.returncode == 0,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'command': ' '.join(result.args) if hasattr(result, 'args') else None,
    }

    # Try to parse output if it is JSON
    if result.stdout:
        try:
            output_data['parsed_output'] = json.loads(result.stdout)
        except json.JSONDecodeError:
            output_data['parsed_output'] = None

    return output_data

def concatenate_df(df_origin, df_calc):
    com_cols = set(df_origin.columns) & set(df_calc.columns)
    df_origin_res = df_origin.drop(columns=com_cols)
    return pd.concat([df_calc,df_origin_res], axis=1)

def convert_results_to_dict(mol: Chem.Mol, path, result):
    res_dict = {}
    if result['success']:
        print("ESNUEL calculation completed successfully!")
        raise_some_error = False
        # For molecules, the results are saved into sdf file.
        for prop in props:
            if os.path.exists(f'{path}_{prop}.pkl'):
                res = pd.read_pickle(f'{path}_{prop}.pkl')
                for _, row in res.iterrows():
                    atom = mol.GetAtomWithIdx(int(row['Atom ID']))
                    atom.SetDoubleProp(props[prop].lower(), float(row[f'{prop} Value [kJ/mol]']))
                    atom.SetProp(f'{props[prop].lower()}_type', '_'.join(str(row['Type']).split()))
                    atom.SetDoubleProp(f'{props[prop].lower()}_est_error', float(row['Est. Error [kJ/mol]']))
                res_dict[f'{prop.lower()}_strings'] = strip_links(str(res.set_index('Atom ID',drop=True).T.to_dict()))
            else:
                raise_some_error = True
                res_dict[f'{prop.lower()}_strings'] = None
        SetAtomPropsToMol(mol)
        res_dict['error'] = 'Failed to create some result file..' if raise_some_error else None
    else:
        print(f"ESNUEL calculation failed with return code: {result['return_code']}")
        for prop in props:
            res_dict[f'{prop.lower()}_strings'] = None
        if result['stderr']:
            print(f"Error message: {result['stderr']}")
            res_dict['error'] = str(result['stderr']).replace('\n','\t')
        else:
            res_dict['error'] = 'Error occurred while processing...'
    return mol, res_dict


def process_single_molecule(name, smiles, sdir, args_dict, per_mol_timeout_sec):
    """Process a single molecule in parallel"""
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp('_Name', str(name))
    df_local = pd.DataFrame({
    args_dict['batch_smiles_col']: smiles
    }, index=[name])

    # Build command
    args_local = argparse.Namespace(**args_dict)
    args_local.smiles = smiles
    args_local.name = name
    cmd = build_esnuelML_command(args_local)

    # Run subprocess with optional per-molecule timeout
    result = run_esnuel_subprocess(
    cmd,
    timeout=per_mol_timeout_sec,
    )

    # Process results
    result = process_results(result, args_local)

    # Convert results
    mol, res_dict = convert_results_to_dict(mol, sdir, result)
    for key, item in res_dict.items():
        df_local[key] = item

    return mol, df_local, result['return_code']


def main(args):
    """
    Main function to run ESNUEL calculator subprocess.
    """
    args_local = argparse.Namespace(
        smiles=None,
        name=None,
        batch=None,
        partition='kemi1',
        parallel_calcs=args.parallel_calcs,
        # Note: per_mol_timeout_sec is wrapper-only and is not forwarded to ESNUEL
    )

    if args.batch is not None:
        df = pd.read_csv(args.batch)
        save_name = args.batch + '_calc.txt'
        n_list = [str(n) for n in df[args.batch_index_col].to_list()]
        s_list = [str(s) for s in df[args.batch_smiles_col].to_list()]
    else:
        save_name = os.path.join(base_pred_dir, args.name + '_calc.txt')
        n_list = [args.name]
        s_list = [args.smiles]
    if args.save_name is not None:
        save_name = args.save_name

    save_dir = Path(save_name)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    d_list = [os.path.join(base_pred_dir, name, name) for name in n_list]
    h_list = [f'{d}.html' for d in d_list]

    try:
        mols = []
        dfs  = []
        pcodes = []

        # Prepare args_dict for parallel processing
        args_dict = vars(args_local).copy()
        args_dict['batch_smiles_col'] = args.batch_smiles_col
        per_mol_timeout = getattr(args, 'per_mol_timeout_sec', None)
        
        # Run in parallel
        max_workers = int(args.parallel_calcs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
            executor.submit(process_single_molecule, name, smiles, sdir, args_dict, per_mol_timeout): 
            (name, smiles, sdir) for name, smiles, sdir in zip(n_list, s_list, d_list)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='Predicting MAA/MCA..'):
                mol, df_local, return_code = future.result()
                mols.append(mol)
                dfs.append(df_local)
                pcodes.append(return_code)

        df_calc = pd.concat(dfs)
        if args.batch is not None:
            df_calc.index.name = args.batch_index_col
            df_calc = concatenate_df(df.set_index(args.batch_index_col, drop=True), df_calc)
        df_calc.to_csv(save_name)

        save_sdf(mols, save_name)

    except Exception as e:
        print(f"ERROR: {e}")
        pcodes.append(1)

    finally:
        # Create tar.gz archive with all directories and HTML files
        archive_name = f"{save_name.replace('.txt', '')}_results.tar.gz"

        with tarfile.open(archive_name, 'w:gz') as tar:
            # Add directories
            for d_path in d_list:
                if os.path.exists(d_path):
                    tar.add(d_path, arcname=os.path.basename(d_path))
                for prop in props:
                    p_path = d_path + f'_{prop}.pkl'
                    if os.path.exists(p_path):
                        tar.add(p_path, arcname=os.path.basename(p_path))

            # Add HTML files
            for h_path in h_list:
                if os.path.exists(h_path):
                    tar.add(h_path, arcname=os.path.basename(h_path))

        print(f"Created archive: {archive_name}")

        # Remove all directories and files
        for d_path in d_list:
            if os.path.exists(d_path):
                shutil.rmtree(d_path)
                print(f"Removed directory: {d_path}")
                for prop in props:
                    p_path = d_path + f'_{prop}.pkl'
                    if os.path.exists(p_path):
                        os.remove(p_path)
                        print(f"Removed file: {p_path}")

        for h_path in h_list:
            if os.path.exists(h_path):
                os.remove(h_path)
                print(f"Removed file: {h_path}")

        sys.exit(max(pcodes))

if __name__ == "__main__":
    args = parse_args()
    main(args)
