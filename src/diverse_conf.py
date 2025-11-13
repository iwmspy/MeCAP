# MIT License
#
# Copyright (c) 2025 Yuto Iwasaki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Generate far conformers from reference optimized conformations in SDF files.

Overview:
- Read all .sdf files under an input directory with explicit hydrogens retained.
- For each file, read all molecules and try to generate a "far" conformer
  relative to the first conformer (assumed optimized reference).
- Write only the selected far conformers (one per input molecule) into an
  output SDF with the same filename under the output directory.
- Emit a CSV summary with columns: filename, success, rmsd.
  success = True if at least one molecule in the file succeeded.
  rmsd = maximum RMSD among successfully processed molecules in the file
         (empty if no molecule succeeded).

Notes:
- Hydrogens are always kept as in the input SDF (removeHs=False). The script never removes hydrogens.
- RMSD is computed on heavy atoms by default unless --all-atoms is given.
- Parallelization is file-level using ProcessPoolExecutor.
"""

import os
import sys
import argparse
import csv
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from core_modules.conformer import diverse_conf_from_sdf_file

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def _process_file_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small wrapper for multiprocessing to process a single file.
    """
    try:
        success, agg_rmsd, error_log = diverse_conf_from_sdf_file(
            infile=job["infile"],
            outfile=job["outfile"],
            min_rmsd=job["min_rmsd"],
            num_confs=job["num_confs"],
            random_seed=job["random_seed"],
        )
        return {
            "filename": job["filename"],
            "success": bool(success),
            "rmsd": (f"{agg_rmsd:.6f}" if (agg_rmsd is not None) else ""),
            "error": error_log
        }
    except Exception:
        # Best-effort clean-up for this outfile
        try:
            if os.path.exists(job["outfile"]):
                os.remove(job["outfile"])
        except Exception:
            pass
        return {"filename": job["filename"], "success": False, "rmsd": "", "error": error_log}


def main():
    parser = argparse.ArgumentParser(
        description="Generate far conformers relative to reference optimized conformations for all SDF files in a directory (hydrogens always kept)."
    )
    parser.add_argument("--input-dir", required=True, help="Input directory containing .sdf files.")
    parser.add_argument("--output-dir", required=True, help="Output directory to write processed .sdf files.")
    parser.add_argument("--summary-csv", required=True, help="Path to write the summary CSV.")
    parser.add_argument("--min-rmsd", type=float, default=1.0, help="Target minimum RMSD in Angstrom.")
    parser.add_argument("--num-confs", type=int, default=20, help="Number of trial conformers per attempt.")
    parser.add_argument("--random-seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count(),8) or 1,
        help="Number of parallel worker processes (default: number of CPU cores)."
    )

    args = parser.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    csv_path = os.path.abspath(args.summary_csv)

    if not os.path.isdir(in_dir):
        print(f"Error: input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    sdf_files = [f for f in os.listdir(in_dir) if f.lower().endswith(".sdf")]
    sdf_files.sort()

    jobs: List[Dict[str, Any]] = []
    for fname in sdf_files:
        infile = os.path.join(in_dir, fname)
        outfile = os.path.join(out_dir, fname)
        jobs.append({
            "filename": fname,
            "infile": infile,
            "outfile": outfile,
            "min_rmsd": args.min_rmsd,
            "num_confs": args.num_confs,
            "random_seed": args.random_seed,
        })

    results: List[Dict[str, Any]] = []

    # Parallel execution per file
    if len(jobs) > 0:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        # with ProcessPoolExecutor(max_workers=1) as ex:
            futures = {ex.submit(_process_file_job, job): job["filename"] for job in jobs}
            suc, fal = 0, 0
            total = len(jobs)
            for fut in as_completed(futures):
                res = fut.result()
                if res['success']:
                    print(f"[ok] name={res['filename']}")
                    suc += 1
                else:
                    print(f"[fail] name={res['filename']}")
                    fal += 1
                results.append(res)
                prog = f"{suc + fal}/{total} done, {fal} failed"
                pct = 100.0 * (suc + fal) / total
                print(f"[prog] {prog} ({pct:.1f}%)")

    # Keep CSV ordered by filename
    results.sort(key=lambda d: d["filename"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "success", "rmsd", "error"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
