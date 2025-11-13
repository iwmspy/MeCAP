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
Batch xTB optimization from SDF directory.
All comments are in English.

- Enumerates all *.sdf (case-insensitive) files under --input-dir (recursively).
- For each SDF, runs two-stage optimization: GFN-FF followed by GFN1.
- Writes optimized SDFs with the same filenames under --out-dir.
- Preserves SDF properties.
- Uses xTB from the active conda environment (prefers $XTBHOME/bin/xtb if set).
"""

import argparse
import sys
from pathlib import Path

from core_modules.conformer import xtb_optimize_sdf_dir  # noqa: F401


# ---------- CLI ----------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch runner that optimizes all SDFs in a directory by xTB (GFN-FF -> GFN1)."
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing SDF files; search is recursive for *.sdf and *.SDF.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Destination directory for optimized SDF files.",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to write a CSV summary of results.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of SDF files to process in parallel (process-based).",
    )
    p.add_argument(
        "--xtb-threads",
        type=int,
        default=2,
        help="Threads per xTB process (exported to OMP_NUM_THREADS and MKL_NUM_THREADS).",
    )
    p.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep per-file working directories for debugging.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[error] Input dir does not exist or is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = xtb_optimize_sdf_dir(
        in_dir=str(input_dir),
        out_dir=str(out_dir),
        max_workers=args.max_workers,
        xtb_threads=args.xtb_threads,
        keep_work=args.keep_work,
        verbose=args.verbose,
    )

    if args.out_csv:
        out_csv_path = Path(args.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False, encoding="utf-8")
        if args.verbose:
            print(f"[done] Wrote CSV: {out_csv_path.resolve()}")

    # Exit code policy:
    # - 0 if at least one file succeeded or partially succeeded.
    # - 1 if all files failed.
    try:
        statuses = set(str(s).lower() for s in df.get("status", []))
    except Exception:
        statuses = set()
    if statuses and statuses != {"fail"}:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
