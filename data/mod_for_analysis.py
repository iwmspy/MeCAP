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

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

from typing import Iterable, Optional, Tuple, Union, Mapping, Any, Dict, Set, List
import ast, io, math, os, re
from collections import deque
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
from PIL import Image, ImageDraw, ImageFont

from core_modules.atom import find_sites, extracted_n_smirks_dict, extracted_e_smirks_dict, added_n_smirks_dict, added_e_smirks_dict

s_params = Chem.SubstructMatchParameters()
s_params.numThreads = min(8, os.cpu_count())
s_params.uniquify = False
thu_aln_smirks_dict = {'thiourea': '[N;X3;H0,H1,H2:2]-[C;X3;v4:1](=[S;X1;v2:3])-[N;X3;H0,H1,H2:4]',
                       'allene': '[C;X2;v4:1](=[!#1:2])=[!#1:3]',
                      }
element_dict        = {'thiourea': 1,
                       'allene': 1,
                      }
substructure_match = lambda mol, submol: mol.GetSubstructMatches(submol,s_params)


def load_csv(path):
    return pd.read_csv(path, index_col=0)

def load_pred_value(path: str):
    l = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l.append(float(line))
    return l

def calculate_stats(x,y):
    r, p = pearsonr(x, y)
    r2 = r2_score(x,y)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    return r, r2, rmse

def stats_and_plot(x, y, hue=None, ax=None, x_label=None, y_label=None, title=None, alpha=0.3, visualize_stats=True):
    """Compute Pearson r, RMSE, and generate a scatter plot."""
    r, r2, rmse = calculate_stats(x, y)

    if ax is None:
        plot, ax = plt.subplots(figsize=(5,5))

    if hue is not None:
        hue_order = sorted(np.unique(hue).tolist())
        sns.scatterplot(x=x, y=y, hue=hue, hue_order=hue_order, alpha=alpha, ax=ax, palette="deep")
    else:
        ax.scatter(x, y, alpha=alpha)

    # Reference y=x line
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--", alpha=0.7)
    ax.set_xlim(lo-50, hi+50)
    ax.set_ylim(lo-50, hi+50)
    ax.set_xlabel("$x$" if x_label is None else x_label)
    ax.set_ylabel("$y$" if y_label is None else y_label)
    if visualize_stats: title = f"{title}\n$r$={r:.3f}, R2={r2:.3f}, RMSE={rmse:.3f}"
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    return ax, r, r2, rmse

def _pca_angle_deg_xy(coords):
    """Return angle (deg) of the first principal axis w.r.t. +x, for 2D coords."""
    # coords: list[(x,y)]
    cx = sum(x for x,_ in coords)/len(coords)
    cy = sum(y for _,y in coords)/len(coords)
    xs = [x-cx for x,_ in coords]
    ys = [y-cy for _,y in coords]
    # 2x2 covariance matrix
    sxx = sum(x*x for x in xs); syy = sum(y*y for y in ys); sxy = sum(x*y for x,y in zip(xs,ys))
    # eigenvector for largest eigenvalue of [[sxx,sxy],[sxy,syy]]
    # Closed-form using atan2 of the eigenvector direction
    # tan(2θ) = 2sxy / (sxx - syy)
    theta = 0.5*math.atan2(2*sxy, (sxx - syy))  # radians
    return math.degrees(theta)

def _rotate_mol2d_inplace(mol, angle_deg, about_centroid=True):
    """Rotate 2D conformer positions (z kept) around centroid by angle_deg."""
    if mol.GetNumConformers() == 0:
        return
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pts = [conf.GetAtomPosition(i) for i in range(n)]
    if about_centroid:
        cx = sum(p.x for p in pts)/n
        cy = sum(p.y for p in pts)/n
    else:
        cx = cy = 0.0
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    for i, p in enumerate(pts):
        x, y = p.x - cx, p.y - cy
        xr =  x*c - y*s
        yr =  x*s + y*c
        conf.SetAtomPosition(i, Point3D(xr+cx, yr+cy, p.z))

def visualize_smiles_highlight(
    smiles: str,
    atom_indices: Union[int, Iterable[int]],
    qm_value: float,
    pred_value: float,
    image_size: Tuple[int, int] = (500, 350),
    caption_fmt: str = "QM-based MCA: {} [kJ/mol]\nPredicted MCA: {} [kJ/mol]",
    caption_extra: Optional[Union[str, Mapping[str, object], Iterable[Tuple[str, object]]]] = None,
    highlight_color: Tuple[int, int, int, int] = (242, 64, 64, 180),
    highlight_radius_px: int = 16,
    highlight_thickness_px: int = 6,
    save_path: Optional[str] = None,
    return_ignored: bool = False,
    auto_rotate: bool = True,
    rotate_deg: float = 0.0,
    rdkit_padding: float = 0.02,   # smaller = tighter fit; default RDKit ≈ 0.05
    font_size = 18,
):
    """
    Draw a SMILES with highlighted atoms, optional PCA-based auto-rotation, and a caption.

    - auto_rotate=True: aligns the longest principal axis to best match canvas aspect
                        (portrait canvas → make molecule tall; landscape → make it wide).
    - rotate_deg: additional manual rotation (degrees, applied after auto-rotation).
    - rdkit_padding: shrink this to make the molecule larger on the canvas.
    """
    # Parse + 2D coords
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)

    # Normalize indices
    if isinstance(atom_indices, int):
        idx_list = [atom_indices]
    else:
        idx_list = [int(i) for i in atom_indices]
    n_atoms = mol.GetNumAtoms()
    valid = [i for i in idx_list if 0 <= i < n_atoms]
    ignored = [i for i in idx_list if not (0 <= i < n_atoms)]

    # Auto-rotate to “nice” orientation using PCA
    if auto_rotate and n_atoms >= 2 and mol.GetNumConformers():
        conf = mol.GetConformer()
        coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(n_atoms)]
        angle_pca = _pca_angle_deg_xy(coords)  # angle vs +x
        W, H = image_size
        # Make the longest axis align with the longer side of the canvas
        # If canvas is portrait, we want the longest axis vertical (≈ +90° from +x).
        target = 90.0 if H >= W else 0.0
        _rotate_mol2d_inplace(mol, target - angle_pca)

    # Optional extra manual rotation
    if abs(rotate_deg) > 1e-6:
        _rotate_mol2d_inplace(mol, rotate_deg)

    # RDKit drawing (tight padding to avoid tiny molecules)
    w, h = image_size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.padding = rdkit_padding       # tighter fit -> molecule appears larger
    opts.bondLineWidth = 2
    opts.addStereoAnnotation = False

    atom_colors = {i: (1.0, 0.2, 0.2) for i in valid} if valid else None
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=valid if valid else None,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    base = Image.open(io.BytesIO(png)).convert("RGBA")

    # Pixel-precise highlight circles
    if valid:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for i in valid:
            x, y = drawer.GetDrawCoords(i)
            l = x - highlight_radius_px; t = y - highlight_radius_px
            r = x + highlight_radius_px; b = y + highlight_radius_px
            d.ellipse([l, t, r, b], outline=highlight_color, width=highlight_thickness_px)
            fill_col = (highlight_color[0], highlight_color[1], highlight_color[2],
                        max(60, highlight_color[3] // 3))
            d.ellipse([l+2, t+2, r-2, b-2], fill=fill_col)
        base = Image.alpha_composite(base, overlay)

    # Caption
    caption = caption_fmt.format(qm_value, pred_value)
    if caption_extra:
        if isinstance(caption_extra, str):
            caption += "\n" + caption_extra
        elif isinstance(caption_extra, Mapping):
            caption += "\n" + "\n".join(f"{k}: {v}" for k, v in caption_extra.items())
        else:
            caption += "\n" + "\n".join(f"{k}: {v}" for k, v in caption_extra)

    pad = 10
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    tmp = ImageDraw.Draw(base)
    try:
        bbox = tmp.multiline_textbbox((0, 0), caption, font=font, spacing=2)
        text_w = bbox[2] - bbox[0]; text_h = bbox[3] - bbox[1]
    except AttributeError:
        lines = caption.split("\n")
        sizes = [tmp.textsize(line, font=font) for line in lines]
        text_w = max(w for w, _ in sizes); text_h = sum(h for _, h in sizes) + 2*(len(lines)-1)

    cap_w = max(base.width, text_w + 2*pad)
    cap_h = text_h + 2*pad
    final_img = Image.new("RGBA", (cap_w, base.height + cap_h), (255, 255, 255, 255))
    final_img.paste(base, ((cap_w - base.width)//2, 0), base)
    draw = ImageDraw.Draw(final_img)
    draw.multiline_text(((cap_w - text_w)//2, base.height + pad), caption, fill=(0,0,0), font=font, align="center", spacing=2)

    if save_path:
        final_img.convert("RGB").save(save_path)
    return (final_img, ignored) if return_ignored else final_img

def extract_best_epoch(log_path: str) -> int:
    """
    Parse a training log file and return the best epoch as an integer.

    Priority:
      1) If a line like 'Test@best_epoch=44' exists, return that number.
      2) Otherwise, use the last occurrence of 'Saved best model at epoch N'.
    Raises:
      ValueError: if no best-epoch information can be found.
    """
    # Read the whole file once
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) Prefer explicit 'Test@best_epoch=...'
    m = re.search(r"Test@best_epoch\s*=\s*(\d+)", text)
    if m:
        return int(m.group(1))

    # 2) Fallback to last 'Saved best model at epoch ...'
    matches = re.findall(r"Saved best model at epoch\s+(\d+)", text)
    if matches:
        return int(matches[-1])

    # If neither pattern is found, raise an informative error
    raise ValueError(
        "Could not determine best epoch. "
        "Expected 'Test@best_epoch=...' or 'Saved best model at epoch ...' in the log."
    )

def calculate_kendall_correlation_by_group(df, group_col, col1, col2):
    """
    Calculate Kendall's tau correlation for each group in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    group_col : str
        Column name to group by
    col1 : str
        First column for correlation calculation
    col2 : str
        Second column for correlation calculation
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with group names, tau values, and p-values
    """
    results = []
    
    for group_name, group_df in df.groupby(group_col):
        # Remove NaN values
        valid_data = group_df[[col1, col2]].dropna()
        
        if len(valid_data) > 1:  # Need at least 2 points for correlation
            tau, p_value = kendalltau(valid_data[col1], valid_data[col2])
            max_index_match = pd.to_numeric(valid_data[col1], errors='coerce').idxmax() == pd.to_numeric(valid_data[col2], errors='coerce').idxmax()
            results.append({
            group_col: group_name,
            'tau': tau,
            'p_value': p_value,
            'n': len(valid_data),
            'max_index_match': max_index_match,
            })
        else:
            results.append({
            group_col: group_name,
            'tau': None,
            'p_value': None,
            'n': len(valid_data),
            'max_index_match': None,
            })
    res_df = pd.DataFrame(results)
    return res_df.set_index(group_col,drop=True)


def find_additional_nuc_sites(rdkit_mol):
    return find_sites(rdkit_mol, (extracted_n_smirks_dict | added_n_smirks_dict))


def find_additional_elec_sites(rdkit_mol):
    return find_sites(rdkit_mol, (extracted_e_smirks_dict | added_e_smirks_dict))


def _load_first_mol_from_sdf(sdf_path: Path, removeHs: bool = False) -> Optional[Chem.Mol]:
    """Load the first valid molecule from an SDF file. Return None on failure."""
    try:
        if not sdf_path.exists():
            return None
        supp = Chem.SDMolSupplier(str(sdf_path), removeHs=removeHs, sanitize=False)
        if supp is None:
            return None
        mols = [m for m in supp if m is not None]
        if not mols:
            return None
        mol = mols[0]
        if mol.GetNumConformers() == 0:
            return None
        return mol
    except Exception:
        return None


def _get_graph_radius_atom_indices(mol: Chem.Mol, center_idx: int, radius: int) -> list[int]:
    """
    Return atom indices within the given graph radius from center_idx.
    The center atom itself is included.
    """
    n_atoms = mol.GetNumAtoms()
    if center_idx < 0 or center_idx >= n_atoms:
        raise IndexError(f"atom_index {center_idx} is out of range for molecule with {n_atoms} atoms")

    if mol.GetNumBonds()==0:
        rdDetermineBonds.DetermineBonds(mol)

    visited = {center_idx}
    q = deque([(center_idx, 0)])

    while q:
        atom_idx, dist = q.popleft()
        if dist == radius:
            continue

        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited:
                visited.add(nbr_idx)
                q.append((nbr_idx, dist + 1))

    return sorted(visited)


def _pairwise_distance_change_for_local_region(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    atom_indices: list[int],
) -> Optional[float]:
    """
    Compute sqrt(mean((d_uv^A - d_uv^B)^2)) over all atom pairs
    within the local atom set.
    """
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        return None    

    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()

    # Return 0.0 when the local set contains fewer than two atoms.
    if len(atom_indices) < 2:
        return 0.0

    sq_diffs = []
    for u, v in combinations(atom_indices, 2):
        pa_u = conf_a.GetAtomPosition(u)
        pa_v = conf_a.GetAtomPosition(v)
        pb_u = conf_b.GetAtomPosition(u)
        pb_v = conf_b.GetAtomPosition(v)

        d_a = ((pa_u.x - pa_v.x) ** 2 + (pa_u.y - pa_v.y) ** 2 + (pa_u.z - pa_v.z) ** 2) ** 0.5
        d_b = ((pb_u.x - pb_v.x) ** 2 + (pb_u.y - pb_v.y) ** 2 + (pb_u.z - pb_v.z) ** 2) ** 0.5
        sq_diffs.append((d_a - d_b) ** 2)

    return float(np.sqrt(np.mean(sq_diffs)))


def compute_local_pairwise_distance_change(
    df: pd.DataFrame,
    dir_a: str | Path,
    dir_b: str | Path,
    radius: int = 1,
    name_col: str = "name",
    atom_index_col: str = "atom_index",
    result_col: str = "local_pairwise_distance_change",
    removeHs: bool = False,
) -> pd.DataFrame:
    """
    For each (name, atom_index) pair in the DataFrame, load {name}.sdf
    from two directories and compute the local alignment-free
    pairwise-distance change.

    The metric is:
        G_i = sqrt( (1 / |P_i|) * sum_{(u,v) in P_i} (d_uv^(A) - d_uv^(B))^2 )

    where:
        - N_i is the set of atoms within the specified graph radius from atom i
        - P_i is the set of all unordered atom pairs in N_i
        - d_uv^(A), d_uv^(B) are Euclidean distances in structures A and B

    Returns a copy of df with an added result column.
    If a file is missing or any error occurs for a row, the result is None.
    """
    if radius not in (1, 2):
        raise ValueError("radius must be 1 or 2")

    dir_a = Path(dir_a)
    dir_b = Path(dir_b)

    out_df = df.copy()

    # Cache molecules to avoid reloading the same SDF multiple times.
    mol_cache_a: dict[str, Optional[Chem.Mol]] = {}
    mol_cache_b: dict[str, Optional[Chem.Mol]] = {}

    results = []

    for _, row in tqdm(out_df.iterrows(),total=out_df.shape[0]):
        name = str(row[name_col])
        atom_index = int(row[atom_index_col])

        if name not in mol_cache_a:
            mol_cache_a[name] = _load_first_mol_from_sdf(dir_a / f"{name}.sdf", removeHs)
        if name not in mol_cache_b:
            mol_cache_b[name] = _load_first_mol_from_sdf(dir_b / f"{name}.sdf", removeHs)

        mol_a = mol_cache_a[name]
        mol_b = mol_cache_b[name]

        # Return None if either file is missing or cannot be parsed.
        if mol_a is None or mol_b is None:
            results.append(None)
            continue

        try:
            local_atom_indices = _get_graph_radius_atom_indices(mol_a, atom_index, radius)

            # This assumes the atom ordering is consistent between the two SDF files.
            value = _pairwise_distance_change_for_local_region(mol_a, mol_b, local_atom_indices)
            results.append(value)
        except Exception:
            results.append(None)

    out_df[result_col] = results
    return out_df


def is_atom_in_pi_system(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetIsAromatic():
        return True
    return any(bond.GetIsConjugated() for bond in atom.GetBonds())

