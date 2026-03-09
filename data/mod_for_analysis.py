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

from typing import Iterable, Optional, Tuple, Union, Mapping
import io, math, os, re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
from PIL import Image, ImageDraw, ImageFont

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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_map_num(smiles: str, atom_idx: int) -> int:
    """Return atom map number at the given atom index from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()


def parse_rxn_id(text: str) -> int:
    """Extract reaction ID from strings like '123reac...'."""
    return int(text.split("reac")[0])


def build_score_map(gdf: pd.DataFrame, score_col: str) -> dict[int, float]:
    """Build raw score map: {MapNum: score}."""
    score_map = {}
    for _, row in gdf.iterrows():
        map_num = int(row["MapNum"])
        if map_num != 0:
            score_map[map_num] = float(row[score_col])
    return score_map


def compare_scores(score_a: float, score_b: float, tol: float = 1e-12) -> int:
    """
    Compare two scores.

    Return:
        1  if score_a > score_b
        0  if abs(score_a - score_b) <= tol
       -1  if score_a < score_b
    """
    diff = score_a - score_b
    if abs(diff) <= tol:
        return 0
    return 1 if diff > 0 else -1


def add_model_scores(
    mapno_data: dict,
    model_name: str,
    df_by_key: dict[str, pd.DataFrame],
    smiles_col: str,
    site_col_by_key: dict[str, str],
    id_col: str,
    score_col_by_key: dict[str, str],
    use_only_succeeded: bool = True,
) -> None:
    """
    Populate:
        mapno_data[rxn_id][model_name][source|sink] = {
            "score_map": {MapNum: raw_score},
            "n_sites": int
        }
    """
    for key, df in df_by_key.items():
        site_col = site_col_by_key[key]
        score_col = score_col_by_key[key]

        temp = df.copy()
        temp["MapNum"] = temp.apply(
            lambda row: get_map_num(row[smiles_col], row[site_col]),
            axis=1,
        )
        temp["rxn_id"] = temp[id_col].apply(parse_rxn_id)

        if use_only_succeeded and "succeeded" in temp.columns:
            temp = temp[temp["succeeded"].fillna(False)].copy()

        for rxn_id, gdf in temp.groupby("rxn_id"):
            gdf_valid = gdf[gdf["MapNum"] != 0].copy()
            if gdf_valid.empty:
                continue

            score_map = build_score_map(gdf_valid, score_col)

            mapno_data.setdefault(rxn_id, {}).setdefault(model_name, {})[key] = {
                "score_map": score_map,
                "n_sites": len(score_map),
            }


def build_pairwise_df(
    df_rxn: pd.DataFrame,
    mapno_data: dict,
    ref_model: str,
    cmp_model: str,
    keys: tuple[str, ...] = ("source", "sink"),
    tol: float = 1e-12,
) -> pd.DataFrame:
    """
    Build pairwise comparison table for:
        target reactive site vs each competitor site

    Pairwise relations are computed on the intersection of sites
    available for both ref_model and cmp_model.
    """
    rows = []

    for rxn_id, model_dict in mapno_data.items():
        if rxn_id not in df_rxn.index:
            continue

        if ref_model not in model_dict or cmp_model not in model_dict:
            continue

        for key in keys:
            if key not in model_dict[ref_model]:
                continue
            if key not in model_dict[cmp_model]:
                continue

            target_map_num = df_rxn.loc[rxn_id, key]
            if pd.isna(target_map_num):
                continue
            target_map_num = int(target_map_num)

            ref_score_map = model_dict[ref_model][key]["score_map"]
            cmp_score_map = model_dict[cmp_model][key]["score_map"]

            common_maps = set(ref_score_map.keys()) & set(cmp_score_map.keys())
            if target_map_num not in common_maps:
                continue

            common_maps.discard(target_map_num)
            if not common_maps:
                continue

            ref_target_score = ref_score_map[target_map_num]
            cmp_target_score = cmp_score_map[target_map_num]

            for competitor_map_num in sorted(common_maps):
                ref_comp_score = ref_score_map[competitor_map_num]
                cmp_comp_score = cmp_score_map[competitor_map_num]

                ref_relation = compare_scores(ref_target_score, ref_comp_score, tol=tol)
                cmp_relation = compare_scores(cmp_target_score, cmp_comp_score, tol=tol)

                rows.append(
                    {
                        "rxn_id": rxn_id,
                        "key": key,
                        "model": cmp_model,
                        "target_map_num": target_map_num,
                        "competitor_map_num": competitor_map_num,
                        "ref_target_score": ref_target_score,
                        "ref_competitor_score": ref_comp_score,
                        "ref_relation": ref_relation,
                        "cmp_target_score": cmp_target_score,
                        "cmp_competitor_score": cmp_comp_score,
                        "cmp_relation": cmp_relation,
                        "agreement": (
                            (ref_relation != 0)
                            and (cmp_relation != 0)
                            and (ref_relation == cmp_relation)
                        ),
                        "ref_tie": ref_relation == 0,
                        "cmp_tie": cmp_relation == 0,
                    }
                )

    return pd.DataFrame(rows)


def summarize_pairwise(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise agreement. Tied comparisons are excluded."""
    if pairwise_df.empty:
        return pd.DataFrame()

    temp = pairwise_df.copy()
    temp["valid_pair"] = ~(temp["ref_tie"] | temp["cmp_tie"])
    temp["agreement_int"] = temp["agreement"].astype(int)

    summary = (
        temp.groupby(["model", "key"])
        .agg(
            n_rxns=("rxn_id", "nunique"),
            n_pairs_total=("valid_pair", "size"),
            n_pairs_valid=("valid_pair", "sum"),
            n_agree=("agreement_int", "sum"),
        )
        .reset_index()
    )

    summary["agreement_rate"] = summary["n_agree"] / summary["n_pairs_valid"]
    return summary
