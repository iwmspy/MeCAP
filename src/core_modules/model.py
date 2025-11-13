# -------------------------
# Model definition (add UniMolV2Model and version-aware builders)
# -------------------------

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unimol_tools.config import MODEL_CONFIG, MODEL_CONFIG_V2
from unimol_tools.data import Dictionary
from unimol_tools.utils import logger, pad_1d_tokens, pad_2d, pad_coords
from unimol_tools.weights import WEIGHT_DIR, weight_download, weight_download_v2
from unimol_tools.models.transformers import TransformerEncoderWithPair
from unimol_tools.models.transformersv2 import (
    TransformerEncoderWithPairV2,
    AtomFeature,
    EdgeFeature,
    MovementPredictionHead,
    SE3InvariantKernel,
)
from unimol_tools.models.unimol import (
    molecule_architecture as molecule_architecture_v1,
    NumericalEmbed,
    NonLinearHead,
    GaussianLayer,
)

# Try to import Uni-Mol v2 module; keep it optional
from unimol_tools.models.unimolv2 import (
    molecule_architecture as molecule_architecture_v2
    )# type: ignore

BACKBONE_V1 = TransformerEncoderWithPair
BACKBONE_V2 = TransformerEncoderWithPairV2


import torch
import torch.nn as nn
from typing import List, Tuple, Optional

def _get_act(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = (name or "gelu").lower()
    if name == "relu":
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    # default
    return nn.GELU()

class SingleAtomRegressionHead(nn.Module):
    """
    Flexible MLP head for single-atom regression.

    Args:
        input_dim (int): Embedding dimension from the backbone.
        hidden_dim (int|None): Backward-compatible shortcut for a single hidden layer.
                               Ignored if `hidden_dims` is provided.
        hidden_dims (list[int]|tuple[int]|None): Sizes of hidden layers (e.g., [512, 256]).
        pooler_dropout (float): Dropout applied before the MLP (and between hidden layers).
        out_dim (int): Output dimension.
        activation (str): One of {"gelu", "relu", "silu", "tanh"}.
        norm (str|None): Optional normalization after each Linear. {"layernorm","batchnorm",None}.
                         BatchNorm1d expects (N, C) tensors (which we provide).
        last_dropout (float): Optional dropout right before the final Linear layer.
    """
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: Optional[List[int] | Tuple[int, ...]] = None,
        pooler_dropout: float = 0.1,
        out_dim: int = 1,
        activation: str = "gelu",
        norm: Optional[str] = None,
        last_dropout: float = 0.0,
    ):
        super().__init__()
        self.pre_dropout = nn.Dropout(p=pooler_dropout)
        act = _get_act(activation)

        # Linear head (no hidden layers)
        if not hidden_dim:
            self.net = nn.Linear(input_dim, out_dim)
            self._is_linear_head = True
            return

        self._is_linear_head = False
        layers: List[nn.Module] = []
        in_dim = int(input_dim)

        # Build hidden stack: [Linear -> (Norm) -> Activation -> Dropout] x L
        for h in hidden_dim:
            h = int(h)
            layers.append(nn.Linear(in_dim, h))
            if norm:
                norm_l = norm.lower()
                if norm_l == "layernorm":
                    layers.append(nn.LayerNorm(h))
                elif norm_l == "batchnorm":
                    layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            layers.append(nn.Dropout(p=pooler_dropout))
            in_dim = h

        # Optional dropout right before the final projection
        if last_dropout and last_dropout > 0:
            layers.append(nn.Dropout(p=last_dropout))

        # Final projection
        layers.append(nn.Linear(in_dim, int(out_dim)))

        self.net = nn.Sequential(*layers)

    def forward(self, atom_repr: torch.Tensor) -> torch.Tensor:
        """atom_repr: (N, D). Returns prediction tensor of shape (N, out_dim)."""
        x = self.pre_dropout(atom_repr)
        return self.net(x)


def _resolve_model_keys(version: str, remove_hs: bool, model_size='84m') -> Tuple[str, str]:
    """
    Resolve MODEL_CONFIG keys for dict/weight robustly across versions.
    We try several candidate names to be compatible with different releases.
    """
    # Candidate name patterns by version
    if version.lower() in ("v2", "unimolv2"):
        name = model_size
    else:
        keys_dict = list(MODEL_CONFIG.get("dict", {}).keys())
        keys_wt = list(MODEL_CONFIG.get("weight", {}).keys())
        cands = [
            "molecule_no_h" if remove_hs else "molecule_all_h",
        ]

        name = None
        for cand in cands:
            if cand in MODEL_CONFIG["dict"] and cand in MODEL_CONFIG["weight"]:
                name = cand
                break
        if name is None:
            raise KeyError(
                f"Cannot resolve MODEL_CONFIG keys for version={version}, remove_hs={remove_hs}. "
                f"Available dict keys: {keys_dict}; weight keys: {keys_wt}"
            )
    return name, name


def _maybe_download_weights(name_key: str):
    """Ensure both dictionary and weight files exist locally."""
    if name_key in MODEL_CONFIG['weight']:
        dict_rel = MODEL_CONFIG['dict'][name_key]
        wt_rel = MODEL_CONFIG['weight'][name_key]
        dict_path = os.path.join(WEIGHT_DIR, dict_rel)
        wt_path = os.path.join(WEIGHT_DIR, wt_rel)
        if not os.path.exists(dict_path):
            weight_download(dict_rel, WEIGHT_DIR)
        if not os.path.exists(wt_path):
            weight_download(wt_rel, WEIGHT_DIR)
        return dict_path, wt_path
    else:
        wt_rel = MODEL_CONFIG_V2['weight'][name_key]
        wt_path = os.path.join(WEIGHT_DIR, wt_rel)
        if not os.path.exists(wt_path):
            weight_download_v2(wt_rel, WEIGHT_DIR)
        return None, wt_path

class _BaseUniMolAtomModel(nn.Module):
    """
    Shared logic for single-atom regression on Uni-Mol backbones.
    Subclasses define: self.args, self.dictionary, self.embed_tokens, self.encoder, self.gbf, self.gbf_proj.
    """

    def __init__(self):
        super().__init__()
    
    def _load_single_atom_head(self, pooler_dropout: float, atom_head_hidden_dim: List[int] = None, atom_out_dim: int = 1):
        self.single_atom_head = SingleAtomRegressionHead(
            input_dim=self.args.encoder_embed_dim,
            hidden_dim=atom_head_hidden_dim,
            pooler_dropout=pooler_dropout,
            out_dim=atom_out_dim,
        )

    def _dist_bias(self, dist, et):
        n_node = dist.size(-1)
        gbf_feature = self.gbf(dist, et)
        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        return graph_attn_bias

    def encode_atoms(self, src_tokens, src_distance, src_coord, src_edge_type):
        """Return per-atom representations (B, L_atoms, D) without head."""
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        graph_attn_bias = self._dist_bias(src_distance, src_edge_type)
        (encoder_rep, _, _, _, _,) = self.encoder(
            x, padding_mask=padding_mask, attn_mask=graph_attn_bias
        )
        return encoder_rep[:, 1:, :]  # drop CLS
    
    def _forward_head(self, encoder_rep, atom_index, target=None):
        token_reprs = encoder_rep[:, 1:, :]  # (B, L-1, D)
        if atom_index is None:
            raise ValueError("atom_index must be provided for single-atom regression")
        B, Lm1, D = token_reprs.size()
        b_idx = torch.arange(B, device=token_reprs.device)
        selected_repr = token_reprs[b_idx, atom_index, :]  # (B, D)
        pred = self.single_atom_head(selected_repr)
        out = {"pred": pred}
        if target is not None:
            if target.dim() == 1:
                target = target.unsqueeze(-1)
            out["loss"] = F.mse_loss(pred, target)
        return out

    def batch_collate_fn(self, samples):
        """Pad and collate Uni-Mol tensors; atom_index and target are handled separately."""
        batch = {}
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'atom_feat':
                v = pad_coords(
                    [torch.tensor(s[0][k]) for s in samples],
                    pad_idx=self.padding_idx,
                    dim=8,
                )
            elif k == 'atom_mask':
                v = pad_1d_tokens(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'edge_feat':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples],
                    pad_idx=self.padding_idx,
                    dim=3,
                )
            elif k == 'shortest_path':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'degree':
                v = pad_1d_tokens(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'pair_type':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples],
                    pad_idx=self.padding_idx,
                    dim=2,
                )
            elif k == 'attn_bias':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k in ('atom_index', 'target'):
                continue
            batch[k] = v

        batch['atom_index'] = torch.tensor([int(s[0]['atom_index']) for s in samples], dtype=torch.long) \
            if 'atom_index' in samples[0][0] else None

        if 'target' in samples[0][0]:
            t = torch.tensor([s[0]['target'] for s in samples], dtype=torch.float32)
            batch['target'] = t if t.dim() > 1 else t  # allow (B,) or (B, K)
        else:
            batch['target'] = None

        return batch, None
    
    def load_pretrained_weights(self, path, strict=False):
        """Load pretrained weights."""
        if path is None:
            return
        logger.info(f"Loading pretrained weights from {path}")
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.load_state_dict(state_dict, strict=strict)


class UniMolV1Model(_BaseUniMolAtomModel):
    """Uni-Mol v1 backbone with a single-atom regression head."""
    def __init__(self, output_dim=2, data_type='molecule', load_original=True, **params):
        self.model_name = 'unimolv1'
        if data_type != 'molecule':
            raise NotImplementedError('Supported data_type = "molecule" only...')
        super().__init__()

        self.args = molecule_architecture_v1()
        self.output_dim = output_dim
        self.data_type = data_type
        self.remove_hs = params.get('remove_hs', False)

        # Resolve model keys and ensure weights
        name_key, name_key_w = _resolve_model_keys("v1", self.remove_hs)
        dict_path, wt_path = _maybe_download_weights(name_key)

        # Dictionary and embeddings
        self.dictionary = Dictionary.load(dict_path)
        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = self.dictionary.pad()


        self.embed_tokens = nn.Embedding(
            len(self.dictionary), self.args.encoder_embed_dim, self.padding_idx
        )
        self.encoder = BACKBONE_V1(
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.delta_pair_repr_norm_loss < 0,
        )
        self._load_single_atom_head(
            pooler_dropout=self.args.pooler_dropout if hasattr(self.args, "pooler_dropout") else 0.1,
            atom_head_hidden_dim=params.get('atom_head_hidden_dim', None),
            atom_out_dim=params.get('atom_out_dim', 1),
            )

        # Edge encoding
        K = 128
        n_edge_type = len(self.dictionary) * len(self.dictionary)
        self.gbf_proj = NonLinearHead(K, self.args.encoder_attention_heads, self.args.activation_fn)
        self.gbf = GaussianLayer(K, n_edge_type) if getattr(self.args, "kernel", "gaussian") == 'gaussian' \
            else NumericalEmbed(K, n_edge_type)

        # Optionally load original pretrain weights
        if load_original:
            self.load_pretrained_weights(path=wt_path)
    
    def forward(self, src_tokens, src_distance, src_coord, src_edge_type,
                atom_index=None, target=None, **kwargs):
        """Row-wise path: select one atom per sample and regress."""
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        graph_attn_bias = self._dist_bias(src_distance, src_edge_type)
        (encoder_rep, _, _, _, _,) = self.encoder(
            x, padding_mask=padding_mask, attn_mask=graph_attn_bias
        )
        return self._forward_head(encoder_rep, atom_index, target)


class UniMolV2Model(_BaseUniMolAtomModel):
    """
    Uni-Mol v2 backbone with a single-atom regression head.
    This class prefers unimol_tools.models.unimolv2 for architecture/backbone if available.
    """
    def __init__(self, output_dim=2, model_size='84m', load_original=True, **params):
        # Get v2 architecture if possible; fallback to v1 hyperparams
        self.model_name = 'unimolv2'
        # Initialize base head
        super().__init__()

        self.args = molecule_architecture_v2(model_size=model_size)
        
        self.output_dim = output_dim
        self.model_size = model_size
        self.remove_hs = params.get('remove_hs', False)

        # Resolve keys and ensure artifacts
        name_key, name_key_w = _resolve_model_keys("v2", self.remove_hs, model_size=model_size)
        dict_path, wt_path = _maybe_download_weights(name_key)

        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = 127
        self.embed_tokens = nn.Embedding(
            self.token_num, self.args.encoder_embed_dim, self.padding_idx
        )

        self.encoder = BACKBONE_V2(
            num_encoder_layers=self.args.num_encoder_layers,
            embedding_dim=self.args.encoder_embed_dim,
            pair_dim=self.args.pair_embed_dim,
            pair_hidden_dim=self.args.pair_hidden_dim,
            ffn_embedding_dim=self.args.ffn_embedding_dim,
            num_attention_heads=self.args.num_attention_heads,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            activation_fn=self.args.activation_fn,
            droppath_prob=self.args.droppath_prob,
            pair_dropout=self.args.pair_dropout,
        )
        self._load_single_atom_head(
            pooler_dropout=self.args.pooler_dropout if hasattr(self.args, "pooler_dropout") else 0.1,
            atom_head_hidden_dim=params.get('atom_head_hidden_dim', None),
            atom_out_dim=params.get('atom_out_dim', 1),
        )

        # Edge encoding (v2 may still accept gaussian/numerical; keep parity)
        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512

        K = 128

        self.atom_feature = AtomFeature(
            num_atom=num_atom,
            num_degree=num_degree,
            hidden_dim=self.args.encoder_embed_dim,
        )

        self.edge_feature = EdgeFeature(
            pair_dim=self.args.pair_embed_dim,
            num_edge=num_edge,
            num_spatial=num_spatial,
        )

        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=self.args.pair_embed_dim,
            num_pair=num_pair,
            num_kernel=K,
            std_width=self.args.gaussian_std_width,
            start=self.args.gaussian_mean_start,
            stop=self.args.gaussian_mean_stop,
        )

        self.movement_pred_head = MovementPredictionHead(
            self.args.encoder_embed_dim,
            self.args.pair_embed_dim,
            self.args.encoder_attention_heads,
        )

        self.dtype = torch.float32

        if load_original:
            self.load_pretrained_weights(path=wt_path)
    
    #'atom_feat', 'atom_mask', 'edge_feat', 'shortest_path', 'degree', 'pair_type', 'attn_bias', 'src_tokens'
    def forward(
        self,
        atom_feat,
        atom_mask,
        edge_feat,
        shortest_path,
        degree,
        pair_type,
        attn_bias,
        src_tokens,
        src_coord,
        atom_index=None, 
        target=None,
        **kwargs
    ):

        pos = src_coord

        n_mol, n_atom = atom_feat.shape[:2]
        token_feat = self.embed_tokens(src_tokens)
        x = self.atom_feature({'atom_feat': atom_feat, 'degree': degree}, token_feat)

        dtype = self.dtype

        x = x.type(dtype)

        attn_mask = attn_bias.clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.args.encoder_attention_heads, 1, 1
        )
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args.pair_embed_dim)
        attn_bias = self.edge_feature(
            {'shortest_path': shortest_path, 'edge_feat': edge_feat}, attn_bias
        )
        attn_mask = attn_mask.type(self.dtype)

        atom_mask_cls = torch.cat(
            [
                torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype),
                atom_mask,
            ],
            dim=1,
        ).type(self.dtype)

        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
        new_attn_bias = attn_bias.clone()
        new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d
        new_attn_bias = new_attn_bias.type(dtype)
        encoder_rep, pair = self.encoder(
            x,
            new_attn_bias,
            atom_mask=atom_mask_cls,
            pair_mask=pair_mask,
            attn_mask=attn_mask,
        )

        return self._forward_head(encoder_rep, atom_index, target)


# -------------------------
# Builders (updated)
# -------------------------

def build_single_atom_model(atom_out_dim: int = 1,
                            atom_head_hidden_dim: List[int] = None,
                            remove_hs: bool = True,
                            load_original: bool = True,
                            device: Optional[str] = None,
                            model_name: str = "unimolv1", model_size: str = "84m") -> Tuple[nn.Module, torch.device]:
    """
    Construct Uni-Mol single-atom model for the requested version.
    model_name in {'unimolv1','unimolv2'}.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = (model_name or "unimolv1").lower()
    if model_name not in {"unimolv1", "unimolv2"}:
        raise ValueError(f"Unsupported model_name: {model_name}")

    common_kwargs = dict(
        output_dim=2,
        data_type='molecule',
        remove_hs=remove_hs,
        atom_out_dim=atom_out_dim,
        load_original=load_original,
        atom_head_hidden_dim=atom_head_hidden_dim,
    )
    if model_name == "unimolv2":
        common_kwargs["model_size"] = model_size
        model = UniMolV2Model(**common_kwargs).to(device)
    else:
        model = UniMolV1Model(**common_kwargs).to(device)
    return model, torch.device(device)

def build_model_from_checkpoint(
    checkpoint_path: str,
    *,
    remove_hs: bool = True,
    device: Optional[str] = None,
) -> Tuple[str, nn.Module, torch.device]:
    """
    Load a checkpoint, infer head config, build the correct Uni-Mol (v1/v2) model accordingly, and load weights.
    """
    # 1) Load state (CPU first)
    meta_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = meta_state['model_name']
    model_size = meta_state['model_size']
    hidden_dim = meta_state['atom_head_hidden_dim']
    out_dim    = meta_state['atom_out_dim']
    if "model" in meta_state:
        state = meta_state["model"]
    elif "state_dict" in meta_state:
        state = meta_state["state_dict"]

    # 3) Guess model version and build
    model, device_ = build_single_atom_model(
        atom_out_dim=out_dim,
        atom_head_hidden_dim=hidden_dim,
        remove_hs=remove_hs,
        load_original=False,   # will load from checkpoint below
        device=device,
        model_name=model_name,
        model_size=model_size,
    )

    # 4) Load weights strictly now that shapes match
    logger.info(f"Loading checkpoint (strict) into {model_name}: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    return meta_state, model, device_
