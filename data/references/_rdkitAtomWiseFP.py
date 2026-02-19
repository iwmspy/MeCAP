from typing import Tuple, List, Dict, Optional
from rdkit import Chem


# Fixed element list for counts (11 dims)
# Order: C, N, O, S, P, F, Cl, Br, I, B, Si
_ELEM_Z: List[int] = [6, 7, 8, 16, 15, 9, 17, 35, 53, 5, 14]
_ELEM_TO_POS: Dict[int, int] = {z: i for i, z in enumerate(_ELEM_Z)}

# Fixed bond type order (4 dims): single, double, triple, aromatic
_BOND_KEYS: Tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


def _bond_type_key(bond: Chem.Bond) -> Optional[str]:
    """Return a normalized bond type key."""
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return "SINGLE"
    if bt == Chem.rdchem.BondType.DOUBLE:
        return "DOUBLE"
    if bt == Chem.rdchem.BondType.TRIPLE:
        return "TRIPLE"
    if bt == Chem.rdchem.BondType.AROMATIC:
        return "AROMATIC"
    return None


def _count_elements(mol: Chem.Mol, atom_indices: List[int]) -> List[int]:
    """Count selected elements in the given atom index list."""
    out = [0] * len(_ELEM_Z)
    for idx in atom_indices:
        z = mol.GetAtomWithIdx(int(idx)).GetAtomicNum()
        pos = _ELEM_TO_POS.get(z, None)
        if pos is not None:
            out[pos] += 1
    return out


def _count_aromatic_and_ring(mol: Chem.Mol, atom_indices: List[int]) -> Tuple[int, int]:
    """Count aromatic atoms and ring atoms in the given atom index list."""
    n_arom = 0
    n_ring = 0
    for idx in atom_indices:
        at = mol.GetAtomWithIdx(int(idx))
        if at.GetIsAromatic():
            n_arom += 1
        if at.IsInRing():
            n_ring += 1
    return n_arom, n_ring


def _count_bonds_between_sets(
    mol: Chem.Mol, left: List[int], right_set: set
) -> List[int]:
    """
    Count bond types for bonds where one endpoint is in 'left' and
    the other endpoint is in 'right_set'. Each bond is counted once.
    """
    out = [0, 0, 0, 0]
    for i in left:
        ai = mol.GetAtomWithIdx(int(i))
        for nb in ai.GetNeighbors():
            j = nb.GetIdx()
            if j not in right_set:
                continue
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            if bond is None:
                continue
            key = _bond_type_key(bond)
            if key is None:
                continue
            out[_BOND_KEYS.index(key)] += 1
    return out


def site_feature_vector_from_mol(mol: Chem.Mol, atom_idx: int) -> Tuple[int, ...]:
    """
    Build a cheap atom-centered local environment vector up to radius 2.

    Vector layout (length 40):
      Center (6):
        [0] atomic_num
        [1] is_aromatic (0/1)
        [2] is_in_ring (0/1)
        [3] total_degree
        [4] total_num_Hs (implicit+explicit)
        [5] formal_charge

      1-hop block (17):
        element counts for _ELEM_Z (11)
        aromatic_count (1)
        ring_count (1)
        bond type counts between center and 1-hop (4)

      2-hop block (17):
        element counts for _ELEM_Z (11)
        aromatic_count (1)
        ring_count (1)
        bond type counts between 1-hop and 2-hop (4)

    Returns:
      tuple of ints (40-dim)
    """
    if mol is None:
        raise ValueError("mol is None")
    if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
        raise IndexError(f"atom_idx out of range: {atom_idx}")

    center = mol.GetAtomWithIdx(int(atom_idx))

    # Center features
    c_atomic_num = center.GetAtomicNum()
    c_is_arom = 1 if center.GetIsAromatic() else 0
    c_in_ring = 1 if center.IsInRing() else 0
    c_degree = int(center.GetTotalDegree())
    c_num_h = int(center.GetTotalNumHs())
    c_charge = int(center.GetFormalCharge())

    # 1-hop indices
    hop1 = [nb.GetIdx() for nb in center.GetNeighbors()]
    hop1_set = set(hop1)

    # 2-hop indices (neighbors of hop1 excluding center and hop1)
    hop2_set = set()
    for i in hop1:
        ai = mol.GetAtomWithIdx(int(i))
        for nb in ai.GetNeighbors():
            j = nb.GetIdx()
            if j == atom_idx:
                continue
            if j in hop1_set:
                continue
            hop2_set.add(j)
    hop2 = sorted(hop2_set)

    # 1-hop counts
    hop1_elem = _count_elements(mol, hop1)
    hop1_arom, hop1_ring = _count_aromatic_and_ring(mol, hop1)
    # Bond types between center and hop1
    hop1_bonds = [0, 0, 0, 0]
    for j in hop1:
        bond = mol.GetBondBetweenAtoms(int(atom_idx), int(j))
        if bond is None:
            continue
        key = _bond_type_key(bond)
        if key is None:
            continue
        hop1_bonds[_BOND_KEYS.index(key)] += 1

    # 2-hop counts
    hop2_elem = _count_elements(mol, hop2)
    hop2_arom, hop2_ring = _count_aromatic_and_ring(mol, hop2)
    # Bond types between hop1 and hop2
    hop2_bonds = _count_bonds_between_sets(mol, hop1, hop2_set)

    vec: List[int] = []
    vec.extend([c_atomic_num, c_is_arom, c_in_ring, c_degree, c_num_h, c_charge])

    vec.extend(hop1_elem)
    vec.extend([hop1_arom, hop1_ring])
    vec.extend(hop1_bonds)

    vec.extend(hop2_elem)
    vec.extend([hop2_arom, hop2_ring])
    vec.extend(hop2_bonds)

    return tuple(int(x) for x in vec)


def site_feature_vector(smiles: str, atom_idx: int) -> Tuple[int, ...]:
    """
    Convenience wrapper: parse SMILES and call site_feature_vector_from_mol.
    For large-scale use, cache RDKit Mol objects and call site_feature_vector_from_mol.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return site_feature_vector_from_mol(mol, atom_idx)
    