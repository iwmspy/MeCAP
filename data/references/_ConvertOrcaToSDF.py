from rdkit import Chem

BOHR_TO_ANG = 0.529177210903


def extract_last_geometry(lines):
    """
    Extract the last 'CARTESIAN COORDINATES' block from ORCA output.
    Returns a list of (symbol, x, y, z) in Angstrom.
    """
    geometries = []  # list of (unit, [(sym, x, y, z), ...])

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if "CARTESIAN COORDINATES" in line.upper():
            # Try to detect units from the same line
            up = line.upper()
            if "ANGSTROEM" in up or "ANGSTROM" in up or "(A)" in up:
                unit = "ANG"
            elif "A.U." in up or "BOHR" in up:
                unit = "AU"
            else:
                unit = "ANG"

            # Skip dashed line if present
            i += 1
            if i < n and set(lines[i].strip()) == {"-"}:
                i += 1

            block = []
            while i < n:
                l = lines[i].rstrip("\n")
                if not l.strip():
                    break
                parts = l.split()
                # Expect at least: symbol x y z
                if len(parts) < 4:
                    break
                try:
                    x, y, z = map(float, parts[-3:])
                except ValueError:
                    break
                symbol = parts[0]
                block.append((symbol, x, y, z))
                i += 1

            if block:
                geometries.append((unit, block))
        else:
            i += 1

    if not geometries:
        raise ValueError("No 'CARTESIAN COORDINATES' block found in ORCA output.")

    unit, coords = geometries[-1]  # use the last geometry (final structure)

    if unit == "ANG":
        factor = 1.0
    elif unit == "AU":
        factor = BOHR_TO_ANG
    else:
        factor = 1.0

    coords_ang = []
    for sym, x, y, z in coords:
        coords_ang.append((sym, x * factor, y * factor, z * factor))

    return coords_ang


def coords_to_rdkit_mol(coords, name="ORCA_structure"):
    """
    Build an RDKit Mol with atoms and 3D coordinates only (no bonds).
    coords: list of (symbol, x, y, z) in Angstrom.
    """
    rw_mol = Chem.RWMol()
    conf = Chem.Conformer(len(coords))

    for idx, (sym, x, y, z) in enumerate(coords):
        atom = Chem.Atom(sym)
        atom_idx = rw_mol.AddAtom(atom)
        # atom_idx should equal idx, but we use atom_idx for clarity
        conf.SetAtomPosition(atom_idx, (x, y, z))

    conf.Set3D(True)
    rw_mol.AddConformer(conf, assignId=True)
    mol = rw_mol.GetMol()

    # Set molecule name (will appear as first line in MOL block)
    mol.SetProp("_Name", name)

    return mol


def write_sdf_with_rdkit(mol, outfile):
    """
    Write the given RDKit Mol to SDF.
    No bonds are added here; if mol has no bonds, the SDF will have 0 bonds.
    """
    writer = Chem.SDWriter(outfile)
    writer.write(mol)
    writer.close()


def orca_out_to_sdf(in_file, out_file):
    with open(in_file, "r") as f:
        lines = f.readlines()

    coords = extract_last_geometry(lines)
    mol = coords_to_rdkit_mol(coords, name=in_file)
    write_sdf_with_rdkit(mol, out_file)
    