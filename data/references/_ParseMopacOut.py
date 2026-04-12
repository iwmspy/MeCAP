import re


KCAL_TO_KJ = 4.184


def extract_last_cartesian_coordinates(lines):
    """
    Extract the last 'CARTESIAN COORDINATES' block from MOPAC output.
    Returns a list of (symbol, x, y, z) in Angstrom.
    """
    geometries = []
    n_lines = len(lines)
    i = 0

    while i < n_lines:
        if "CARTESIAN COORDINATES" not in lines[i].upper():
            i += 1
            continue

        i += 1
        while i < n_lines and not lines[i].strip():
            i += 1

        if i < n_lines and "NO." in lines[i].upper() and "ATOM" in lines[i].upper():
            i += 1

        block = []
        while i < n_lines:
            line = lines[i].strip()
            if not line:
                break

            parts = line.split()
            if len(parts) < 5:
                break

            try:
                int(parts[0])
                x, y, z = map(float, parts[-3:])
            except ValueError:
                break

            block.append((parts[1], x, y, z))
            i += 1

        if block:
            geometries.append(block)

    if not geometries:
        raise ValueError("No 'CARTESIAN COORDINATES' block found in MOPAC output.")

    return geometries[-1]


def ParseMopacOut(fname):
    """
    Parse a MOPAC .out file into a lightweight dictionary.

    Returns keys:
      - success
      - total_energy_kjmol
      - heat_of_formation_kcalmol
      - coordinates
      - charge
      - method
      - solvent_model
      - dielectric_constant
    """
    mopac_results = {
        "success": False,
        "total_energy_kjmol": None,
        "heat_of_formation_kcalmol": None,
        "coordinates": None,
        "charge": None,
        "method": "",
        "solvent_model": "",
        "dielectric_constant": None,
    }

    with open(fname, "r", errors="replace") as f:
        lines = f.readlines()

    text = "".join(lines)
    upper_text = text.upper()

    mopac_results["success"] = "JOB ENDED NORMALLY" in upper_text

    charge_match = re.search(r"CHARGE ON SYSTEM\s*=\s*([+-]?\d+)", text)
    if charge_match:
        mopac_results["charge"] = int(charge_match.group(1))

    method_match = re.search(r"^\s*(AM1|PM3|PM6|PM7|RM1|MNDO)\s+CALCULATION RESULTS", text, re.MULTILINE)
    if method_match:
        mopac_results["method"] = method_match.group(1)

    if "COSMO" in upper_text:
        mopac_results["solvent_model"] = "COSMO"

    eps_match = re.search(r"\bEPS\s*=\s*([-+]?\d*\.?\d+)", text)
    if eps_match:
        mopac_results["dielectric_constant"] = float(eps_match.group(1))

    hof_match = re.search(
        r"FINAL HEAT OF FORMATION\s*=\s*"
        r"([-+]?\d*\.?\d+)\s*KCAL/MOL"
        r"(?:\s*=\s*([-+]?\d*\.?\d+)\s*KJ/MOL)?",
        text,
    )
    if hof_match:
        kcalmol = float(hof_match.group(1))
        kjmol = float(hof_match.group(2)) if hof_match.group(2) is not None else kcalmol * KCAL_TO_KJ
        mopac_results["heat_of_formation_kcalmol"] = kcalmol
        mopac_results["total_energy_kjmol"] = kjmol

    try:
        mopac_results["coordinates"] = extract_last_cartesian_coordinates(lines)
    except ValueError:
        mopac_results["coordinates"] = None

    return mopac_results
