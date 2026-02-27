# MIT License
#
# Copyright (c) 2026 Yuto Iwasaki
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
#
# -------------------------------------------------------------------------------
#
# MIT License
#
# Copyright (c) 2023 Nicolai Ree
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

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def convert_xyz_to_sdf(input_file, output_file=None):
    READ_FCN = Chem.MolFromXYZFile if input_file.endswith('.xyz') else lambda infile: [m for m in Chem.SDMolSupplier(infile)][0]
    rdkit_mol = READ_FCN(input_file)
    out_mol = Chem.Mol(rdkit_mol)

    rdDetermineBonds.DetermineConnectivity(out_mol, useHueckel=True)
    
    if len(Chem.MolToSmiles(out_mol).split('.')) != 1:
        # print('OBS! Trying to detemine bonds without Hueckel')
        out_mol = Chem.Mol(rdkit_mol)
        rdDetermineBonds.DetermineConnectivity(out_mol, useHueckel=False)

    if len(Chem.MolToSmiles(out_mol).split('.')) != 1:
        # print('OBS! Trying to detemine bonds without Hueckel and covFactor=1.35')
        out_mol = Chem.Mol(rdkit_mol)
        rdDetermineBonds.DetermineConnectivity(out_mol, useHueckel=False, covFactor=1.35)

    if output_file:
      writer = Chem.rdmolfiles.SDWriter(output_file)
      writer.write(out_mol)
      writer.close()
    
    return out_mol


def get_bonds(sdf_file):
    """ The count line is; aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv,
    where aaa is atom count and bbb is bond count.
    """

    atoms = 0
    bond_list = []

    searchlines = open(sdf_file, 'r').readlines()

    for i, line in enumerate(searchlines):
        words = line.split() #split line into words
        if len(words) < 1:
            continue
        if i == 3:
            atoms = int(line[0:3])
            bonds = int(line[3:6])
        if 'Pd' in words: #find atom index of Pd
            transistion_metal_idx = i - 3
        else:
            transistion_metal_idx = -1
        if i > atoms+3 and i <= atoms+bonds+3:
            atom_1 = int(line[0:3])
            atom_2 = int(line[3:6])
            if (atom_1 == transistion_metal_idx) or (atom_2 == transistion_metal_idx): #skip bonds to Pd
                continue
            if atom_2 > atom_1:
                bond_list.append(tuple((atom_1,atom_2)))
            else:
                bond_list.append(tuple((atom_2,atom_1)))

    bond_list.sort()

    return bond_list


def get_bonds_molblock(molblock):
    """ The count line is; aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv,
    where aaa is atom count and bbb is bond count.
    """

    atoms = 0
    bond_list = []

    searchlines = molblock.split('\n')

    for i, line in enumerate(searchlines):
        words = line.split() #split line into words
        if len(words) < 1:
            continue
        if i == 3:
            atoms = int(line[0:3])
            bonds = int(line[3:6])
        if 'Pd' in words: #find atom index of Pd
            transistion_metal_idx = i - 3
        else:
            transistion_metal_idx = -1
        if i > atoms+3 and i <= atoms+bonds+3:
            atom_1 = int(line[0:3])
            atom_2 = int(line[3:6])
            if (atom_1 == transistion_metal_idx) or (atom_2 == transistion_metal_idx): #skip bonds to Pd
                continue
            if atom_2 > atom_1:
                bond_list.append(tuple((atom_1,atom_2)))
            else:
                bond_list.append(tuple((atom_2,atom_1)))

    bond_list.sort()

    return bond_list


def compare_sdf_structure(start, end, molblockStart=False, molblockEnd=False):
    """
    Returns True if structures are the same

    Return False if there has been a proton transfer
    """
    
    if molblockStart:
        bond_start = get_bonds_molblock(start)
    else:
        bond_start = get_bonds(start)
    
    if molblockEnd:
        bond_end = get_bonds_molblock(end)
    else:    
        bond_end = get_bonds(end)

    return bond_start == bond_end


if __name__ == "__main__":
    
    import sys

    print(compare_sdf_structure(sys.argv[1], sys.argv[2]))
