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

import os
from rdkit import Chem

s_params = Chem.SubstructMatchParameters()
s_params.numThreads = min(8, os.cpu_count())
s_params.uniquify = False
substructure_match = lambda mol, submol, only_has: mol.HasSubstructMatch(submol,s_params) if only_has else mol.GetSubstructMatches(submol,s_params)

# Rules were defined by Ree. et. al.
# https://github.com/jensengroup/ESNUEL/blob/d7eead26d9646c98076d8f9962dc64e2b4a0b792/src/esnuel/locate_atom_sites.py

n_smirks_dict = {'Ether': '[OX2:1]([#6;!$(C([OX2])[#7,#8,#15,#16,F,Cl,Br,I]);!$([#6]=[#8]):2])[#6;!$(C([OX2])[#7,#8,#15,#16]);!$([#6]=[#8]):3]>>[CH3][OX3+:1]([*:2])[*:3]',
                 'Ketone': '[OX1H0:1]=[#6X3:2]([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]>>[CH3][OX2H0:1][#6X3+:2]([*:3])[*:4]',
                 'Amide': '[OX1:1]=[CX3;$([CX3][#6]),$([CX3H]):2][#7X3;!R:3]>>[CH3][OX2:1][CX3:2]=[#7X3+:3]',
                 'Enolate': '[#6;$([#6]=,:[#6]-[#8-]),$([#6-]-[#6]=,:[#8]):1]~[#6:2]~[#8;$([#8-]-[#6]=,:[#6]),$([#8]=,:[#6]-[#6-]):3]>>[CH3][#6+0:1][*:2]=[#8+0:3]',
                 'Aldehyde': '[OX1:1]=[$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):2]>>[CH3][OX2:1][#6+:2]',
                 'Imine': '[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):1]=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):2]>>[CH3][NX3+:1]=[*:2]',
                 'Nitranion': '[#7X2-:1]>>[CH3][#7X3+0:1]',
                 'Carbanion': '[#6-;!$([#6X1-]#[#7,#8,#15,#16]):1]>>[CH3][#6+0:1]',
                 'Nitronate': '[#6:1]=[#7+:2](-[#8-:3])-[#8-:4]>>[CH3][#6:1][#7+:2](=[#8+0:3])-[*:4]',
                 'Ester': '[OX1:1]=[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]):2][#8X2H0:3][#6;!$(C=[O,N,S]):4]>>[CH3][OX2:1][#6X3+:2][*:3][*:4]',
                 'Carboxylic acid': '[OX1:1]=[CX3;$([R0][#6]),$([H1R0]):2][$([OX2H]),$([OX1-]):3]>>[CH3][OX2:1][CX3+:2][*:3]',
                 'Amine': '[#7+0;$([N;R;!$([#7X2]);$(N-[#6]);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]),$([NX3+0;!$([#7X3][CX3;$([CX3][#6]),$([CX3H])]=[OX1])]),$([NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]):1]>>[CH3][#7+:1]',
                 'Cyanoalkyl/nitrile anion': '[C:1]=[C:2]=[#7X1-:3]>>[CH3][C:1][C:2]#[#7X1+0:3]',
                 'Nitrile': '[NX1:1]#[CX2;!$(CC=C=[#7X1-]);!$(CC=C):2]>>[CH3][NX2+:1]#[*:2]',
                 'Isonitrile': '[CX1-:1]#[NX2+:2]>>[CH3][CX2+0:1]#[NX2+:2]',
                 'Phenol': '[OX2H:1][$(c(c)c),$([#6X3;R](=[#6X3;R])[#6X3;R]):2]>>[CH3][OX3+1:1][*:2]', # added due to rxn100
                 'Silyl_ether': '[#8X2H0:1][#14X4:2]([!#1:3])([!#1:4])[!#1:5]>>[CH3][#8X3H0+:1][*:2]([*:3])([*:4])[*:5]', # added due to rxn100
                 'Pyridine_like_nitrogen': '[#7X2;$([nX2](:*):*),$([#7X2;R](=[*;R])[*;R]):1]>>[CH3][#7X3+:1]', # added due to rxn100
                 'anion_with_charge_minus1': '[*-:1]>>[CH3][*+0:1]', # added to capture additional sites 
                 'double_bond': '[*;!$([!X4;!#1;!#6:1])+0:1]=[*+0:2]>>[CH3][*:1]-[*+1:2]', # added to capture additional sites
                 'double_bond_neighbouratom_with_charge_plus1': '[*;!$([!X4;!#1;!#6:1])+0:1]=[*+1:2]>>[CH3][*:1]-[*+2:2]', # added to capture additional sites
                 'triple_bond': '[*;!$([!X4;!#1;!#6:1])+0:1]#[*+0:2]>>[CH3][*:1]=[*+1:2]', # added to capture additional sites 
                 'triple_bond_neighbouratom_with_charge_plus1': '[*;!$([!X4;!#1;!#6:1])+0:1]#[*+1:2]>>[CH3][*:1]=[*+2:2]', # added to capture additional sites
                 'atom_with_lone_pair': '[!X4;!#1;!#6:1]>>[CH3][*+1:1]', # added to capture additional sites
                }

extracted_n_smirks_dict = {
                 'Ether': (
                    '[OX2:1]([#6;!$(C([OX2])[#7,#8,#15,#16,F,Cl,Br,I]);!$([#6]=[#8]):2])[#6;!$(C([OX2])[#7,#8,#15,#16]);!$([#6]=[#8]):3]'
                    ),
                 'Ketone_O': (
                    '[OX1H0:1]=[#6X3:2]([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]'
                    ),
                 'Amide_O': (
                    '[OX1:1]=[CX3;$([CX3][#6]),$([CX3H]):2][#7X3;!R:3]'
                    ),
                 'Enolate': (
                    '[#6;!a;$([#6]=,:[#6]-[#8-]),$([#6-]-[#6]=,:[#8]):1]~[#6:2]~[#8;$([#8-]-[#6]=,:[#6]),$([#8]=,:[#6]-[#6-]):3]'
                    ),  # Refined: Phenoxide excluded
                 'Aldehyde_O': (
                    '[OX1:1]=[$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):2]'
                    ),
                 'Imine': (
                    '[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):1]=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):2]'
                    ),
                 'Nitronate': (
                    '[#6:1]=[#7+:2](-[#8-:3])-[#8-:4]'
                    ),
                 'Ester_O': (
                    '[OX1:1]=[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]):2][#8X2H0:3][#6;!$(C=[O,N,S]):4]'
                    ),
                 'Carboxylic_acid': (
                    '[OX1:1]=[CX3;$([R0][#6]),$([H1R0]):2][$([OX2H]),$([OX1-]):3]'
                    ),
                 'Amine': (
                    '[#7+0;!$([#7X2]);$(N-[#6]);!$(N-[!#6;!#1]);!$(N-C=[O,N,S]):1]'
                    ),  # Refined: Simplified rule; ammonium excluded
                 'Cyanoalkyl_nitrile_anion': (
                    '[C:1]=[C:2]=[#7X1-:3]'
                    ),
                 'Nitrile_N': (
                    '[NX1:1]#[CX2;!$(CC=C=[#7X1-]);!$(CC=C):2]'
                    ),
                 'Silyl_ether': (
                    '[#8X2H0:1][#14X4:2]([!#1:3])([!#1:4])[!#1:5]'
                    ),
                 'Pyridine_like_nitrogen': (
                    '[#7X2;$([nX2](:*):*),$([#7X2;R](=[*;R])[*;R]):1]'
                    ),
                 'Nitranion': (
                    '[#7-;X2,X1;!$([#7-]=[#7]);!$([#7-][C,N+,S,P]=O);!$([#7-]C#N):1]'
                    ),  # Refined: Diazo, nitroso, and beta-carbonyl-like motifs (e.g., sulfonamide) excluded
                 'Carbanion': (
                    '[#6-;!$([#6X1-]#[#8,#15,#16]);!$([#6X1-]#[#7+]);!$([#6-]-[#7+]=[#7]);!$([#6-]-[#7]=[#7+]);!$([#6-]-[C](=O)):1]'
                    ),  # Refined: Cyanide included; isonitrile, diazo, and alpha-carbonyl carbanions excluded
                 'Double_bond': (
                    '[CX3+0;!$([C]-[C,N+,S,P](=O));!$([C]-C#N):1]=[CX3+0;!$([C]-[C,N+,S,P](=O));!$([C]-C#N):2]'
                    ),  # Refined: Nucleophilic pi bond (alkene; aromatic excluded). Strictly C=C (exclude C=O, C=N, etc.). 
                        # Exclude EWG-conjugated alkenes (e.g., Michael acceptors).
                 'Triple_bond': (
                    '[CX2+0;!$([C]-[C,N+,S,P](=O));!$([C]-C#N):1]#[CX2+0;!$([C]-[C,N+,S,P](=O));!$([C]-C#N):2]'
                    ),  # Refined: Nucleophilic pi bond (alkyne; aromatic excluded). Strictly C#C (exclude nitriles, etc.). 
                        # Exclude EWG-activated alkynes (e.g., ynones).
                 }

added_n_smirks_dict = {
                 'Aryl_pi_EDG': (
                    '[cH:1](~[a:3])~[c;D3:2]([$([#6X4;H3]),$([O-&X1,N-&X2]),$([OX2H]),$([OX2;H0][#6]),$([NX3;!$([N]C(=O));!$([N]~[!#6])]):7])~[a:4]'
                    ),  # EDG-substituted aryl pi site (EDG: phenoxide, nitroxide, phenol, aryl ether, aniline-like). 
                        # Note: Ortho-only rule; meta rules tend to introduce out-of-distribution substrates relative to the training data.
                 'Alcohol': (
                    '[$([OX2H;+0;!$([O;X1]=[*]);!$([O][C](=O));!$([O][S](=O));!$([O][S](=O)(=O));!$([O][P](=O))]):1]'
                    ),  # Alcohol oxygen (exclude carbonyl O and acyl/phosphoryl/sulfonyl O)
                 'Oxide_anion': (
                    '[O-;X1;!$([O-][C](=O));!$([O-][N+]);!$([O-][n+]);!$([O-][S](=O)(=O));!$([O-][P](=O));!$([O-][Cl,Br,I]):1]'
                    ),  # Alkoxide oxygen (exclude carbonyl O and nitronate/sulfonate/phosphate/perchlorate O)
                 'Sulfur_with_lone_pair': (
                    '[#16;X1,X2;+0;!a;!$([#16](=O));!$([#16][N,F,Cl,Br,I]):1]'
                    ),  # General neutral sulfur with a lone pair (exclude sulfoxides/sulfones, sulfenamides, and sulfenyl halides)
                 'Phosphorus_with_lone_pair': (
                    '[#15;X3;+0;!a;!$([#15](=[S,O]));!$([#15]-[F,Cl,Br,I]):1]'
                    ),  # Phosphorus(III) (exclude phosphoryl P=O / P=S and P-halides)
                 'Halide': (
                    '[F-,Cl-,Br-,I-:1]'
                    ),  # Halide anion (charge -1)
                 'Thiolate': (
                    '[S-:1][#6:2]'
                    ),  # Thiolate sulfur (including thiocyanate)
                 'Hydrosilane_Si': (
                     '[#14X4;H1,H2,H3,H4;+0:1]'
                     ), # H-Si hydride donors (hydrosilanes)
                 'Hydrophosphorus_P': (
                     '[#15;H1,H2,H3;+0:1]'
                     ), # H-P hydride donors (hydrophosphines / phosphine hydrides)
                 'Borohydride_B': (
                     '[#5X4-;H4:1]'
                     ), # H-B hydride donors
                 'Boron_hydride_B': (
                     '[#5;H1,H2,H3,H4:1]'
                     ), # H-B hydride donors
                 'Hydrostannane_Sn': (
                     '[#50X4;H1,H2,H3,H4:1]'
                     ), # H-Sn hydride donors
                 'Hydrogermane_Ge': (
                     '[#32X4;H1,H2,H3,H4:1]'
                     ), # H-Ge hydride donors
                 'Dihydropyridine_C4': (
                     '[#6X4H1:1]1-[#6X3]=[#6X3]-[#7X3]-[#6X3]=[#6X3]-[#6X4]-1'
                     ), # H-C hydride donors: 1,4-dihydropyridine-like (Hantzsch ester / NADH-like)
                 'Carboxylate_O': (
                     '[O-:1][CX3](=O)[#6,#1]'
                     ), # Carboxylates(O- nucleophiles)
                 'Carbonate_O': (
                     '[O-:1][CX3](=O)[OX2]'
                     ), # Carbonates (O- nucleophiles)
                 'Amidate_N': (
                     '[#7-:1][CX3](=O)[#6,#1]'
                     ), # Amide anion (N nucleophile under strong base)
                 'Indole_like_pi_C': (
                     '[cH:1]1[nH,n]c2ccccc2c1'
                     ), # Indole-like motif; maps a pyrrolic-ring carbon adjacent to [nH]/[n]
                 'Pyrrole_like_pi_C': (
                     '[cH:1]1[nH,n]ccc1'
                     ), # Pyrrole-like motif; maps a carbon adjacent to [nH]/[n]
                 'Nitronate_O': (
                     '[#8-:1]-[#7+;X3](=[#6])[#8-]'
                     ), # Nitronate O- (maps the O- as :1; this single pattern can match either O- in symmetric cases)
                 'Nitroalkyl_C_anion': (
                     '[#6-:1]-[#7+;X3](=O)[#8-]'
                     ), # Nitroalkyl anion resonance form (alpha-carbon anion next to nitro)
                 'Nitrite_O': (
                     '[#8-:1]-[#7](=O)'
                     ), # Nitrite (NO2-): ambident; include both O-site and N-site candidates
                 'Nitrite_N': (
                     '[#7:1](=O)[#8-]'
                     ), # Nitrite (NO2-): ambident; include both O-site and N-site candidates
                 'Sulfinate_O': (
                     '[#8-:1]-[#16](=O)[#6,#1]'
                     ), # Sulfinate (RSO2-): ambident; include O-site and S-site candidates
                 'Sulfinate_S': (
                     '[#16:1](=O)([#8-])[#6,#1]'
                     ), # Sulfinate (RSO2-): ambident; include O-site and S-site candidates
                 'Oximate_O': (
                     '[#8-:1]-[#7]=[#6]'
                     ), # Oximate (O-centered)
                 'Hydroxamate_O': (
                     '[#8-:1]-[#7]-[#6](=O)'
                     ), # Hydroxamate (O-centered)
                 'Peroxy_O': (
                     '[#8-:1]-[#8]-[#6,#1]'
                     ), # Peroxy anion
                }

# ### BEGIN SMe (OBS! Require changes to run_rxn() in molecule_formats.py) ###
# n_smirks_dict = {'Ether': '[OX2:1]([#6;!$(C([OX2])[#7,#8,#15,#16,F,Cl,Br,I]);!$([#6]=[#8]):2])[#6;!$(C([OX2])[#7,#8,#15,#16]);!$([#6]=[#8]):3]>>[CH3][#16X2][OX3+:1]([*:2])[*:3]',
#                  'Ketone': '[OX1H0:1]=[#6X3:2]([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]>>[CH3][#16X2][OX2H0:1][#6X3+:2]([*:3])[*:4]',
#                  'Amide': '[OX1:1]=[CX3;$([CX3][#6]),$([CX3H]):2][#7X3;!R:3]>>[CH3][#16X2][OX2:1][CX3:2]=[#7X3+:3]',
#                  'Enolate': '[#6;$([#6]=,:[#6]-[#8-]),$([#6-]-[#6]=,:[#8]):1]~[#6:2]~[#8;$([#8-]-[#6]=,:[#6]),$([#8]=,:[#6]-[#6-]):3]>>[CH3][#16X2][#6+0:1][*:2]=[#8+0:3]',
#                  'Aldehyde': '[OX1:1]=[$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):2]>>[CH3][#16X2][OX2:1][#6+:2]',
#                  'Imine': '[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):1]=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):2]>>[CH3][#16X2][NX3+:1]=[*:2]',
#                  'Nitranion': '[#7X2-:1]>>[CH3][#16X2][#7X3+0:1]',
#                  'Carbanion': '[#6-;!$([#6X1-]#[#7,#8,#15,#16]):1]>>[CH3][#16X2][#6+0:1]',
#                  'Nitronate': '[#6:1]=[#7+:2](-[#8-:3])-[#8-:4]>>[CH3][#16X2][#6:1][#7+:2](=[#8+0:3])-[*:4]',
#                  'Ester': '[OX1:1]=[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]):2][#8X2H0:3][#6;!$(C=[O,N,S]):4]>>[CH3][#16X2][OX2:1][#6X3+:2][*:3][*:4]',
#                  'Carboxylic acid': '[OX1:1]=[CX3;$([R0][#6]),$([H1R0]):2][$([OX2H]),$([OX1-]):3]>>[CH3][#16X2][OX2:1][CX3+:2][*:3]',
#                  'Amine': '[#7+0;$([N;R;!$([#7X2]);$(N-[#6]);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]),$([NX3+0;!$([#7X3][CX3;$([CX3][#6]),$([CX3H])]=[OX1])]),$([NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]):1]>>[CH3][#16X2][#7+:1]',
#                  'Cyanoalkyl/nitrile anion': '[C:1]=[C:2]=[#7X1-:3]>>[CH3][#16X2][C:1][C:2]#[#7X1+0:3]',
#                  'Nitrile': '[NX1:1]#[CX2;!$(CC=C=[#7X1-]);!$(CC=C):2]>>[CH3][#16X2][NX2+:1]#[*:2]',
#                  'Isonitrile': '[CX1-:1]#[NX2+:2]>>[CH3][#16X2][CX2+0:1]#[NX2+:2]',
#                  'Phenol': '[OX2H:1][$(c(c)c),$([#6X3;R](=[#6X3;R])[#6X3;R]):2]>>[CH3][#16X2][OX3+1:1][*:2]', # added due to rxn100
#                  'Silyl_ether': '[#8X2H0:1][#14X4:2]([!#1:3])([!#1:4])[!#1:5]>>[CH3][#16X2][#8X3H0+:1][*:2]([*:3])([*:4])[*:5]', # added due to rxn100
#                  'Pyridine_like_nitrogen': '[#7X2;$([nX2](:*):*),$([#7X2;R](=[*;R])[*;R]):1]>>[CH3][#16X2][#7X3+:1]', # added due to rxn100
#                  'anion_with_charge_minus1': '[*-:1]>>[CH3][#16X2][*+0:1]', # added to capture additional sites 
#                  'double_bond': '[*;!$([!X4;!#1;!#6:1])+0:1]=[*+0:2]>>[CH3][#16X2][*:1]-[*+1:2]', # added to capture additional sites 
#                  'double_bond_neighbouratom_with_charge_plus1': '[*;!$([!X4;!#1;!#6:1])+0:1]=[*+1:2]>>[CH3][#16X2][*:1]-[*+2:2]', # added to capture additional sites 
#                  'triple_bond': '[*;!$([!X4;!#1;!#6:1])+0:1]#[*+0:2]>>[CH3][#16X2][*:1]=[*+1:2]', # added to capture additional sites 
#                  'triple_bond_neighbouratom_with_charge_plus1': '[*;!$([!X4;!#1;!#6:1])+0:1]#[*+1:2]>>[CH3][#16X2][*:1]=[*+2:2]', # added to capture additional sites 
#                  'atom_with_lone_pair': '[!X4;!#1;!#6:1]>>[CH3][#16X2][*+1:1]', # added to capture additional sites 
#                 }
# ### END SMe ###


e_smirks_dict = {'Oxonium': '[#6:1]=[O+;!$([O]~[!#6]);!$([S]*~[#7,#8,#15,#16]):2]>>[CH3][#6:1][O+0:2]',
                 'Carbocation': '[#6+:1]>>[CH3][#6+0:1]',
                 'Ketone': '[#6X3:1](=[OX1:2])([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]>>[CH3][#6X4:1](-[OX1-:2])([*:3])[*:4]',
                 'Amide': '[CX3;$([CX3][#6]),$([CX3H]):1](=[OX1:2])[#7X3;!R:3]>>[CH3][CX4:1](-[OX1-:2])[*:3]',
                 'Ester': '[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]),$([#6X3][OX2H0]):1](=[OX1:2])[#8X2H0:3][#6;!$(C=[O,N,S]):4]>>[CH3][#6X4:1](-[OX1-:2])[*:3][*:4]',
                 'Iminium': '[CX3:1]=[NX3+;!$(N([#8-])[#8-]):2]>>[CH3][CX4:1]-[NX3+0:2]',
                 'Michael acceptor': '[CX3;!R:1]=[CX3:2][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]):3]>>[CH3][CX4:1]-[CX3-:2][*:3]',
                 'Imine': '[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):1]=[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):2]>>[CH3][CX4:1]-[NX2-:2]',
                 'Aldehyde': '[CX3;$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):1]=[OX1:2]>>[CH3][CX4:1]-[OX1-:2]',
                 'Anhydride': '[CX3:1](=[OX1:2])[OX2:3][CX3:4]=[OX1:5]>>[CH3][CX4:1](-[OX1-:2])[*:3][*:4]=[*:5]', # added due to rxn100
                 'Acyl Halide': '[CX3:1](=[OX1:2])[ClX1,BrX1,IX1:3]>>[CH3][CX4:1](-[OX1-:2])[*:3]', # added due to rxn100
                 'cation_with_charge_plus1': '[*+:1]>>[CH3][*+0:1]', # added to capture additional sites 
                 'double_bond': '[*+0:1]=[*+0:2]>>[CH3][*:1]-[*-1:2]', # added to capture additional sites 
                 'double_bond_neighbouratom_with_charge_plus1': '[*+0:1]=[*+1:2]>>[CH3][*:1]-[*+0:2]', # added to capture additional sites
                 'triple_bond': '[*+0:1]#[*+0:2]>>[CH3][*:1]=[*-1:2]', # added to capture additional sites 
                 'triple_bond_neighbouratom_with_charge_plus1': '[*+0:1]#[*+1:2]>>[CH3][*:1]=[*+0:2]', # added to capture additional sites
                }

extracted_e_smirks_dict = {
                 'Oxonium': (
                    '[#6:1]=[O+;!$([O]~[!#6]);!$([S]*~[#7,#8,#15,#16]):2]'
                    ),
                 'Carbocation': (
                    '[#6+:1]'
                    ),
                 'Ketone_C': (
                    '[#6X3:1](=[OX1:2])([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]'
                    ),
                 'Amide_C': (
                    '[CX3;$([CX3][#6]),$([CX3H]):1](=[OX1:2])[#7X3;!R:3]'
                    ),
                 'Ester_C': (
                    '[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]),$([#6X3][OX2H0]):1](=[OX1:2])[#8X2H0:3][#6;!$(C=[O,N,S]):4]'
                    ),
                 'Iminium': (
                    '[CX3:1]=[NX3+;!$(N([#8-])[#8-]):2]'
                    ),
                 'Michael_acceptor': (
                    '[CX3:1]=[CX3:2][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]):3]'
                    ),  # Refined: Include cyclic michael acceptors
                 'Imine': (
                    '[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):1]=[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):2]'
                    ),
                 'Aldehyde_C': (
                    '[CX3;$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):1]=[OX1:2]'
                    ),
                 'Anhydride_C': (
                    '[CX3:1](=[OX1:2])[OX2:3][CX3:4]=[OX1:5]'
                    ), 
                 'Acyl_Halide': (
                    '[CX3:1](=[OX1:2])[ClX1,BrX1,IX1:3]'
                    ), 
                 }

added_e_smirks_dict = {
                 'Cumulene': (
                    '[CX2:1](=[$([NX2]),$([#6;!a]):2])=[$([OX1]),$([SX1]),$([NX2]):3]'
                    ),  # Cumulene-center C (e.g. iso(thio)cyanate, carbodiimide, ketene, ketenimine)
                 'Nitroso_azo': (
                    '[NX2:1]=[$([OX1]),$([NX2]):2]'
                    ),  # Nitroso and azo nitrogen
                 'Nitrile_C': (
                    '[CX2:1]#[NX1:2]'
                    ),  # Nitrile carbon
                 'Activated_sulfur_oxo': (
                    '[#16+0;!$(S(=O)[O-,OX2]);!$(S(=O)[NX3]):1](=[O:2])([!#6:3])'
                    ),  # Activated only: 
                 'Sulfenyl_halide': (
                    '[#16X2+0:1]([F,Cl,Br,I:2])[!#1;!a;!$([#16](=O)):3]'
                    ), 
                 'Phosphoryl': (
                    '[#15+0:1](=[O,S;X1:2])([!#6:3])'
                    ), 
                 'Haloaryl': (
                    '[#6X3;a;R:1]([F,Cl,Br,I:2])(-[*;a;R:4])=[*;a;R:3]'
                    ),
                 }

# ### BEGIN SMe (OBS! Require changes to run_rxn() in molecule_formats.py) ###
# e_smirks_dict = {'Oxonium': '[#6:1]=[O+;!$([O]~[!#6]);!$([S]*~[#7,#8,#15,#16]):2]>>[CH3][#16X2][#6:1][O+0:2]',
#                  'Carbocation': '[#6+:1]>>[CH3][#16X2][#6+0:1]',
#                  'Ketone': '[#6X3:1](=[OX1:2])([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]>>[CH3][#16X2][#6X4:1](-[OX1-:2])([*:3])[*:4]',
#                  'Amide': '[CX3;$([CX3][#6]),$([CX3H]):1](=[OX1:2])[#7X3;!R:3]>>[CH3][#16X2][CX4:1](-[OX1-:2])[*:3]',
#                  'Ester': '[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]),$([#6X3][OX2H0]):1](=[OX1:2])[#8X2H0:3][#6;!$(C=[O,N,S]):4]>>[CH3][#16X2][#6X4:1](-[OX1-:2])[*:3][*:4]',
#                  'Iminium': '[CX3:1]=[NX3+;!$(N([#8-])[#8-]):2]>>[CH3][#16X2][CX4:1]-[NX3+0:2]',
#                  'Michael acceptor': '[CX3;!R:1]=[CX3:2][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]):3]>>[CH3][#16X2][CX4:1]-[CX3-:2][*:3]',
#                  'Imine': '[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):1]=[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):2]>>[CH3][#16X2][CX4:1]-[NX2-:2]',
#                  'Aldehyde': '[CX3;$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):1]=[OX1:2]>>[CH3][#16X2][CX4:1]-[OX1-:2]',
#                  'Anhydride': '[CX3:1](=[OX1:2])[OX2:3][CX3:4]=[OX1:5]>>[CH3][#16X2][CX4:1](-[OX1-:2])[*:3][*:4]=[*:5]', # added due to rxn100
#                  'Acyl Halide': '[CX3:1](=[OX1:2])[ClX1,BrX1,IX1:3]>>[CH3][#16X2][CX4:1](-[OX1-:2])[*:3]', # added due to rxn100
#                  'cation_with_charge_plus1': '[*+:1]>>[CH3][#16X2][*+0:1]', # added to capture additional sites 
#                  'double_bond': '[*+0:1]=[*+0:2]>>[CH3][#16X2][*:1]-[*-1:2]', # added to capture additional sites 
#                  'double_bond_neighbouratom_with_charge_plus1': '[*+0:1]=[*+1:2]>>[CH3][#16X2][*:1]-[*+0:2]', # added to capture additional sites 
#                  'triple_bond': '[*+0:1]#[*+0:2]>>[CH3][#16X2][*:1]=[*-1:2]', # added to capture additional sites 
#                  'triple_bond_neighbouratom_with_charge_plus1': '[*+0:1]#[*+1:2]>>[CH3][#16X2][*:1]=[*+0:2]', # added to capture additional sites 
#                 }
# ### END SMe ###

def find_electrophilic_sites(rdkit_mol):
    copy_rdkit_mol = Chem.Mol(rdkit_mol, True)
    copy_rdkit_mol = Chem.AddHs(copy_rdkit_mol)
    Chem.Kekulize(copy_rdkit_mol)

    elec_sites = []
    elec_names = []
    elec_smirks = []
    for name, smirks in e_smirks_dict.items():
        smarts = smirks.split('>>')[0]
        subs = substructure_match(copy_rdkit_mol, Chem.MolFromSmarts(smarts), False)
        if subs:
            sites = [x[0] for x in subs]
            for site in sites:
                if site not in elec_sites:
                    elec_sites.append(site)
                    elec_names.append(name)
                    elec_smirks.append(smirks)
    
    return elec_sites, elec_names, elec_smirks


def find_nucleophilic_sites(rdkit_mol):
    copy_rdkit_mol = Chem.Mol(rdkit_mol, True)
    copy_rdkit_mol = Chem.AddHs(copy_rdkit_mol)
    Chem.Kekulize(copy_rdkit_mol)

    nuc_sites = []
    nuc_names = []
    nuc_smirks = []
    for name, smirks in n_smirks_dict.items():
        smarts = smirks.split('>>')[0]
        subs = substructure_match(copy_rdkit_mol, Chem.MolFromSmarts(smarts), False)
        if subs:
            sites = [x[0] for x in subs]
            # sites = remove_identical_atoms(copy_rdkit_mol, sites)
            for site in sites:
                if site not in nuc_sites:
                    nuc_sites.append(site)
                    nuc_names.append(name)
                    nuc_smirks.append(smirks)

    return nuc_sites, nuc_names, nuc_smirks
