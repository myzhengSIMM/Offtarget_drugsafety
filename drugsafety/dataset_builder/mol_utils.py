from typing import Dict
import os
import csv
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover
from argparse import ArgumentParser
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

Molecule = Chem.Mol


def neutralize_atoms(mol: Molecule) -> Molecule:

    """ See https://www.rdkit.org/docs/Cookbook.html
    :param mol: Chem.Mol
    :return: uncharged Chem.Mol
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def remove_salts_solvents(smiles: str) -> str:
    """Remove solvents and ions. This function returns only one fragment in molecule that largest
    number of heavy atoms and the removed fragment might not be an actual solvent or salt

    Parameters
    ----------
    smiles : str
        SMILES

    Returns
    -------
    str
        smiles
    """
    mols = []
    num_heavy_atoms = []
    mol_complex = smiles.split(".")
    if len(mol_complex) > 1:
        for el in mol_complex:
            mol = Chem.MolFromSmiles(str(el))
            mols.append(mol)
            num_heavy_atoms.append(mol.GetNumHeavyAtoms())
        smiles = Chem.MolToSmiles(mols[np.argmax(num_heavy_atoms)])

    return smiles


def wash_smiles(smiles: str, neutralize: bool = False) -> str:
    """
    Filters out invalid SMILES.

    :param neutralize:
    :param smiles: A str of SMILES.
    :return: A MoleculeDataset with only valid molecules.
    """

    if smiles == '':
        return None
    else:
        smiles_update = remove_salts_solvents(smiles)
        mol = Chem.MolFromSmiles(smiles_update)
        if neutralize:
            mol = neutralize_atoms(mol)
        smiles_update = Chem.MolToSmiles(mol,
                                         # isomericSmiles=False,
                                         # allBondsExplicit=True,
                                         # allHsExplicit=True,
                                         kekuleSmiles=True
                                         )

        if mol is None:
            return None
        if mol.GetNumHeavyAtoms() == 0:
            return None

        return smiles_update




