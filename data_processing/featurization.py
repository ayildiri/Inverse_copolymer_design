from typing import List, Tuple, Union
from itertools import zip_longest
from copy import deepcopy
from collections import Counter
import logging
import re

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np

from data_processing.rdkit_poly import make_mol, make_polymer_mol


class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """

    def __init__(self) -> None:

        # Atom feature sizes
        # self.MAX_ATOMIC_NUM = 100
        self.MAX_ATOMIC_NUM = 8
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'stoichiometry': [0.25, 0.5, 0.75],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(
            range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(
            len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 1 #excluding mass
            # len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.POLYMER = False
        self.ADDING_H = False


# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()


def reset_featurization_parameters(logger: logging.Logger = None) -> None:
    """
    Function resets feature parameter values to defaults by replacing the parameters instance.
    """
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
    debug('Setting molecule featurization parameters to default.')
    global PARAMS
    PARAMS = Featurization_parameters()


def get_atom_fdim(overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the atom feature vector.
    """
    return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + PARAMS.EXTRA_ATOM_FDIM


def set_explicit_h(explicit_h: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with explicit Hs.

    :param explicit_h: Boolean whether to keep explicit Hs from input.
    """
    PARAMS.EXPLICIT_H = explicit_h


def set_adding_hs(adding_hs: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with adding the Hs to them.

    :param adding_hs: Boolean whether to add Hs to the molecule.
    """
    PARAMS.ADDING_H = adding_hs


def set_polymer(polymer: bool) -> None:
    """
    Sets whether RDKit molecules are two monomers of a co-polymer.

    :param polymer: Boolean whether input is two monomer units of a co-polymer.
    """
    PARAMS.POLYMER = polymer


def set_reaction(reaction: bool, mode: str) -> None:
    """
    Sets whether to use a reaction or molecule as input and adapts feature dimensions.

    :param reaction: Boolean whether to except reactions as input.
    :param mode: Reaction mode to construct atom and bond feature vectors.

    """
    PARAMS.REACTION = reaction
    if reaction:
        PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1
        PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
        PARAMS.REACTION_MODE = mode


def is_explicit_h() -> bool:
    r"""Returns whether to use retain explicit Hs"""
    return PARAMS.EXPLICIT_H


def is_adding_hs() -> bool:
    r"""Returns whether to add explicit Hs to the mol"""
    return PARAMS.ADDING_H


def is_reaction() -> bool:
    r"""Returns whether to use reactions as input"""
    return PARAMS.REACTION


def is_polymer() -> bool:
    r"""Returns whether to the molecule is a polymer"""
    return PARAMS.POLYMER


def reaction_mode() -> str:
    r"""Returns the reaction mode"""
    return PARAMS.REACTION_MODE


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    PARAMS.EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the bond feature vector.
    """

    return (not overwrite_default_bond) * PARAMS.BOND_FDIM + PARAMS.EXTRA_BOND_FDIM + \
           (not atom_messages) * \
        get_atom_fdim(overwrite_default_atom=overwrite_default_atom)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    PARAMS.EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def onek_encoding_unk_atoms(value: int) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    Modified version only for atoms

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    # All atomic numbers in the dataset (only 8 types of atoms)
    g_truth = [6, 7, 8, 9, 16, 17, 35, 53]
    encoding = [0] * (len(g_truth) + 1)
    index = g_truth.index(value) if value in g_truth else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        try:
            # features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            features = onek_encoding_unk_atoms(atom.GetAtomicNum()) + \
                onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
                onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
                onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
                onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
                onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
                onek_encoding_unk(atom.GetDoubleProp('w_frag'), PARAMS.ATOM_FEATURES['stoichiometry']) + \
                [1 if atom.GetIsAromatic() else 0]
                # [1 if atom.GetIsAromatic() else 0] + \
                # [atom.GetMass() * 0.01]  # scaled to about the same range as other features
            
            if functional_groups is not None:
                features += functional_groups
        except Exception as e:
            print(f"Error computing atom features for {atom.GetSymbol()}: {str(e)}")
            # Provide default features
            features = [0] * PARAMS.ATOM_FDIM
            features[-1] = 1  # Set unknown flag
            
    return features


def atom_features_zeros(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM -
                   1)  # set other features to zero
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
    else:
        try:
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)
            ]
            fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        except Exception as e:
            print(f"Error computing bond features: {str(e)}")
            # Default bond features
            fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
            
    return fbond


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol):
    """
    Build a dictionary of mapping atom indices in the reactants to the products.

    :param mol_reac: An RDKit molecule of the reactants.
    :param mol_prod: An RDKit molecule of the products.
    :return: A dictionary of corresponding reactant and product atom indices.
    """
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()])
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx())
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())
    return reac_id_to_prod_id, only_prod_ids, only_reac_ids


def tag_atoms_in_repeating_unit(mol):
    """
    Tags atoms that are part of the core units, as well as atoms serving to identify attachment points. In addition,
    create a map of bond types based on what bonds are connected to R groups in the input.
    """
    try:
        atoms = [a for a in mol.GetAtoms()]
        neighbor_map = {}  # map R group to index of atom it is attached to
        r_bond_types = {}  # map R group to bond type

        # Show molecule info for diagnostics
        smiles = Chem.MolToSmiles(mol)
        print(f"Tagging atoms in molecule: {smiles}")
        attachment_points = re.findall(r'\[\*:(\d+)\]', smiles)
        
        if attachment_points:
            print(f"Found attachment points: {attachment_points}")
        else:
            print("Warning: No attachment points found in molecule")
            
        # Detect potential homopolymer structure
        if '.' in smiles:
            parts = smiles.split('.')
            if len(parts) == 2:
                # Remove attachment points for comparison
                base_part1 = re.sub(r'\[\*:\d+\]', '[*]', parts[0])
                base_part2 = re.sub(r'\[\*:\d+\]', '[*]', parts[1])
                
                if base_part1 == base_part2:
                    print("Detected homopolymer structure")

        # go through each atoms and: (i) get index of attachment atoms, (ii) tag all non-R atoms
        for atom in atoms:
            try:
                # if R atom
                if '*' in atom.GetSmarts():
                    # get index of atom it is attached to
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) != 1:
                        print(f"Warning: R atom at index {atom.GetIdx()} has {len(neighbors)} neighbors (expected 1)")
                        
                        # Use first neighbor if available
                        if len(neighbors) > 0:
                            neighbor_idx = neighbors[0].GetIdx()
                        else:
                            # No neighbors, skip this atom
                            print(f"Skipping R atom with no neighbors at index {atom.GetIdx()}")
                            atom.SetBoolProp('core', False)
                            continue
                    else:
                        neighbor_idx = neighbors[0].GetIdx()
                    
                    # *1, *2, ...
                    r_tag = atom.GetSmarts().strip('[]').replace(':', '')
                    print(f"Found R group {r_tag} at index {atom.GetIdx()}, connected to atom {neighbor_idx}")
                    
                    neighbor_map[r_tag] = neighbor_idx
                    # tag it as non-core atom
                    atom.SetBoolProp('core', False)
                    
                    # create a map R --> bond type
                    try:
                        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
                        if bond is None:
                            print(f"Warning: No bond found between atoms {atom.GetIdx()} and {neighbor_idx}")
                            r_bond_types[r_tag] = Chem.rdchem.BondType.SINGLE
                        else:
                            r_bond_types[r_tag] = bond.GetBondType()
                            print(f"Bond type for {r_tag}: {bond.GetBondType()}")
                    except Exception as bond_error:
                        print(f"Error getting bond type: {str(bond_error)}")
                        # Default to SINGLE
                        r_bond_types[r_tag] = Chem.rdchem.BondType.SINGLE
                # if not R atom
                else:
                    # tag it as core atom
                    atom.SetBoolProp('core', True)
            except Exception as atom_error:
                print(f"Error processing atom {atom.GetIdx()}: {str(atom_error)}")
                # Default to safe option
                atom.SetBoolProp('core', False)

        # use the map created to tag attachment atoms
        for atom in atoms:
            try:
                if atom.GetIdx() in neighbor_map.values():
                    r_tags = [k for k, v in neighbor_map.items() if v == atom.GetIdx()]
                    atom.SetProp('R', ''.join(r_tags))
                else:
                    atom.SetProp('R', '')
            except Exception as tag_error:
                print(f"Error setting R property for atom {atom.GetIdx()}: {str(tag_error)}")
                atom.SetProp('R', '')

        # Check results
        r_atoms = [a.GetIdx() for a in atoms if a.HasProp('core') and not a.GetBoolProp('core')]
        print(f"Tagged {len(r_atoms)} atoms as R groups")
        print(f"R group to atom mapping: {neighbor_map}")
        
        return mol, r_bond_types
    except Exception as e:
        print(f"Error in tag_atoms_in_repeating_unit: {str(e)}")
        # Attempt recovery with minimal info
        for atom in mol.GetAtoms():
            atom.SetBoolProp('core', '*' not in atom.GetSmarts())
            if atom.GetBoolProp('core'):
                atom.SetProp('R', '')
            else:
                atom.SetProp('R', atom.GetSmarts().strip('[]').replace(':', ''))
                
        # Create a basic r_bond_types dictionary
        r_bond_types = {}
        for atom in mol.GetAtoms():
            if '*' in atom.GetSmarts():
                r_tag = atom.GetSmarts().strip('[]').replace(':', '')
                r_bond_types[r_tag] = Chem.rdchem.BondType.SINGLE
                
        return mol, r_bond_types


def remove_wildcard_atoms(rwmol):
    """
    Removes wildcard atoms ([*]) from a molecule with proper handling of aromaticity and bonds.
    
    Args:
        rwmol: RDKit RWMol object
        
    Returns:
        RDKit RWMol object with wildcards removed and proper structure
    """
    try:
        # Create a backup copy for fallback
        mol_backup = Chem.Mol(rwmol)
        # Get indices of wildcard atoms
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
        
        # Pre-adjust all aromatic atoms to help with kekulization
        for atom in rwmol.GetAtoms():
            if atom.GetIsAromatic():
                # Ensure aromatic atoms have proper valence
                atom.SetNoImplicit(False)
        
        # Track neighbors of wildcard atoms for post-processing
        neighbor_indices = []
        for idx in indices:
            atom = rwmol.GetAtomWithIdx(idx)
            neighbors = atom.GetNeighbors()
            neighbor_indices.extend([n.GetIdx() for n in neighbors])
        neighbor_indices = list(set(neighbor_indices))
        
        # Pre-kekulize to preserve aromatic systems
        try:
            # Careful kekulization - maintain aromatic flags
            Chem.Kekulize(rwmol, clearAromaticFlags=False)
        except Exception as e:
            print(f"Pre-kekulization warning (this is normal for complex aromatics): {str(e)}")
            # Try partial sanitization
            Chem.SanitizeMol(rwmol, 
                Chem.SanitizeFlags.SANITIZE_ALL ^ 
                Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                catchErrors=True)
        
        # Remove wildcard atoms one by one
        removed_count = 0
        while indices:
            idx = indices[0]
            atom = rwmol.GetAtomWithIdx(idx)
            
            # Store neighbor info before removal for valence adjustment
            neighbors = []
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                # Adjust index for atoms removed already
                if neighbor_idx > idx:
                    neighbor_idx -= removed_count
                neighbors.append(neighbor_idx)
            
            # Remove the atom
            rwmol.RemoveAtom(idx)
            removed_count += 1
            
            # Refresh indices list
            indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
            
            # Process each neighbor to fix valence
            for neighbor_idx in neighbors:
                if neighbor_idx < rwmol.GetNumAtoms():
                    neighbor = rwmol.GetAtomWithIdx(neighbor_idx)
                    neighbor.SetNoImplicit(False)
                    # Adjust explicit Hs to maintain proper valence
                    try:
                        neighbor.UpdatePropertyCache(strict=False)
                    except:
                        # If valence issues, just ensure it has valid structure
                        neighbor.SetNumExplicitHs(0)
                        
        # Final sanitization with careful handling
        try:
            # Try partial sanitization first (skip kekulization)
            Chem.SanitizeMol(rwmol, 
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS | 
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | 
                Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | 
                Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                catchErrors=True)
            
            # Try kekulization last
            try:
                Chem.Kekulize(rwmol, clearAromaticFlags=False)
            except:
                print("Warning: Kekulization failed after wildcard removal (not critical)")
                # Just set aromaticity
                AllChem.SetAromaticity(rwmol)
        except Exception as e:
            print(f"Warning: Sanitization failed after removing wildcards: {str(e)}")
            print("Maintaining basic aromaticity only")
            AllChem.SetAromaticity(rwmol)
        
        # Ensure all atoms have valid valence
        for atom in rwmol.GetAtoms():
            atom.SetNoImplicit(False)
            
        return rwmol
    except Exception as e:
        print(f"Error in remove_wildcard_atoms: {str(e)}")
        print("Using fallback approach")
        
        # Simple fallback approach
        try:
            # Just remove the wildcard atoms without complex processing
            indices = [a.GetIdx() for a in mol_backup.GetAtoms() if '*' in a.GetSmarts()]
            rwmol_simple = Chem.RWMol(mol_backup)
            
            # Remove from the end to avoid index issues
            for idx in sorted(indices, reverse=True):
                rwmol_simple.RemoveAtom(idx)
            
            # Set aromaticity without full sanitization
            AllChem.SetAromaticity(rwmol_simple)
            return rwmol_simple
        except:
            print("Fallback also failed - using minimal processing")
            # Return the original with minimal processing
            return mol_backup


def detect_homopolymer(smiles):
    """
    Detect if a polymer SMILES string represents a homopolymer with 
    repeated attachment points.
    
    Args:
        smiles: SMILES string with attachment points
        
    Returns:
        bool: True if homopolymer detected, False otherwise
    """
    # Extract attachment points
    attachment_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', smiles)])
    if not attachment_points:
        return False
    
    # Check homopolymer by structure
    if '.' in smiles:
        parts = smiles.split('.')
        if len(parts) == 2:
            # Remove attachment points for comparison
            base_part1 = re.sub(r'\[\*:\d+\]', '[*]', parts[0])
            base_part2 = re.sub(r'\[\*:\d+\]', '[*]', parts[1])
            
            if base_part1 == base_part2:
                print(f"Detected homopolymer with base structure: {base_part1}")
                return True
        
    # Check if it's a homopolymer (only attachments 1,2 are present)
    if len(attachment_points) <= 2 and all(ap <= 2 for ap in attachment_points):
        print("Detected homopolymer with attachments 1,2")
        return True
        
    # Check if it looks like our renumbered format (1,2,3,4)
    if len(attachment_points) == 4 and attachment_points == [1, 2, 3, 4]:
        print("Detected homopolymer with properly numbered attachments 1,2,3,4")
        return True
    
    return False


def parse_polymer_rules(rules):
    """
    Parse polymer connectivity rules with robust handling for various formats.
    
    Args:
        rules: List of connectivity rules (strings)
        
    Returns:
        tuple: (polymer_info, degree_of_polymerization)
    """
    polymer_info = []
    counter = Counter()  # used for validating the input
    
    # Print input for diagnostics
    rules_str = ", ".join(rules) if rules else "empty rules"
    print(f"Parsing polymer rules: {rules_str}")

    # check if deg of polymerization is provided
    if rules and '~' in rules[-1]:
        Xn = float(rules[-1].split('~')[1])
        rules[-1] = rules[-1].split('~')[0]
    else:
        Xn = 1.

    # Try to detect if this is a homopolymer from the rules
    is_homopolymer = any(('1-3:' in rule or '1.3:' in rule or 
                          '2-4:' in rule or '2.4:' in rule) for rule in rules)
    
    for rule in rules:
        # handle edge case where we have no rules, and rule is empty string
        if rule == "":
            continue
            
        try:
            # QC of input string - try with original format first
            if len(rule.split(':')) != 3:
                print(f'Warning: incorrect format for input information "{rule}", using default 1-1:1:1')
                idx1, idx2 = "1", "1"  # Default connectivity between positions 1 and 1
                w12 = 1.0  # Default weight
                w21 = 1.0  # Default weight
            else:
                # Try to split by hyphen, but handle case where there's no hyphen
                connection_part = rule.split(':')[0]
                if '-' in connection_part:
                    # Normal hyphen format
                    idx1, idx2 = connection_part.split('-')
                elif '.' in connection_part:
                    # Handle older dot format
                    print(f'Warning: connection format "{connection_part}" uses dots instead of hyphens')
                    idx1, idx2 = connection_part.split('.')
                else:
                    print(f'Warning: connection format "{connection_part}" does not contain hyphen or dot, using default 1-1')
                    idx1, idx2 = "1", "1"  # Default connectivity between positions 1 and 1
                
                # Apply homopolymer remapping if needed
                if is_homopolymer:
                    original_idx1, original_idx2 = idx1, idx2
                    # For homopolymers, allow mapping between 1↔3 and 2↔4
                    if idx1 == "3":
                        idx1 = "1"
                    elif idx1 == "4":
                        idx1 = "2"
                        
                    if idx2 == "3":
                        idx2 = "1"
                    elif idx2 == "4":
                        idx2 = "2"
                    
                    if idx1 != original_idx1 or idx2 != original_idx2:
                        print(f'Remapped homopolymer connection {original_idx1}-{original_idx2} to {idx1}-{idx2}')
                
                # Extract weights
                weights_parts = rule.split(':')[1:]
                if len(weights_parts) >= 2:
                    try:
                        w12 = float(weights_parts[0])  # weight for bond R_idx1 -> R_idx2
                        w21 = float(weights_parts[1])  # weight for bond R_idx2 -> R_idx1
                    except ValueError:
                        print(f'Warning: invalid weight values in "{rule}", using default weights 1.0')
                        w12 = 1.0
                        w21 = 1.0
                else:
                    print(f'Warning: missing weight information in "{rule}", using default weights 1.0')
                    w12 = 1.0
                    w21 = 1.0
        except Exception as e:
            print(f'Error parsing rule "{rule}": {str(e)}. Using default values 1-1:1:1')
            idx1, idx2 = "1", "1"  # Default connectivity
            w12 = 1.0  # Default weight
            w21 = 1.0  # Default weight
            
        # Add to polymer info and counter
        polymer_info.append((idx1, idx2, w12, w21))
        counter[idx1] += float(w21)
        counter[idx2] += float(w12)

    # validate input: sum of incoming weights should be one for each vertex
    for k, v in counter.items():
        if np.isclose(v, 1.0) is False:
            print(f'Warning: sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]. Proceeding anyway.')
    
    # Show final parsed rules
    print(f"Parsed {len(polymer_info)} polymer connections:")
    for i, (idx1, idx2, w12, w21) in enumerate(polymer_info):
        print(f"  {i+1}. {idx1}-{idx2} with weights {w12:.2f}, {w21:.2f}")
            
    return polymer_info, 1. + np.log10(Xn)


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating
        """
        self.is_polymer = is_polymer()
        self.polymer_info = []
        self.is_reaction = is_reaction()
        self.is_explicit_h = is_explicit_h()
        self.is_adding_hs = is_adding_hs()
        self.reaction_mode = reaction_mode()

        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            if self.is_reaction:
                mol = (make_mol(mol.split(">")[0], self.is_explicit_h, self.is_adding_hs),
                       make_mol(mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs))
            elif self.is_polymer:
                # TODO: use BigSMILES notation as input for polymers with a dedicated parser
                mol = (make_polymer_mol(mol.split("|")[0], self.is_explicit_h, self.is_adding_hs,  # smiles
                                        fragment_weights=mol.split("|")[1:-1]),  # fraction of each fragment
                       mol.split("<")[1:])  # edges between fragments
            else:
                mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs)
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.degree_of_polym = 1  # degree of polymerization
        self.f_atoms = []  # mapping from atom index to atom features
        # mapping from bond index to concat(in_atom, bond) features
        self.f_bonds = []
        self.w_bonds = []  # mapping from bond index to bond weight
        self.w_atoms = []  # mapping from atom index to atom weight
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features

        # =============
        # Standard mode
        # =============
        if not self.is_reaction and not self.is_polymer:
            # Get atom features
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
            self.w_atoms = [1.] * len(mol.GetAtoms())
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist()
                                    for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms,
                                    descs in zip(self.f_atoms, atom_features_extra)]

            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra atom features')

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.w_bonds.extend([1.0, 1.0])  # edge weights of 1.
                    self.n_bonds += 2

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra bond features')

        # ============
        # Polymer mode
        # ============
        if not self.is_reaction and self.is_polymer:
            # TODO: infer bond type from bond to R groups
            m = mol[0]  # RDKit Mol object
            rules = mol[1]  # [str], list of rules
            
            # Check for homopolymer
            smiles_part = Chem.MolToSmiles(m)
            is_homopolymer = detect_homopolymer(smiles_part)
            
            # parse rules on monomer connections
            self.polymer_info, self.degree_of_polym = parse_polymer_rules(
                rules)
            # make molecule editable
            rwmol = Chem.rdchem.RWMol(m)
            # tag (i) attachment atoms and (ii) atoms for which features needs to be computed
            # also get map of R groups to bonds types, e.f. r_bond_types[*1] -> SINGLE
            rwmol, r_bond_types = tag_atoms_in_repeating_unit(rwmol)

            # -----------------
            # Get atom features
            # -----------------
            # for all 'core' atoms, i.e. not R groups, as tagged before. Do this here so that atoms linked to
            # R groups have the correct saturation
            self.f_atoms = [atom_features(
                atom) for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]
            self.w_atoms = [atom.GetDoubleProp(
                'w_frag') for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]

            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist()
                                    for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms,
                                    descs in zip(self.f_atoms, atom_features_extra)]

            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(rwmol)} is different from the length of '
                                 f'the extra atom features')

            # remove R groups -> now atoms in rdkit Mol object have the same order as self.f_atoms
            rwmol = remove_wildcard_atoms(rwmol)

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # ---------------------------------------
            # Get bond features for separate monomers
            # ---------------------------------------
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = rwmol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.w_bonds.extend([1.0, 1.0])  # edge weights of 1.
                    self.n_bonds += 2

            # ---------------------------------------------------
            # Get bond features for bonds between repeating units
            # ---------------------------------------------------
            # we duplicate the monomers present to allow (i) creating bonds that exist already within the same
            # molecule, and (ii) collect the correct bond features, e.g., for bonds that would otherwise be
            # considered in a ring when they are not, when e.g. creating a bond between 2 atoms in the same ring.
            rwmol_copy = deepcopy(rwmol)
            _ = [a.SetBoolProp('OrigMol', True) for a in rwmol.GetAtoms()]
            _ = [a.SetBoolProp('OrigMol', False)
                 for a in rwmol_copy.GetAtoms()]
            # create an editable combined molecule
            cm = Chem.CombineMols(rwmol, rwmol_copy)
            cm = Chem.RWMol(cm)

            # Check available attachment points from R tags
            r_tags = set()
            for atom in cm.GetAtoms():
                if atom.HasProp('R') and atom.GetProp('R'):
                    for tag in atom.GetProp('R').split('*')[1:]:
                        r_tags.add(tag)
            
            print(f"Available R tags in molecule: {sorted(list(r_tags))}")

            # Process each connection with proper error handling
            processed_connections = []
            
            # for all possible bonds between monomers:
            # add bond -> compute bond features -> add to bond list -> remove bond
            for r1, r2, w_bond12, w_bond21 in self.polymer_info:
                print(f"Processing connection {r1}-{r2}")
                
                # Skip duplicate connections (can happen with homopolymer remapping)
                connection_key = f"{r1}-{r2}"
                if connection_key in processed_connections:
                    print(f"Skipping duplicate connection {connection_key}")
                    continue
                processed_connections.append(connection_key)
                
                # get index of attachment atoms
                a1 = None  # idx of atom 1 in rwmol
                a2 = None  # idx of atom 1 in rwmol --> to be used by MolGraph
                _a2 = None  # idx of atom 1 in cm --> to be used by RDKit
                
                # Check if the R groups actually exist in the molecule
                r1_exists = False
                r2_exists = False
                
                for atom in cm.GetAtoms():
                    # Check if the R groups exist
                    if atom.HasProp('R'):
                        r_prop = atom.GetProp('R')
                        if f'*{r1}' in r_prop:
                            r1_exists = True
                        if f'*{r2}' in r_prop:
                            r2_exists = True
                        
                        # take a1 from a fragment in the original molecule object
                        if f'*{r1}' in r_prop and atom.GetBoolProp('OrigMol') is True:
                            a1 = atom.GetIdx()
                        # take _a2 from a fragment in the copied molecule object, but a2 from the original
                        if f'*{r2}' in r_prop:
                            if atom.GetBoolProp('OrigMol') is True:
                                a2 = atom.GetIdx()
                            elif atom.GetBoolProp('OrigMol') is False:
                                _a2 = atom.GetIdx()

                # Handle homopolymer case by remapping attachment points if needed
                if is_homopolymer:
                    # For homopolymer with only attachment points 1,2
                    # Try to map attachment points 3,4 to 1,2
                    if int(r1) > 2 and not r1_exists:
                        remapped_r1 = str((int(r1) - 2) if int(r1) <= 4 else r1)
                        print(f"Remapping attachment point {r1} to {remapped_r1} for homopolymer")
                        
                        for atom in cm.GetAtoms():
                            if atom.HasProp('R'):
                                r_prop = atom.GetProp('R')
                                if f'*{remapped_r1}' in r_prop and atom.GetBoolProp('OrigMol') is True:
                                    a1 = atom.GetIdx()
                                    r1_exists = True
                    
                    if int(r2) > 2 and not r2_exists:
                        remapped_r2 = str((int(r2) - 2) if int(r2) <= 4 else r2)
                        print(f"Remapping attachment point {r2} to {remapped_r2} for homopolymer")
                        
                        for atom in cm.GetAtoms():
                            if atom.HasProp('R'):
                                r_prop = atom.GetProp('R')
                                if f'*{remapped_r2}' in r_prop:
                                    if atom.GetBoolProp('OrigMol') is True:
                                        a2 = atom.GetIdx()
                                    elif atom.GetBoolProp('OrigMol') is False:
                                        _a2 = atom.GetIdx()
                                        r2_exists = True

                if not r1_exists:
                    print(f'Warning: attachment point [*:{r1}] not found in molecule. Skipping connection {r1}-{r2}.')
                    continue  # Skip this connection and move to the next
                    
                if not r2_exists:
                    print(f'Warning: attachment point [*:{r2}] not found in molecule. Skipping connection {r1}-{r2}.')
                    continue  # Skip this connection and move to the next

                if a1 is None:
                    print(f'Warning: cannot find atom attached to [*:{r1}]. Skipping connection {r1}-{r2}.')
                    continue  # Skip this connection
                    
                if a2 is None or _a2 is None:
                    print(f'Warning: cannot find atom attached to [*:{r2}]. Skipping connection {r1}-{r2}.')
                    continue  # Skip this connection

                try:
                    # Try to get bond types - handle missing keys for homopolymers
                    if f'*{r1}' in r_bond_types:
                        order1 = r_bond_types[f'*{r1}']
                    elif is_homopolymer and f'*{str((int(r1) - 2))}' in r_bond_types and int(r1) <= 4:
                        # Map 3->1, 4->2 for homopolymers
                        remapped_r1 = str((int(r1) - 2))
                        order1 = r_bond_types[f'*{remapped_r1}']
                    else:
                        print(f"Cannot find bond type for *{r1}, using SINGLE")
                        order1 = Chem.rdchem.BondType.SINGLE
                        
                    if f'*{r2}' in r_bond_types:
                        order2 = r_bond_types[f'*{r2}']
                    elif is_homopolymer and f'*{str((int(r2) - 2))}' in r_bond_types and int(r2) <= 4:
                        # Map 3->1, 4->2 for homopolymers
                        remapped_r2 = str((int(r2) - 2))
                        order2 = r_bond_types[f'*{remapped_r2}']
                    else:
                        print(f"Cannot find bond type for *{r2}, using SINGLE")
                        order2 = Chem.rdchem.BondType.SINGLE
                        
                    if order1 != order2:
                        print(f'Warning: two atoms are trying to be bonded with different bond types: '
                                    f'{order1} vs {order2}. Using {order1}.')
                        
                    # create bond
                    cm.AddBond(a1, _a2, order=order1)
                    
                    try:
                        # Try various sanitization approaches
                        try:
                            Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)
                        except:
                            # Try more flexible sanitization
                            Chem.SanitizeMol(cm, 
                                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                                catchErrors=True)
                    except Exception as e:
                        print(f"Warning: Sanitization failed after adding bond: {str(e)}")
                        # Just set aromaticity
                        AllChem.SetAromaticity(cm)

                    # get bond object and features
                    bond = cm.GetBondBetweenAtoms(a1, _a2)
                    if bond is None:
                        print(f"Warning: Could not find bond between atoms {a1} and {_a2}.")
                        # Create a default bond feature vector
                        f_bond = [0] * PARAMS.BOND_FDIM
                        f_bond[0] = 1  # Mark as None bond
                    else:
                        f_bond = bond_features(bond)
                        
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.w_bonds.extend([w_bond12, w_bond21])  # add edge weights
                    self.n_bonds += 2

                    # remove the bond
                    cm.RemoveBond(a1, _a2)
                    
                    try:
                        Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except:
                        # Just set aromaticity if sanitization fails
                        AllChem.SetAromaticity(cm)
                    
                except Exception as e:
                    print(f"Error adding bond {r1}-{r2}: {str(e)}. Skipping.")
                    continue

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                print(f'Warning: The number of bonds in processed polymer ({self.n_bonds/2}) is different from the length of the extra bond features ({len(bond_features_extra)})')

        # =============
        # Reaction mode
        # =============
        # TODO: add compatibility of reaction mode with new polymer code that assumes presence of
        #  self.w_bonds and self.w_atoms
        elif self.is_reaction and not self.is_polymer:
            if atom_features_extra is not None:
                raise NotImplementedError(
                    'Extra atom features are currently not supported for reactions')
            if bond_features_extra is not None:
                raise NotImplementedError(
                    'Extra bond features are currently not supported for reactions')

            mol_reac = mol[0]
            mol_prod = mol[1]
            ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)

            # Get atom features
            if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_prod']:
                # Reactant: regular atom features for each atom in the reactants, as well as zero features for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms(
                )] + [atom_features_zeros(mol_prod.GetAtomWithIdx(index)) for index in pio]

                # Product: regular atom features for each atom that is in both reactants and products (not in rio), other atom features zero,
                # regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features_zeros(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            else:  # balance
                # Reactant: regular atom features for each atom in the reactants, copy features from product side for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms(
                )] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]

                # Product: regular atom features for each atom that is in both reactants and products (not in rio), copy features from reactant side for
                # other atoms, regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]

            if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                f_atoms_diff = [list(map(lambda x, y: x - y, ii, jj))
                                for ii, jj in zip(f_atoms_prod, f_atoms_reac)]
            if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:]
                                for x, y in zip(f_atoms_reac, f_atoms_prod)]
            elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:]
                                for x, y in zip(f_atoms_reac, f_atoms_diff)]
            elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:]
                                for x, y in zip(f_atoms_prod, f_atoms_diff)]
            self.n_atoms = len(self.f_atoms)
            n_atoms_reac = mol_reac.GetNumAtoms()

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    if a1 >= n_atoms_reac and a2 >= n_atoms_reac:  # Both atoms only in product
                        bond_prod = mol_prod.GetBondBetweenAtoms(
                            pio[a1 - n_atoms_reac], pio[a2 - n_atoms_reac])
                        if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                            bond_reac = bond_prod
                        else:
                            bond_reac = None
                    elif a1 < n_atoms_reac and a2 >= n_atoms_reac:  # One atom only in product
                        bond_reac = None
                        if a1 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(
                                ri2pi[a1], pio[a2 - n_atoms_reac])
                        else:
                            bond_prod = None  # Atom atom only in reactant, the other only in product
                    else:
                        bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                        if a1 in ri2pi.keys() and a2 in ri2pi.keys():
                            # Both atoms in both reactant and product
                            bond_prod = mol_prod.GetBondBetweenAtoms(
                                ri2pi[a1], ri2pi[a2])
                        else:
                            if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                                if a1 in ri2pi.keys() or a2 in ri2pi.keys():
                                    bond_prod = None  # One atom only in reactant
                                else:
                                    bond_prod = bond_reac  # Both atoms only in reactant
                            else:
                                bond_prod = None  # One or both atoms only in reactant

                    if bond_reac is None and bond_prod is None:
                        continue

                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                        f_bond_diff = [y - x for x,
                                       y in zip(f_bond_reac, f_bond_prod)]
                    if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                        f_bond = f_bond_reac + f_bond_prod
                    elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                        f_bond = f_bond_reac + f_bond_diff
                    elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                        f_bond = f_bond_prod + f_bond_diff
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2


class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features
        self.atom_fdim = get_atom_fdim(
            overwrite_default_atom=self.overwrite_default_atom_features)
        self.bond_fdim = get_bond_fdim(overwrite_default_bond=self.overwrite_default_bond_features,
                                       overwrite_default_atom=self.overwrite_default_atom_features)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_atoms = 1
        # number of bonds (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1
        # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.a_scope = []
        # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.b_scope = []
        # list of floats with degree of polymerization used when --polymer
        self.degree_of_polym = []

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        w_atoms = [0]  # mapping from atom index to vertex weight
        w_bonds = [0]  # mapping from bond index to edge weight
        # mapping from bond index to the index of the atom the bond is coming from
        b2a = [0]
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            w_atoms.extend(mol_graph.w_atoms)
            w_bonds.extend(mol_graph.w_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

            self.degree_of_polym.append(mol_graph.degree_of_polym)

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.w_atoms = torch.FloatTensor(w_atoms)
        self.w_bonds = torch.FloatTensor(w_bonds)
        self.a2b = torch.LongTensor(
            [a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]
                                                                        ], List[Tuple[int, int]],
                                                                   List[float]]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.w_atoms, self.w_bonds, self.a2b, self.b2a, self.b2revb, \
            self.a_scope, self.b_scope, self.degree_of_polym

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(
                1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
              overwrite_default_atom_features: bool = False,
              overwrite_default_bond_features: bool = False
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol, af, bf,
                                   overwrite_default_atom_features=overwrite_default_atom_features,
                                   overwrite_default_bond_features=overwrite_default_bond_features)
                          for mol, af, bf
                          in zip_longest(mols, atom_features_batch, bond_features_batch)])
