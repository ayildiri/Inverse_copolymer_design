import data_processing.featurization as ft
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from data_processing.rdkit_poly import make_polymer_mol, make_mol
from copy import deepcopy
from torch_geometric.data import Data
import torch
import re

def extract_attachment_points(smiles):
    """
    Extract attachment point numbers from SMILES
    
    Args:
        smiles: SMILES string with [*:X] attachment points
        
    Returns:
        List of attachment point numbers
    """
    attachment_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', smiles)])
    return attachment_points

def is_homopolymer(poly_input):
    """
    Determine if the polymer input represents a homopolymer
    
    Args:
        poly_input: The polymer input string
        
    Returns:
        bool: True if homopolymer, False if copolymer
    """
    parts = poly_input.split('|')[0].split('.')
    if len(parts) != 2:
        return False
        
    # Extract original monomers (before attachment point renumbering)
    mona = parts[0]
    monb = parts[1]
    
    # Remove attachment point numbers and compare base structures
    base_mona = re.sub(r'\[\*:\d+\]', '[*]', mona)
    base_monb = re.sub(r'\[\*:\d+\]', '[*]', monb)
    
    return base_mona == base_monb

def poly_smiles_to_graph(poly_input, poly_label1=None, poly_label2=None, poly_input_nocan=None, property_values=None):
    '''
    Turns adjusted polymer smiles string into PyG data objects with robust handling
    
    Args:
        poly_input: Polymer SMILES string in proper format
        poly_label1: First property value (for backward compatibility)
        poly_label2: Second property value (for backward compatibility)
        poly_input_nocan: Non-canonical version (optional)
        property_values: List of property values (new flexible approach)
        
    Returns:
        PyG Data object with flexible property attributes
    '''
    # Check polymer input format and report diagnostics
    parts = poly_input.split('|')
    if len(parts) < 3:
        print(f"Warning: Polymer input doesn't have enough parts: {poly_input}")
        print("Expected format: MonA.MonB|stoichiometry|connectivity")
        # Add missing parts if needed
        if len(parts) == 1:
            poly_input = f"{parts[0]}|0.5|0.5|<1-1:1.0:1.0"
        elif len(parts) == 2:
            poly_input = f"{parts[0]}|{parts[1]}|<1-1:1.0:1.0"
        parts = poly_input.split('|')
    
    # Extract SMILES part and analyze attachment points
    smiles_part = parts[0]
    attachment_points = extract_attachment_points(smiles_part)
    homopolymer_detected = is_homopolymer(poly_input)
    
    print(f"Processing polymer: {poly_input}")
    print(f"Found attachment points in SMILES: {attachment_points}")
    if homopolymer_detected:
        print("Detected homopolymer structure")

    # Handle flexible property input
    if property_values is not None:
        # New flexible approach - use property_values list
        if poly_label1 is not None or poly_label2 is not None:
            print("Warning: Both property_values and individual labels provided. Using property_values.")
        properties = property_values
    else:
        # Backward compatibility - use individual labels
        properties = []
        if poly_label1 is not None:
            properties.append(poly_label1)
        if poly_label2 is not None:
            properties.append(poly_label2)

    # Pre-process connectivity to ensure hyphens, not dots
    if len(parts) >= 3:
        parts[2] = parts[2].replace('.', '-')
        poly_input = '|'.join(parts)

    # Turn into RDKIT mol object with careful error handling
    try:
        mol = (make_polymer_mol(poly_input.split("|")[0], 0, 0,  # smiles
                            fragment_weights=poly_input.split("|")[1:-1]),  # fraction of each fragment
           poly_input.split("<")[1:])
    except Exception as e:
        print(f"Error creating polymer molecule: {str(e)}")
        raise ValueError(f"Failed to create polymer molecule from {poly_input}")

    # Set some variables needed later
    n_atoms = 0  # number of atoms
    n_bonds = 0  # number of bonds
    f_atoms = []  # mapping from atom index to atom features
    f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
    w_bonds = []  # mapping from bond index to bond weight
    w_atoms = []  # mapping from atom index to atom weight
    a2b = []  # mapping from atom index to incoming bond indices
    b2a = []  # mapping from bond index to the index of the atom the bond is coming from
    b2revb = []  # mapping from bond index to the index of the reverse bond

    # ============
    # Polymer mode
    # ============
    m = mol[0]  # RDKit Mol object
    rules = mol[1]  # [str], list of rules
    
    try:
        # Pre-kekulize molecule to avoid kekulization issues later
        # This is critical for aromatic structures
        Chem.Kekulize(m, clearAromaticFlags=False)
    except Exception as e:
        print(f"Warning: Kekulization issue detected (this is common for complex aromatics): {str(e)}")
        print("Will use partial sanitization instead")
        try:
            # Try partial sanitization without kekulization
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception as e2:
            print(f"Warning: Partial sanitization also failed: {str(e2)}")
            print("Proceeding without full sanitization")
    
    # Parse rules on monomer connections
    # Use robust rule parsing that handles unexpected formats
    try:
        polymer_info, degree_of_polym = ft.parse_polymer_rules(rules)
        print(f"Polymer connections: {polymer_info}")
    except Exception as e:
        print(f"Error parsing polymer rules: {str(e)}")
        # Provide fallback connectivity
        print("Using default connectivity rules")
        polymer_info = [('1', '1', 0.5, 0.5)]
        degree_of_polym = 1.0
    
    # Make molecule editable
    rwmol = Chem.rdchem.RWMol(m)
    
    # Tag atoms for features and get bond types
    try:
        rwmol, r_bond_types = ft.tag_atoms_in_repeating_unit(rwmol)
    except Exception as e:
        print(f"Error in tagging atoms: {str(e)}")
        raise ValueError(f"Failed to tag atoms in polymer molecule")

    # -----------------
    # Get atom features with error handling
    # -----------------
    try:
        f_atoms = [ft.atom_features(atom) for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]
        w_atoms = [atom.GetDoubleProp('w_frag') for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]
    except Exception as e:
        print(f"Error computing atom features: {str(e)}")
        raise ValueError(f"Failed to compute atom features")

    n_atoms = len(f_atoms)
    if n_atoms == 0:
        raise ValueError(f"No core atoms found in polymer molecule")

    # Remove wildcard atoms cleanly
    try:
        # Create a copy for backup in case of errors
        rwmol_backup = Chem.Mol(rwmol)
        
        # Wildcard removal with careful handling
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
        while indices:
            # Get neighboring atom to preserve valence
            atom = rwmol.GetAtomWithIdx(indices[0])
            neighbors = atom.GetNeighbors()
            neighbor_idx = None
            if neighbors:
                neighbor_idx = neighbors[0].GetIdx()
                # Ensure neighbor has correct valence
                neighbor = rwmol.GetAtomWithIdx(neighbor_idx)
                # Store original valence
                original_expl_h = neighbor.GetNumExplicitHs()
                
            # Remove the wildcard atom
            rwmol.RemoveAtom(indices[0])
            
            # Fix neighbor atom if needed
            if neighbor_idx is not None:
                # Adjust index if needed
                if neighbor_idx > indices[0]:
                    neighbor_idx -= 1
                if neighbor_idx < rwmol.GetNumAtoms():
                    neighbor = rwmol.GetAtomWithIdx(neighbor_idx)
                    # Try to maintain valid valence
                    try:
                        neighbor.UpdatePropertyCache()
                    except:
                        # If valence issue, try adjusting H count
                        neighbor.SetNumExplicitHs(original_expl_h)
                        
            # Update indices
            indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
        
        # Sanitize without kekulization (to avoid errors with aromatic systems)
        try:
            Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception as e:
            print(f"Warning: Sanitizing molecule after removing wildcards failed: {str(e)}")
            print("Using simplified sanitization")
            try:
                Chem.SanitizeMol(rwmol, 
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS | 
                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | 
                    Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                    catchErrors=True)
            except:
                print("Warning: Using original molecule structure with only wildcard atoms removed")
                # Keep aromaticity flags intact
                AllChem.SetAromaticity(rwmol)
    except Exception as e:
        print(f"Error removing wildcard atoms: {str(e)}")
        print("Fallback to original molecule")
        rwmol = rwmol_backup
        
    # Initialize atom to bond mapping for each atom
    for _ in range(n_atoms):
        a2b.append([])

    # ---------------------------------------
    # Get bond features for separate monomers
    # ---------------------------------------
    try:
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                bond = rwmol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                # get bond features
                f_bond = ft.bond_features(bond)

                # append bond features twice
                f_bonds.append(f_bond)
                f_bonds.append(f_bond)
                # Update index mappings
                b1 = n_bonds
                b2 = b1 + 1
                a2b[a2].append(b1)  # b1 = a1 --> a2
                b2a.append(a1)
                a2b[a1].append(b2)  # b2 = a2 --> a1
                b2a.append(a2)
                b2revb.append(b2)
                b2revb.append(b1)
                w_bonds.extend([1.0, 1.0])  # edge weights of 1.
                n_bonds += 2
    except Exception as e:
        print(f"Error processing intra-monomer bonds: {str(e)}")
        raise ValueError(f"Failed to process bonds within monomers")

    # ---------------------------------------------------
    # Get bond features for bonds between repeating units
    # ---------------------------------------------------
    try:
        # we duplicate the monomers present to allow (i) creating bonds that exist already within the same
        # molecule, and (ii) collect the correct bond features, e.g., for bonds that would otherwise be
        # considered in a ring when they are not, when e.g. creating a bond between 2 atoms in the same ring.
        rwmol_copy = deepcopy(rwmol)
        _ = [a.SetBoolProp('OrigMol', True) for a in rwmol.GetAtoms()]
        _ = [a.SetBoolProp('OrigMol', False) for a in rwmol_copy.GetAtoms()]
        
        # create an editable combined molecule
        cm = Chem.CombineMols(rwmol, rwmol_copy)
        cm = Chem.RWMol(cm)
        
        # Process all connections between monomers
        processed_connections = []
        
        # Check available R tags
        r_tags = set()
        for atom in cm.GetAtoms():
            if atom.HasProp('R') and atom.GetProp('R'):
                for tag in atom.GetProp('R').split('*')[1:]:
                    r_tags.add(tag)
        print(f"Available R tags in molecule: {sorted(list(r_tags))}")
    
        # For all possible bonds between monomers:
        # add bond -> compute bond features -> add to bond list -> remove bond
        for r1, r2, w_bond12, w_bond21 in polymer_info:
            print(f"Processing connection {r1}-{r2}")
            
            # Skip duplicate connections 
            connection_key = f"{r1}-{r2}"
            if connection_key in processed_connections:
                print(f"Skipping duplicate connection {connection_key}")
                continue
            processed_connections.append(connection_key)
            
            # Find atoms connected to each attachment point
            a1 = None  # idx of atom 1 in rwmol
            a2 = None  # idx of atom 1 in rwmol --> to be used by MolGraph
            _a2 = None  # idx of atom 1 in cm --> to be used by RDKit
            
            # Also track which R groups are actually in the molecule
            r1_exists = False
            r2_exists = False
            
            for atom in cm.GetAtoms():
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
            
            # Remap connection points if needed (especially for homopolymers)
            if homopolymer_detected:
                # For attachment points that don't exist, try mapping
                # 3→1 and 4→2 for homopolymers
                if not r1_exists and int(r1) > 2:
                    remapped_r1 = str((int(r1) - 2) if int(r1) <= 4 else r1)
                    print(f"Remapping homopolymer connection {r1}-{r2} to {remapped_r1}-{r2}")
                    
                    for atom in cm.GetAtoms():
                        if atom.HasProp('R'):
                            r_prop = atom.GetProp('R')
                            if f'*{remapped_r1}' in r_prop and atom.GetBoolProp('OrigMol') is True:
                                a1 = atom.GetIdx()
                                r1_exists = True
                
                if not r2_exists and int(r2) > 2:
                    remapped_r2 = str((int(r2) - 2) if int(r2) <= 4 else r2)
                    print(f"Remapping homopolymer connection {r1}-{r2} to {r1}-{remapped_r2}")
                    
                    for atom in cm.GetAtoms():
                        if atom.HasProp('R'):
                            r_prop = atom.GetProp('R')
                            if f'*{remapped_r2}' in r_prop:
                                if atom.GetBoolProp('OrigMol') is True:
                                    a2 = atom.GetIdx()
                                elif atom.GetBoolProp('OrigMol') is False:
                                    _a2 = atom.GetIdx()
                                    r2_exists = True
            
            # Skip if necessary atoms not found
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
            
            # Determine bond type
            order1 = None
            order2 = None
            try:
                # Get bond types with safe fallbacks
                if f'*{r1}' in r_bond_types:
                    order1 = r_bond_types[f'*{r1}']
                elif homopolymer_detected and int(r1) > 2 and f'*{str(int(r1)-2)}' in r_bond_types:
                    # For homopolymers, try mapping 3→1, 4→2
                    order1 = r_bond_types[f'*{str(int(r1)-2)}']
                else:
                    print(f"Warning: No bond type for *{r1} found, using SINGLE")
                    order1 = Chem.rdchem.BondType.SINGLE
                    
                if f'*{r2}' in r_bond_types:
                    order2 = r_bond_types[f'*{r2}']
                elif homopolymer_detected and int(r2) > 2 and f'*{str(int(r2)-2)}' in r_bond_types:
                    # For homopolymers, try mapping 3→1, 4→2
                    order2 = r_bond_types[f'*{str(int(r2)-2)}']
                else:
                    print(f"Warning: No bond type for *{r2} found, using SINGLE")
                    order2 = Chem.rdchem.BondType.SINGLE
                
                # Ensure bond types match or use a default
                if order1 != order2:
                    print(f'Warning: two atoms are trying to be bonded with different bond types: '
                                f'{order1} vs {order2}. Using {order1}.')
                
                # Create bond and validate molecule
                cm.AddBond(a1, _a2, order=order1)
                try:
                    # Use more flexible sanitization
                    Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception as e:
                    print(f"Warning: Failed to sanitize molecule after adding bond: {str(e)}")
                    print("Using simplified sanitization")
                    Chem.SanitizeMol(cm, 
                        Chem.SanitizeFlags.SANITIZE_FINDRADICALS | 
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                        catchErrors=True)

                # Get bond features
                bond = cm.GetBondBetweenAtoms(a1, _a2)
                if bond is None:
                    print(f"Warning: Could not find bond between {a1} and {_a2}. Skipping connection.")
                    continue
                f_bond = ft.bond_features(bond)

                # Add to graph structure
                f_bonds.append(f_bond)
                f_bonds.append(f_bond)

                # Update index mappings
                b1 = n_bonds
                b2 = b1 + 1
                a2b[a2].append(b1)  # b1 = a1 --> a2
                b2a.append(a1)
                a2b[a1].append(b2)  # b2 = a2 --> a1
                b2a.append(a2)
                b2revb.append(b2)
                b2revb.append(b1)
                w_bonds.extend([w_bond12, w_bond21])  # add edge weights
                n_bonds += 2

                # remove the bond to prepare for next connection
                cm.RemoveBond(a1, _a2)
                try:
                    Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except:
                    print("Warning: Failed to sanitize after removing bond. Continuing anyway.")
                    # Try just fixing aromaticity
                    AllChem.SetAromaticity(cm)
                
            except Exception as e:
                print(f"Error processing connection {r1}-{r2}: {str(e)}")
                print("Skipping this connection")
                continue
    except Exception as e:
        print(f"Error processing inter-monomer bonds: {str(e)}")
        # We don't raise an error here, as we can still proceed with intra-monomer bonds

    # ------------------
    # Make ensemble molecular weight for self-supervised learning
    # ------------------
    try:
        monomer_smiles = poly_input.split("|")[0].split('.')
        # if smiles are canonicalized we want to also keep the non-canonical input
        if poly_input_nocan:
            monomer_smiles_nocan = poly_input_nocan.split("|")[0].split('.')
        else: 
            monomer_smiles_nocan = None
            
        monomer_weights = poly_input.split("|")[1:-1]

        mol_mono_1 = make_mol(monomer_smiles[0], 0, 0)
        mol_mono_2 = make_mol(monomer_smiles[1], 0, 0)

        M_ensemble = float(monomer_weights[0]) * Descriptors.ExactMolWt(
            mol_mono_1) + float(monomer_weights[1]) * Descriptors.ExactMolWt(mol_mono_2)
    except Exception as e:
        print(f"Error calculating molecular weight: {str(e)}")
        print("Using default molecular weight")
        monomer_smiles = poly_input.split("|")[0].split('.')
        monomer_smiles_nocan = None
        M_ensemble = 100.0  # default value

    # -------------------------------------------
    # Make own PyTorch geometric data object
    # -------------------------------------------
    # PyG data object is: Data(x, edge_index, edge_attr, y, **kwargs)
    try:
        # create node feature matrix,
        X = torch.empty(n_atoms, len(f_atoms[0]))
        for i in range(n_atoms):
            X[i, :] = torch.FloatTensor(f_atoms[i])
        # associated atom weights we already found
        W_atoms = torch.FloatTensor(w_atoms)

        # get edge_index and associated edge attribute and edge weight
        # edge index is of shape [2, num_edges],  edge_attribute of shape [num_edges, num_node_features], edge_weights = [num_edges]
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, len(f_bonds[0]))
        W_bonds = torch.empty(0, dtype=torch.float)
        for i in range(n_atoms):
            # pick atom
            atom = torch.LongTensor([i])
            # find number of bonds to that atom. a2b is mapping from atom to bonds
            num_bonds = len(a2b[i])

            # create graph connectivity for that atom
            atom_repeat = atom.repeat(1, num_bonds)
            # a2b is mapping from atom to incoming bonds, need b2a to map these bonds to atoms they originated from
            neigh_atoms = [b2a[bond] for bond in a2b[i]]
            edges = torch.LongTensor(neigh_atoms).reshape(1, num_bonds)
            edge_idx_atom = torch.cat((atom_repeat, edges), dim=0)
            # append connectivity of atom to edge_index
            edge_index = torch.cat((edge_index, edge_idx_atom), dim=1)

            # Find weight of bonds
            # weight of bonds attached to atom
            W_bond_atom = torch.FloatTensor([w_bonds[bond] for bond in a2b[i]])
            W_bonds = torch.cat((W_bonds, W_bond_atom), dim=0)

            # find edge attribute
            edge_attr_atom = torch.FloatTensor([f_bonds[bond] for bond in a2b[i]])
            edge_attr = torch.cat((edge_attr, edge_attr_atom), dim=0)

        # Create PyG Data object with flexible properties
        graph_data = {
            'x': X,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'monomer_smiles': monomer_smiles,
            'W_atoms': W_atoms,
            'W_bonds': W_bonds,
            'M_ensemble': M_ensemble,
            'monomer_smiles_nocan': monomer_smiles_nocan
        }
        
        # Add properties dynamically
        for i, prop_value in enumerate(properties):
            if i == 0:
                graph_data['y1'] = prop_value
            elif i == 1:
                graph_data['y2'] = prop_value
            else:
                # For properties beyond y2, use y3, y4, etc.
                graph_data[f'y{i+1}'] = prop_value
        
        # Create the graph object
        graph = Data(**graph_data)
        
        print(f"Successfully created graph with {n_atoms} atoms and {n_bonds//2} bonds")
        return graph
        
    except Exception as e:
        print(f"Error creating PyG data object: {str(e)}")
        raise ValueError(f"Failed to create graph object from polymer")


# Legacy wrapper function for backward compatibility
def poly_smiles_to_graph_legacy(poly_input, poly_label1, poly_label2, poly_input_nocan=None):
    """
    Legacy wrapper for backward compatibility.
    Use poly_smiles_to_graph with property_values parameter for new code.
    """
    return poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)


# New flexible function for multiple properties
def poly_smiles_to_graph_flexible(poly_input, property_values, poly_input_nocan=None):
    """
    Create polymer graph with flexible number of properties.
    
    Args:
        poly_input: Polymer SMILES string
        property_values: List of property values
        poly_input_nocan: Non-canonical version (optional)
    
    Returns:
        PyG Data object with flexible property attributes
    """
    return poly_smiles_to_graph(poly_input, None, None, poly_input_nocan, property_values)
