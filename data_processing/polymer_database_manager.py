"""
Polymer Database Manager
========================

A comprehensive toolkit for processing polymer datasets and managing polymer databases
with ChemProp input generation, chemical validation, and template management.

Place this file in: /content/Inverse_copolymer_design/data_processing/

Command Line Usage:
    !python data_processing/polymer_database_manager.py -i input.csv -o output.csv
    
Programmatic Usage:
    import sys
    sys.path.append('/content/Inverse_copolymer_design')
    from data_processing.polymer_database_manager import PolymerDatabaseManager
    
    manager = PolymerDatabaseManager()
    processed_df = manager.process_new_dataset('your_data.csv')
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
import re
import logging
from typing import List, Tuple, Optional, Dict, Any, Union

# Handle imports for both command line and programmatic usage
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"Error importing RDKit: {e}")
    print("Please install RDKit with: pip install rdkit-pypi")
    sys.exit(1)

try:
    from natsort import natsorted
    NATSORT_AVAILABLE = True
except ImportError:
    print("Warning: natsort not available, using regular sort")
    natsorted = sorted
    NATSORT_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolymerDatabaseManager:
    """
    Main class for managing polymer databases with comprehensive processing capabilities
    """
    
    def __init__(self, template_path: str = None, verbose: bool = True):
        """
        Initialize the Polymer Database Manager
        
        Args:
            template_path: Path to existing template CSV file
            verbose: Whether to print detailed logs
        """
        self.template_path = template_path
        self.template_df = None
        self.verbose = verbose
        
        if template_path and os.path.exists(template_path):
            self.template_df = pd.read_csv(template_path)
            if verbose:
                logger.info(f"Loaded template with {len(self.template_df)} rows and {len(self.template_df.columns)} columns")
        
        # Default polymer configurations - NOT hardcoded, can be modified
        self.default_poly_types = ['alternating', 'block', 'random']
        self.default_compositions = ['4A_4B', '6A_2B', '2A_6B']
        self.comp_fracs = {
            '4A_4B': (0.5, 0.5),
            '6A_2B': (0.75, 0.25),
            '2A_6B': (0.25, 0.75)
        }
    
    # ========================
    # Chemical Processing Core
    # ========================
    
    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Canonicalize SMILES and convert attachment points to numbered format
        Ensures chemical validity is preserved and consistent formatting
        """
        try:
            # ✅ NEW: Pre-filter obviously invalid patterns
            if not smiles or smiles.strip() == '':
                return None
                
            # Count parentheses
            open_parens = smiles.count('(')
            close_parens = smiles.count(')')
            if abs(open_parens - close_parens) > 3:  # Too unbalanced
                if self.verbose:
                    logger.warning(f"Severely unbalanced parentheses in SMILES: {smiles}")
                return None
            
            # Handle different attachment point formats
            numbered_smiles = smiles
            
            # Convert [*] to [*:1], [*:2] etc.
            counter = 1
            while '[*]' in numbered_smiles:
                numbered_smiles = numbered_smiles.replace('[*]', f'[*:{counter}]', 1)
                counter += 1
            
            # Clean up any malformed attachment points
            numbered_smiles = re.sub(r'\*([^:\[])', r'[*:1]\1', numbered_smiles)  # Handle bare *
            
            # ✅ IMPROVED: Try original first, then attempt repair if needed
            mol = Chem.MolFromSmiles(numbered_smiles)
            
            if mol is None:
                # ✅ ENHANCED: Try auto-repair with multiple strategies
                repaired_smiles = self._attempt_smiles_repair(numbered_smiles)
                if repaired_smiles != numbered_smiles:
                    mol = Chem.MolFromSmiles(repaired_smiles)
                    if mol is not None:
                        numbered_smiles = repaired_smiles
                        if self.verbose:
                            logger.info(f"Auto-repaired SMILES: {smiles} → {repaired_smiles}")
                    else:
                        # ✅ ENHANCED: Try multiple repair strategies
                        for attempt in range(3):
                            if attempt == 0:
                                # Strategy 1: Remove all ()
                                test_smiles = repaired_smiles.replace('()', '')
                            elif attempt == 1:
                                # Strategy 2: Replace () with C
                                test_smiles = repaired_smiles.replace('()', 'C')
                            else:
                                # Strategy 3: More aggressive repair
                                test_smiles = self._aggressive_repair(repaired_smiles)
                            
                            mol = Chem.MolFromSmiles(test_smiles)
                            if mol is not None:
                                numbered_smiles = test_smiles
                                if self.verbose:
                                    logger.info(f"Auto-repaired SMILES (attempt {attempt+2}): {smiles} → {test_smiles}")
                                break
            
            if mol is None:
                return None
                
            # Ensure aromatic atoms have proper valence before canonicalization
            try:
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                
                # Ensure consistent attachment point numbering
                canonical_smiles = self._standardize_attachment_points(canonical_smiles)
                
                return canonical_smiles
            except:
                # If sanitization fails, try to fix the structure
                for atom in mol.GetAtoms():
                    if atom.GetIsAromatic():
                        atom.SetNumExplicitHs(0)
                        atom.SetNoImplicit(False)
                try:
                    Chem.SanitizeMol(mol)
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    canonical_smiles = self._standardize_attachment_points(canonical_smiles)
                    return canonical_smiles
                except:
                    return None
        except:
            return None

    def _standardize_attachment_points(self, smiles: str) -> str:
        """
        Standardize attachment point numbering to ensure consistency
        """
        # Find all attachment points
        attachment_points = re.findall(r'\[\*:(\d+)\]', smiles)
        if not attachment_points:
            return smiles
        
        # Create mapping for sequential numbering
        unique_points = sorted(set(map(int, attachment_points)))
        point_mapping = {old: new for new, old in enumerate(unique_points, 1)}
        
        # Replace with standardized numbering
        result = smiles
        for old_num, new_num in point_mapping.items():
            result = result.replace(f'[*:{old_num}]', f'[*:{new_num}]')
        
        return result

    def _attempt_smiles_repair(self, smiles: str) -> str:
        """
        Enhanced intelligent SMILES repair with pre-filtering
        """
        if '()' not in smiles:
            return smiles  # No empty parentheses to fix
        
        repaired = smiles
        
        # ✅ NEW: Pre-filter obviously corrupted patterns
        corrupted_patterns = [
            (r'\(\)\(\)', ''),                          # ()() → remove
            (r'\(\)\(\)\(\)', ''),                      # ()()() → remove
        ]
        
        for pattern, replacement in corrupted_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        # ✅ COMPREHENSIVE: Apply repairs in order of specificity
        repaired = self._repair_aromatic_rings(repaired)
        repaired = self._repair_aliphatic_rings(repaired)
        repaired = self._repair_general_patterns(repaired)
        repaired = self._repair_simple_patterns(repaired)
        
        return repaired
    
    def _repair_aromatic_rings(self, smiles: str) -> str:
        """
        Repair aromatic ring systems with comprehensive pattern matching
        """
        repaired = smiles
        
        # ✅ COMPREHENSIVE: Handle numbered aromatic rings first (most specific)
        numbered_ring_patterns = [
            # Single digit rings: c1...()...c1, c2...()...c2, etc.
            (r'c(\d)([^c]*?)c\(\)([^c]*?)c\1', r'c\1\2c\3c\1'),
            
            # Handle cases like c1cc()ccc1, c1c()cccc1, etc.
            (r'c(\d)((?:c{0,5}|[sno])*?)\(\)((?:c{0,5}|[sno])*?)c\1', r'c\1\2c\3c\1'),
        ]
        
        for pattern, replacement in numbered_ring_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        # ✅ COMPREHENSIVE: Handle aromatic chains without ring numbers
        aromatic_chain_patterns = [
            # Basic aromatic chains
            (r'cc\(\)cc', 'cccc'),           # cc()cc → cccc
            (r'ccc\(\)cc', 'ccccc'),         # ccc()cc → ccccc  
            (r'cc\(\)ccc', 'ccccc'),         # cc()ccc → ccccc
            (r'ccc\(\)c', 'cccc'),           # ccc()c → cccc
            (r'c\(\)ccc', 'cccc'),           # c()ccc → cccc
            (r'cc\(\)c', 'ccc'),             # cc()c → ccc
            (r'c\(\)cc', 'ccc'),             # c()cc → ccc
            (r'c\(\)c', 'cc'),               # c()c → cc
            
            # With heteroatoms (s, n, o)
            (r'cc\(\)([sno])', r'cc\1'),     # cc()s → ccs
            (r'c\(\)([sno])', r'c\1'),       # c()s → cs
            (r'([sno])\(\)c', r'\1c'),       # s()c → sc
            (r'([sno])c\(\)', r'\1c'),       # sc() → sc
            
            # Mixed aromatic/heteroaromatic
            (r'([cn])\(\)([cn])', r'\1c\2'), # n()c → ncc, c()n → ccn
            (r'([cn])c\(\)([cn])', r'\1cc\2'), # nc()c → ncc, cc()n → ccn
        ]
        
        for pattern, replacement in aromatic_chain_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        return repaired
    
    def _repair_aromatic_chains(self, smiles: str) -> str:
        """
        Fix aromatic chains and partial rings
        """
        repaired = smiles
        
        # ✅ COMPREHENSIVE: Cover all the patterns we're seeing
        aromatic_patterns = [
            # Benzene ring patterns (most common)
            (r'ccc\(\)cc', 'ccccc'),                # ccc()cc → ccccc
            (r'cc\(\)cc', 'cccc'),                  # cc()cc → cccc  
            (r'c\(\)cc', 'ccc'),                    # c()cc → ccc
            (r'cc\(\)c', 'ccc'),                    # cc()c → ccc
            (r'c\(\)c', 'cc'),                      # c()c → cc
            
            # Thiophene patterns 
            (r'ccc\(\)s', 'cccs'),                  # ccc()s → cccs
            (r'cc\(\)s', 'ccs'),                    # cc()s → ccs
            (r'c\(\)s', 'cs'),                      # c()s → cs
            
            # Pyridine patterns
            (r'ccc\(\)n', 'cccn'),                  # ccc()n → cccn
            (r'cc\(\)n', 'ccn'),                    # cc()n → ccn
            (r'c\(\)n', 'cn'),                      # c()n → cn
            
            # With substituents (like your error cases)
            (r'([cn])c\(\)([cn])', r'\1cc\2'),       # Insert carbon between aromatics
            (r'([cn])\(\)([cn])', r'\1c\2'),         # Insert missing carbon
            
            # Complex ring systems (like c2ccc()cc2)
            (r'c(\d+)ccc\(\)cc\1', r'c\1ccccc\1'),  # c2ccc()cc2 → c2ccccc2
            (r'c(\d+)cc\(\)cc\1', r'c\1cccc\1'),    # c2cc()cc2 → c2cccc2
            (r'c(\d+)c\(\)cc\1', r'c\1ccc\1'),      # c2c()cc2 → c2ccc2
            
            # Numbered ring with heteroatoms
            (r'c(\d+)ccc\(\)s\1', r'c\1cccs\1'),    # c1ccc()s1 → c1cccs1
            (r'c(\d+)cc\(\)s\1', r'c\1ccs\1'),      # c1cc()s1 → c1ccs1
        ]
        
        for pattern, replacement in aromatic_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        return repaired

    def _repair_aliphatic_rings(self, smiles: str) -> str:
        """
        Repair aliphatic ring systems and chains
        """
        repaired = smiles
        
        # ✅ COMPREHENSIVE: Handle numbered aliphatic rings
        aliphatic_ring_patterns = [
            # Single digit rings: C1...()...C1, C2...()...C2, etc.
            (r'C(\d)([^C]*?)C\(\)([^C]*?)C\1', r'C\1\2C\3C\1'),
            
            # Handle mixed case in rings
            (r'C(\d)((?:C{0,10})*?)\(\)((?:C{0,10})*?)C\1', r'C\1\2C\3C\1'),
        ]
        
        for pattern, replacement in aliphatic_ring_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        # ✅ COMPREHENSIVE: Handle aliphatic chains
        aliphatic_chain_patterns = [
            # Basic aliphatic chains
            (r'CC\(\)CC', 'CCCC'),           # CC()CC → CCCC
            (r'CCC\(\)CC', 'CCCCC'),         # CCC()CC → CCCCC
            (r'CC\(\)CCC', 'CCCCC'),         # CC()CCC → CCCCC
            (r'CC\(\)C', 'CCC'),             # CC()C → CCC
            (r'C\(\)CC', 'CCC'),             # C()CC → CCC
            (r'C\(\)C', 'CC'),               # C()C → CC
            
            # With substituents
            (r'C\(([^)]+)\)\(\)', r'C(\1)C'), # C(substituent)() → C(substituent)C
            (r'\(\)C\(([^)]+)\)', r'CC(\1)'), # ()C(substituent) → CC(substituent)
        ]
        
        for pattern, replacement in aliphatic_chain_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        return repaired
    
    def _repair_general_patterns(self, smiles: str) -> str:
        """
        Fix general SMILES corruption patterns using chemical intelligence
        """
        repaired = smiles
        
        # ✅ COMPREHENSIVE: Handle complex functional groups
        functional_group_patterns = [
            # Carbonyl groups
            (r'O=C\(\)([CO])', r'O=C\1'),           # O=C()O → O=CO, O=C()C → O=CC
            (r'([CO])C\(\)=O', r'\1C=O'),           # CC()=O → CC=O
            
            # Double bonds
            (r'=C\(\)C', '=CC'),                     # =C()C → =CC
            (r'C\(\)=C', 'C=C'),                     # C()=C → C=C
            (r'CC\(\)=C', 'CC=C'),                   # CC()=C → CC=C
            (r'=C\(\)CC', '=CCC'),                   # =C()CC → =CCC
            
            # Heteroatoms in chains
            (r'([SNO])\(\)([CNS])', r'\1\2'),        # S()C → SC, N()C → NC, etc.
            (r'([CNS])\(\)([SNO])', r'\1\2'),        # C()S → CS, N()O → NO, etc.
            
            # Ring junction patterns
            (r'([^=])\(\)([123456789])', r'\1\2'),   # Remove () before ring numbers
            (r'([123456789])\(\)([^=])', r'\1\2'),   # Remove () after ring numbers
        ]
        
        for pattern, replacement in functional_group_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        return repaired

    def _repair_simple_patterns(self, smiles: str) -> str:
        """
        Simple fallback patterns for any remaining () issues
        """
        repaired = smiles
        
        # ✅ FALLBACK: Last resort patterns - context-aware replacement
        remaining_patterns = [
            # In aromatic contexts, replace () with c
            (r'([a-z])\(\)([a-z])', r'\1c\2'),      # aromatic()aromatic → aromaticccaromatic
            (r'([a-z])\(\)', r'\1c'),               # aromatic() → aromaticc
            (r'\(\)([a-z])', r'c\1'),               # ()aromatic → caromatic
            
            # In aliphatic contexts, replace () with C  
            (r'([A-Z])\(\)([A-Z])', r'\1C\2'),      # ALIPHATIC()ALIPHATIC → ALIPHATICCALIPHATIC
            (r'([A-Z])\(\)', r'\1C'),               # ALIPHATIC() → ALIPHATICC
            (r'\(\)([A-Z])', r'C\1'),               # ()ALIPHATIC → CALIPHATIC
            
            # Mixed contexts - be conservative, use C
            (r'([A-Za-z])\(\)([A-Za-z])', r'\1C\2'), # letter()letter → letterCletter
            (r'([A-Za-z])\(\)', r'\1C'),            # letter() → letterC
            (r'\(\)([A-Za-z])', r'C\1'),            # ()letter → Cletter
            
            # Final fallback - just remove empty parentheses
            (r'\(\)', 'C'),                         # () → C (conservative default)
        ]
        
        for pattern, replacement in remaining_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        return repaired

    def _aggressive_repair(self, smiles: str) -> str:
        """
        Aggressive repair strategy for severely corrupted SMILES
        """
        repaired = smiles
        
        # Remove multiple consecutive empty parentheses
        repaired = re.sub(r'\(\)\(\)+', '', repaired)
        
        # Remove empty parentheses at start/end
        repaired = re.sub(r'^\(\)', '', repaired)
        repaired = re.sub(r'\(\)$', '', repaired)
        
        # Fix common ring patterns
        repaired = re.sub(r'([A-Za-z])([0-9]+)\(\)([A-Za-z])', r'\1\2\3', repaired)
        
        # Remove remaining empty parentheses
        repaired = repaired.replace('()', '')
        
        return repaired
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """
        Quick validation check for SMILES
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def get_iupac_name(self, smiles: str) -> str:
        """
        Get IUPAC name for a SMILES string with multiple fallback strategies
        """
        try:
            # Remove attachment points for name generation
            clean_smiles = re.sub(r'\[\*:\d+\]', '', smiles)
            clean_smiles = re.sub(r'\[\*\]', '', clean_smiles)
            clean_smiles = re.sub(r'\*', '', clean_smiles)
            
            mol = Chem.MolFromSmiles(clean_smiles)
            if mol is None:
                return "Invalid_SMILES"
            
            # Strategy 1: Try to use a lookup table for common polymers
            iupac_name = self._lookup_common_polymer_names(clean_smiles)
            if iupac_name:
                return iupac_name
            
            # Strategy 2: Try PubChem lookup (if available)
            iupac_name = self._try_pubchem_lookup(clean_smiles)
            if iupac_name:
                return iupac_name
            
            # Strategy 3: Generate descriptive name based on functional groups
            descriptive_name = self._generate_descriptive_name(mol, clean_smiles)
            if descriptive_name:
                return descriptive_name
            
            # Strategy 4: Fallback to molecular formula
            formula = CalcMolFormula(mol)
            return f"Polymer_{formula}"
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not generate IUPAC name for {smiles}: {e}")
            return "Unknown_compound"

    def _lookup_common_polymer_names(self, smiles: str) -> Optional[str]:
        """
        Lookup table for common polymer monomers
        """
        # Common polymer monomer names
        common_names = {
            'CCc1ccc(CC)cc1': 'diethylbenzene',
            'c1ccc2c(c1)cccc2': 'naphthalene',
            'CC(C)c1ccc(C(C)C)cc1': 'diisopropylbenzene',
            'Nc1ccccc1': 'aniline',
            'CCC': 'propane',
            'CCCC': 'butane',
            'CCCCC': 'pentane',
            'C1CCCCC1': 'cyclohexane',
            'C1CCC1': 'cyclobutane',
            'CC(C)C': 'isobutane',
            'CC(CC)C': 'isopentane',
            'CC': 'ethane',
            'C': 'methane',
            'c1ccccc1': 'benzene',
            'CC(C)(C)c1ccc(C(C)(C)C)cc1': 'di-tert-butylbenzene',
            'Oc1ccccc1': 'phenol',
            'Nc1cc(N)ccc1': 'diaminobenzene',
            'Fc1ccc(F)cc1': 'difluorobenzene',
            'Clc1ccc(Cl)cc1': 'dichlorobenzene',
            'Brc1ccc(Br)cc1': 'dibromobenzene',
        }
        
        return common_names.get(smiles)

    def _try_pubchem_lookup(self, smiles: str) -> Optional[str]:
        """
        Try to lookup IUPAC name from PubChem (placeholder for future implementation)
        """
        # This would require PubChemPy or similar
        # For now, return None to fall back to other methods
        try:
            # Example implementation (would need pubchempy):
            # import pubchempy as pcp
            # compound = pcp.get_compounds(smiles, namespace='smiles')
            # if compound and compound[0].iupac_name:
            #     return compound[0].iupac_name
            pass
        except:
            pass
        return None

    def _generate_descriptive_name(self, mol: Chem.Mol, smiles: str) -> Optional[str]:
        """
        Generate a descriptive name based on molecular features
        """
        try:
            # Basic descriptive naming based on functional groups and structure
            name_parts = []
            
            # Check for rings
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            
            if num_rings > 0:
                # Check for aromatic rings
                aromatic_atoms = [atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]
                if aromatic_atoms:
                    if num_rings == 1 and len(aromatic_atoms) == 6:
                        name_parts.append("benzene")
                    elif num_rings == 2:
                        name_parts.append("naphthalene")
                    else:
                        name_parts.append(f"aromatic_{num_rings}ring")
                else:
                    name_parts.append(f"cyclic_{num_rings}ring")
            else:
                # Aliphatic compound
                carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
                if carbon_count <= 10:
                    alkane_names = {1: "methane", 2: "ethane", 3: "propane", 4: "butane", 
                                  5: "pentane", 6: "hexane", 7: "heptane", 8: "octane",
                                  9: "nonane", 10: "decane"}
                    base_name = alkane_names.get(carbon_count, f"C{carbon_count}_alkane")
                    name_parts.append(base_name)
                else:
                    name_parts.append(f"C{carbon_count}_alkane")
            
            # Check for functional groups
            functional_groups = []
            
            # Common functional group SMARTS patterns
            fg_patterns = {
                'amine': '[NX3;H2,H1;!$(NC=O)]',
                'alcohol': '[OX2H]',
                'carboxylic_acid': '[CX3](=O)[OX2H1]',
                'ester': '[#6][CX3](=O)[OX2H0][#6]',
                'nitro': '[NX3+](=O)[O-]',
                'sulfonyl': '[SX4](=O)(=O)',
                'halide': '[F,Cl,Br,I]',
                'nitrile': '[CX2]#[NX1]',
            }
            
            for fg_name, pattern in fg_patterns.items():
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    functional_groups.append(fg_name)
            
            if functional_groups:
                name_parts.extend(functional_groups)
            
            if name_parts:
                return "_".join(name_parts) + "_derivative"
            
            # Final fallback
            formula = CalcMolFormula(mol)
            return f"compound_{formula}"
            
        except:
            return None

    def check_polymer_validity(self, mona_smiles: str, monb_smiles: str) -> Tuple[bool, str]:
        """
        Check if polymer monomers are valid and can form a proper polymer structure
        """
        # Verify monomers have attachment points
        mona_valid = mona_smiles is not None and '[*:' in mona_smiles
        monb_valid = monb_smiles is not None and '[*:' in monb_smiles
        
        if not mona_valid or not monb_valid:
            return False, "Invalid monomer SMILES or missing attachment points"
        
        # Count attachment points
        mona_points = len(re.findall(r'\[\*:\d+\]', mona_smiles))
        monb_points = len(re.findall(r'\[\*:\d+\]', monb_smiles))
        
        if mona_points < 1 or monb_points < 1:
            return False, "Insufficient attachment points"
            
        return True, "Valid polymer structure"

    def rm_duplicate_mols(self, mols: List[Chem.Mol]) -> List[Chem.Mol]:
        """Remove duplicate molecules"""
        smiles = list(set([Chem.MolToSmiles(m, canonical=True) for m in mols if m is not None]))
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        return [m for m in mols if m is not None]

    def protect_CBr(self, m: Chem.Mol) -> Chem.Mol:
        """Protect C-Br bonds by replacing with astatine"""
        while m.HasSubstructMatch(Chem.MolFromSmarts('cCBr')):
            smarts = "[*:1]CBr>>[*:1]C[At]"
            rxn = AllChem.ReactionFromSmarts(smarts)
            ps = rxn.RunReactants((m,))
            if ps:
                products = self.rm_duplicate_mols([m[0] for m in ps])
                if products:
                    m = products[0]
                else:
                    break
            else:
                break
        return m

    def deprotect_CBr(self, m: Chem.Mol) -> Chem.Mol:
        """Deprotect C-At bonds back to C-Br"""
        while m.HasSubstructMatch(Chem.MolFromSmarts('C[At]')):
            smarts = "[*:1]C[At]>>[*:1]CBr"
            rxn = AllChem.ReactionFromSmarts(smarts)
            ps = rxn.RunReactants((m,))
            if ps:
                products = self.rm_duplicate_mols([m[0] for m in ps])
                if products:
                    m = products[0]
                else:
                    break
            else:
                break
        return m

    def rm_termini(self, m: Chem.Mol) -> Chem.Mol:
        """Remove terminal groups (Br and BOO)"""
        if m is None:
            return None
            
        # Remove all Br (protect C-Br first)
        m = self.protect_CBr(m)
        while m.HasSubstructMatch(Chem.MolFromSmarts('cBr')):
            smarts = "[*:1]Br>>[*:1]"
            rxn = AllChem.ReactionFromSmarts(smarts)
            ps = rxn.RunReactants((m,))
            if ps:
                products = self.rm_duplicate_mols([m[0] for m in ps])
                if products:
                    m = products[0]
                else:
                    break
            else:
                break
        m = self.deprotect_CBr(m)
        
        # Remove all BOO
        while m.HasSubstructMatch(Chem.MolFromSmarts('[B](-O)(-O)')):
            smarts = "[*:1]([B](-O)(-O))>>[*:1]"
            rxn = AllChem.ReactionFromSmarts(smarts)
            ps = rxn.RunReactants((m,))
            if ps:
                products = self.rm_duplicate_mols([m[0] for m in ps])
                if products:
                    m = products[0]
                else:
                    break
            else:
                break
                
        return m

    def prepare_homopolymer(self, monomer_smiles: str) -> Tuple[str, str]:
        """
        Properly prepare a homopolymer by creating consistent attachment points
        """
        # Extract current attachment points
        attachment_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', monomer_smiles)])
        
        if not attachment_points:
            return monomer_smiles, monomer_smiles
        
        # For the second unit, shift attachment points by largest number
        max_point = max(attachment_points)
        monb_smiles = monomer_smiles
        
        # Replace each attachment point with offset version
        for point in sorted(attachment_points, reverse=True):
            monb_smiles = monb_smiles.replace(f'[*:{point}]', f'[*:{point + max_point}]')
        
        return monomer_smiles, monb_smiles

    def create_polymer_attachment_scheme(self, is_homopolymer: bool, mona_pts: List[int], 
                                       monb_pts: List[int], weights: Union[Tuple, float] = None) -> str:
        """
        Create a proper attachment scheme based on monomer structures
        """
        # Default weights
        if weights is None:
            a_weight = b_weight = 0.5
        elif isinstance(weights, (list, tuple)) and len(weights) >= 2:
            a_weight, b_weight = float(weights[0]), float(weights[1])
        else:
            a_weight = b_weight = float(weights)

        # Normalize weights
        total = a_weight + b_weight
        if total == 0:
            a_weight = b_weight = 0.5
        else:
            a_weight = a_weight / total
            b_weight = b_weight / total

        connectivity = ""
        for a_pt in mona_pts:
            for b_pt in monb_pts:
                connectivity += f"<{a_pt}-{b_pt}:{a_weight:.3f}:{b_weight:.3f}"
        return connectivity

    # =============================
    # ChemProp Input Generation
    # =============================
    
    def make_master_chemprop_input(self, smiA: str, smiB: str) -> str:
        """Generate master ChemProp input by removing termini and concatenating monomers"""
        try:
            mA = Chem.MolFromSmiles(smiA)
            mB = Chem.MolFromSmiles(smiB)
            
            if mA is None or mB is None:
                if self.verbose:
                    logger.warning(f"Invalid SMILES: {smiA} or {smiB}")
                return f"{smiA}.{smiB}"
            
            mA = self.rm_termini(mA)
            mB = self.rm_termini(mB)
            
            if mA is None or mB is None:
                if self.verbose:
                    logger.warning(f"Failed to process termini for: {smiA} or {smiB}")
                return f"{smiA}.{smiB}"
            
            smiA_clean = Chem.MolToSmiles(mA, canonical=True)
            smiB_clean = Chem.MolToSmiles(mB, canonical=True)
            
            return f'{smiA_clean}.{smiB_clean}'
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Error in make_master_chemprop_input: {e}")
            return f"{smiA}.{smiB}"

    def make_poly_chemprop_input(self, mona: str, monb: str, poly_type: str, 
                               fracA: float = 0.5, selfedges: bool = True) -> Optional[str]:
        """
        Create properly formatted poly_chemprop_input string with chemical validity checks
        """
        try:
            # Canonicalize monomers
            can_mona = self.canonicalize_smiles(mona)
            is_homopolymer = (monb == mona)

            if is_homopolymer:
                can_mona, can_monb = self.prepare_homopolymer(can_mona)
            else:
                can_monb = self.canonicalize_smiles(monb)

            if can_mona is None or can_monb is None:
                return None

            # For traditional polymer processing (BOO/Br system), process attachment points
            if '[*:' not in can_mona and '[*:' not in can_monb:
                # Process using original termini removal system
                mA = Chem.MolFromSmiles(mona)
                mB = Chem.MolFromSmiles(monb)
                
                if mA is None or mB is None:
                    return None
                
                # Replace BOO in monoA with attachment points
                m = mA
                for i in [1, 2]:
                    smarts = f"[*:1]([B](-O)(-O))>>[*:1]-[*{i}]"
                    rxn = AllChem.ReactionFromSmarts(smarts)
                    ps = rxn.RunReactants((m,))
                    if ps:
                        products = self.rm_duplicate_mols([m[0] for m in ps])
                        if products:
                            m = products[0]
                
                smiA_proc = Chem.MolToSmiles(m, canonical=True)
                smiA_proc = smiA_proc.replace('1*', '*:1').replace('2*', '*:2')
                
                # Replace Br in monoB with attachment points
                m = mB
                m = self.protect_CBr(m)
                for i in [3, 4]:
                    smarts = f"[*:1]Br>>[*:1]-[*{i}]"
                    rxn = AllChem.ReactionFromSmarts(smarts)
                    ps = rxn.RunReactants((m,))
                    if ps:
                        products = self.rm_duplicate_mols([m[0] for m in ps])
                        if products:
                            m = products[0]
                m = self.deprotect_CBr(m)
                
                smiB_proc = Chem.MolToSmiles(m, canonical=True)
                smiB_proc = smiB_proc.replace('3*', '*:3').replace('4*', '*:4')
                
                can_mona, can_monb = smiA_proc, smiB_proc

            # Validate structure
            is_valid, _ = self.check_polymer_validity(can_mona, can_monb)
            if not is_valid:
                return None

            # Extract attachment point numbers
            mona_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', can_mona)])
            monb_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', can_monb)])

            # Build connectivity based on polymer type
            fracB = 1.0 - fracA
            stoich = f"{fracA:.3f}|{fracB:.3f}"
            
            if poly_type == 'alternating':
                edges = '<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5'
            elif poly_type == 'block':
                if selfedges:
                    edges = [(1, 2, 3/8, 3/8), (1, 1, 3/8, 3/8), (2, 2, 3/8, 3/8),
                           (3, 4, 3/8, 3/8), (3, 3, 3/8, 3/8), (4, 4, 1/8, 1/8),
                           (1, 3, 1/8, 1/8), (1, 4, 1/8, 1/8), (2, 3, 1/8, 1/8), (2, 4, 1/8, 1/8)]
                else:
                    edges = [(1, 2, 6/8, 6/8), (3, 4, 6/8, 6/8),
                           (1, 3, 1/8, 1/8), (1, 4, 1/8, 1/8), (2, 3, 1/8, 1/8), (2, 4, 1/8, 1/8)]
                edges = "".join([f"<{e[0]}-{e[1]}:{e[2]:.3f}:{e[3]:.3f}" for e in edges])
            elif poly_type == 'random':
                if selfedges:
                    edges = '<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25'
                else:
                    edges = '<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.5:0.5<3-4:0.5:0.5'
            else:
                # Use generic attachment scheme
                weights = (fracA, fracB)
                edges = self.create_polymer_attachment_scheme(is_homopolymer, mona_points, monb_points, weights)

            return f"{can_mona}.{can_monb}|{stoich}|{edges}"
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Error in make_poly_chemprop_input: {e}")
            return None

    def _extract_monomers_from_poly_input(self, poly_input: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract monomer SMILES from poly_chemprop_input format
        Format: monA.monB|stoich|connectivity
        """
        try:
            if pd.isna(poly_input) or not isinstance(poly_input, str):
                return None, None
            
            # Split by first pipe to get monomers part
            parts = poly_input.split('|')
            if len(parts) < 1:
                return None, None
            
            monomers_part = parts[0]
            
            # Split by dot to get individual monomers
            if '.' in monomers_part:
                monomers = monomers_part.split('.')
                if len(monomers) >= 2:
                    return monomers[0].strip(), monomers[1].strip()
                elif len(monomers) == 1:
                    # Homopolymer case
                    return monomers[0].strip(), monomers[0].strip()
            
            return None, None
        except:
            return None, None

    # ========================
    # Database Management
    # ========================
    
    def generate_poly_ids(self, df: pd.DataFrame, existing_ids: set = None) -> List[str]:
        """Generate unique polymer IDs that continue from existing template"""
        if existing_ids is None:
            existing_ids = set()
        
        # ✅ NEW: Parse existing IDs to find the highest number and detect naming pattern
        max_id = -1
        id_prefix = ""
        id_format = "numeric"  # Default format
        
        if existing_ids:
            for existing_id in existing_ids:
                try:
                    # ✅ NEW: Handle different ID formats
                    if existing_id.startswith('p_'):
                        # Format: p_12345
                        id_prefix = "p_"
                        id_format = "p_prefix"
                        num = int(existing_id.replace('p_', ''))  # ✅ Extract 42966 from p_42966
                        max_id = max(max_id, num)
                    elif '_' in existing_id:
                        # Format: 12345_6 
                        parts = existing_id.split('_')
                        if len(parts) >= 2 and parts[0].isdigit():
                            num = int(parts[0])
                            max_id = max(max_id, num)
                            id_format = "underscore"
                    elif existing_id.isdigit():
                        # Format: 12345
                        num = int(existing_id)
                        max_id = max(max_id, num)
                        id_format = "numeric"
                except (ValueError, AttributeError):
                    continue
        
        # ✅ NEW: Generate new IDs continuing from the highest found
        poly_ids = []
        unique_pairs = df[['monoA', 'monoB']].drop_duplicates()
        
        # ✅ NEW: Start from the next number after max_id
        current_id = max_id + 1  # ✅ Start from 42967 if max was 42966
        
        for idx, (_, row) in enumerate(unique_pairs.iterrows()):
            # ✅ NEW: Generate ID based on detected format
            if id_format == "p_prefix":
                new_id = f"p_{current_id}"  # ✅ Generates p_42967, p_42968, etc.
            elif id_format == "underscore":
                new_id = f"{current_id}_{idx}"
            else:
                new_id = str(current_id)
            
            # ✅ NEW: Ensure uniqueness
            while new_id in existing_ids:
                current_id += 1
                if id_format == "p_prefix":
                    new_id = f"p_{current_id}"
                elif id_format == "underscore":
                    new_id = f"{current_id}_{idx}"
                else:
                    new_id = str(current_id)
            
            poly_ids.append(new_id)
            existing_ids.add(new_id)
            current_id += 1
        
        # Map back to original dataframe
        pair_to_id = dict(zip(zip(unique_pairs['monoA'], unique_pairs['monoB']), poly_ids))
        result_ids = [pair_to_id[(row['monoA'], row['monoB'])] for _, row in df.iterrows()]
    
        # ✅ NEW: Logging for debugging
        if self.verbose:
            logger.info(f"Generated poly_ids continuing from {max_id} using format: {id_format}")
            logger.info(f"New ID range: {poly_ids[0] if poly_ids else 'None'} to {poly_ids[-1] if poly_ids else 'None'}")
        
        return result_ids

    def expand_polymer_variants(self, df: pd.DataFrame, poly_types: List[str] = None, 
                              compositions: List[str] = None) -> pd.DataFrame:
        """
        Expand dataset to include different polymer types and compositions
        FLEXIBLE: Can override default polymer types and compositions
        """
        if poly_types is None:
            poly_types = self.default_poly_types
        if compositions is None:
            compositions = self.default_compositions
            
        expanded_rows = []
        
        for _, row in df.iterrows():
            # Check if it's a homopolymer
            is_homo = row.get('monoA', '') == row.get('monoB', '') or pd.isna(row.get('monoB', ''))
            
            if is_homo:
                # For homopolymers, only use alternating with 50:50 ratio
                for poly_type in ['alternating']:
                    for comp in ['4A_4B']:
                        new_row = row.copy()
                        new_row['poly_type'] = f"{poly_type} (homopolymer)"  # ✅ NEW: Add (homopolymer)
                        new_row['comp'] = comp
                        new_row['fracA'] = 0.5
                        new_row['fracB'] = 0.5
                        # Ensure monoB is same as monoA for homopolymer
                        if pd.isna(new_row.get('monoB', '')) or new_row.get('monoB', '') == '':
                            new_row['monoB'] = new_row['monoA']
                        expanded_rows.append(new_row)
            else:
                # For copolymers, use specified combinations
                for poly_type in poly_types:
                    # Alternating only supports 4A_4B
                    if poly_type == 'alternating':
                        comps = ['4A_4B']
                    else:
                        comps = compositions
                    
                    for comp in comps:
                        new_row = row.copy()
                        new_row['poly_type'] = poly_type
                        new_row['comp'] = comp
                        new_row['fracA'], new_row['fracB'] = self.comp_fracs.get(comp, (0.5, 0.5))
                        expanded_rows.append(new_row)
        
        return pd.DataFrame(expanded_rows)

    # ========================
    # Interactive Processing
    # ========================
    
    def detect_target_columns(self, df: pd.DataFrame, exclude_columns: List[str] = None, 
                         auto_exclude_patterns: List[str] = None) -> List[str]:
        """
        Automatically detect potential target columns with flexible exclusion patterns
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # ✅ NEW: Auto-exclude common non-target patterns
        if auto_exclude_patterns is None:
            auto_exclude_patterns = [
                # Structure columns
                'id', 'smiles', 'canonical', 'iupac', 'formula',
                # Polymer-specific columns  
                'mono', 'poly', 'frac', 'comp', 'stoich', 'connectivity',
                # ChemProp columns
                'chemprop', 'master', 'input',
                # Common metadata
                'source', 'reference', 'notes', 'comments', 'url'
            ]
        
        # ✅ NEW: Build comprehensive exclusion list
        comprehensive_exclude = set(exclude_columns)
        
        # Add columns that match patterns (case-insensitive)
        for col in df.columns:
            col_lower = col.lower()
            for pattern in auto_exclude_patterns:
                if pattern in col_lower:
                    comprehensive_exclude.add(col)
                    break
        
        # Find numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        
        # Remove excluded columns
        potential_targets = [col for col in numeric_columns if col not in comprehensive_exclude]
        
        # ✅ NEW: Log what was excluded for transparency
        if self.verbose:
            excluded_found = [col for col in df.columns if col in comprehensive_exclude]
            if excluded_found:
                logger.info(f"Auto-excluded columns: {excluded_found}")
        
        return potential_targets

    def interactive_column_selection(self, df: pd.DataFrame, interactive: bool = True) -> Tuple[List[str], Dict[str, str]]:
        """
        Interactively select target columns and their new names
        FLEXIBLE: Returns whatever columns user selects
        """
        if not interactive:
            # Auto-detect mode
            potential_targets = self.detect_target_columns(df)
            if not potential_targets:
                raise ValueError("No suitable target columns found!")
            
            # Create default mapping (keep original names)
            column_mapping = {col: col for col in potential_targets}
            return potential_targets, column_mapping
        
        # Interactive mode
        print("\nAvailable columns in your data:")
        print("-" * 50)
        all_columns = df.columns.tolist()
        potential_targets = self.detect_target_columns(df)
        
        for i, col in enumerate(all_columns):
            col_type = str(df[col].dtype)
            is_potential = col in potential_targets
            marker = "→ " if is_potential else "  "
            print(f"{marker}{i+1:2d}. {col:<25} ({col_type})")
        
        print(f"\nColumns marked with → are automatically detected as potential targets")
        print(f"Auto-detected targets: {potential_targets}")
        
        # Ask user to select columns
        print("\nHow do you want to select target columns?")
        print("1. Use all auto-detected columns")
        print("2. Select specific columns by number")
        print("3. Enter column names manually")
        
        while True:
            try:
                choice = input("\nEnter your choice (1, 2, or 3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                print("Please enter 1, 2, or 3")
            except KeyboardInterrupt:
                raise
            except:
                print("Please enter 1, 2, or 3")
        
        selected_columns = []
        
        if choice == '1':
            selected_columns = potential_targets
        elif choice == '2':
            print("\nEnter the numbers of columns you want to use (comma-separated):")
            print("Example: 1,3,5")
            while True:
                try:
                    numbers = input("Column numbers: ").strip().split(',')
                    selected_columns = []
                    for num in numbers:
                        idx = int(num.strip()) - 1
                        if 0 <= idx < len(all_columns):
                            selected_columns.append(all_columns[idx])
                        else:
                            print(f"Invalid number: {num}")
                            selected_columns = []
                            break
                    if selected_columns:
                        break
                except:
                    print("Please enter valid numbers separated by commas")
        else:  # choice == '3'
            print("\nEnter column names (comma-separated):")
            print("Example: value,band_gap,property1")
            names_input = input("Column names: ").strip()
            names = names_input.split(',')
            selected_columns = []
            for name in names:
                name = name.strip()
                if name in all_columns:
                    selected_columns.append(name)
                else:
                    print(f"Column '{name}' not found!")
        
        print(f"\nSelected columns: {selected_columns}")
        
        # Ask for new names
        column_mapping = {}
        print("\nNow specify what you want to call these columns in the output:")
        print("(Press Enter to keep the original name)")
        
        for col in selected_columns:
            while True:
                try:
                    new_name = input(f"'{col}' → ").strip()
                    if not new_name:
                        new_name = col
                    column_mapping[col] = new_name
                    break
                except:
                    print("Please enter a valid name or press Enter")
        
        print(f"\nFinal column mapping: {column_mapping}")
        return selected_columns, column_mapping

    # ========================
    # Main Processing Methods - FULLY FLEXIBLE
    # ========================
    
    def _process_existing_poly_chemprop_dataset(self, new_df: pd.DataFrame, 
                                              target_columns: List[str] = None,
                                              column_mapping: Dict[str, str] = None,
                                              exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Process a dataset that already contains poly_chemprop_input
        """
        # Handle target column selection - COMPLETELY FLEXIBLE
        if target_columns is None or column_mapping is None:
            target_columns, column_mapping = self.interactive_column_selection(new_df, interactive=True)
        
        # Rename target columns according to mapping
        for old_name, new_name in column_mapping.items():
            if old_name != new_name and old_name in new_df.columns:
                new_df.rename(columns={old_name: new_name}, inplace=True)
        
        # Update target_columns to use new names
        final_target_columns = [column_mapping[col] for col in target_columns]
        
        # Generate poly_ids for existing poly_chemprop_inputs
        existing_ids = set()
        if self.template_df is not None and 'poly_id' in self.template_df.columns:
            existing_ids = set(self.template_df['poly_id'].unique())
        
        # Create unique identifiers based on poly_chemprop_input
        unique_polys = new_df['poly_chemprop_input'].drop_duplicates()
        poly_id_mapping = {}
        
        # ✅ NEW: Parse existing IDs to find the highest number and detect naming pattern
        max_id = -1
        id_format = "numeric"  # Default format
        
        if existing_ids:
            for existing_id in existing_ids:
                try:
                    if existing_id.startswith('p_'):
                        # Format: p_12345
                        id_format = "p_prefix"
                        num = int(existing_id.replace('p_', ''))
                        max_id = max(max_id, num)
                    elif '_' in existing_id:
                        # Format: 12345_6 
                        parts = existing_id.split('_')
                        if len(parts) >= 2 and parts[0].isdigit():
                            num = int(parts[0])
                            max_id = max(max_id, num)
                            id_format = "underscore"
                    elif existing_id.isdigit():
                        # Format: 12345
                        num = int(existing_id)
                        max_id = max(max_id, num)
                        id_format = "numeric"
                except (ValueError, AttributeError):
                    continue
        
        # Generate new IDs
        current_id = max_id + 1
        for poly_input in unique_polys:
            if id_format == "p_prefix":
                new_id = f"p_{current_id}"
            elif id_format == "underscore":
                new_id = f"{current_id}_0"
            else:
                new_id = str(current_id)
            
            while new_id in existing_ids:
                current_id += 1
                if id_format == "p_prefix":
                    new_id = f"p_{current_id}"
                elif id_format == "underscore":
                    new_id = f"{current_id}_0"
                else:
                    new_id = str(current_id)
            
            poly_id_mapping[poly_input] = new_id
            existing_ids.add(new_id)
            current_id += 1
        
        # Map poly_ids to dataframe
        new_df['poly_id'] = new_df['poly_chemprop_input'].map(poly_id_mapping)
        
        # Extract monomers from poly_chemprop_input if possible
        if 'monoA' not in new_df.columns or 'monoB' not in new_df.columns:
            if self.verbose:
                logger.info("Attempting to extract monomer information from poly_chemprop_input...")
            
            monomers_extracted = new_df['poly_chemprop_input'].apply(self._extract_monomers_from_poly_input)
            new_df['monoA'] = monomers_extracted.apply(lambda x: x[0] if x else None)
            new_df['monoB'] = monomers_extracted.apply(lambda x: x[1] if x else None)
        
        # Add missing polymer-related columns with defaults
        if 'poly_type' not in new_df.columns:
            new_df['poly_type'] = 'unknown'
        if 'comp' not in new_df.columns:
            new_df['comp'] = 'unknown'
        if 'fracA' not in new_df.columns:
            new_df['fracA'] = 0.5
        if 'fracB' not in new_df.columns:
            new_df['fracB'] = 0.5
        
        # Generate master_chemprop_input if missing
        if 'master_chemprop_input' not in new_df.columns:
            if 'monoA' in new_df.columns and 'monoB' in new_df.columns:
                new_df['master_chemprop_input'] = [
                    self.make_master_chemprop_input(sA, sB) if pd.notna(sA) and pd.notna(sB) else None
                    for sA, sB in zip(new_df['monoA'], new_df['monoB'])
                ]
            else:
                new_df['master_chemprop_input'] = None
        
        # Generate IUPAC names if monomers are available
        if 'monoA' in new_df.columns and new_df['monoA'].notna().any():
            if 'monoA_IUPAC' not in new_df.columns:
                if self.verbose:
                    logger.info("Generating IUPAC names for monoA...")
                new_df['monoA_IUPAC'] = new_df['monoA'].apply(
                    lambda x: self.get_iupac_name(x) if pd.notna(x) else "Unknown_compound"
                )
            
            if 'monoB_IUPAC' not in new_df.columns:
                if self.verbose:
                    logger.info("Generating IUPAC names for monoB...")
                new_df['monoB_IUPAC'] = new_df['monoB'].apply(
                    lambda x: self.get_iupac_name(x) if pd.notna(x) else "Unknown_compound"
                )
        else:
            # Set default IUPAC names if monomers unavailable
            if 'monoA_IUPAC' not in new_df.columns:
                new_df['monoA_IUPAC'] = "Unknown_compound"
            if 'monoB_IUPAC' not in new_df.columns:
                new_df['monoB_IUPAC'] = "Unknown_compound"
        
        if self.verbose:
            logger.info(f"Successfully processed poly_chemprop_input dataset with {len(new_df)} rows")
            logger.info(f"Target columns preserved: {final_target_columns}")
        
        return new_df
    
    # ✅ NEW: Add exclude_columns parameter
    def process_new_dataset(self, input_path: str = None, df: pd.DataFrame = None,
                          expand_variants: bool = True, generate_iupac: bool = True,
                          interactive: bool = True, target_columns: List[str] = None,
                          column_mapping: Dict[str, str] = None, 
                          poly_types: List[str] = None, compositions: List[str] = None,
                          exclude_columns: List[str] = None):
        
        # ✅ NEW: Store for later use
        self._exclude_columns = exclude_columns if exclude_columns else []
        """
        Process a new dataset and prepare it for appending to template
        COMPLETELY FLEXIBLE - no hardcoded assumptions about target properties
        WITH CONSISTENT CANONICALIZATION
        Can handle both monomer-based datasets and pre-processed poly_chemprop_input datasets
        """
        # Load data
        if df is not None:
            new_df = df.copy()
            if self.verbose:
                logger.info(f"Processing provided DataFrame with {len(new_df)} rows")
        elif input_path:
            new_df = pd.read_csv(input_path)
            if self.verbose:
                logger.info(f"Loaded {len(new_df)} rows from {input_path}")
        else:
            raise ValueError("Either input_path or df must be provided")
        
        # ✅ NEW: Check if dataset already has poly_chemprop_input
        has_poly_chemprop = 'poly_chemprop_input' in new_df.columns
        has_monomers = any(col in new_df.columns for col in ['monoA', 'smiles', 'MonA', 'SMILES', 'Smiles'])
        
        if has_poly_chemprop and not has_monomers:
            if self.verbose:
                logger.info("Dataset contains poly_chemprop_input - processing as pre-processed polymer data")
            return self._process_existing_poly_chemprop_dataset(new_df, target_columns, column_mapping, exclude_columns)
        
        # Standardize column names - flexible mappings
        column_mapping_standard = {
            'smiles': 'monoA',
            'MonA': 'monoA',
            'MonB': 'monoB',
            'SMILES': 'monoA',
            'Smiles': 'monoA'
        }
        
        for old_name, new_name in column_mapping_standard.items():
            if old_name in new_df.columns and new_name not in new_df.columns:
                new_df.rename(columns={old_name: new_name}, inplace=True)
        
        # Validate required columns
        if 'monoA' not in new_df.columns:
            raise ValueError("No 'monoA', 'smiles', 'MonA', or 'poly_chemprop_input' column found!")
        
        # CANONICALIZE monoA SMILES IMMEDIATELY
        if self.verbose:
            logger.info("Canonicalizing monoA SMILES...")
        new_df['monoA'] = new_df['monoA'].apply(
            lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else x
        )
        
        # Remove rows where canonicalization failed
        initial_count = len(new_df)
        new_df = new_df[new_df['monoA'].notna()]
        if len(new_df) != initial_count and self.verbose:
            logger.warning(f"Removed {initial_count - len(new_df)} rows due to invalid monoA SMILES")
        
        # Handle monoB for homopolymers
        if 'monoB' not in new_df.columns:
            new_df['monoB'] = new_df['monoA']  # Homopolymer: monoB = monoA
            if self.verbose:
                logger.info("Added monoB column (same as monoA for homopolymers)")
        else:
            # Fill missing monoB with monoA (homopolymers)
            new_df['monoB'] = new_df['monoB'].fillna(new_df['monoA'])
            
            # CANONICALIZE monoB SMILES
            if self.verbose:
                logger.info("Canonicalizing monoB SMILES...")
            new_df['monoB'] = new_df['monoB'].apply(
                lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else x
            )
        
        # Handle target column selection - COMPLETELY FLEXIBLE
        if target_columns is None or column_mapping is None:
            target_columns, column_mapping = self.interactive_column_selection(new_df, interactive=interactive)
        
        # Rename target columns according to mapping
        for old_name, new_name in column_mapping.items():
            if old_name != new_name and old_name in new_df.columns:
                new_df.rename(columns={old_name: new_name}, inplace=True)
        
        # Update target_columns to use new names
        final_target_columns = [column_mapping[col] for col in target_columns]
        
        # Expand polymer variants if requested - with flexible options
        if expand_variants:
            new_df = self.expand_polymer_variants(new_df, poly_types=poly_types, compositions=compositions)
            if self.verbose:
                logger.info(f"Expanded to {len(new_df)} rows with polymer variants")
        else:
            # Ensure required columns exist for non-expanded datasets
            if 'poly_type' not in new_df.columns:
                new_df['poly_type'] = 'alternating'  # Default
            if 'comp' not in new_df.columns:
                new_df['comp'] = '4A_4B'  # Default
            if 'fracA' not in new_df.columns:
                new_df['fracA'] = 0.5
            if 'fracB' not in new_df.columns:
                new_df['fracB'] = 0.5
        
        # Generate poly_ids
        existing_ids = set()
        if self.template_df is not None and 'poly_id' in self.template_df.columns:
            existing_ids = set(self.template_df['poly_id'].unique())
        
        new_df['poly_id'] = self.generate_poly_ids(new_df, existing_ids)
        
        # Generate IUPAC names if requested (IMPROVED)
        if generate_iupac:
            if 'monoA_IUPAC' not in new_df.columns:
                if self.verbose:
                    logger.info("Generating IUPAC names for monoA...")
                new_df['monoA_IUPAC'] = new_df['monoA'].apply(self.get_iupac_name)
            
            if 'monoB_IUPAC' not in new_df.columns:
                if self.verbose:
                    logger.info("Generating IUPAC names for monoB...")
                new_df['monoB_IUPAC'] = new_df['monoB'].apply(self.get_iupac_name)
        
        # Generate ChemProp inputs with CONSISTENT PROCESSING
        if self.verbose:
            logger.info("Generating ChemProp inputs...")
        
        new_df['master_chemprop_input'] = [
            self.make_master_chemprop_input(sA, sB) 
            for sA, sB in zip(new_df['monoA'], new_df['monoB'])
        ]
        
        new_df['poly_chemprop_input'] = [
            self.make_poly_chemprop_input(sA, sB, t, fA, selfedges=True)
            for sA, sB, t, fA in zip(
                new_df['monoA'], new_df['monoB'], 
                new_df['poly_type'], new_df['fracA']
            )
        ]
        
        # Remove rows where ChemProp input generation failed
        initial_count = len(new_df)
        new_df = new_df[new_df['poly_chemprop_input'].notnull()]
        final_count = len(new_df)
        
        if initial_count != final_count and self.verbose:
            logger.warning(f"Removed {initial_count - final_count} rows due to ChemProp input generation failures")
        
        if self.verbose:
            logger.info(f"Successfully processed dataset with {len(new_df)} rows")
            logger.info(f"Target columns preserved: {final_target_columns}")
        
        return new_df

    def append_to_template(self, new_df: pd.DataFrame, output_path: str = None, 
                      exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Append new dataset to existing template with optional column cleanup
        """
        if exclude_columns is None:
            exclude_columns = []
        
        if self.template_df is None:
            combined_df = new_df.copy()
        else:
            # Align columns - preserves ALL columns from both datasets
            all_columns = list(set(self.template_df.columns) | set(new_df.columns))
            
            # Add missing columns to both dataframes
            for col in all_columns:
                if col not in self.template_df.columns:
                    self.template_df[col] = None
                if col not in new_df.columns:
                    new_df[col] = None
            
            # Reorder columns to match template preference, then add new columns
            template_cols = list(self.template_df.columns)
            new_cols = [col for col in all_columns if col not in template_cols]
            final_columns = template_cols + new_cols
            
            self.template_df = self.template_df[final_columns]
            new_df = new_df[final_columns]
            
            # Combine dataframes
            combined_df = pd.concat([self.template_df, new_df], ignore_index=True)
        
        # ✅ NEW: Remove unwanted columns from final output
        if exclude_columns:
            columns_to_remove = [col for col in exclude_columns if col in combined_df.columns]
            if columns_to_remove:
                combined_df = combined_df.drop(columns=columns_to_remove)
                if self.verbose:
                    logger.info(f"Removed unwanted columns: {columns_to_remove}")
        
        if output_path:
            combined_df.to_csv(output_path, index=False)  # ✅ Saves only wanted columns
            if self.verbose:
                logger.info(f"Saved combined dataset to {output_path}")
        
        if self.verbose:
            logger.info(f"Combined dataset has {len(combined_df)} rows and {len(combined_df.columns)} columns")
        
        return combined_df

    # ========================
    # Convenience Methods
    # ========================
    
    def quick_process(self, input_path: str, output_path: str, 
                     expand_variants: bool = True, interactive: bool = True,
                     poly_types: List[str] = None, compositions: List[str] = None) -> pd.DataFrame:
        """
        Quick processing method that combines process_new_dataset and append_to_template
        FLEXIBLE: Can customize polymer types and compositions
        """
        processed_df = self.process_new_dataset(
            input_path=input_path,
            expand_variants=expand_variants,
            interactive=interactive,
            poly_types=poly_types,
            compositions=compositions
        )
        
        combined_df = self.append_to_template(processed_df, output_path)
        return combined_df

    def cleanup_existing_database(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Clean up an existing database to ensure consistent formatting
        Useful for fixing databases created with earlier versions
        """
        if self.verbose:
            logger.info(f"Cleaning up existing database: {input_path}")
        
        df = pd.read_csv(input_path)
        original_count = len(df)
        
        # Canonicalize existing SMILES
        if 'monoA' in df.columns:
            if self.verbose:
                logger.info("Re-canonicalizing monoA SMILES...")
            df['monoA'] = df['monoA'].apply(
                lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else x
            )
        
        if 'monoB' in df.columns:
            if self.verbose:
                logger.info("Re-canonicalizing monoB SMILES...")
            df['monoB'] = df['monoB'].apply(
                lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else x
            )
        
        # Re-generate IUPAC names with improved algorithm
        if 'monoA_IUPAC' in df.columns:
            if self.verbose:
                logger.info("Re-generating IUPAC names for monoA...")
            df['monoA_IUPAC'] = df['monoA'].apply(self.get_iupac_name)
        
        if 'monoB_IUPAC' in df.columns:
            if self.verbose:
                logger.info("Re-generating IUPAC names for monoB...")
            df['monoB_IUPAC'] = df['monoB'].apply(self.get_iupac_name)
        
        # Re-generate ChemProp inputs for consistency
        if all(col in df.columns for col in ['monoA', 'monoB']):
            if self.verbose:
                logger.info("Re-generating master ChemProp inputs...")
            df['master_chemprop_input'] = [
                self.make_master_chemprop_input(sA, sB) 
                for sA, sB in zip(df['monoA'], df['monoB'])
            ]
        
        if all(col in df.columns for col in ['monoA', 'monoB', 'poly_type', 'fracA']):
            if self.verbose:
                logger.info("Re-generating poly ChemProp inputs...")
            df['poly_chemprop_input'] = [
                self.make_poly_chemprop_input(sA, sB, t, fA, selfedges=True)
                for sA, sB, t, fA in zip(
                    df['monoA'], df['monoB'], 
                    df['poly_type'], df['fracA']
                )
            ]
        
        # Remove any rows that failed processing
        df = df[df['poly_chemprop_input'].notna()] if 'poly_chemprop_input' in df.columns else df
        final_count = len(df)
        
        # Save cleaned database
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            logger.info(f"Database cleanup complete!")
            logger.info(f"Original rows: {original_count}")
            logger.info(f"Final rows: {final_count}")
            logger.info(f"Cleaned database saved to: {output_path}")
        
        return df

# ========================
# Additional Utility Functions
# ========================

def cleanup_database(input_path: str, output_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Standalone function to clean up an existing polymer database
    
    Args:
        input_path: Path to existing database CSV
        output_path: Path to save cleaned database
        verbose: Whether to show detailed output
        
    Returns:
        Cleaned DataFrame
    """
    manager = PolymerDatabaseManager(verbose=verbose)
    return manager.cleanup_existing_database(input_path, output_path)

def fix_database_consistency(database_path: str, backup: bool = True) -> str:
    """
    Fix consistency issues in an existing database (in-place with backup)
    
    Args:
        database_path: Path to database to fix
        backup: Whether to create backup before fixing
        
    Returns:
        Path to the fixed database
    """
    import shutil
    from datetime import datetime
    
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = database_path.replace('.csv', f'_backup_{timestamp}.csv')
        shutil.copy2(database_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Clean the database
    manager = PolymerDatabaseManager(verbose=True)
    cleaned_df = manager.cleanup_existing_database(database_path, database_path)
    
    print(f"Fixed database saved to: {database_path}")
    return database_path

# ========================
# Convenience Functions - NO HARDCODING
# ========================

def create_database_manager(template_path: str = None) -> PolymerDatabaseManager:
    """Create a new database manager instance"""
    return PolymerDatabaseManager(template_path)

def quick_process_dataset(input_path: str, output_path: str, template_path: str = None,
                         expand_variants: bool = True, interactive: bool = True,
                         poly_types: List[str] = None, compositions: List[str] = None) -> pd.DataFrame:
    """
    Quick function to process a dataset with minimal setup
    FLEXIBLE: Can override all default behaviors
    """
    manager = PolymerDatabaseManager(template_path)
    return manager.quick_process(input_path, output_path, expand_variants, interactive, poly_types, compositions)

# ========================
# Configuration Updates - RUNTIME FLEXIBILITY
# ========================

def update_default_polymer_configs(manager: PolymerDatabaseManager, 
                                 poly_types: List[str] = None,
                                 compositions: List[str] = None,
                                 comp_fracs: Dict[str, Tuple[float, float]] = None):
    """
    Update default polymer configurations at runtime
    FLEXIBLE: No need to hardcode anything
    """
    if poly_types is not None:
        manager.default_poly_types = poly_types
    if compositions is not None:
        manager.default_compositions = compositions
    if comp_fracs is not None:
        manager.comp_fracs.update(comp_fracs)

# ========================
# Command Line Interface
# ========================

def main():
    """
    Command line interface for the Polymer Database Manager
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Polymer Database Manager - Process polymer datasets with ChemProp input generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - interactive column selection
  !python data_processing/polymer_database_manager.py -i input.csv -o output.csv
  
  # Process dataset with existing poly_chemprop_input
  !python data_processing/polymer_database_manager.py -i poly_dataset.csv -o output.csv -t template.csv
  
  # With existing template
  !python data_processing/polymer_database_manager.py -i new_data.csv -o updated_db.csv -t existing_template.csv
  
  # Non-interactive with output directory
  !python data_processing/polymer_database_manager.py -i data.csv -o results/ --non-interactive
  
  # Clean up existing database for consistency
  !python data_processing/polymer_database_manager.py --cleanup input_db.csv -o cleaned_db.csv
  
  # Custom polymer types and compositions
  !python data_processing/polymer_database_manager.py -i data.csv -o output.csv --poly-types alternating block --compositions 4A_4B 6A_2B
  
  # Specify target columns and their new names
  !python data_processing/polymer_database_manager.py -i data.csv -o output.csv --target-columns band_gap conductivity --target-names Band_Gap_eV Conductivity_S_cm
  
  # No polymer variant expansion (keep as-is)
  !python data_processing/polymer_database_manager.py -i data.csv -o output.csv --no-expand
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', 
                       help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output CSV file path or directory')
    
    # Optional arguments
    parser.add_argument('-t', '--template',
                       help='Existing template CSV file path to append to')
    
    # Cleanup mode
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up existing database for consistency (use with -i for input database)')
    
    # Processing options
    parser.add_argument('--no-expand', action='store_true',
                       help='Do not expand polymer variants (keep original data structure)')
    parser.add_argument('--no-iupac', action='store_true',
                       help='Do not generate IUPAC names')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Use non-interactive mode (auto-detect all numeric columns)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disable verbose output')
    
    # Target column specification
    parser.add_argument('--target-columns', nargs='+',
                       help='Specific target column names to use')
    parser.add_argument('--target-names', nargs='+',
                       help='New names for target columns (must match --target-columns length)')
    
    # ✅ NEW: More flexible exclusion options
    parser.add_argument('--exclude-columns', nargs='+',
                       help='Specific column names to exclude from auto-detection')
    parser.add_argument('--exclude-patterns', nargs='+',
                       help='Column name patterns to exclude (e.g., "id" excludes hp_id, pol_id)')
    parser.add_argument('--keep-all-columns', action='store_true',
                       help='Keep ALL original columns (disable auto-cleanup)')
    parser.add_argument('--remove-empty-columns', action='store_true',
                       help='Remove columns that are mostly empty/null')
    
    # Polymer configuration
    parser.add_argument('--poly-types', nargs='+', 
                       default=['alternating', 'block', 'random'],
                       help='Polymer types to generate (default: alternating block random)')
    parser.add_argument('--compositions', nargs='+',
                       default=['4A_4B', '6A_2B', '2A_6B'],
                       help='Polymer compositions to generate (default: 4A_4B 6A_2B 2A_6B)')
    
    # Advanced options
    parser.add_argument('--selfedges', action='store_true', default=True,
                       help='Include self-edges in polymer connectivity (default: True)')
    parser.add_argument('--no-selfedges', action='store_false', dest='selfedges',
                       help='Exclude self-edges in polymer connectivity')
    
    # Information options
    parser.add_argument('--list-columns', action='store_true',
                       help='List all columns in the input file and potential targets, then exit')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Handle cleanup mode
    if args.cleanup:
        if not args.input:
            print("Error: --cleanup requires --input to specify the database to clean")
            return 1
        
        try:
            cleaned_df = cleanup_database(args.input, args.output, verbose=not args.quiet)
            print(f"✓ Database cleanup completed!")
            print(f"✓ Cleaned database saved to: {args.output}")
            return 0
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return 1
    
    # Regular processing mode requires input
    if not args.input:
        print("Error: --input is required for regular processing")
        return 1
    
    # Handle conflicting arguments
    if args.quiet:
        args.verbose = False
    
    # Handle output path
    output_path = args.output
    if os.path.isdir(output_path):
        # If output is a directory, create filename
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(output_path, f"processed_{input_basename}_polymer_db.csv")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if args.verbose:
            print(f"Created output directory: {output_dir}")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # List columns mode
    if args.list_columns:
        try:
            df = pd.read_csv(args.input)
            manager = PolymerDatabaseManager(verbose=False)
            # ✅ NEW: Flexible exclusion handling
            exclude_patterns = args.exclude_patterns if hasattr(args, 'exclude_patterns') else None
            potential_targets = manager.detect_target_columns(
                df, 
                exclude_columns=args.exclude_columns,
                auto_exclude_patterns=exclude_patterns if not args.keep_all_columns else []
            )
            
            print(f"Columns in {args.input}:")
            print("-" * 50)
            for i, col in enumerate(df.columns):
                col_type = str(df[col].dtype)
                is_target = " (potential target)" if col in potential_targets else ""
                print(f"{i+1:2d}. {col:<25} ({col_type}){is_target}")
            
            print(f"\nAuto-detected target columns: {potential_targets}")
            return 0
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    
    # Validate target columns and names
    target_columns = args.target_columns
    column_mapping = None
    
    if target_columns and args.target_names:
        if len(target_columns) != len(args.target_names):
            print("Error: Number of target columns must match number of target names")
            return 1
        column_mapping = dict(zip(target_columns, args.target_names))
    elif target_columns:
        # Keep original names
        column_mapping = {col: col for col in target_columns}
    
    try:
        # Initialize manager
        if args.verbose:
            print(f"Initializing Polymer Database Manager...")
            if args.template:
                print(f"Using template: {args.template}")
        
        manager = PolymerDatabaseManager(template_path=args.template, verbose=args.verbose)
        
        # Update configurations if provided
        if args.poly_types != ['alternating', 'block', 'random']:
            manager.default_poly_types = args.poly_types
        if args.compositions != ['4A_4B', '6A_2B', '2A_6B']:
            manager.default_compositions = args.compositions
            
        # Add custom compositions if needed
        for comp in args.compositions:
            if comp not in manager.comp_fracs:
                # Try to parse composition to determine fractions
                if 'A' in comp and 'B' in comp:
                    try:
                        a_part = comp.split('A_')[0]
                        b_part = comp.split('A_')[1].replace('B', '')
                        a_frac = float(a_part) / (float(a_part) + float(b_part))
                        b_frac = 1.0 - a_frac
                        manager.comp_fracs[comp] = (a_frac, b_frac)
                        if args.verbose:
                            print(f"Added composition {comp}: A={a_frac:.2f}, B={b_frac:.2f}")
                    except:
                        if args.verbose:
                            print(f"Warning: Could not parse composition {comp}, using 50:50")
                        manager.comp_fracs[comp] = (0.5, 0.5)
        
        # Process the dataset
        if args.verbose:
            print(f"Processing dataset: {args.input}")
            
        processed_df = manager.process_new_dataset(
            input_path=args.input,
            expand_variants=not args.no_expand,
            generate_iupac=not args.no_iupac,
            interactive=not args.non_interactive,
            target_columns=target_columns,
            column_mapping=column_mapping,
            exclude_columns=args.exclude_columns,  # ✅ NEW: Pass exclude_columns
            poly_types=args.poly_types if args.poly_types != ['alternating', 'block', 'random'] else None,
            compositions=args.compositions if args.compositions != ['4A_4B', '6A_2B', '2A_6B'] else None
        )
        
        # ✅ NEW: Use stored exclude_columns when appending
        exclude_cols = getattr(manager, '_exclude_columns', [])
        combined_df = manager.append_to_template(processed_df, output_path, exclude_columns=exclude_cols)
                
        if args.verbose:
            print(f"\n✓ Processing completed successfully!")
            print(f"✓ Input file: {args.input}")
            print(f"✓ Output file: {output_path}")
            print(f"✓ Processed {len(processed_df)} new rows")
            print(f"✓ Final database: {len(combined_df)} total rows")
            
            # Show target properties
            target_cols = [col for col in combined_df.columns 
                          if col not in ['poly_id', 'poly_type', 'comp', 'fracA', 'fracB', 'monoA', 'monoB', 
                                        'monoA_IUPAC', 'monoB_IUPAC', 'master_chemprop_input', 'poly_chemprop_input']]
            if target_cols:
                print(f"✓ Target properties: {target_cols}")
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def run_interactive_mode():
    """
    Interactive mode for Jupyter/Colab environments
    """
    print("Polymer Database Manager - Interactive Mode")
    print("=" * 50)
    
    # Get input file
    input_file = input("Enter input CSV file path: ").strip()
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return None
    
    # Get output file
    output_file = input("Enter output CSV file path: ").strip()
    
    # Get template file (optional)
    template_file = input("Enter template CSV file path (press Enter to skip): ").strip()
    if not template_file:
        template_file = None
    elif not os.path.exists(template_file):
        print(f"Warning: Template file not found: {template_file}")
        template_file = None
    
    # Processing options
    expand_variants = input("Expand polymer variants? (y/n, default: y): ").strip().lower()
    expand_variants = expand_variants != 'n'
    
    generate_iupac = input("Generate IUPAC names? (y/n, default: y): ").strip().lower()
    generate_iupac = generate_iupac != 'n'
    
    interactive = input("Use interactive column selection? (y/n, default: y): ").strip().lower()
    interactive = interactive != 'n'
    
    # Initialize and process
    manager = PolymerDatabaseManager(template_path=template_file, verbose=True)
    
    processed_df = manager.process_new_dataset(
        input_path=input_file,
        expand_variants=expand_variants,
        generate_iupac=generate_iupac,
        interactive=interactive
    )
    
    combined_df = manager.append_to_template(processed_df, output_file)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Final database: {len(combined_df)} rows")
    
    return combined_df

# Version info
__version__ = "1.0.0"
__author__ = "Polymer Database Toolkit"

if __name__ == "__main__":
    sys.exit(main())
