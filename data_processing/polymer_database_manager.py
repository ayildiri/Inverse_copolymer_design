#!/usr/bin/env python3
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
        Ensures chemical validity is preserved
        """
        try:
            # Convert [*] to [*:1], [*:2] etc.
            numbered_smiles = smiles
            counter = 1
            while '[*]' in numbered_smiles:
                numbered_smiles = numbered_smiles.replace('[*]', f'[*:{counter}]', 1)
                counter += 1
            
            # Canonicalize with RDKit
            mol = Chem.MolFromSmiles(numbered_smiles)
            if mol is None:
                return None
                
            # Ensure aromatic atoms have proper valence before canonicalization
            try:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol, canonical=True)
            except:
                # If sanitization fails, try to fix the structure
                for atom in mol.GetAtoms():
                    if atom.GetIsAromatic():
                        atom.SetNumExplicitHs(0)
                        atom.SetNoImplicit(False)
                try:
                    Chem.SanitizeMol(mol)
                    return Chem.MolToSmiles(mol, canonical=True)
                except:
                    return None
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

    # ========================
    # Database Management
    # ========================
    
    def generate_poly_ids(self, df: pd.DataFrame, existing_ids: set = None) -> List[str]:
        """Generate unique polymer IDs"""
        if existing_ids is None:
            existing_ids = set()
        
        poly_ids = []
        base_id = len(existing_ids)
        
        unique_pairs = df[['monoA', 'monoB']].drop_duplicates()
        
        for idx, (_, row) in enumerate(unique_pairs.iterrows()):
            while f"{base_id}_{idx}" in existing_ids:
                base_id += 1
            poly_ids.append(f"{base_id}_{idx}")
            existing_ids.add(f"{base_id}_{idx}")
        
        # Map back to original dataframe
        pair_to_id = dict(zip(zip(unique_pairs['monoA'], unique_pairs['monoB']), poly_ids))
        result_ids = [pair_to_id[(row['monoA'], row['monoB'])] for _, row in df.iterrows()]
        
        return result_ids

    def get_iupac_name(self, smiles: str) -> str:
        """Get IUPAC name for a SMILES string (placeholder implementation)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid_SMILES"
            
            # Generate a simple name based on molecular formula as placeholder
            formula = CalcMolFormula(mol)
            return f"Compound_{formula}"
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not generate IUPAC name for {smiles}: {e}")
            return "Unknown_compound"

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
                        new_row['poly_type'] = poly_type
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
    
    def detect_target_columns(self, df: pd.DataFrame, exclude_columns: List[str] = None) -> List[str]:
        """
        Automatically detect potential target columns (numeric columns)
        FLEXIBLE: Can provide custom exclude list
        """
        if exclude_columns is None:
            # Default exclusions - but this can be overridden
            exclude_columns = ['pol_id', 'hp_id', 'poly_id', 'MonA', 'MonB', 'monoA', 'monoB', 
                             'stoich', 'connectivity', 'smiles', 'canonical_smiles', 'fracA', 'fracB',
                             'poly_type', 'comp', 'master_chemprop_input', 'poly_chemprop_input',
                             'monoA_IUPAC', 'monoB_IUPAC']
        
        # Find numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        
        # Remove excluded columns
        potential_targets = [col for col in numeric_columns if col not in exclude_columns]
        
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
    
    def process_new_dataset(self, input_path: str = None, df: pd.DataFrame = None,
                          expand_variants: bool = True, generate_iupac: bool = True,
                          interactive: bool = True, target_columns: List[str] = None,
                          column_mapping: Dict[str, str] = None, 
                          poly_types: List[str] = None, compositions: List[str] = None) -> pd.DataFrame:
        """
        Process a new dataset and prepare it for appending to template
        COMPLETELY FLEXIBLE - no hardcoded assumptions about target properties
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
            raise ValueError("No 'monoA', 'smiles', or 'MonA' column found!")
        
        # Handle monoB for homopolymers
        if 'monoB' not in new_df.columns:
            new_df['monoB'] = new_df['monoA']
            if self.verbose:
                logger.info("Added monoB column (same as monoA for homopolymers)")
        else:
            # Fill missing monoB with monoA (homopolymers)
            new_df['monoB'] = new_df['monoB'].fillna(new_df['monoA'])
        
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
        
        # Generate IUPAC names if requested
        if generate_iupac:
            if 'monoA_IUPAC' not in new_df.columns:
                if self.verbose:
                    logger.info("Generating IUPAC names for monoA...")
                new_df['monoA_IUPAC'] = new_df['monoA'].apply(self.get_iupac_name)
            
            if 'monoB_IUPAC' not in new_df.columns:
                if self.verbose:
                    logger.info("Generating IUPAC names for monoB...")
                new_df['monoB_IUPAC'] = new_df['monoB'].apply(self.get_iupac_name)
        
        # Generate ChemProp inputs
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

    def append_to_template(self, new_df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Append new dataset to existing template
        FLEXIBLE: Preserves ALL columns from both datasets
        """
        if self.template_df is None:
            if self.verbose:
                logger.info("No existing template found, using new data as template")
            combined_df = new_df.copy()
        else:
            if self.verbose:
                logger.info(f"Appending {len(new_df)} rows to existing template with {len(self.template_df)} rows")
            
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
        
        if output_path:
            combined_df.to_csv(output_path, index=False)
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
  
  # With existing template
  !python data_processing/polymer_database_manager.py -i new_data.csv -o updated_db.csv -t existing_template.csv
  
  # Non-interactive with output directory
  !python data_processing/polymer_database_manager.py -i data.csv -o results/ --non-interactive
  
  # Custom polymer types and compositions
  !python data_processing/polymer_database_manager.py -i data.csv -o output.csv --poly-types alternating block --compositions 4A_4B 6A_2B
  
  # Specify target columns and their new names
  !python data_processing/polymer_database_manager.py -i data.csv -o output.csv --target-columns band_gap conductivity --target-names Band_Gap_eV Conductivity_S_cm
  
  # No polymer variant expansion (keep as-is)
  !python data_processing/polymer_database_manager.py -i data.csv -o output.csv --no-expand
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                       help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output CSV file path or directory')
    
    # Optional arguments
    parser.add_argument('-t', '--template',
                       help='Existing template CSV file path to append to')
    
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
    parser.add_argument('--exclude-columns', nargs='+',
                       help='Column names to exclude from auto-detection')
    
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
            potential_targets = manager.detect_target_columns(df, exclude_columns=args.exclude_columns)
            
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
            poly_types=args.poly_types if args.poly_types != ['alternating', 'block', 'random'] else None,
            compositions=args.compositions if args.compositions != ['4A_4B', '6A_2B', '2A_6B'] else None
        )
        
        # Append to template and save
        combined_df = manager.append_to_template(processed_df, output_path)
        
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
