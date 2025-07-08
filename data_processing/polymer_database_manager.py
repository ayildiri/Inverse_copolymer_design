"""
Universal Polymer Database Manager - Enhanced Version 3.0
=========================================================

A comprehensive toolkit for processing ANY polymer dataset format with automatic
detection, repair, and intelligent merging capabilities.

Enhanced to handle:
- Datasets with only poly_chemprop_input (Type A)
- Datasets with structure but no poly_chemprop_input (Type B)  
- Monomer datasets needing polymer expansion (Type C)
- Any combination of the above in a single command

Place this file in: /content/Inverse_copolymer_design/data_processing/

Command Line Usage:
    # Smart merge different dataset types
    !python data_processing/polymer_database_manager.py --smart-merge data1.csv data2.csv data3.csv -o merged.csv
    
    # Regular processing with auto-repair
    !python data_processing/polymer_database_manager.py -i input.csv -o output.csv --repair-missing
    
Programmatic Usage:
    from polymer_database_manager import PolymerDatabaseManager
    manager = PolymerDatabaseManager()
    manager.smart_merge_datasets(['data1.csv', 'data2.csv', 'data3.csv'], 'output.csv')
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
import re
import logging
import shutil  # Added missing import
from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Handle imports for both command line and programmatic usage
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit import RDLogger
    RDKIT_AVAILABLE = True

    # Suppress RDKit warnings during SMILES parsing
    RDLogger.DisableLog('rdApp.*')

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

class DatasetType(Enum):
    """Enum for different dataset types"""
    TYPE_A = "poly_chemprop_only"  # Has poly_chemprop_input but missing structure
    TYPE_B = "structure_only"       # Has structure but missing poly_chemprop_input
    TYPE_C = "monomers_only"        # Has only monomers, needs expansion
    TYPE_COMPLETE = "complete"      # Has everything
    TYPE_UNKNOWN = "unknown"        # Can't determine

class MergeStrategy(Enum):
    """Strategies for merging duplicate properties"""
    FIRST = "first"      # Keep first non-null value
    LAST = "last"        # Keep last non-null value
    MEAN = "mean"        # Average of all non-null values
    MAX = "max"          # Maximum of all non-null values
    MIN = "min"          # Minimum of all non-null values

class PolymerDatabaseManager:
    """
    Universal Polymer Database Manager with automatic dataset repair and smart merging
    """
    
    def __init__(self, template_path: str = None, verbose: bool = True, 
                 clean_template: bool = True, fix_unknowns: bool = True,
                 auto_repair: bool = True):
        """
        Initialize the Universal Polymer Database Manager
        
        Args:
            template_path: Path to existing template CSV file
            verbose: Whether to print detailed logs
            clean_template: Whether to clean poly_chemprop_input in template
            fix_unknowns: Whether to attempt fixing unknown values
            auto_repair: Whether to automatically repair missing columns
        """
        self.template_path = template_path
        self.template_df = None
        self.verbose = verbose
        self.fix_unknowns = fix_unknowns
        self.auto_repair = auto_repair
        
        # Customizable identifier columns for merge operations
        # These columns will always use 'first' strategy, not averaged
        self.identifier_columns = {
            # Structure and ID columns
            'poly_id', 'poly_type', 'comp', 'monoA', 'monoB', 
            'monoA_IUPAC', 'monoB_IUPAC', 'master_chemprop_input',
            'fracA', 'fracB', 'polymer_ID', 'monomer_ID', 'id', 'ID',
            'polymer_class', 'source', 'reference', 'notes', 'comments',
            'tacticity', 'polymer_name', 'name', 'Name',
            # Experimental conditions and molecular descriptors
            'temp', 'press', 'DP', 'Mn', 'mol_weight_monomer'
        }
        
        if template_path and os.path.exists(template_path):
            self.template_df = pd.read_csv(template_path)
            if verbose:
                logger.info(f"Loaded template with {len(self.template_df)} rows and {len(self.template_df.columns)} columns")
            
            # Auto-repair template if needed
            if auto_repair:
                self.template_df = self._detect_and_repair_dataset(self.template_df, "template")
            
            # Clean template poly_chemprop_input if requested
            if clean_template and 'poly_chemprop_input' in self.template_df.columns:
                if verbose:
                    logger.info("Cleaning template poly_chemprop_input data...")
                
                # Count corrupted entries in template
                corrupted_count = self.template_df['poly_chemprop_input'].astype(str).str.contains('~', na=False).sum()
                if corrupted_count > 0 and verbose:
                    logger.info(f"Found {corrupted_count} corrupted entries in template to clean")
                
                # Clean template data
                self.template_df['poly_chemprop_input'] = self.template_df['poly_chemprop_input'].apply(
                    lambda x: self.clean_poly_chemprop_input(x, remove_trailing_values=True)
                )
                
                # Remove rows where cleaning failed
                initial_count = len(self.template_df)
                self.template_df = self.template_df[self.template_df['poly_chemprop_input'].notna()]
                if len(self.template_df) != initial_count and verbose:
                    logger.warning(f"Removed {initial_count - len(self.template_df)} corrupted rows from template")
            
            # Fix unknowns in template if requested
            if fix_unknowns and self.template_df is not None:
                self._fix_unknown_values(self.template_df)
        
        # Default polymer configurations - NOT hardcoded, can be modified
        self.default_poly_types = ['alternating', 'block', 'random']
        self.default_compositions = ['4A_4B', '6A_2B', '2A_6B']
        self.comp_fracs = {
            '4A_4B': (0.5, 0.5),
            '6A_2B': (0.75, 0.25),
            '2A_6B': (0.25, 0.75)
        }
        
        # Enhanced patterns for polymer type detection
        self.polymer_type_patterns = {
            'alternating': [
                r'<1-3:0\.5:0\.5.*<1-4:0\.5:0\.5.*<2-3:0\.5:0\.5.*<2-4:0\.5:0\.5(?!.*<1-1)',
                r'<1-3:0\.500:0\.500.*<1-4:0\.500:0\.500.*<2-3:0\.500:0\.500.*<2-4:0\.500:0\.500(?!.*<1-1)'
            ],
            'block': [
                r'<1-2:.*<1-1:.*<2-2:.*<3-4:.*<3-3:.*<4-4:',
                r'<1-2:0\.375:0\.375.*<1-1:0\.375:0\.375.*<2-2:0\.375:0\.375',
                r'<1-2:0\.750:0\.750.*<3-4:0\.750:0\.750(?!.*<1-1:0\.25)'
            ],
            'random': [
                r'<1-3:0\.25:0\.25.*<1-4:0\.25:0\.25.*<2-3:0\.25:0\.25.*<2-4:0\.25:0\.25.*<1-2:0\.25:0\.25.*<3-4:0\.25:0\.25.*<1-1:0\.25:0\.25.*<2-2:0\.25:0\.25.*<3-3:0\.25:0\.25.*<4-4:0\.25:0\.25',
                r'<1-3:0\.250:0\.250.*<1-4:0\.250:0\.250.*<2-3:0\.250:0\.250.*<2-4:0\.250:0\.250.*<1-1:',
                r'<1-3:0\.25:0\.25.*<1-4:0\.25:0\.25.*<2-3:0\.25:0\.25.*<2-4:0\.25:0\.25.*<1-2:0\.5:0\.5.*<3-4:0\.5:0\.5'
            ]
        }

    def set_rdkit_verbosity(self, verbose: bool = False):
        """
        Control RDKit error message verbosity
        
        Args:
            verbose: If True, show RDKit errors. If False, suppress them.
        """
        if not verbose:
            RDLogger.DisableLog('rdApp.*')
        else:
            RDLogger.EnableLog('rdApp.*')
    
    def customize_merge_behavior(self, columns_to_average: List[str] = None, 
                                columns_to_keep_first: List[str] = None):
        """
        Customize which columns are averaged vs kept as first value during merge
        
        Args:
            columns_to_average: List of column names that should be averaged (removed from identifier_columns)
            columns_to_keep_first: List of column names that should keep first value (added to identifier_columns)
            
        Example:
            # Make temperature and pressure get averaged instead of keeping first
            manager.customize_merge_behavior(columns_to_average=['temp', 'press'])
            
            # Make certain properties keep first value instead of averaging
            manager.customize_merge_behavior(columns_to_keep_first=['density', 'viscosity'])
        """
        if columns_to_average:
            # Remove these columns from identifier set so they get averaged
            for col in columns_to_average:
                self.identifier_columns.discard(col)
            if self.verbose:
                logger.info(f"Columns set to average: {columns_to_average}")
        
        if columns_to_keep_first:
            # Add these columns to identifier set so they keep first value
            for col in columns_to_keep_first:
                self.identifier_columns.add(col)
            if self.verbose:
                logger.info(f"Columns set to keep first: {columns_to_keep_first}")
    
    def reset_merge_behavior(self):
        """Reset merge behavior to defaults"""
        self.identifier_columns = {
            # Structure and ID columns
            'poly_id', 'poly_type', 'comp', 'monoA', 'monoB', 
            'monoA_IUPAC', 'monoB_IUPAC', 'master_chemprop_input',
            'fracA', 'fracB', 'polymer_ID', 'monomer_ID', 'id', 'ID',
            'polymer_class', 'source', 'reference', 'notes', 'comments',
            'tacticity', 'polymer_name', 'name', 'Name',
            # Experimental conditions and molecular descriptors
            'temp', 'press', 'DP', 'Mn', 'mol_weight_monomer'
        }
        if self.verbose:
            logger.info("Reset merge behavior to defaults")
            
    # ========================
    # Dataset Type Detection and Repair
    # ========================
    
    def _detect_dataset_type(self, df: pd.DataFrame) -> DatasetType:
        """
        Detect the type of polymer dataset
        
        Returns:
            DatasetType enum indicating the dataset structure
        """
        has_poly_chemprop = 'poly_chemprop_input' in df.columns and df['poly_chemprop_input'].notna().any()
        has_monomers = any(col in df.columns for col in ['monoA', 'monoB', 'smiles', 'MonA', 'MonB', 'SMILES'])
        has_structure = all(col in df.columns for col in ['poly_type', 'comp', 'fracA', 'fracB'])
        
        # Check if it's just monomers (no polymer info at all)
        if not has_poly_chemprop and any(col in df.columns for col in ['smiles', 'SMILES', 'Smiles']) and 'poly_type' not in df.columns:
            return DatasetType.TYPE_C
        
        # Has poly_chemprop_input but missing structure
        if has_poly_chemprop and not has_monomers and not has_structure:
            return DatasetType.TYPE_A
        
        # Has structure but missing poly_chemprop_input
        if not has_poly_chemprop and has_monomers and has_structure:
            return DatasetType.TYPE_B
        
        # Has everything
        if has_poly_chemprop and has_monomers and has_structure:
            return DatasetType.TYPE_COMPLETE
        
        # Edge cases
        if has_poly_chemprop and has_monomers and not has_structure:
            # Has poly_chemprop and monomers but missing structure info
            return DatasetType.TYPE_A  # Treat as Type A, will extract structure
        
        return DatasetType.TYPE_UNKNOWN
    
    def _detect_and_repair_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Automatically detect dataset type and repair missing columns
        
        Args:
            df: Input dataframe
            dataset_name: Name for logging purposes
            
        Returns:
            Repaired dataframe with all necessary columns
        """
        if not self.auto_repair:
            return df
        
        dataset_type = self._detect_dataset_type(df)
        
        if self.verbose:
            logger.info(f"Dataset '{dataset_name}' detected as: {dataset_type.value}")
        
        if dataset_type == DatasetType.TYPE_COMPLETE:
            return df
        
        elif dataset_type == DatasetType.TYPE_A:
            # Has poly_chemprop_input but missing structure
            return self._repair_type_a_dataset(df)
        
        elif dataset_type == DatasetType.TYPE_B:
            # Has structure but missing poly_chemprop_input
            return self._repair_type_b_dataset(df)
        
        elif dataset_type == DatasetType.TYPE_C:
            # Monomers only - this is handled differently (needs expansion)
            if self.verbose:
                logger.info(f"Dataset '{dataset_name}' contains only monomers - will be expanded during processing")
            return df
        
        else:
            if self.verbose:
                logger.warning(f"Unknown dataset type for '{dataset_name}' - returning as-is")
            return df
    
    def _repair_type_a_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Repair Type A dataset: has poly_chemprop_input but missing structural columns
        """
        if self.verbose:
            logger.info("Repairing Type A dataset (extracting structure from poly_chemprop_input)...")
        
        repaired_df = df.copy()
        
        # Extract monomers if missing
        if 'monoA' not in repaired_df.columns or 'monoB' not in repaired_df.columns:
            if self.verbose:
                logger.info("Extracting monomers from poly_chemprop_input...")
            
            monomers_extracted = repaired_df['poly_chemprop_input'].apply(self._extract_monomers_from_poly_input)
            repaired_df['monoA'] = monomers_extracted.apply(lambda x: x[0] if x else None)
            repaired_df['monoB'] = monomers_extracted.apply(lambda x: x[1] if x else None)
        
        # Extract poly_type if missing
        if 'poly_type' not in repaired_df.columns:
            if self.verbose:
                logger.info("Detecting polymer types from connectivity patterns...")
            
            repaired_df['poly_type'] = repaired_df['poly_chemprop_input'].apply(
                lambda x: self._detect_poly_type_from_poly_input(x)
            )
        
        # Extract comp if missing
        if 'comp' not in repaired_df.columns:
            if self.verbose:
                logger.info("Detecting compositions from stoichiometry...")
            
            repaired_df['comp'] = repaired_df['poly_chemprop_input'].apply(
                lambda x: self._detect_comp_from_poly_input(x)
            )
        
        # Extract fracA/fracB if missing
        if 'fracA' not in repaired_df.columns or 'fracB' not in repaired_df.columns:
            if self.verbose:
                logger.info("Extracting stoichiometry fractions...")
            
            fractions = repaired_df['poly_chemprop_input'].apply(
                lambda x: self._extract_fractions_from_poly_input(x)
            )
            repaired_df['fracA'] = fractions.apply(lambda x: x[0] if x else 0.5)
            repaired_df['fracB'] = fractions.apply(lambda x: x[1] if x else 0.5)
        
        # Generate poly_id if missing
        if 'poly_id' not in repaired_df.columns:
            existing_ids = set()
            if self.template_df is not None and 'poly_id' in self.template_df.columns:
                existing_ids = set(self.template_df['poly_id'].unique())
            repaired_df['poly_id'] = self.generate_poly_ids(repaired_df, existing_ids)
        
        # Generate IUPAC names if missing
        for col, base_col in [('monoA_IUPAC', 'monoA'), ('monoB_IUPAC', 'monoB')]:
            if col not in repaired_df.columns and base_col in repaired_df.columns:
                if self.verbose:
                    logger.info(f"Generating {col}...")
                repaired_df[col] = repaired_df[base_col].apply(self.get_iupac_name)
        
        # Generate master_chemprop_input if missing
        if 'master_chemprop_input' not in repaired_df.columns:
            if 'monoA' in repaired_df.columns and 'monoB' in repaired_df.columns:
                if self.verbose:
                    logger.info("Generating master_chemprop_input...")
                repaired_df['master_chemprop_input'] = [
                    self.make_master_chemprop_input(sA, sB) if pd.notna(sA) and pd.notna(sB) else None
                    for sA, sB in zip(repaired_df['monoA'], repaired_df['monoB'])
                ]
        
        return repaired_df
    
    def _repair_type_b_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Repair Type B dataset: has structure but missing poly_chemprop_input
        """
        if self.verbose:
            logger.info("Repairing Type B dataset (generating poly_chemprop_input)...")
        
        repaired_df = df.copy()
        
        # Ensure all required columns exist with defaults
        if 'poly_type' not in repaired_df.columns:
            repaired_df['poly_type'] = 'alternating'
        if 'comp' not in repaired_df.columns:
            repaired_df['comp'] = '4A_4B'
        if 'fracA' not in repaired_df.columns:
            repaired_df['fracA'] = 0.5
        if 'fracB' not in repaired_df.columns:
            repaired_df['fracB'] = 0.5
        
        # Generate poly_chemprop_input
        if self.verbose:
            logger.info("Generating poly_chemprop_input for all polymers...")
        
        poly_inputs = []
        for idx, row in repaired_df.iterrows():
            if pd.notna(row.get('monoA')) and pd.notna(row.get('monoB')):
                poly_input = self.make_poly_chemprop_input(
                    row['monoA'],
                    row['monoB'],
                    row['poly_type'],
                    row['fracA']
                )
                poly_inputs.append(poly_input)
            else:
                poly_inputs.append(None)
            
            if self.verbose and idx % 1000 == 0:
                logger.info(f"Processed {idx}/{len(repaired_df)} polymers")
        
        repaired_df['poly_chemprop_input'] = poly_inputs
        
        # Generate other missing columns
        if 'poly_id' not in repaired_df.columns:
            existing_ids = set()
            if self.template_df is not None and 'poly_id' in self.template_df.columns:
                existing_ids = set(self.template_df['poly_id'].unique())
            repaired_df['poly_id'] = self.generate_poly_ids(repaired_df, existing_ids)
        
        # Generate IUPAC names if missing
        for col, base_col in [('monoA_IUPAC', 'monoA'), ('monoB_IUPAC', 'monoB')]:
            if col not in repaired_df.columns and base_col in repaired_df.columns:
                if self.verbose:
                    logger.info(f"Generating {col}...")
                repaired_df[col] = repaired_df[base_col].apply(self.get_iupac_name)
        
        # Generate master_chemprop_input if missing
        if 'master_chemprop_input' not in repaired_df.columns:
            if self.verbose:
                logger.info("Generating master_chemprop_input...")
            repaired_df['master_chemprop_input'] = [
                self.make_master_chemprop_input(sA, sB) if pd.notna(sA) and pd.notna(sB) else None
                for sA, sB in zip(repaired_df['monoA'], repaired_df['monoB'])
            ]
        
        return repaired_df
    
    def _detect_poly_type_from_poly_input(self, poly_input: str) -> str:
        """
        Detect polymer type from poly_chemprop_input connectivity pattern
        """
        if pd.isna(poly_input) or not isinstance(poly_input, str):
            return "unknown"
        
        # Clean the input
        poly_input = self.clean_poly_chemprop_input(poly_input)
        if not poly_input:
            return "unknown"
        
        # Extract connectivity pattern
        parts = poly_input.split('|')
        if len(parts) < 3:
            return "unknown"
        
        connectivity = parts[2]
        
        # Use the enhanced detection method
        return self._detect_polymer_type_from_connectivity(connectivity)
        
    def _analyze_connectivity_pattern(self, connectivity: str) -> Dict[str, any]:
        """
        Analyze a connectivity pattern to extract key features
        """
        if not connectivity:
            return {}
        
        # Extract all edges and weights
        edge_pattern = r'<(\d+)-(\d+):([0-9.]+):([0-9.]+)'
        matches = re.findall(edge_pattern, connectivity)
        
        if not matches:
            return {}
        
        edges = []
        weights = []
        
        for match in matches:
            n1, n2, w1, w2 = match
            edges.append((int(n1), int(n2)))
            weights.append(float(w1))
        
        # Analyze edge types
        self_edges = [(n1, n2) for n1, n2 in edges if n1 == n2]
        cross_edges = [(n1, n2) for n1, n2 in edges if n1 != n2]
        
        # Get unique weights
        unique_weights = sorted(set(weights))
        
        # Get node groups
        nodes = set()
        for n1, n2 in edges:
            nodes.add(n1)
            nodes.add(n2)
        
        # Likely monomer A nodes: 1, 2
        # Likely monomer B nodes: 3, 4 (or higher)
        a_nodes = {n for n in nodes if n <= 2}
        b_nodes = {n for n in nodes if n > 2}
        
        # Count edge types
        aa_edges = [(n1, n2) for n1, n2 in edges if n1 in a_nodes and n2 in a_nodes]
        bb_edges = [(n1, n2) for n1, n2 in edges if n1 in b_nodes and n2 in b_nodes]
        ab_edges = [(n1, n2) for n1, n2 in edges if (n1 in a_nodes and n2 in b_nodes) or (n1 in b_nodes and n2 in a_nodes)]
        
        return {
            'total_edges': len(edges),
            'self_edges': len(self_edges),
            'cross_edges': len(cross_edges),
            'unique_weights': unique_weights,
            'aa_edges': len(aa_edges),
            'bb_edges': len(bb_edges),
            'ab_edges': len(ab_edges),
            'nodes': nodes
        }
        
    def _detect_comp_from_poly_input(self, poly_input: str) -> str:
        """
        Detect composition from poly_chemprop_input stoichiometry
        """
        if pd.isna(poly_input) or not isinstance(poly_input, str):
            return "unknown"
        
        # Clean the input
        poly_input = self.clean_poly_chemprop_input(poly_input)
        if not poly_input:
            return "unknown"
        
        # Extract stoichiometry
        parts = poly_input.split('|')
        if len(parts) < 2:
            return "unknown"
        
        stoich = parts[1]
        return self._detect_composition_from_stoichiometry(stoich)
    
    def _extract_fractions_from_poly_input(self, poly_input: str) -> Tuple[float, float]:
        """
        Extract fracA and fracB from poly_chemprop_input
        """
        if pd.isna(poly_input) or not isinstance(poly_input, str):
            return (0.5, 0.5)
        
        # Clean the input
        poly_input = self.clean_poly_chemprop_input(poly_input)
        if not poly_input:
            return (0.5, 0.5)
        
        # Extract stoichiometry
        parts = poly_input.split('|')
        if len(parts) < 2:
            return (0.5, 0.5)
        
        stoich = parts[1]
        
        try:
            if '|' in stoich:
                fracA = float(stoich.split('|')[0])
                fracB = float(stoich.split('|')[1])
            else:
                # Try to parse as single fraction
                fracA = float(stoich)
                fracB = 1.0 - fracA
            
            return (fracA, fracB)
        except:
            return (0.5, 0.5)
    
    # ========================
    # Enhanced Smart Merging
    # ========================
    
    def smart_merge_datasets(self, dataset_paths: List[str], output_path: str,
                       merge_strategy: Union[str, MergeStrategy] = MergeStrategy.FIRST,
                       repair_missing: bool = True,
                       expand_monomers: bool = True,
                       remove_duplicates: bool = True,
                       fix_unknowns: bool = True,
                       custom_identifier_columns: set = None) -> pd.DataFrame:
        """
        Smart merge multiple polymer datasets of different types
        
        Args:
            dataset_paths: List of paths to datasets to merge
            output_path: Path to save merged dataset
            merge_strategy: How to handle duplicate properties
            repair_missing: Whether to repair missing columns before merging
            expand_monomers: Whether to expand monomer datasets to polymers
            remove_duplicates: Whether to remove duplicate polymers
            fix_unknowns: Whether to fix unknown values after merging
            custom_identifier_columns: Optional custom set of columns to always use 'first' strategy
            
        Returns:
            Merged dataframe
        """
        if self.verbose:
            logger.info(f"Smart merging {len(dataset_paths)} datasets...")
        
        # Convert merge_strategy to enum if string
        if isinstance(merge_strategy, str):
            merge_strategy = MergeStrategy(merge_strategy)
        
        all_dfs = []
        dataset_types = []
        
        # Load and analyze each dataset
        for path in dataset_paths:
            if not os.path.exists(path):
                if self.verbose:
                    logger.warning(f"File not found: {path}")
                continue
            
            df = pd.read_csv(path)
            dataset_type = self._detect_dataset_type(df)
            
            if self.verbose:
                logger.info(f"Loaded {path.split('/')[-1]}: {len(df)} rows, type: {dataset_type.value}")
            
            # Handle different dataset types
            if dataset_type == DatasetType.TYPE_C and expand_monomers:
                # Expand monomers to polymers
                if self.verbose:
                    logger.info(f"Expanding monomers to polymers for {path.split('/')[-1]}...")
                
                # Process as new dataset with expansion
                processed_df = self.process_new_dataset(
                    df=df,
                    expand_variants=True,
                    interactive=False,
                    fix_existing_unknowns=fix_unknowns
                )
                all_dfs.append(processed_df)
            else:
                # Repair if needed
                if repair_missing:
                    df = self._detect_and_repair_dataset(df, path.split('/')[-1])
                all_dfs.append(df)
            
            dataset_types.append(dataset_type)
        
        if not all_dfs:
            raise ValueError("No valid datasets found to merge")
        
        # Determine all unique columns
        all_columns = set()
        for df in all_dfs:
            all_columns.update(df.columns)
        
        # Convert to sorted list for consistent ordering
        all_columns = sorted(list(all_columns))
        
        # Standardize all dataframes to have the same columns (efficiently)
        standardized_dfs = []
        for df in all_dfs:
            # Find missing columns
            missing_cols = [col for col in all_columns if col not in df.columns]
            
            if missing_cols:
                # Create a DataFrame with all missing columns at once
                missing_df = pd.DataFrame(
                    index=df.index,
                    columns=missing_cols,
                    data=np.nan
                )
                # Concatenate original df with missing columns df
                df = pd.concat([df, missing_df], axis=1)
            
            # Reorder columns to match all_columns
            df = df[all_columns]
            standardized_dfs.append(df)
        
        all_dfs = standardized_dfs
                               
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if self.verbose:
            logger.info(f"Combined dataset has {len(combined_df)} rows before deduplication")
        
        # Handle duplicates with smart merging
        if remove_duplicates and 'poly_chemprop_input' in combined_df.columns:
            merged_df = self._smart_merge_duplicates(combined_df, merge_strategy, custom_identifier_columns)
        else:
            merged_df = combined_df
        
        # ALWAYS regenerate poly_ids for clean sequential numbering from 1
        if self.verbose:
            logger.info("Regenerating all poly_ids for clean sequential numbering...")
        
        # First, order by existing poly_id if possible to maintain some ordering
        if 'poly_id' in merged_df.columns:
            # Try to extract numeric values for sorting
            def extract_numeric(poly_id):
                try:
                    poly_id_str = str(poly_id)
                    if poly_id_str.lower() in ['unknown', 'nan', 'none', '']:
                        return float('inf')  # Put unknowns at the end
                    
                    # Handle underscore formats
                    if '_' in poly_id_str:
                        parts = poly_id_str.split('_')
                        for part in reversed(parts):
                            if part.isdigit():
                                return int(part)
                    
                    if poly_id_str.isdigit():
                        return int(poly_id_str)
                    
                    # Extract any number
                    numbers = re.findall(r'\d+', poly_id_str)
                    if numbers:
                        return max(int(n) for n in numbers)
                    
                    return float('inf')
                except:
                    return float('inf')
            
            # Sort by extracted numeric value
            merged_df['_sort_key'] = merged_df['poly_id'].apply(extract_numeric)
            merged_df = merged_df.sort_values('_sort_key').drop('_sort_key', axis=1)
        
        # ALWAYS regenerate poly_ids starting from 1
        merged_df['poly_id'] = self.generate_poly_ids(merged_df, set())
        
        if self.verbose:
            logger.info(f"Generated sequential poly_ids from 1 to {len(merged_df)}")
        
        # Fix other unknown values if requested
        if fix_unknowns:
            # Create a copy to fix unknowns
            fixed_df = merged_df.copy()
            
            # Fix unknowns in poly_type, comp, and IUPAC columns
            # Fix poly_type
            if 'poly_type' in fixed_df.columns and 'poly_chemprop_input' in fixed_df.columns:
                unknown_mask = fixed_df['poly_type'].isin(['unknown', 'Unknown', None, '']) | fixed_df['poly_type'].isna()
                
                if unknown_mask.any():
                    def extract_and_detect_poly_type(poly_input):
                        if pd.notna(poly_input):
                            parts = str(poly_input).split('|')
                            if len(parts) >= 3:
                                connectivity = parts[2]
                                return self._detect_polymer_type_from_connectivity(connectivity)
                        return "unknown"
                    
                    detected_types = fixed_df.loc[unknown_mask, 'poly_chemprop_input'].apply(extract_and_detect_poly_type)
                    fixed_mask = detected_types != "unknown"
                    fixed_df.loc[unknown_mask & fixed_mask, 'poly_type'] = detected_types[fixed_mask]
            
            # Fix comp
            if 'comp' in fixed_df.columns and 'poly_chemprop_input' in fixed_df.columns:
                unknown_mask = fixed_df['comp'].isin(['unknown', 'Unknown', None, '']) | fixed_df['comp'].isna()
                
                if unknown_mask.any():
                    def extract_and_detect_comp(poly_input):
                        if pd.notna(poly_input):
                            parts = str(poly_input).split('|')
                            if len(parts) >= 2:
                                stoich = parts[1]
                                return self._detect_composition_from_stoichiometry(stoich)
                        return "unknown"
                    
                    detected_comps = fixed_df.loc[unknown_mask, 'poly_chemprop_input'].apply(extract_and_detect_comp)
                    fixed_mask = detected_comps != "unknown"
                    fixed_df.loc[unknown_mask & fixed_mask, 'comp'] = detected_comps[fixed_mask]
            
            # Keep our regenerated poly_ids
            fixed_df['poly_id'] = merged_df['poly_id']
            merged_df = fixed_df
        
        # Order columns nicely
        merged_df = self._order_columns(merged_df)
        
        # Apply comprehensive fixes for truncated SMILES and unknown poly_types
        if self.verbose:
            logger.info("Applying comprehensive dataset fixes...")
        merged_df = self.post_merge_cleanup(merged_df)
        
        merged_df.to_csv(output_path, index=False)
                               
        if self.verbose:
            logger.info(f"Smart merge complete!")
            logger.info(f"Final dataset: {len(merged_df)} polymers")
            logger.info(f"Poly IDs: Sequential from 1 to {len(merged_df)}")
            logger.info(f"Saved to: {output_path}")
            
            # Report property coverage
            self._report_property_coverage(merged_df)
        
        return merged_df
    
    def _smart_merge_duplicates(self, df: pd.DataFrame, merge_strategy: MergeStrategy,
                                identifier_columns: set = None) -> pd.DataFrame:
        """
        Intelligently merge duplicate polymers using specified strategy
        
        Args:
            df: DataFrame with potential duplicates
            merge_strategy: Strategy for merging (first/last/mean/max/min)
            identifier_columns: Optional custom set of columns to always use 'first' strategy
        """
        if self.verbose:
            logger.info(f"Merging duplicates using strategy: {merge_strategy.value}")
        
        # Group by poly_chemprop_input
        grouped = df.groupby('poly_chemprop_input')
        
        # Extended list of identifier/metadata columns that should not be averaged
        # CUSTOMIZE THIS SET TO CONTROL MERGE BEHAVIOR
        if identifier_columns is None:
            # Use instance's default identifier columns
            identifier_columns = self.identifier_columns
        
        # Define aggregation functions based on strategy
        agg_dict = {}
        
        for col in df.columns:
            if col == 'poly_chemprop_input':
                continue
            
            # Check if column is in identifier list (case-insensitive)
            col_lower = col.lower()
            is_identifier = col in identifier_columns or any(
                id_col.lower() in col_lower for id_col in ['_id', '_name', '_class', '_type', '_ref']
            )
            
            if is_identifier:
                # Identifier columns - always take first
                agg_dict[col] = 'first'
            else:
                # Check if column is numeric
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                
                if not is_numeric or merge_strategy == MergeStrategy.FIRST:
                    # Non-numeric or FIRST strategy - take first non-null value
                    agg_dict[col] = lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
                elif merge_strategy == MergeStrategy.LAST:
                    agg_dict[col] = lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
                elif merge_strategy == MergeStrategy.MEAN:
                    # For numeric columns, calculate mean
                    def safe_mean(x):
                        try:
                            numeric_vals = pd.to_numeric(x.dropna(), errors='coerce')
                            numeric_vals = numeric_vals.dropna()
                            return numeric_vals.mean() if len(numeric_vals) > 0 else np.nan
                        except:
                            # If mean fails, fall back to first value
                            return x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
                    agg_dict[col] = safe_mean
                elif merge_strategy == MergeStrategy.MAX:
                    def safe_max(x):
                        try:
                            numeric_vals = pd.to_numeric(x.dropna(), errors='coerce')
                            numeric_vals = numeric_vals.dropna()
                            return numeric_vals.max() if len(numeric_vals) > 0 else np.nan
                        except:
                            return x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
                    agg_dict[col] = safe_max
                elif merge_strategy == MergeStrategy.MIN:
                    def safe_min(x):
                        try:
                            numeric_vals = pd.to_numeric(x.dropna(), errors='coerce')
                            numeric_vals = numeric_vals.dropna()
                            return numeric_vals.min() if len(numeric_vals) > 0 else np.nan
                        except:
                            return x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
                    agg_dict[col] = safe_min
        
        # Debug: Print merge strategy for each column
        if self.verbose:
            logger.info("\nMerge strategy for each column:")
            logger.info("-" * 60)
            strategy_summary = {'first': [], 'numeric': []}
            
            for col, func in agg_dict.items():
                if func == 'first' or not callable(func):
                    strategy_summary['first'].append(col)
                else:
                    strategy_summary['numeric'].append(col)
            
            logger.info(f"Columns using 'first' strategy ({len(strategy_summary['first'])} columns):")
            if len(strategy_summary['first']) <= 20:
                logger.info(f"  {strategy_summary['first']}")
            else:
                logger.info(f"  {strategy_summary['first'][:20]}...")
                logger.info(f"  ... and {len(strategy_summary['first']) - 20} more")
            
            logger.info(f"\nColumns using '{merge_strategy.value}' strategy ({len(strategy_summary['numeric'])} columns):")
            if len(strategy_summary['numeric']) <= 20:
                logger.info(f"  {strategy_summary['numeric']}")
            else:
                logger.info(f"  {strategy_summary['numeric'][:20]}...")
                logger.info(f"  ... and {len(strategy_summary['numeric']) - 20} more")
            logger.info("-" * 60)
        
        # Apply aggregation with error handling
        try:
            merged_df = grouped.agg(agg_dict).reset_index()
        except Exception as e:
            if self.verbose:
                logger.error(f"Error during smart merge: {e}")
                logger.info("Falling back to 'first' strategy for problematic columns")
            
            # Fallback: identify problematic columns and use 'first' for them
            safe_agg_dict = {}
            for col, func in agg_dict.items():
                try:
                    # Test the aggregation on a small sample
                    test_group = grouped.get_group(list(grouped.groups.keys())[0])
                    if callable(func):
                        func(test_group[col])
                    safe_agg_dict[col] = func
                except:
                    # Use 'first' for problematic columns
                    safe_agg_dict[col] = 'first'
                    if self.verbose:
                        logger.warning(f"Column '{col}' cannot use {merge_strategy.value} strategy, using 'first' instead")
            
            merged_df = grouped.agg(safe_agg_dict).reset_index()
        
        if self.verbose:
            original_count = len(df)
            final_count = len(merged_df)
            logger.info(f"Reduced from {original_count} to {final_count} polymers ({original_count - final_count} duplicates merged)")
        
        return merged_df

    def _is_numeric_column(self, series: pd.Series, threshold: float = 0.5) -> bool:
        """
        Check if a column contains mostly numeric data
        
        Args:
            series: Pandas series to check
            threshold: Minimum fraction of numeric values to consider column numeric
            
        Returns:
            True if column is numeric, False otherwise
        """
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        # Try to convert to numeric and see how many succeed
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_numeric = numeric_series.notna().sum()
            total_non_null = series.notna().sum()
            
            if total_non_null == 0:
                return False
                
            return (valid_numeric / total_non_null) >= threshold
        except:
            return False
            
    def _order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order columns in a logical way
        """
        # Define preferred column order
        preferred_order = [
            'poly_id', 'poly_type', 'comp', 'fracA', 'fracB',
            'monoA', 'monoA_IUPAC', 'monoB', 'monoB_IUPAC',
            'master_chemprop_input', 'poly_chemprop_input'
        ]
        
        # Build final column order
        ordered_columns = []
        
        # First add preferred columns that exist
        for col in preferred_order:
            if col in df.columns:
                ordered_columns.append(col)
        
        # Then add any remaining columns
        remaining_columns = [col for col in df.columns if col not in ordered_columns]
        
        # Sort remaining columns for consistency
        remaining_columns = natsorted(remaining_columns) if NATSORT_AVAILABLE else sorted(remaining_columns)
        
        final_columns = ordered_columns + remaining_columns
        
        return df[final_columns]
    
    def _report_property_coverage(self, df: pd.DataFrame):
        """
        Report coverage statistics for properties in the dataset
        """
        print("\nProperty Coverage Report:")
        print("=" * 60)
        
        # Identify property columns
        structural_cols = ['poly_id', 'poly_type', 'comp', 'fracA', 'fracB', 
                          'monoA', 'monoA_IUPAC', 'monoB', 'monoB_IUPAC',
                          'master_chemprop_input', 'poly_chemprop_input']
        
        property_cols = [col for col in df.columns if col not in structural_cols]
        
        if not property_cols:
            print("No property columns found")
            return
        
        # Sort properties by coverage
        coverage_stats = []
        for col in property_cols:
            non_null = df[col].notna().sum()
            coverage = non_null / len(df) * 100
            coverage_stats.append((col, non_null, coverage))
        
        coverage_stats.sort(key=lambda x: x[2], reverse=True)
        
        # Display statistics
        for col, non_null, coverage in coverage_stats:
            print(f"{col:<40} {non_null:>7,}/{len(df):<7,} ({coverage:>5.1f}%)")
        
        # Summary statistics
        print("\nSummary:")
        print(f"Total polymers: {len(df):,}")
        print(f"Total properties: {len(property_cols)}")
        
        # Count polymers with at least one property
        has_any_property = df[property_cols].notna().any(axis=1).sum()
        print(f"Polymers with at least one property: {has_any_property:,} ({has_any_property/len(df)*100:.1f}%)")
        
        # Count fully characterized polymers
        fully_characterized = df[property_cols].notna().all(axis=1).sum()
        if fully_characterized > 0:
            print(f"Fully characterized polymers: {fully_characterized:,} ({fully_characterized/len(df)*100:.1f}%)")
    
    # ========================
    # Enhanced Chemical Processing (inherited from previous version)
    # ========================
    
    def canonicalize_smiles(self, smiles: str, verbose_conversion: bool = False) -> Optional[str]:
        """
        Enhanced canonicalize SMILES with better error handling and repair
        
        Args:
            smiles: SMILES string to canonicalize
            verbose_conversion: Whether to log bare asterisk conversions (default: False for bulk operations)
        """
        try:
            # Pre-filter obviously invalid patterns
            if not smiles or smiles.strip() == '' or smiles == 'nan':
                return None
                
            # Pre-clean common corruption patterns
            smiles = self._pre_clean_smiles(smiles)
            
            # Pre-repair empty parentheses if still present
            if '()' in smiles:
                # Try to fix empty parentheses before sending to RDKit
                smiles = self._attempt_smiles_repair(smiles)
                
            # Count parentheses
            open_parens = smiles.count('(')
            close_parens = smiles.count(')')
            if abs(open_parens - close_parens) > 3:  # Too unbalanced
                if self.verbose:
                    logger.warning(f"Severely unbalanced parentheses in SMILES: {smiles}")
                return None
            
            # Handle different attachment point formats
            numbered_smiles = smiles
            
            # ENHANCED: Handle bare asterisks - convert ALL of them to numbered format
            if '*' in numbered_smiles and '[*:' not in numbered_smiles:
                # Count bare asterisks (not inside brackets)
                import re
                bare_asterisks = re.findall(r'(?<!\[)\*(?![:\]])', numbered_smiles)
                
                # Replace each bare asterisk with numbered attachment point
                counter = 1
                while True:
                    # Find next bare asterisk
                    match = re.search(r'(?<!\[)\*(?![:\]])', numbered_smiles)
                    if not match:
                        break
                    # Replace it with numbered attachment point
                    numbered_smiles = numbered_smiles[:match.start()] + f'[*:{counter}]' + numbered_smiles[match.end():]
                    counter += 1
            
            # Convert [*] to [*:1], [*:2] etc.
            counter = 1
            while '[*]' in numbered_smiles:
                numbered_smiles = numbered_smiles.replace('[*]', f'[*:{counter}]', 1)
                counter += 1
            
            # Try original first, then attempt repair if needed
            mol = Chem.MolFromSmiles(numbered_smiles)
            
            if mol is None:
                # Enhanced repair attempts
                repaired_smiles = self._attempt_smiles_repair(numbered_smiles)
                if repaired_smiles != numbered_smiles:
                    mol = Chem.MolFromSmiles(repaired_smiles)
                    if mol is not None:
                        numbered_smiles = repaired_smiles
                        if self.verbose:
                            logger.info(f"Auto-repaired SMILES: {smiles} → {repaired_smiles}")
                    else:
                        # Try multiple repair strategies
                        for attempt in range(5):  # Increased attempts
                            if attempt == 0:
                                # Strategy 1: Remove all ()
                                test_smiles = repaired_smiles.replace('()', '')
                            elif attempt == 1:
                                # Strategy 2: Replace () with C
                                test_smiles = repaired_smiles.replace('()', 'C')
                            elif attempt == 2:
                                # Strategy 3: More aggressive repair
                                test_smiles = self._aggressive_repair(repaired_smiles)
                            elif attempt == 3:
                                # Strategy 4: Fix double bonds
                                test_smiles = self._fix_double_bonds(repaired_smiles)
                            else:
                                # Strategy 5: Last resort - simplify
                                test_smiles = self._simplify_smiles(repaired_smiles)
                            
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
                # FIXED: Added try-except around SanitizeMol
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                
                # Ensure consistent attachment point numbering
                canonical_smiles = self._standardize_attachment_points(canonical_smiles)
                
                return canonical_smiles
            except Exception as e:
                # If sanitization fails, try to fix the structure
                try:
                    for atom in mol.GetAtoms():
                        if atom.GetIsAromatic():
                            atom.SetNumExplicitHs(0)
                            atom.SetNoImplicit(False)
                    Chem.SanitizeMol(mol)
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    canonical_smiles = self._standardize_attachment_points(canonical_smiles)
                    return canonical_smiles
                except:
                    return None
        except:
            return None
    # Add these enhanced methods to your PolymerDatabaseManager class

    def validate_and_fix_smiles(self, smiles: str) -> Tuple[str, bool]:
        """
        Validate SMILES and attempt to fix common issues
        
        Returns:
            Tuple of (fixed_smiles, is_valid)
        """
        if not smiles or pd.isna(smiles):
            return None, False
        
        smiles = str(smiles).strip()
        
        # Count parentheses
        open_count = smiles.count('(')
        close_count = smiles.count(')')
        
        # Fix truncated SMILES by adding missing closing parentheses
        if open_count > close_count:
            smiles += ')' * (open_count - close_count)
            if self.verbose:
                logger.info(f"Fixed truncated SMILES by adding {open_count - close_count} closing parentheses")
        
        # Fix common syntax errors
        # Fix C4=O( pattern -> C4(=O)
        smiles = re.sub(r'([A-Za-z]\d+)=O\(', r'\1(=O)', smiles)
        
        # Fix =O( at the end -> (=O)
        smiles = re.sub(r'=O\($', '(=O)', smiles)
        
        # Validate with RDKit
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return smiles, True
            else:
                return smiles, False
        except:
            return smiles, False
    
    def enhanced_detect_polymer_type(self, connectivity: str) -> str:
        """
        Enhanced polymer type detection with more flexible pattern matching
        """
        if not connectivity:
            return "unknown"
        
        # Count edge types
        edges = re.findall(r'<(\d+)-(\d+):', connectivity)
        if not edges:
            return "unknown"
        
        # Create edge sets
        self_edges = set()
        cross_edges = set()
        
        for n1, n2 in edges:
            n1, n2 = int(n1), int(n2)
            if n1 == n2:
                self_edges.add((n1, n2))
            else:
                # Normalize cross edges (smaller number first)
                cross_edges.add((min(n1, n2), max(n1, n2)))
        
        # Decision logic
        if len(self_edges) == 0:
            # No self-edges = alternating
            return "alternating"
        
        elif len(self_edges) >= 4:
            # Many self-edges (1-1, 2-2, 3-3, 4-4) = random
            return "random"
        
        elif len(self_edges) > 0 and len(self_edges) < 4:
            # Some self-edges but not all = block
            return "block"
        
        # Fallback to edge count
        total_edges = len(edges)
        if total_edges == 4 and len(self_edges) == 0:
            return "alternating"
        elif total_edges >= 10:
            return "random"
        elif total_edges >= 6:
            return "block"
        
        return "unknown"
    
    def fix_dataset_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive fix for dataset issues including truncated SMILES and unknown poly_types
        """
        fixed_df = df.copy()
        fixes_made = {
            'truncated_smiles': 0,
            'invalid_smiles': 0,
            'unknown_poly_types': 0,
            'missing_poly_inputs': 0
        }
        
        if self.verbose:
            logger.info("Starting comprehensive dataset fix...")
        
        # 1. Fix truncated/invalid SMILES
        for col in ['monoA', 'monoB']:
            if col in fixed_df.columns:
                for idx, smiles in enumerate(fixed_df[col]):
                    if pd.notna(smiles) and smiles:
                        fixed_smiles, is_valid = self.validate_and_fix_smiles(smiles)
                        if fixed_smiles != smiles:
                            fixed_df.at[idx, col] = fixed_smiles
                            fixes_made['truncated_smiles'] += 1
                        if not is_valid:
                            fixes_made['invalid_smiles'] += 1
        
        # 2. Re-canonicalize SMILES after fixes
        if 'monoA' in fixed_df.columns:
            fixed_df['monoA'] = fixed_df['monoA'].apply(
                lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else None
            )
        
        if 'monoB' in fixed_df.columns:
            fixed_df['monoB'] = fixed_df['monoB'].apply(
                lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else None
            )
        
        # 3. Regenerate IUPAC names for fixed SMILES
        if 'monoA_IUPAC' in fixed_df.columns:
            mask = fixed_df['monoA_IUPAC'] == 'Invalid_SMILES'
            if mask.any():
                fixed_df.loc[mask, 'monoA_IUPAC'] = fixed_df.loc[mask, 'monoA'].apply(self.get_iupac_name)
        
        if 'monoB_IUPAC' in fixed_df.columns:
            mask = fixed_df['monoB_IUPAC'] == 'Invalid_SMILES'
            if mask.any():
                fixed_df.loc[mask, 'monoB_IUPAC'] = fixed_df.loc[mask, 'monoB'].apply(self.get_iupac_name)
        
        # 4. Fix unknown poly_types using enhanced detection
        if 'poly_type' in fixed_df.columns and 'poly_chemprop_input' in fixed_df.columns:
            unknown_mask = fixed_df['poly_type'].isin(['unknown', 'Unknown', None, ''])
            
            for idx in fixed_df[unknown_mask].index:
                poly_input = fixed_df.at[idx, 'poly_chemprop_input']
                if pd.notna(poly_input):
                    parts = str(poly_input).split('|')
                    if len(parts) >= 4:  # monomers|fracA|fracB|connectivity
                        connectivity = parts[3]
                        detected_type = self.enhanced_detect_polymer_type(connectivity)
                        if detected_type != "unknown":
                            fixed_df.at[idx, 'poly_type'] = detected_type
                            fixes_made['unknown_poly_types'] += 1
        
        # 5. Regenerate poly_chemprop_input for rows with valid monomers but missing/invalid poly_input
        if all(col in fixed_df.columns for col in ['monoA', 'monoB', 'poly_type', 'fracA']):
            missing_mask = fixed_df['poly_chemprop_input'].isna()
            
            for idx in fixed_df[missing_mask].index:
                monoA = fixed_df.at[idx, 'monoA']
                monoB = fixed_df.at[idx, 'monoB'] 
                poly_type = fixed_df.at[idx, 'poly_type']
                fracA = fixed_df.at[idx, 'fracA']
                
                if all(pd.notna(x) for x in [monoA, monoB, poly_type, fracA]):
                    poly_input = self.make_poly_chemprop_input(monoA, monoB, poly_type, fracA)
                    if poly_input:
                        fixed_df.at[idx, 'poly_chemprop_input'] = poly_input
                        fixes_made['missing_poly_inputs'] += 1
        
        # Report fixes
        if self.verbose:
            logger.info("Dataset fixes completed:")
            for fix_type, count in fixes_made.items():
                if count > 0:
                    logger.info(f"  - {fix_type}: {count} fixes")
        
        return fixed_df
    
    # Enhanced function to use after smart merge
    def post_merge_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive cleanup after merging datasets
        """
        if self.verbose:
            logger.info("Running post-merge cleanup...")
        
        # 1. Fix dataset issues
        df = self.fix_dataset_issues(df)
        
        # 2. Remove rows with unfixable issues
        initial_count = len(df)
        
        # Remove rows where both monomers are invalid
        if all(col in df.columns for col in ['monoA', 'monoB']):
            valid_mask = df['monoA'].notna() | df['monoB'].notna()
            df = df[valid_mask]
        
        # Remove rows without poly_chemprop_input
        if 'poly_chemprop_input' in df.columns:
            df = df[df['poly_chemprop_input'].notna()]
        
        final_count = len(df)
        if initial_count != final_count and self.verbose:
            logger.info(f"Removed {initial_count - final_count} unfixable rows")
        
        # 3. Final sequential poly_id regeneration
        df['poly_id'] = range(1, len(df) + 1)
        
        return df
    
    def _fix_double_bonds(self, smiles: str) -> str:
        """Fix common double bond issues"""
        fixed = smiles
        # Fix disconnected double bonds
        fixed = re.sub(r'=\(\)', '=C', fixed)
        fixed = re.sub(r'\(\)=', 'C=', fixed)
        # Fix double bond valence issues
        fixed = re.sub(r'C\(=\)C', 'C=C', fixed)
        return fixed
    
    def _simplify_smiles(self, smiles: str) -> str:
        """Simplify complex SMILES as last resort"""
        # Remove all empty parentheses
        simplified = re.sub(r'\(\)+', '', smiles)
        # Fix common patterns
        simplified = re.sub(r'([A-Za-z])([A-Za-z])', r'\1C\2', simplified)
        return simplified

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
        
        # Pre-filter obviously corrupted patterns
        corrupted_patterns = [
            (r'\(\)\(\)', ''),                          # ()() → remove
            (r'\(\)\(\)\(\)', ''),                      # ()()() → remove
            (r'\(\)\(\)\(\)\(\)', ''),                  # ()()()() → remove
        ]
        
        for pattern, replacement in corrupted_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        # Apply repairs in order of specificity
        repaired = self._repair_aromatic_rings(repaired)
        repaired = self._repair_aliphatic_rings(repaired)
        repaired = self._repair_general_patterns(repaired)
        repaired = self._repair_simple_patterns(repaired)
        
        return repaired
        
    def _pre_clean_smiles(self, smiles: str) -> str:
        """
        Pre-clean SMILES before sending to RDKit to avoid parse errors
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Pre-cleaned SMILES string
        """
        if not smiles or not isinstance(smiles, str):
            return smiles
            
        # Quick fixes for common corruption patterns
        cleaned = smiles
        
        # Remove multiple consecutive empty parentheses
        cleaned = re.sub(r'\(\){2,}', '', cleaned)
        
        # Fix incomplete stereochemistry (trailing slashes)
        cleaned = re.sub(r'/$', '', cleaned)  # Remove trailing forward slash
        cleaned = re.sub(r'\\$', '', cleaned)  # Remove trailing backslash
        cleaned = re.sub(r'^/', '', cleaned)  # Remove leading forward slash
        cleaned = re.sub(r'^\\', '', cleaned)  # Remove leading backslash
        
        # Fix empty parentheses in rings
        cleaned = re.sub(r'(\d)\(\)([A-Za-z])', r'\1\2', cleaned)  # e.g., C1()C → C1C
        cleaned = re.sub(r'([A-Za-z])\(\)(\d)', r'\1\2', cleaned)  # e.g., C()1 → C1
        
        # Fix common aromatic ring corruptions
        if '()' in cleaned:
            # Simple aromatic patterns
            cleaned = re.sub(r'c\(\)c', 'cc', cleaned)
            cleaned = re.sub(r'c\(\)n', 'cn', cleaned)
            cleaned = re.sub(r'n\(\)c', 'nc', cleaned)
            
            # Aromatic with numbers
            cleaned = re.sub(r'c(\d+)c\(\)c', r'c\1cc', cleaned)
            cleaned = re.sub(r'cc\(\)c(\d+)', r'ccc\1', cleaned)
        
        return cleaned

    def _repair_aromatic_rings(self, smiles: str) -> str:
        """
        Repair aromatic ring systems with comprehensive pattern matching
        """
        repaired = smiles
        
        # Handle numbered aromatic rings first (most specific)
        numbered_ring_patterns = [
            # Single digit rings: c1...()...c1, c2...()...c2, etc.
            (r'c(\d)([^c]*?)c\(\)([^c]*?)c\1', r'c\1\2c\3c\1'),
            
            # Handle cases like c1cc()ccc1, c1c()cccc1, etc.
            (r'c(\d)((?:c{0,5}|[sno])*?)\(\)((?:c{0,5}|[sno])*?)c\1', r'c\1\2c\3c\1'),
        ]
        
        for pattern, replacement in numbered_ring_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        # Handle aromatic chains without ring numbers
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

    def _repair_aliphatic_rings(self, smiles: str) -> str:
        """
        Repair aliphatic ring systems and chains
        """
        repaired = smiles
        
        # Handle numbered aliphatic rings
        aliphatic_ring_patterns = [
            # Single digit rings: C1...()...C1, C2...()...C2, etc.
            (r'C(\d)([^C]*?)C\(\)([^C]*?)C\1', r'C\1\2C\3C\1'),
            
            # Handle mixed case in rings
            (r'C(\d)((?:C{0,10})*?)\(\)((?:C{0,10})*?)C\1', r'C\1\2C\3C\1'),
        ]
        
        for pattern, replacement in aliphatic_ring_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        # Handle aliphatic chains
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
        
        # Handle complex functional groups
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
        
        # Fallback: Last resort patterns - context-aware replacement
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
        Enhanced IUPAC name generation with better fallbacks
        """
        # Temporarily suppress RDKit errors
        self.set_rdkit_verbosity(False)
        
        try:
            # Handle empty or invalid input
            if pd.isna(smiles) or smiles == '' or smiles == 'nan':
                return "Unknown_compound"
                
            # Remove attachment points for name generation
            clean_smiles = re.sub(r'\[\*:\d+\]', '', smiles)
            clean_smiles = re.sub(r'\[\*\]', '', clean_smiles)
            clean_smiles = re.sub(r'\*', '', clean_smiles)
            
            # IMPORTANT: Remove empty parentheses that result from removing attachment points
            clean_smiles = re.sub(r'\(\)', '', clean_smiles)
            # Also handle cases where there might be nested empty parentheses
            while '()' in clean_smiles:
                clean_smiles = clean_smiles.replace('()', '')
            
            # Fix incomplete stereochemistry
            clean_smiles = re.sub(r'/C\(=C\\', 'C(=C', clean_smiles)
            clean_smiles = re.sub(r'\\C\(=C/', 'C(=C', clean_smiles)
            clean_smiles = re.sub(r'/$', '', clean_smiles)
            clean_smiles = re.sub(r'\\$', '', clean_smiles)
            
            if not clean_smiles or clean_smiles.strip() == '':
                return "Unknown_compound"
            
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
        finally:
            # Re-enable verbosity if it was on
            if self.verbose:
                self.set_rdkit_verbosity(True)

    def _lookup_common_polymer_names(self, smiles: str) -> Optional[str]:
        """
        Enhanced lookup table for common polymer monomers
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
            'c1ccc(cc1)c2ccccc2': 'biphenyl',
            'c1ccc(cc1)Cc2ccccc2': 'diphenylmethane',
            'c1ccc(cc1)C(c2ccccc2)c3ccccc3': 'triphenylmethane',
            'c1cc2cc3ccccc3cc2cc1': 'anthracene',
            'c1ccc2cc3ccccc3cc2c1': 'phenanthrene',
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
        Enhanced descriptive name generation based on molecular features
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
                    elif num_rings == 3:
                        name_parts.append("anthracene")
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
            
            # Enhanced functional group SMARTS patterns
            fg_patterns = {
                'amine': '[NX3;H2,H1;!$(NC=O)]',
                'alcohol': '[OX2H]',
                'carboxylic_acid': '[CX3](=O)[OX2H1]',
                'ester': '[#6][CX3](=O)[OX2H0][#6]',
                'nitro': '[NX3+](=O)[O-]',
                'sulfonyl': '[SX4](=O)(=O)',
                'fluoride': '[F]',
                'chloride': '[Cl]',
                'bromide': '[Br]',
                'iodide': '[I]',
                'nitrile': '[CX2]#[NX1]',
                'thiol': '[SH]',
                'ether': '[OD2]([#6])[#6]',
                'ketone': '[CX3](=O)[#6]',
                'aldehyde': '[CX3H1](=O)',
            }
            
            for fg_name, pattern in fg_patterns.items():
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    functional_groups.append(fg_name)
            
            if functional_groups:
                name_parts.extend(functional_groups[:3])  # Limit to 3 functional groups
            
            if name_parts:
                return "_".join(name_parts) + "_derivative"
            
            # Final fallback
            formula = CalcMolFormula(mol)
            return f"compound_{formula}"
            
        except:
            return None

    def check_polymer_validity(self, mona_smiles: str, monb_smiles: str) -> Tuple[bool, str]:
        """
        Enhanced polymer validity checking
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
        
        # Check if attachment points are properly numbered
        mona_nums = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', mona_smiles)])
        monb_nums = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', monb_smiles)])
        
        if not mona_nums or not monb_nums:
            return False, "Attachment points not properly numbered"
            
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
        
        # For homopolymers, we need different numbering for the two units
        # First unit keeps original numbering (e.g., 1, 2)
        # Second unit gets offset numbering (e.g., 3, 4)
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
    # Enhanced ChemProp Input Generation
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
            # Canonicalize monomers (this will convert bare * to [*:n])
            can_mona = self.canonicalize_smiles(mona)
            is_homopolymer = (monb == mona)

            if is_homopolymer:
                # For homopolymers, ensure we have two differently numbered units
                if can_mona and '[*:' in can_mona:
                    can_mona, can_monb = self.prepare_homopolymer(can_mona)
                else:
                    can_monb = can_mona
            else:
                can_monb = self.canonicalize_smiles(monb)

            if can_mona is None or can_monb is None:
                if self.verbose:
                    logger.debug(f"Canonicalization failed: monoA={can_mona}, monoB={can_monb}")
                return None

            # Check if monomers already have attachment points
            has_attachment_a = '[*:' in can_mona if can_mona else False
            has_attachment_b = '[*:' in can_monb if can_monb else False
            
            # If both already have attachment points, use them directly
            if has_attachment_a and has_attachment_b:
                # Already handled in prepare_homopolymer for homopolymers
                pass
                        
            # For traditional polymer processing (BOO/Br system), process attachment points
            elif '[*:' not in can_mona and '[*:' not in can_monb:
                # Process using original termini removal system
                mA = Chem.MolFromSmiles(can_mona)
                mB = Chem.MolFromSmiles(can_monb)
                
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
            is_valid, error_msg = self.check_polymer_validity(can_mona, can_monb)
            if not is_valid:
                if self.verbose:
                    logger.debug(f"Polymer validation failed: {error_msg}")
                    logger.debug(f"MonoA: {can_mona}, MonoB: {can_monb}")
                return None

            # Extract attachment point numbers
            mona_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', can_mona)])
            monb_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', can_monb)])

            # Build connectivity based on polymer type
            fracB = 1.0 - fracA
            stoich = f"{fracA:.3f}|{fracB:.3f}"
            
            if poly_type == 'alternating' or poly_type == 'alternating (homopolymer)':
                # For standard 2-attachment point monomers
                if len(mona_points) == 2 and len(monb_points) == 2:
                    edges = '<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5'
                else:
                    # Generic alternating for any number of attachment points
                    edges = self.create_polymer_attachment_scheme(is_homopolymer, mona_points, monb_points, (fracA, fracB))
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

            # FIXED: Validate the generated poly_chemprop_input
            poly_input = f"{can_mona}.{can_monb}|{stoich}|{edges}"
            
            # Basic validation
            if not poly_input or poly_input.count('|') != 3:
                if self.verbose:
                    logger.debug(f"Invalid poly_chemprop_input format: {poly_input}")
                return None
                
            return poly_input
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Error in make_poly_chemprop_input: {e}")
                logger.debug(f"Input: monoA={mona}, monoB={monb}, poly_type={poly_type}")
            return None

    def _extract_monomers_from_poly_input(self, poly_input: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Enhanced extraction of monomer SMILES from poly_chemprop_input format
        """
        try:
            if pd.isna(poly_input) or not isinstance(poly_input, str):
                return None, None
            
            # Clean the input first
            poly_input = self.clean_poly_chemprop_input(poly_input)
            if not poly_input:
                return None, None
            
            # Split by first pipe to get monomers part
            parts = poly_input.split('|')
            
            # FIXED: Check if parts is empty
            if not parts or len(parts) < 1:
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

    def clean_poly_chemprop_input(self, poly_input: str, remove_trailing_values: bool = True) -> Optional[str]:
        """
        Enhanced cleaning of poly_chemprop_input with better pattern matching
        """
        if pd.isna(poly_input) or not isinstance(poly_input, str):
            return None
        
        cleaned = poly_input.strip()
        
        if remove_trailing_values:
            import re
            # Only remove values that start with ~ (tilde)
            cleaned = re.sub(r'~[0-9]*\.?[0-9]*$', '', cleaned).strip()
            # Also remove any trailing numeric values without tilde
            cleaned = re.sub(r'\s+[0-9]+\.?[0-9]*$', '', cleaned).strip()
        
        # Validate basic structure
        if '|' not in cleaned:
            return None
            
        return cleaned if cleaned else None

    def _detect_polymer_type_from_connectivity(self, connectivity: str) -> str:
        """
        FIXED VERSION: Detect polymer type from connectivity pattern using smart heuristics
        """
        # Use the new enhanced method instead
        return self.enhanced_detect_polymer_type(connectivity)
    
    def _detect_composition_from_stoichiometry(self, stoich: str) -> str:
        """
        Detect composition from stoichiometry values
        """
        if not stoich:
            return "unknown"
        
        try:
            if '|' in stoich:
                parts = stoich.split('|')
                fracA = float(parts[0])
            else:
                # Handle case where stoich is just the fraction
                fracA = float(stoich.split(':')[0] if ':' in stoich else stoich)
            
            # Match to closest known composition
            if abs(fracA - 0.5) < 0.01:
                return "4A_4B"
            elif abs(fracA - 0.75) < 0.01:
                return "6A_2B"
            elif abs(fracA - 0.25) < 0.01:
                return "2A_6B"
            else:
                # Create custom composition label
                fracB = 1.0 - fracA
                return f"{int(fracA*8)}A_{int(fracB*8)}B"
        except:
            return "unknown"
    
    def _fix_unknown_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix unknown values in poly_type, comp, IUPAC names, and poly_id
        """
        if not self.fix_unknowns:
            return df
        
        fixed_count = 0
        
        # Fix poly_id FIRST - regenerate any unknown/invalid poly_ids
        if 'poly_id' in df.columns:
            # Check for unknown, empty, or invalid poly_ids (convert to string for comparison)
            df['poly_id'] = df['poly_id'].astype(str)  # Ensure string type for comparison
            
            invalid_mask = (
                df['poly_id'].isna() | 
                (df['poly_id'] == 'unknown') | 
                (df['poly_id'] == 'Unknown') | 
                (df['poly_id'] == '') |
                (df['poly_id'] == 'nan') |
                (df['poly_id'] == 'None') |
                (df['poly_id'] == 'NaN')
            )
            
            if invalid_mask.any():
                if self.verbose:
                    logger.info(f"Found {invalid_mask.sum()} invalid poly_id values to fix")
                
                # Get existing valid IDs to continue from
                valid_mask = ~invalid_mask
                if valid_mask.any():
                    valid_ids = df.loc[valid_mask, 'poly_id'].astype(str).unique()
                    # Filter out any that might still be invalid
                    valid_ids = [pid for pid in valid_ids if pid not in ['unknown', 'Unknown', '', 'nan', 'None', 'NaN']]
                    existing_ids = set(valid_ids)
                else:
                    # All IDs are invalid, start fresh
                    existing_ids = set()
                
                # Generate new IDs for all rows (to ensure sequential)
                if invalid_mask.all():
                    # All are invalid - regenerate entire column
                    df['poly_id'] = self.generate_poly_ids(df, set())
                    fixed_count += len(df)
                else:
                    # Only some are invalid - generate for those
                    new_ids = self.generate_poly_ids(df[invalid_mask], existing_ids)
                    df.loc[invalid_mask, 'poly_id'] = new_ids
                    fixed_count += invalid_mask.sum()
                
                if self.verbose:
                    logger.info(f"Fixed {invalid_mask.sum()} invalid poly_id values")
        
        # FIXED: Use vectorized operations for better performance
        if 'poly_type' in df.columns and 'poly_chemprop_input' in df.columns:
            unknown_mask = df['poly_type'].isin(['unknown', 'Unknown', None, '']) | df['poly_type'].isna()
            
            if unknown_mask.any():
                # Vectorized extraction of connectivity patterns
                def extract_and_detect_poly_type(poly_input):
                    if pd.notna(poly_input):
                        parts = str(poly_input).split('|')
                        if len(parts) >= 3:
                            connectivity = parts[2]
                            return self._detect_polymer_type_from_connectivity(connectivity)
                    return "unknown"
                
                detected_types = df.loc[unknown_mask, 'poly_chemprop_input'].apply(extract_and_detect_poly_type)
                fixed_mask = detected_types != "unknown"
                df.loc[unknown_mask & fixed_mask, 'poly_type'] = detected_types[fixed_mask]
                fixed_count += fixed_mask.sum()
        
        # Fix comp
        if 'comp' in df.columns and 'poly_chemprop_input' in df.columns:
            unknown_mask = df['comp'].isin(['unknown', 'Unknown', None, '']) | df['comp'].isna()
            
            if unknown_mask.any():
                # Vectorized extraction of stoichiometry
                def extract_and_detect_comp(poly_input):
                    if pd.notna(poly_input):
                        parts = str(poly_input).split('|')
                        if len(parts) >= 2:
                            stoich = parts[1]
                            return self._detect_composition_from_stoichiometry(stoich)
                    return "unknown"
                
                detected_comps = df.loc[unknown_mask, 'poly_chemprop_input'].apply(extract_and_detect_comp)
                fixed_mask = detected_comps != "unknown"
                df.loc[unknown_mask & fixed_mask, 'comp'] = detected_comps[fixed_mask]
                fixed_count += fixed_mask.sum()
        
        # Fix IUPAC names
        for col in ['monoA_IUPAC', 'monoB_IUPAC']:
            if col in df.columns:
                smiles_col = col.replace('_IUPAC', '')
                if smiles_col in df.columns:
                    invalid_mask = df[col].isin(['Invalid_SMILES', 'Unknown_compound', None, '']) | df[col].isna()
                    
                    if invalid_mask.any():
                        # Process in batches for better performance
                        batch_size = 1000
                        indices = df[invalid_mask].index
                        
                        for i in range(0, len(indices), batch_size):
                            batch_indices = indices[i:i+batch_size]
                            batch_smiles = df.loc[batch_indices, smiles_col]
                            batch_iupac = batch_smiles.apply(self.get_iupac_name)
                            
                            valid_iupac = ~batch_iupac.isin(['Invalid_SMILES', 'Unknown_compound'])
                            df.loc[batch_indices[valid_iupac], col] = batch_iupac[valid_iupac]
                            fixed_count += valid_iupac.sum()
        
        if self.verbose and fixed_count > 0:
            logger.info(f"Fixed {fixed_count} unknown values in the dataset")
        
        return df
    
    # ========================
    # Database Management
    # ========================
    
    def generate_poly_ids(self, df: pd.DataFrame, existing_ids: set = None) -> List[str]:
        """Generate unique polymer IDs that continue from existing template"""
        if existing_ids is None:
            existing_ids = set()
        
        # Find the highest numeric ID from existing IDs
        max_id = 0
        
        if existing_ids:
            for existing_id in existing_ids:
                try:
                    # Convert to string and extract numeric part
                    existing_id_str = str(existing_id)
                    
                    # Skip invalid IDs
                    if existing_id_str.lower() in ['unknown', 'nan', 'none', '']:
                        continue
                    
                    # Handle underscore formats like "1_435" - extract last number
                    if '_' in existing_id_str:
                        parts = existing_id_str.split('_')
                        for part in reversed(parts):
                            if part.isdigit():
                                num = int(part)
                                max_id = max(max_id, num)
                                break
                    elif existing_id_str.isdigit():
                        num = int(existing_id_str)
                        max_id = max(max_id, num)
                    else:
                        # Try to extract any number
                        numbers = re.findall(r'\d+', existing_id_str)
                        if numbers:
                            # Take the largest number found
                            num = max(int(n) for n in numbers)
                            max_id = max(max_id, num)
                except:
                    continue
        
        # If no existing IDs, start from 1 (not 0)
        if max_id == 0:
            start_id = 1
        else:
            start_id = max_id + 1
        
        # Generate new sequential IDs
        poly_ids = []
        current_id = start_id
        
        # Generate IDs for each row (not just unique pairs)
        for idx in range(len(df)):
            poly_ids.append(str(current_id))  # FIXED: Always return strings
            current_id += 1
        
        if self.verbose:
            if existing_ids:
                logger.info(f"Generated poly_ids continuing from {max_id}, new IDs: {start_id} to {current_id - 1}")
            else:
                logger.info(f"Generated new poly_ids: {start_id} to {current_id - 1}")
        
        return poly_ids

    def expand_polymer_variants(self, df: pd.DataFrame, poly_types: List[str] = None, 
                              compositions: List[str] = None) -> pd.DataFrame:
        """
        Expand dataset to include different polymer types and compositions
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
                        new_row['poly_type'] = f"{poly_type} (homopolymer)"
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
        
        # Auto-exclude common non-target patterns
        if auto_exclude_patterns is None:
            auto_exclude_patterns = [
                # Structure columns
                'id', 'smiles', 'canonical', 'iupac', 'formula',
                # Polymer-specific columns  
                'mono', 'poly', 'frac', 'comp', 'stoich', 'connectivity',
                # ChemProp columns
                'chemprop', 'master', 'input',
                # Common metadata
                'source', 'reference', 'notes', 'comments', 'url', 'index',
                # PI1070 specific
                'monomer_ID', 'mol_weight', 'atomic_weight', 'temp', 'press',
                'tacticity', 'DP', 'n_mol', 'n_atom', 'Mn', 'polymer_class',
                # Additional ID patterns
                'polymer_ID', 'polymer_id', 'sample_id', 'sample_ID',
                'batch', 'batch_id', 'experiment_id', 'run_id',
                'name', 'polymer_name', 'sample_name', 'material_name',
                # PI-specific patterns
                'PI', 'pi_id', 'pi_code', 'material_code'
            ]
        
        # Build comprehensive exclusion list
        comprehensive_exclude = set(exclude_columns)
        
        # Add columns that match patterns (case-insensitive)
        for col in df.columns:
            col_lower = col.lower()
            for pattern in auto_exclude_patterns:
                if pattern.lower() in col_lower:
                    comprehensive_exclude.add(col)
                    break
        
        # Find numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        
        # Remove excluded columns
        potential_targets = [col for col in numeric_columns if col not in comprehensive_exclude]
        
        # Log what was excluded for transparency
        if self.verbose:
            excluded_found = [col for col in df.columns if col in comprehensive_exclude]
            if excluded_found and len(excluded_found) < 20:  # Don't print too many
                logger.info(f"Auto-excluded columns: {excluded_found[:20]}...")
        
        return potential_targets

    def interactive_column_selection(self, df: pd.DataFrame, interactive: bool = True) -> Tuple[List[str], Dict[str, str]]:
        """
        Interactively select target columns and their new names
        """
        if not interactive:
            # Auto-detect mode
            potential_targets = self.detect_target_columns(df)
            if not potential_targets:
                # If no numeric columns found, return empty lists
                return [], {}
            
            # Create default mapping (keep original names)
            column_mapping = {col: col for col in potential_targets}
            return potential_targets, column_mapping
        
        # Interactive mode
        print("\nAvailable columns in your data:")
        print("-" * 50)
        all_columns = df.columns.tolist()
        potential_targets = self.detect_target_columns(df)
        
        # Limit display if too many columns
        if len(all_columns) > 50:
            print(f"Note: Dataset has {len(all_columns)} columns. Showing first 50...")
            display_columns = all_columns[:50]
        else:
            display_columns = all_columns
        
        for i, col in enumerate(display_columns):
            col_type = str(df[col].dtype)
            is_potential = col in potential_targets
            marker = "→ " if is_potential else "  "
            print(f"{marker}{i+1:2d}. {col:<25} ({col_type})")
        
        if len(all_columns) > 50:
            print(f"... and {len(all_columns) - 50} more columns")
        
        print(f"\nColumns marked with → are automatically detected as potential targets")
        print(f"Auto-detected {len(potential_targets)} target columns")
        
        # Ask user to select columns
        print("\nHow do you want to select target columns?")
        print("1. Use all auto-detected columns")
        print("2. Select specific columns by number")
        print("3. Enter column names manually")
        print("4. Skip target column selection (no properties)")
        
        while True:
            try:
                choice = input("\nEnter your choice (1, 2, 3, or 4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    break
                print("Please enter 1, 2, 3, or 4")
            except KeyboardInterrupt:
                raise
            except:
                print("Please enter 1, 2, 3, or 4")
        
        if choice == '4':
            return [], {}
        
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
        
        if not selected_columns:
            return [], {}
        
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
    # Main Processing Methods
    # ========================
    
    def _process_existing_poly_chemprop_dataset(self, new_df: pd.DataFrame, 
                                          target_columns: List[str] = None,
                                          column_mapping: Dict[str, str] = None,
                                          exclude_columns: List[str] = None,
                                          clean_poly_inputs: bool = True,
                                          fix_existing_unknowns: bool = True) -> pd.DataFrame:
        """
        Enhanced processing of datasets that already contain poly_chemprop_input
        Now with automatic repair of missing structural information
        """
        # Clean poly_chemprop_input data if requested
        if clean_poly_inputs and 'poly_chemprop_input' in new_df.columns:
            if self.verbose:
                logger.info("Cleaning poly_chemprop_input data (removing trailing property values)...")
            
            # Count corrupted entries before cleaning
            corrupted_count = new_df['poly_chemprop_input'].astype(str).str.contains('~', na=False).sum()
            if corrupted_count > 0 and self.verbose:
                logger.info(f"Found {corrupted_count} entries with trailing property values to clean")
            
            # Clean the data
            new_df['poly_chemprop_input'] = new_df['poly_chemprop_input'].apply(
                lambda x: self.clean_poly_chemprop_input(x, remove_trailing_values=True)
            )
            
            # Remove rows where cleaning failed
            initial_count = len(new_df)
            new_df = new_df[new_df['poly_chemprop_input'].notna()]
            if len(new_df) != initial_count and self.verbose:
                logger.warning(f"Removed {initial_count - len(new_df)} rows due to poly_chemprop_input cleaning failures")
        
        # Auto-repair if enabled
        if self.auto_repair:
            new_df = self._detect_and_repair_dataset(new_df, "input")
        
        # Handle target column selection
        if target_columns is None or column_mapping is None:
            target_columns, column_mapping = self.interactive_column_selection(new_df, interactive=True)
        
        # Rename target columns according to mapping
        for old_name, new_name in column_mapping.items():
            if old_name != new_name and old_name in new_df.columns:
                new_df.rename(columns={old_name: new_name}, inplace=True)
        
        # Update target_columns to use new names
        final_target_columns = [column_mapping[col] for col in target_columns]
        
        # Fix unknown values if requested
        if fix_existing_unknowns and self.fix_unknowns:
            new_df = self._fix_unknown_values(new_df)
        
        if self.verbose:
            logger.info(f"Successfully processed poly_chemprop_input dataset with {len(new_df)} rows")
            if final_target_columns:
                logger.info(f"Target columns preserved: {final_target_columns}")
        
        return new_df
    
    def process_new_dataset(self, input_path: str = None, df: pd.DataFrame = None,
                      expand_variants: bool = True, generate_iupac: bool = True,
                      interactive: bool = True, target_columns: List[str] = None,
                      column_mapping: Dict[str, str] = None, 
                      poly_types: List[str] = None, compositions: List[str] = None,
                      exclude_columns: List[str] = None, clean_poly_inputs: bool = True,
                      fix_existing_unknowns: bool = True, repair_missing: bool = None):
        """
        Enhanced process new dataset with automatic dataset repair
        """
        # Store for later use
        self._exclude_columns = exclude_columns if exclude_columns else []
        
        # Use global auto_repair if repair_missing not specified
        if repair_missing is None:
            repair_missing = self.auto_repair
        
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
        
        # Auto-detect and repair if enabled
        if repair_missing:
            dataset_type = self._detect_dataset_type(new_df)
            if self.verbose:
                logger.info(f"Dataset type detected: {dataset_type.value}")
            
            # Repair if needed
            if dataset_type == DatasetType.TYPE_B:
                new_df = self._repair_type_b_dataset(new_df)
        
        # Check if dataset already has poly_chemprop_input
        has_poly_chemprop = 'poly_chemprop_input' in new_df.columns
        has_monomers = any(col in new_df.columns for col in ['monoA', 'smiles', 'MonA', 'SMILES', 'Smiles'])
        
        if has_poly_chemprop and not has_monomers:
            if self.verbose:
                logger.info("Dataset contains poly_chemprop_input - processing as pre-processed polymer data")
            return self._process_existing_poly_chemprop_dataset(
                new_df, target_columns, column_mapping, exclude_columns, 
                clean_poly_inputs=clean_poly_inputs, fix_existing_unknowns=fix_existing_unknowns
            )
        
        # Standardize column names  
        # FIXED: Handle inconsistent column name handling
        column_mapping_standard = {
            'smiles': 'monoA',
            'MonA': 'monoA',
            'MonB': 'monoB',
            'SMILES': 'monoA',
            'Smiles': 'monoA',
            'monA': 'monoA',  # Added
            'monB': 'monoB'   # Added
        }
        
        for old_name, new_name in column_mapping_standard.items():
            if old_name in new_df.columns and new_name not in new_df.columns:
                new_df.rename(columns={old_name: new_name}, inplace=True)
        
        # Validate required columns
        if 'monoA' not in new_df.columns:
            raise ValueError("No 'monoA', 'smiles', 'MonA', or 'poly_chemprop_input' column found!")
        
        # CANONICALIZE monoA SMILES
        if self.verbose:
            logger.info("Canonicalizing monoA SMILES...")
        
        # Temporarily suppress RDKit errors during bulk canonicalization
        self.set_rdkit_verbosity(False)
        
        # Keep track of conversion count for summary
        conversion_count = 0
        processed_count = 0
        
        canonicalized = new_df['monoA'].apply(
            lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else x
        )
        
        # Count bare asterisk conversions
        if self.verbose:
            for orig, canon in zip(new_df['monoA'], canonicalized):
                if pd.notna(orig) and pd.notna(canon):
                    processed_count += 1
                    if '*' in str(orig) and '[*:' not in str(orig) and '[*:' in str(canon):
                        conversion_count += 1
            
            if conversion_count > 0:
                logger.info(f"Converted {conversion_count} monomers with bare asterisk (*) attachment points")
        
        new_df['monoA'] = canonicalized
        
        # Re-enable if verbose mode
        if self.verbose:
            self.set_rdkit_verbosity(True)
        
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
            
            # Temporarily suppress RDKit errors during bulk canonicalization
            self.set_rdkit_verbosity(False)
            
            new_df['monoB'] = new_df['monoB'].apply(
                lambda x: self.canonicalize_smiles(str(x)) if pd.notna(x) else x
            )
            
            # Re-enable if verbose mode
            if self.verbose:
                self.set_rdkit_verbosity(True)
        
        # Handle target column selection
        if target_columns is None or column_mapping is None:
            target_columns, column_mapping = self.interactive_column_selection(new_df, interactive=interactive)
        
        # Rename target columns according to mapping
        for old_name, new_name in column_mapping.items():
            if old_name != new_name and old_name in new_df.columns:
                new_df.rename(columns={old_name: new_name}, inplace=True)
        
        # Update target_columns to use new names
        final_target_columns = [column_mapping[col] for col in target_columns]
        
        # Expand polymer variants if requested
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
        if 'poly_chemprop_input' not in new_df.columns:
            if self.verbose:
                logger.info("Generating ChemProp inputs...")
            
            # Suppress RDKit errors during ChemProp generation
            self.set_rdkit_verbosity(False)
            
            new_df['master_chemprop_input'] = [
                self.make_master_chemprop_input(sA, sB) 
                for sA, sB in zip(new_df['monoA'], new_df['monoB'])
            ]
            
            # Re-enable if verbose
            if self.verbose:
                self.set_rdkit_verbosity(True)
                
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
        
        # Fix unknown values if any were generated
        if fix_existing_unknowns and self.fix_unknowns:
            new_df = self._fix_unknown_values(new_df)
        
        if self.verbose:
            logger.info(f"Successfully processed dataset with {len(new_df)} rows")
            if final_target_columns:
                logger.info(f"Target columns preserved: {final_target_columns}")
        
        return new_df

    def append_to_template(self, new_df: pd.DataFrame, output_path: str = None, 
                          exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Enhanced append to template with better column management
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # FIXED: Corrected the indentation logic
        if self.template_df is None:
            combined_df = new_df.copy()
        else:
            # If template exists and has poly_ids, ensure new_df continues from the highest ID
            if 'poly_id' in self.template_df.columns and 'poly_id' in new_df.columns:
                # Get existing IDs from template
                existing_ids = set(self.template_df['poly_id'].astype(str).unique())
                
                # Regenerate IDs for new_df to continue from template's highest ID
                new_df['poly_id'] = self.generate_poly_ids(new_df, existing_ids)
            
            # Align columns - preserves ALL columns from both datasets
            all_columns = list(set(self.template_df.columns) | set(new_df.columns))
            
            # Add missing columns to both dataframes
            for col in all_columns:
                if col not in self.template_df.columns:
                    self.template_df[col] = None
                if col not in new_df.columns:
                    new_df[col] = None
            
            # Order columns nicely
            combined_df = pd.concat([self.template_df, new_df], ignore_index=True)
            combined_df = self._order_columns(combined_df)
        
        # Remove unwanted columns from final output
        if exclude_columns:
            columns_to_remove = [col for col in exclude_columns if col in combined_df.columns]
            if columns_to_remove:
                combined_df = combined_df.drop(columns=columns_to_remove)
                if self.verbose:
                    logger.info(f"Removed unwanted columns: {columns_to_remove}")
        
        # Fix any unknown values in the combined dataset
        if self.fix_unknowns:
            combined_df = self._fix_unknown_values(combined_df)
        
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
                     poly_types: List[str] = None, compositions: List[str] = None,
                     fix_unknowns: bool = True) -> pd.DataFrame:
        """
        Quick processing method that combines process_new_dataset and append_to_template
        """
        processed_df = self.process_new_dataset(
            input_path=input_path,
            expand_variants=expand_variants,
            interactive=interactive,
            poly_types=poly_types,
            compositions=compositions,
            fix_existing_unknowns=fix_unknowns
        )
        
        combined_df = self.append_to_template(processed_df, output_path)
        return combined_df

    def cleanup_existing_database(self, input_path: str, output_path: str, 
                                fix_unknowns: bool = True, repair_missing: bool = True) -> pd.DataFrame:
        """
        Enhanced cleanup of existing database with unknown value fixing and column repair
        """
        if self.verbose:
            logger.info(f"Cleaning up existing database: {input_path}")
        
        df = pd.read_csv(input_path)
        original_count = len(df)
        
        # Auto-repair missing columns if enabled
        if repair_missing:
            df = self._detect_and_repair_dataset(df, input_path)
        
        # Clean poly_chemprop_input if present
        if 'poly_chemprop_input' in df.columns:
            if self.verbose:
                logger.info("Cleaning poly_chemprop_input values...")
            df['poly_chemprop_input'] = df['poly_chemprop_input'].apply(
                lambda x: self.clean_poly_chemprop_input(x, remove_trailing_values=True)
            )
        
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
        
        # Fix unknown values
        if fix_unknowns and self.fix_unknowns:
            df = self._fix_unknown_values(df)
        
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

    def update_existing_entries(self, database_path: str, update_unknowns: bool = True,
                              update_iupac: bool = True, update_poly_inputs: bool = False,
                              repair_missing: bool = True) -> pd.DataFrame:
        """
        Update existing database entries without adding new ones
        Now with option to repair missing columns
        """
        if self.verbose:
            logger.info(f"Updating existing entries in: {database_path}")
        
        df = pd.read_csv(database_path)
        updates_made = 0
        
        # Repair missing columns if enabled
        if repair_missing:
            df = self._detect_and_repair_dataset(df, database_path)
        
        # Update unknown polymer types and compositions
        if update_unknowns:
            initial_unknowns = 0
            if 'poly_type' in df.columns:
                initial_unknowns += (df['poly_type'] == 'unknown').sum()
            if 'comp' in df.columns:
                initial_unknowns += (df['comp'] == 'unknown').sum()
            
            df = self._fix_unknown_values(df)
            
            final_unknowns = 0
            if 'poly_type' in df.columns:
                final_unknowns += (df['poly_type'] == 'unknown').sum()
            if 'comp' in df.columns:
                final_unknowns += (df['comp'] == 'unknown').sum()
            
            updates_made += initial_unknowns - final_unknowns
        
        # Update IUPAC names
        if update_iupac:
            for col in ['monoA_IUPAC', 'monoB_IUPAC']:
                if col in df.columns:
                    invalid_count = df[col].isin(['Invalid_SMILES', 'Unknown_compound']).sum()
                    if invalid_count > 0:
                        smiles_col = col.replace('_IUPAC', '')
                        if smiles_col in df.columns:
                            df[col] = df[smiles_col].apply(self.get_iupac_name)
                            new_invalid = df[col].isin(['Invalid_SMILES', 'Unknown_compound']).sum()
                            updates_made += invalid_count - new_invalid
        
        # Update poly_chemprop_inputs if requested (careful - can change results)
        if update_poly_inputs and all(col in df.columns for col in ['monoA', 'monoB', 'poly_type', 'fracA']):
            if self.verbose:
                logger.warning("Updating poly_chemprop_inputs - this may change model inputs!")
            
            df['poly_chemprop_input'] = [
                self.make_poly_chemprop_input(sA, sB, t, fA, selfedges=True)
                for sA, sB, t, fA in zip(
                    df['monoA'], df['monoB'], 
                    df['poly_type'], df['fracA']
                )
            ]
            updates_made += len(df)
        
        # Save updated database
        df.to_csv(database_path, index=False)
        
        if self.verbose:
            logger.info(f"Updated {updates_made} entries in the database")
            logger.info(f"Database saved to: {database_path}")
        
        return df

    def test_bare_asterisk_conversion(self, test_smiles: List[str] = None):
        """
        Test conversion of bare asterisk attachment points
        
        Args:
            test_smiles: List of SMILES to test, or uses default examples
        """
        if test_smiles is None:
            test_smiles = [
                "*C(C*)C(c1ccccc1)C",
                "*C1CC(CC1)C*",
                "*C(C*)CCCCCC",
                "C(C)(C)c1ccc(*)cc1*"
            ]
        
        print("Testing bare asterisk conversion:")
        print("-" * 60)
        
        for smiles in test_smiles:
            canonical = self.canonicalize_smiles(smiles, verbose_conversion=True)
            print(f"Original:  {smiles}")
            print(f"Converted: {canonical}")
            
            if canonical:
                # Test if it can generate poly_chemprop_input
                poly_input = self.make_poly_chemprop_input(
                    canonical, canonical, 'alternating', 0.5
                )
                if poly_input:
                    print(f"✓ Valid poly_chemprop_input generated")
                else:
                    print(f"✗ Failed to generate poly_chemprop_input")
            else:
                print(f"✗ Failed to canonicalize")
            print()

# ========================
# Additional Utility Functions
# ========================

def test_pi1070_format():
    """Quick test for PI1070 monomer format with bare asterisks"""
    test_monomers = [
        "*C(C*)C(c1ccccc1)C",
        "*C1CC(CC1)C*",
        "*C(C*)CCCCCC"
    ]
    
    print("Testing PI1070 monomer format conversion:")
    print("=" * 70)
    
    manager = PolymerDatabaseManager(verbose=False)
    
    for monomer in test_monomers:
        print(f"\nOriginal: {monomer}")
        
        # Test canonicalization
        canonical = manager.canonicalize_smiles(monomer, verbose_conversion=True)
        print(f"Canonical: {canonical}")
        
        if canonical:
            # Test poly_chemprop_input generation
            poly_input = manager.make_poly_chemprop_input(
                canonical, canonical, 'alternating', 0.5
            )
            
            if poly_input:
                print(f"✓ Success! Poly_chemprop_input: {poly_input[:80]}...")
            else:
                print("✗ Failed to generate poly_chemprop_input")
        else:
            print("✗ Failed to canonicalize")
    
    return manager

def debug_pi1070_issue():
    """Debug why PI1070 monomers are failing"""
    import pandas as pd
    
    # Test monomers from PI1070
    test_monomers = [
        "*C(C*)C(c1ccccc1)C",
        "*C1CC(CC1)C*", 
        "*C(C*)CCCCCC",
        "*C(C*)CCCCCCCCC",
        "*C(C*)C(CC)C"
    ]
    
    print("Debugging PI1070 Monomer Processing:")
    print("=" * 80)
    
    # Set up manager with verbose logging
    # Temporarily change log level to DEBUG
    import logging
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.DEBUG)
    
    manager = PolymerDatabaseManager(verbose=True)
    
    for i, monomer in enumerate(test_monomers):
        print(f"\n{i+1}. Testing monomer: {monomer}")
        print("-" * 60)
        
        # Step 1: Canonicalize
        canonical = manager.canonicalize_smiles(monomer, verbose_conversion=True)
        print(f"   Canonicalized: {canonical}")
        
        if not canonical:
            print("   ✗ Canonicalization failed!")
            continue
            
        # Step 2: Check attachment points
        import re
        attach_points = re.findall(r'\[\*:(\d+)\]', canonical)
        print(f"   Attachment points: {attach_points}")
        
        # Step 3: Prepare homopolymer
        mona, monb = manager.prepare_homopolymer(canonical)
        print(f"   MonoA: {mona}")
        print(f"   MonoB: {monb}")
        
        # Step 4: Check validity
        is_valid, msg = manager.check_polymer_validity(mona, monb)
        print(f"   Validity check: {is_valid} - {msg}")
        
        # Step 5: Try to make poly_chemprop_input
        for poly_type in ['alternating']:
            poly_input = manager.make_poly_chemprop_input(
                canonical, canonical, poly_type, 0.5
            )
            
            if poly_input:
                print(f"   ✓ {poly_type}: {poly_input[:100]}...")
            else:
                print(f"   ✗ {poly_type}: Failed to generate")
    
    # Restore original log level            
    logging.getLogger().setLevel(original_level)
    
    return manager

def cleanup_database(input_path: str, output_path: str, verbose: bool = True,
                    fix_unknowns: bool = True, repair_missing: bool = True) -> pd.DataFrame:
    """
    Standalone function to clean up an existing polymer database
    
    Args:
        input_path: Path to existing database CSV
        output_path: Path to save cleaned database
        verbose: Whether to show detailed output
        fix_unknowns: Whether to fix unknown values
        repair_missing: Whether to repair missing columns
        
    Returns:
        Cleaned DataFrame
    """
    manager = PolymerDatabaseManager(verbose=verbose, fix_unknowns=fix_unknowns, auto_repair=repair_missing)
    return manager.cleanup_existing_database(input_path, output_path, fix_unknowns=fix_unknowns, repair_missing=repair_missing)

def fix_database_consistency(database_path: str, backup: bool = True) -> str:
    """
    Fix consistency issues in an existing database (in-place with backup)
    
    Args:
        database_path: Path to database to fix
        backup: Whether to create backup before fixing
        
    Returns:
        Path to the fixed database
    """
    from datetime import datetime
    
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = database_path.replace('.csv', f'_backup_{timestamp}.csv')
        shutil.copy2(database_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Clean the database
    manager = PolymerDatabaseManager(verbose=True, fix_unknowns=True, auto_repair=True)
    cleaned_df = manager.cleanup_existing_database(database_path, database_path, fix_unknowns=True, repair_missing=True)
    
    print(f"Fixed database saved to: {database_path}")
    return database_path

def merge_databases(database_paths: List[str], output_path: str, 
                   remove_duplicates: bool = True, repair_missing: bool = True,
                   merge_strategy: str = "first",
                   columns_to_average: List[str] = None,
                   columns_to_keep_first: List[str] = None) -> pd.DataFrame:
    """
    Merge multiple polymer databases into one
    
    Args:
        database_paths: List of paths to databases to merge
        output_path: Path to save merged database
        remove_duplicates: Whether to remove duplicate entries
        repair_missing: Whether to repair missing columns before merging
        merge_strategy: How to handle duplicates (first/last/mean/max/min)
        columns_to_average: Columns that should be averaged during merge
        columns_to_keep_first: Columns that should keep first value during merge
        
    Returns:
        Merged DataFrame
    """
    manager = PolymerDatabaseManager(verbose=True, fix_unknowns=True, auto_repair=repair_missing)
    
    # Apply custom merge behavior if specified
    if columns_to_average or columns_to_keep_first:
        manager.customize_merge_behavior(
            columns_to_average=columns_to_average,
            columns_to_keep_first=columns_to_keep_first
        )
    
    return manager.smart_merge_datasets(
        database_paths, 
        output_path, 
        merge_strategy=merge_strategy,
        repair_missing=repair_missing,
        remove_duplicates=remove_duplicates
    )

def smart_merge_polymer_datasets(dataset_paths: List[str], output_path: str,
                               merge_strategy: str = "first") -> pd.DataFrame:
    """
    Smart merge that handles different dataset types automatically
    
    Args:
        dataset_paths: List of dataset paths (can be Type A, B, or C)
        output_path: Path to save merged result
        merge_strategy: How to merge duplicate properties
        
    Returns:
        Merged DataFrame
    """
    manager = PolymerDatabaseManager(verbose=True, fix_unknowns=True, auto_repair=True)
    return manager.smart_merge_datasets(
        dataset_paths,
        output_path,
        merge_strategy=merge_strategy,
        repair_missing=True,
        expand_monomers=True,
        remove_duplicates=True,
        fix_unknowns=True
    )

# ========================
# Convenience Functions
# ========================

def create_database_manager(template_path: str = None, clean_template: bool = True,
                          fix_unknowns: bool = True, auto_repair: bool = True) -> PolymerDatabaseManager:
    """Create a new database manager instance"""
    return PolymerDatabaseManager(template_path, clean_template=clean_template, 
                                fix_unknowns=fix_unknowns, auto_repair=auto_repair)

def quick_process_dataset(input_path: str, output_path: str, template_path: str = None,
                         expand_variants: bool = True, interactive: bool = True,
                         poly_types: List[str] = None, compositions: List[str] = None,
                         clean_template: bool = True, fix_unknowns: bool = True,
                         repair_missing: bool = True) -> pd.DataFrame:
    """
    Quick function to process a dataset with minimal setup
    """
    manager = PolymerDatabaseManager(template_path, clean_template=clean_template, 
                                   fix_unknowns=fix_unknowns, auto_repair=repair_missing)
    return manager.quick_process(input_path, output_path, expand_variants, interactive, 
                               poly_types, compositions, fix_unknowns)

# ========================
# Configuration Updates
# ========================

def update_default_polymer_configs(manager: PolymerDatabaseManager, 
                                 poly_types: List[str] = None,
                                 compositions: List[str] = None,
                                 comp_fracs: Dict[str, Tuple[float, float]] = None):
    """
    Update default polymer configurations at runtime
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
    Enhanced command line interface for the Universal Polymer Database Manager
    """
    import argparse
    
    # First, check if this is just a simple info command that doesn't need full parsing
    if len(sys.argv) == 2 and sys.argv[1] in ['--test-bare-asterisk', '--debug-pi1070']:
        if sys.argv[1] == '--test-bare-asterisk':
            test_pi1070_format()
            return 0
        elif sys.argv[1] == '--debug-pi1070':
            debug_pi1070_issue()
            return 0
    
    parser = argparse.ArgumentParser(
        description='Universal Polymer Database Manager v3.0 - Process ANY polymer dataset format with automatic repair',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart merge different dataset types (NEW!)
  !python polymer_database_manager.py --smart-merge dataset1.csv dataset2.csv dataset3.csv -o merged.csv
  
  # Smart merge with custom column handling (NEW!)
  !python polymer_database_manager.py --smart-merge data1.csv data2.csv -o merged.csv \
      --merge-strategy mean \
      --columns-to-average temp press \
      --columns-to-keep-first density viscosity
  
  # Process with automatic repair (NEW!)
  !python polymer_database_manager.py -i data.csv -o output.csv --repair-missing
  
  # Basic usage - interactive column selection
  !python polymer_database_manager.py -i input.csv -o output.csv
  
  # Process dataset with existing poly_chemprop_input
  !python polymer_database_manager.py -i poly_dataset.csv -o output.csv -t template.csv
  
  # With existing template
  !python polymer_database_manager.py -i new_data.csv -o updated_db.csv -t existing_template.csv
  
  # Non-interactive with output directory
  !python polymer_database_manager.py -i data.csv -o results/ --non-interactive
  
  # Clean up existing database with repair
  !python polymer_database_manager.py --cleanup input_db.csv -o cleaned_db.csv --repair-missing
  
  # Fix unknown values in existing database
  !python polymer_database_manager.py --fix-unknowns database.csv
  
  # Merge multiple databases with smart merge
  !python polymer_database_manager.py --merge db1.csv db2.csv db3.csv -o merged.csv --merge-strategy mean
  
  # Custom polymer types and compositions
  !python polymer_database_manager.py -i data.csv -o output.csv --poly-types alternating block --compositions 4A_4B 6A_2B
  
  # Specify target columns and their new names
  !python polymer_database_manager.py -i data.csv -o output.csv --target-columns band_gap conductivity --target-names Band_Gap_eV Conductivity_S_cm
  
  # No polymer variant expansion (keep as-is)
  !python polymer_database_manager.py -i data.csv -o output.csv --no-expand
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
    
    # Special modes
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up existing database for consistency (use with -i for input database)')
    parser.add_argument('--fix-unknowns', action='store_true',
                       help='Fix unknown values in existing database (use with -i)')
    parser.add_argument('--merge', nargs='+', metavar='DB',
                       help='Merge multiple databases into one (legacy mode)')
    parser.add_argument('--smart-merge', nargs='+', metavar='DB',
                       help='Smart merge that handles different dataset types (NEW!)')
    
    # Processing options
    parser.add_argument('--no-expand', action='store_true',
                       help='Do not expand polymer variants (keep original data structure)')
    parser.add_argument('--no-iupac', action='store_true',
                       help='Do not generate IUPAC names')
    parser.add_argument('--no-clean-poly', action='store_true',
                       help='Do not clean poly_chemprop_input in input data and template')
    parser.add_argument('--no-fix-unknowns', action='store_true',
                       help='Do not attempt to fix unknown values')
    parser.add_argument('--repair-missing', action='store_true', default=True,
                       help='Automatically repair missing columns (default: True)')
    parser.add_argument('--no-repair', action='store_false', dest='repair_missing',
                       help='Disable automatic column repair')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Use non-interactive mode (auto-detect all numeric columns)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disable verbose output')
    
    # Merge options
    parser.add_argument('--merge-strategy', choices=['first', 'last', 'mean', 'max', 'min'],
                       default='first',
                       help='Strategy for merging duplicate properties (default: first)')
    parser.add_argument('--columns-to-average', nargs='+',
                       help='Columns that should be averaged during merge (removes from identifier list)')
    parser.add_argument('--columns-to-keep-first', nargs='+',
                       help='Columns that should keep first value during merge (adds to identifier list)')
    
    # Target column specification
    parser.add_argument('--target-columns', nargs='+',
                       help='Specific target column names to use')
    parser.add_argument('--target-names', nargs='+',
                       help='New names for target columns (must match --target-columns length)')
    
    # Column management
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
    parser.add_argument('--check-unknowns', action='store_true',
                       help='Check for unknown values in the database and report statistics')
    parser.add_argument('--detect-type', action='store_true',
                       help='Detect dataset type and report what repairs would be made')
    parser.add_argument('--test-bare-asterisk', action='store_true',
                       help='Test conversion of bare asterisk (*) attachment points')
    parser.add_argument('--debug-pi1070', action='store_true',
                       help='Debug PI1070 monomer processing issues')
    parser.add_argument('--version', action='version', version=f'%(prog)s 3.0.0')
    
    args = parser.parse_args()
    
    # Handle test bare asterisk mode
    if hasattr(args, 'test_bare_asterisk') and args.test_bare_asterisk:
        test_pi1070_format()
        return 0
    
    # Handle debug PI1070 mode
    if hasattr(args, 'debug_pi1070') and args.debug_pi1070:
        debug_pi1070_issue()
        return 0
    
    # Check if output is required for remaining operations
    # Info modes and in-place operations don't need output
    info_modes = ['list_columns', 'check_unknowns', 'detect_type', 'fix_unknowns', 
                  'test_bare_asterisk', 'debug_pi1070']
    if not args.output and not any(getattr(args, mode, False) for mode in info_modes):
        # For cleanup without output, it will default to input
        if not args.cleanup:
            print("Error: --output is required for this operation")
            parser.print_help()
            return 1
    
    # Handle smart merge mode (NEW!)
    if args.smart_merge:
        if not args.output:
            print("Error: --output is required for smart merge mode")
            return 1
        
        try:
            manager = PolymerDatabaseManager(verbose=not args.quiet, fix_unknowns=not args.no_fix_unknowns, 
                                           auto_repair=args.repair_missing)
            
            # Apply custom merge behavior if specified
            if args.columns_to_average or args.columns_to_keep_first:
                manager.customize_merge_behavior(
                    columns_to_average=args.columns_to_average,
                    columns_to_keep_first=args.columns_to_keep_first
                )
            
            merged_df = manager.smart_merge_datasets(
                args.smart_merge,
                args.output,
                merge_strategy=args.merge_strategy,
                repair_missing=args.repair_missing,
                expand_monomers=not args.no_expand,
                remove_duplicates=True,
                fix_unknowns=not args.no_fix_unknowns
            )
            print(f"✓ Smart merge completed successfully!")
            print(f"✓ Merged database saved to: {args.output}")
            return 0
        except TypeError as e:
            if "Could not convert string" in str(e) and "to numeric" in str(e):
                print(f"\nError: Cannot use '{args.merge_strategy}' strategy with non-numeric columns.")
                print("The dataset contains ID or text columns that cannot be averaged.")
                print("\nSuggestions:")
                print("1. Use --merge-strategy first (or last) instead of mean/max/min")
                print("2. Or manually exclude non-numeric columns before merging")
                print("\nRerun with: --merge-strategy first")
            else:
                print(f"Error during smart merge: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        except Exception as e:
            print(f"Error during smart merge: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    # Handle legacy merge mode
    if args.merge:
        if not args.output:
            print("Error: --output is required for merge mode")
            return 1
        
        try:
            merged_df = merge_databases(
                args.merge, 
                args.output, 
                remove_duplicates=True,
                repair_missing=args.repair_missing,
                merge_strategy=args.merge_strategy,
                columns_to_average=args.columns_to_average,
                columns_to_keep_first=args.columns_to_keep_first
            )
            print(f"✓ Databases merged successfully!")
            print(f"✓ Merged database saved to: {args.output}")
            return 0
        except Exception as e:
            print(f"Error during merge: {e}")
            return 1
    
    # Handle cleanup mode
    if args.cleanup:
        if not args.input:
            print("Error: --cleanup requires --input to specify the database to clean")
            return 1
        
        # Default output to input if not specified (in-place cleanup)
        output_path = args.output if args.output else args.input
        
        try:
            cleaned_df = cleanup_database(args.input, output_path, 
                                        verbose=not args.quiet, 
                                        fix_unknowns=not args.no_fix_unknowns,
                                        repair_missing=args.repair_missing)
            print(f"✓ Database cleanup completed!")
            print(f"✓ Cleaned database saved to: {output_path}")
            return 0
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return 1
    
    # Handle fix-unknowns mode
    if args.fix_unknowns:
        if not args.input:
            print("Error: --fix-unknowns requires --input to specify the database")
            return 1
        
        try:
            manager = PolymerDatabaseManager(verbose=not args.quiet, fix_unknowns=True, 
                                           auto_repair=args.repair_missing)
            df = manager.update_existing_entries(args.input, update_unknowns=True, 
                                               update_iupac=True, update_poly_inputs=False,
                                               repair_missing=args.repair_missing)
            print(f"✓ Unknown values fixed!")
            print(f"✓ Updated database saved to: {args.input}")
            return 0
        except Exception as e:
            print(f"Error fixing unknowns: {e}")
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
    
    # Detect type mode (NEW!)
    if args.detect_type:
        try:
            df = pd.read_csv(args.input)
            manager = PolymerDatabaseManager(verbose=False, auto_repair=False)
            dataset_type = manager._detect_dataset_type(df)
            
            print(f"Dataset Type Analysis for: {args.input}")
            print("=" * 60)
            print(f"Detected type: {dataset_type.value}")
            print(f"Shape: {df.shape}")
            
            # Report what would be done
            if dataset_type == DatasetType.TYPE_A:
                print("\nThis dataset has poly_chemprop_input but missing structural info.")
                print("With --repair-missing, the following would be extracted:")
                print("  - monoA and monoB from poly_chemprop_input")
                print("  - poly_type from connectivity patterns")
                print("  - comp from stoichiometry")
                print("  - fracA and fracB from stoichiometry")
                print("  - poly_id (generated)")
                print("  - IUPAC names for monomers")
            elif dataset_type == DatasetType.TYPE_B:
                print("\nThis dataset has structural info but missing poly_chemprop_input.")
                print("With --repair-missing, the following would be generated:")
                print("  - poly_chemprop_input for all polymers")
                print("  - poly_id (if missing)")
                print("  - IUPAC names (if missing)")
            elif dataset_type == DatasetType.TYPE_C:
                print("\nThis dataset contains only monomers.")
                print("Will be expanded to polymers during processing.")
            elif dataset_type == DatasetType.TYPE_COMPLETE:
                print("\nThis dataset is complete - no repairs needed!")
            
            # Check for properties
            potential_targets = manager.detect_target_columns(df)
            if potential_targets:
                print(f"\nDetected {len(potential_targets)} potential target properties:")
                for prop in potential_targets[:10]:  # Show first 10
                    non_null = df[prop].notna().sum()
                    print(f"  - {prop}: {non_null}/{len(df)} non-null")
                if len(potential_targets) > 10:
                    print(f"  ... and {len(potential_targets) - 10} more")
            
            return 0
        except Exception as e:
            print(f"Error detecting type: {e}")
            return 1
    
    # Check unknowns mode
    if args.check_unknowns:
        try:
            df = pd.read_csv(args.input)
            unknown_stats = {}
            
            if 'poly_type' in df.columns:
                unknown_stats['poly_type'] = (df['poly_type'] == 'unknown').sum()
            if 'comp' in df.columns:
                unknown_stats['comp'] = (df['comp'] == 'unknown').sum()
            if 'monoA_IUPAC' in df.columns:
                unknown_stats['monoA_IUPAC'] = df['monoA_IUPAC'].isin(['Invalid_SMILES', 'Unknown_compound']).sum()
            if 'monoB_IUPAC' in df.columns:
                unknown_stats['monoB_IUPAC'] = df['monoB_IUPAC'].isin(['Invalid_SMILES', 'Unknown_compound']).sum()
            
            print(f"Unknown values in {args.input}:")
            print("-" * 50)
            total_unknowns = 0
            for col, count in unknown_stats.items():
                print(f"{col}: {count} unknown values ({count/len(df)*100:.1f}%)")
                total_unknowns += count
            print(f"\nTotal unknown values: {total_unknowns}")
            
            # Check missing columns
            manager = PolymerDatabaseManager(verbose=False, auto_repair=False)
            dataset_type = manager._detect_dataset_type(df)
            if dataset_type != DatasetType.TYPE_COMPLETE:
                print(f"\nDataset type: {dataset_type.value}")
                print("Use --repair-missing to fix missing columns")
            
            return 0
        except Exception as e:
            print(f"Error checking unknowns: {e}")
            return 1
    
    # List columns mode
    if args.list_columns:
        try:
            df = pd.read_csv(args.input)
            manager = PolymerDatabaseManager(verbose=False, auto_repair=False)
            exclude_patterns = args.exclude_patterns if hasattr(args, 'exclude_patterns') else None
            potential_targets = manager.detect_target_columns(
                df, 
                exclude_columns=args.exclude_columns,
                auto_exclude_patterns=exclude_patterns if not args.keep_all_columns else []
            )
            
            print(f"Columns in {args.input}:")
            print("-" * 50)
            
            # Limit display for very wide datasets
            max_display = 100
            display_cols = df.columns[:max_display]
            
            for i, col in enumerate(display_cols):
                col_type = str(df[col].dtype)
                is_target = " (potential target)" if col in potential_targets else ""
                non_null = df[col].notna().sum()
                null_pct = (1 - non_null/len(df)) * 100
                print(f"{i+1:3d}. {col:<30} ({col_type:<10}) [{non_null:>6}/{len(df):<6} non-null, {null_pct:>5.1f}% null]{is_target}")
            
            if len(df.columns) > max_display:
                print(f"\n... and {len(df.columns) - max_display} more columns")
            
            print(f"\nAuto-detected target columns: {len(potential_targets)}")
            if potential_targets:
                print("Target columns:", potential_targets[:20])
                if len(potential_targets) > 20:
                    print(f"... and {len(potential_targets) - 20} more")
            
            print(f"\nDataset shape: {df.shape}")
            
            # Detect type
            dataset_type = manager._detect_dataset_type(df)
            print(f"Dataset type: {dataset_type.value}")
            
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
            print(f"Initializing Universal Polymer Database Manager v3.0...")
            if args.template:
                print(f"Using template: {args.template}")
        
        manager = PolymerDatabaseManager(
            template_path=args.template, 
            verbose=args.verbose, 
            clean_template=not args.no_clean_poly,
            fix_unknowns=not args.no_fix_unknowns,
            auto_repair=args.repair_missing
        )
        
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
            exclude_columns=args.exclude_columns,
            clean_poly_inputs=not args.no_clean_poly,
            fix_existing_unknowns=not args.no_fix_unknowns,
            repair_missing=args.repair_missing,
            poly_types=args.poly_types if args.poly_types != ['alternating', 'block', 'random'] else None,
            compositions=args.compositions if args.compositions != ['4A_4B', '6A_2B', '2A_6B'] else None
        )
        
        # Use stored exclude_columns when appending
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
                print(f"✓ Target properties: {len(target_cols)} properties")
                if len(target_cols) <= 20:
                    print(f"  Properties: {target_cols}")
                else:
                    print(f"  Properties: {target_cols[:20]}...")
                    print(f"  ... and {len(target_cols) - 20} more")
            
            # Report on unknown values if any
            unknown_count = 0
            if 'poly_type' in combined_df.columns:
                unknown_count += (combined_df['poly_type'] == 'unknown').sum()
            if 'comp' in combined_df.columns:
                unknown_count += (combined_df['comp'] == 'unknown').sum()
            
            if unknown_count > 0:
                print(f"\n⚠ Note: {unknown_count} unknown values remain. Use --fix-unknowns to resolve.")
        
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
    print("Universal Polymer Database Manager v3.0 - Interactive Mode")
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
    
    fix_unknowns = input("Fix unknown values? (y/n, default: y): ").strip().lower()
    fix_unknowns = fix_unknowns != 'n'
    
    repair_missing = input("Repair missing columns? (y/n, default: y): ").strip().lower()
    repair_missing = repair_missing != 'n'
    
    interactive = input("Use interactive column selection? (y/n, default: y): ").strip().lower()
    interactive = interactive != 'n'
    
    # Initialize and process
    manager = PolymerDatabaseManager(template_path=template_file, verbose=True, 
                                   fix_unknowns=fix_unknowns, auto_repair=repair_missing)
    
    # Detect dataset type
    df = pd.read_csv(input_file)
    dataset_type = manager._detect_dataset_type(df)
    print(f"\nDetected dataset type: {dataset_type.value}")
    
    processed_df = manager.process_new_dataset(
        input_path=input_file,
        expand_variants=expand_variants,
        generate_iupac=generate_iupac,
        interactive=interactive,
        fix_existing_unknowns=fix_unknowns,
        repair_missing=repair_missing
    )
    
    combined_df = manager.append_to_template(processed_df, output_file)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Final database: {len(combined_df)} rows")
    
    # Report unknown values
    unknown_stats = {}
    if 'poly_type' in combined_df.columns:
        unknown_stats['poly_type'] = (combined_df['poly_type'] == 'unknown').sum()
    if 'comp' in combined_df.columns:
        unknown_stats['comp'] = (combined_df['comp'] == 'unknown').sum()
    
    if any(unknown_stats.values()):
        print("\nUnknown values remaining:")
        for col, count in unknown_stats.items():
            if count > 0:
                print(f"  {col}: {count} unknowns")
    
    return combined_df

def fix_existing_database(database_path: str, output_path: str = None):
    """
    Fix an existing database with truncated SMILES and unknown poly_types
    """
    if output_path is None:
        output_path = database_path.replace('.csv', '_fixed.csv')
    
    # Load database
    df = pd.read_csv(database_path)
    print(f"Loaded database with {len(df)} rows")
    
    # Count issues before fixing
    issues_before = {
        'invalid_smiles': 0,
        'unknown_poly_types': 0
    }
    
    if 'monoA_IUPAC' in df.columns:
        issues_before['invalid_smiles'] += (df['monoA_IUPAC'] == 'Invalid_SMILES').sum()
    if 'monoB_IUPAC' in df.columns:
        issues_before['invalid_smiles'] += (df['monoB_IUPAC'] == 'Invalid_SMILES').sum()
    if 'poly_type' in df.columns:
        issues_before['unknown_poly_types'] = df['poly_type'].isin(['unknown', 'Unknown']).sum()
    
    print(f"\nIssues before fixing:")
    print(f"  - Invalid SMILES: {issues_before['invalid_smiles']}")
    print(f"  - Unknown poly_types: {issues_before['unknown_poly_types']}")
    
    # Initialize manager and fix
    manager = PolymerDatabaseManager(verbose=True)
    fixed_df = manager.fix_dataset_issues(df)
    fixed_df = manager.post_merge_cleanup(fixed_df)
    
    # Count issues after fixing
    issues_after = {
        'invalid_smiles': 0,
        'unknown_poly_types': 0
    }
    
    if 'monoA_IUPAC' in fixed_df.columns:
        issues_after['invalid_smiles'] += (fixed_df['monoA_IUPAC'] == 'Invalid_SMILES').sum()
    if 'monoB_IUPAC' in fixed_df.columns:
        issues_after['invalid_smiles'] += (fixed_df['monoB_IUPAC'] == 'Invalid_SMILES').sum()
    if 'poly_type' in fixed_df.columns:
        issues_after['unknown_poly_types'] = fixed_df['poly_type'].isin(['unknown', 'Unknown']).sum()
    
    print(f"\nIssues after fixing:")
    print(f"  - Invalid SMILES: {issues_after['invalid_smiles']} (fixed {issues_before['invalid_smiles'] - issues_after['invalid_smiles']})")
    print(f"  - Unknown poly_types: {issues_after['unknown_poly_types']} (fixed {issues_before['unknown_poly_types'] - issues_after['unknown_poly_types']})")
    
    # Save fixed database
    fixed_df.to_csv(output_path, index=False)
    print(f"\nFixed database saved to: {output_path}")
    print(f"Final row count: {len(fixed_df)}")
    
    return fixed_df
    
# Version info
__version__ = "3.0.0"
__author__ = "Universal Polymer Database Toolkit"

if __name__ == "__main__":
    sys.exit(main())
