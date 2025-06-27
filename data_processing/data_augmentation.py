import os, sys
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
import numpy as np
import torch
from torch.utils.data import Dataset
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.data_utils import *
import pickle
from statistics import mean
import pandas as pd
import re
import random
import argparse

def get_property_columns(df, property_names=None):
    """
    Identify property columns in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        property_names: Optional list of property names to look for
    
    Returns:
        List of property column names found in the DataFrame
    """
    if property_names:
        # Use provided property names
        property_columns = []
        for prop_name in property_names:
            # Try exact match first
            if prop_name in df.columns:
                property_columns.append(prop_name)
            else:
                # Try to find columns containing the property name
                matching_cols = [col for col in df.columns if prop_name.lower() in col.lower()]
                if matching_cols:
                    property_columns.append(matching_cols[0])
                    print(f"Note: Using column '{matching_cols[0]}' for property '{prop_name}'")
                else:
                    print(f"Warning: Property '{prop_name}' not found in columns: {list(df.columns)}")
        return property_columns
    else:
        # Auto-detect property columns (exclude the SMILES input column)
        smiles_cols = ['poly_chemprop_input', 'smiles', 'input', 'polymer_smiles']
        property_columns = [col for col in df.columns if col not in smiles_cols]
        return property_columns

def main():
    parser = argparse.ArgumentParser(description='Augment polymer dataset with new monomer combinations')
    parser.add_argument("--input_file", type=str, default="dataset-poly_chemprop.csv",
                        help="Input CSV file name (relative to /data directory)")
    parser.add_argument("--output_suffix", type=str, default="augmented",
                        help="Suffix for output files (e.g., 'augmented' -> 'dataset-augmented-poly_chemprop.csv')")
    parser.add_argument("--property_names", type=str, nargs='+', default=None,
                        help="Names of property columns to include. If not specified, auto-detects all non-SMILES columns")
    parser.add_argument("--n_combinations", type=int, default=50,
                        help="Number of random B monomer combinations to generate for each B monomer")
    parser.add_argument("--smiles_column", type=str, default="poly_chemprop_input",
                        help="Name of the column containing polymer SMILES strings")
    
    args = parser.parse_args()
    
    # Load input dataset
    input_path = os.path.join(main_dir_path, 'data', args.input_file)
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with {len(df)} entries from {args.input_file}")
    
    # Validate SMILES column exists
    if args.smiles_column not in df.columns:
        print(f"Error: SMILES column '{args.smiles_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Identify property columns
    property_columns = get_property_columns(df, args.property_names)
    if not property_columns:
        print("Error: No property columns found")
        return
    
    print(f"Using property columns: {property_columns}")
    
    # Extract monomers and combinations
    monA_list = []
    monB_list = []
    stoichiometry_connectivity_combs = []
    
    for i in range(len(df)):
        poly_input = df.loc[i, args.smiles_column]
        
        # Parse polymer string
        try:
            monA, monB = poly_input.split("|")[0].split(".")
            stoichiometry_connectivity_combs.append("|" + "|".join(poly_input.split("|")[1:]))
            monA_list.append(monA)
            monB_list.append(monB)
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse polymer string at row {i}: {poly_input}")
            continue
    
    # Get unique monomers and combinations
    monAs = list(set(monA_list))
    monBs = list(set(monB_list))
    stoichiometry_connectivity_combs = list(set(stoichiometry_connectivity_combs))
    
    print(f"Found {len(monAs)} unique A monomers and {len(monBs)} unique B monomers")
    print(f"Found {len(stoichiometry_connectivity_combs)} unique stoichiometry/connectivity combinations")
    
    # Build augmented dataset
    n = args.n_combinations
    
    # Create copy of B monomers and change the wildcards to be [*:1] and [*:2] 
    monBs_mod = monBs.copy()
    rep = {"[*:3]": "[*:1]", "[*:4]": "[*:2]"}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    
    for i, m in enumerate(monBs_mod): 
        monBs_mod[i] = pattern.sub(lambda x: rep[re.escape(x.group(0))], m)
    
    # Generate new polymer combinations
    new_entries = []
    total_combinations = 0
    
    for b1 in monBs_mod:
        for nn in range(n):
            b2 = random.choice(monB_list)
            new_mon_comb = ".".join([b1, b2])
            
            for stoich_con in stoichiometry_connectivity_combs:
                new_poly = new_mon_comb + stoich_con
                
                # Create new entry with NaN for all property columns
                new_entry = {args.smiles_column: new_poly}
                for prop_col in property_columns:
                    new_entry[prop_col] = np.NaN
                
                new_entries.append(new_entry)
                total_combinations += 1
    
    print(f"Generated {total_combinations} new polymer combinations")
    
    # Create DataFrame for new entries
    df_new = pd.DataFrame(new_entries)
    
    # Save augmented dataset (only new entries)
    base_name = args.input_file.replace('.csv', '')
    augmented_file = f"{base_name}-{args.output_suffix}-poly_chemprop.csv"
    augmented_path = os.path.join(main_dir_path, 'data', augmented_file)
    df_new.to_csv(augmented_path, index=False)
    print(f"Saved augmented dataset: {augmented_file}")
    
    # Combine original and augmented data
    df_combined = pd.concat([df, df_new], axis=0, ignore_index=True)
    combined_file = f"{base_name}-combined-poly_chemprop.csv"
    combined_path = os.path.join(main_dir_path, 'data', combined_file)
    df_combined.to_csv(combined_path, index=False)
    print(f"Saved combined dataset: {combined_file}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Original dataset: {len(df)} entries")
    print(f"Augmented entries: {len(df_new)} entries") 
    print(f"Combined dataset: {len(df_combined)} entries")
    print(f"Properties included: {property_columns}")
    
    # Show sample of new entries
    print(f"\nSample of new entries:")
    print(df_new.head(10))
    
    print('Done!')

if __name__ == "__main__":
    main()
