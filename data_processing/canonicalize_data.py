import pickle
import pandas as pd
import re
import sys, os
import argparse
from functools import partial
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from data_processing.Smiles_enum_canon import SmilesEnumCanon

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
        # Auto-detect property columns (exclude the SMILES input column and non-canonical column)
        smiles_cols = ['poly_chemprop_input', 'poly_chemprop_input_nocan', 'smiles', 'input', 'polymer_smiles']
        property_columns = [col for col in df.columns if col not in smiles_cols]
        return property_columns

def main():
    parser = argparse.ArgumentParser(description='Canonicalize and/or enumerate polymer dataset')
    parser.add_argument("--input_file", type=str, default="dataset-combined-poly_chemprop.csv",
                        help="Input CSV file name (relative to /data directory)")
    parser.add_argument("--output_suffix", type=str, default="enumerated2",
                        help="Suffix for output file (e.g., 'canonical', 'enumerated', 'enumerated2')")
    parser.add_argument("--property_names", type=str, nargs='+', default=None,
                        help="Names of property columns to include. If not specified, auto-detects all non-SMILES columns")
    parser.add_argument("--smiles_column", type=str, default="poly_chemprop_input",
                        help="Name of the column containing polymer SMILES strings")
    parser.add_argument("--nr_enumerations", type=int, default=1,
                        help="Number of enumerations to generate per molecule")
    parser.add_argument("--mode", type=str, default="enumeration2", 
                        choices=["canonical", "enumeration", "enumeration2"],
                        help="Type of processing: canonical, enumeration, or enumeration2")
    
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
    
    # Extract data from DataFrame
    all_poly_inputs = []
    all_property_values = {col: [] for col in property_columns}
    all_mono_combs = []
    
    for i in range(len(df)):
        poly_input = df.loc[i, args.smiles_column]
        all_poly_inputs.append(poly_input)
        all_mono_combs.append(poly_input.split("|")[0])
        
        # Extract all property values
        for prop_col in property_columns:
            all_property_values[prop_col].append(df.loc[i, prop_col])
    
    print(f"Extracted {len(all_poly_inputs)} polymer entries")
    print(f"Properties: {list(property_columns)}")
    
    # Process based on mode
    if args.mode == "canonical":
        print("Processing: Canonicalization only")
        sm_can = SmilesEnumCanon()
        new_polys = list(map(sm_can.canonicalize, all_poly_inputs))
        new_entries = []
        
        for i, canonical_poly_sm in enumerate(new_polys):
            new_entry = {
                args.smiles_column: canonical_poly_sm,
                f"{args.smiles_column}_nocan": df.loc[i, args.smiles_column]
            }
            # Add all property values
            for prop_col in property_columns:
                new_entry[prop_col] = all_property_values[prop_col][i]
            new_entries.append(new_entry)
    
    elif args.mode == "enumeration":
        print(f"Processing: Enumeration with {args.nr_enumerations} enumerations per molecule")
        sm_en = SmilesEnumCanon()
        randomize_smiles_fixed_enums = partial(sm_en.randomize_smiles, nr_enum=args.nr_enumerations)
        new_polys = list(map(randomize_smiles_fixed_enums, all_poly_inputs))  # List of lists
        new_entries = []
        
        for i, enumerated_smiles_list in enumerate(new_polys):
            # For each original datapoint, create multiple enumerations with same labels
            for enumerated_smiles in enumerated_smiles_list:
                new_entry = {
                    args.smiles_column: enumerated_smiles,
                    f"{args.smiles_column}_nocan": df.loc[i, args.smiles_column]
                }
                # Add all property values
                for prop_col in property_columns:
                    new_entry[prop_col] = all_property_values[prop_col][i]
                new_entries.append(new_entry)
    
    elif args.mode == "enumeration2":
        print(f"Processing: Enumeration2 - keeping monomer order, {args.nr_enumerations} enumerations")
        sm_en = SmilesEnumCanon()
        replacement_mon_comb = {}
        all_mono_combs_unique = list(set(all_mono_combs))
        
        # Create enumerated versions of unique monomer combinations
        for c in all_mono_combs_unique:
            # Split monomers and enumerate them once
            monA = c.split('.')[0]
            monB = c.split('.')[1]
            monA_en = sm_en.randomize_smiles(monA, nr_enum=1, renumber_poly_position=True, stoich_con_info=False)[0]
            monB_en = sm_en.randomize_smiles(monB, nr_enum=1, renumber_poly_position=True, stoich_con_info=False)[0]
            replacement_mon_comb[c] = '.'.join([monA_en, monB_en])
        
        # Replace monomer combinations while keeping stoichiometry and connectivity
        all_mono_combs_en = [replacement_mon_comb[item] for item in all_mono_combs]
        new_polys = ["|".join([all_mono_combs_en[i], item.split("|", 1)[1]]) for i, item in enumerate(all_poly_inputs)]
        
        new_entries = []
        for i, new_poly in enumerate(new_polys):
            new_entry = {
                args.smiles_column: new_poly,
                f"{args.smiles_column}_nocan": df.loc[i, args.smiles_column]
            }
            # Add all property values
            for prop_col in property_columns:
                new_entry[prop_col] = all_property_values[prop_col][i]
            new_entries.append(new_entry)
    
    # Create output DataFrame and save
    df_new = pd.DataFrame(new_entries)
    
    # Generate output filename
    base_name = args.input_file.replace('.csv', '')
    output_file = f"{base_name}-{args.output_suffix}-poly_chemprop.csv"
    output_path = os.path.join(main_dir_path, 'data', output_file)
    
    df_new.to_csv(output_path, index=False)
    print(f"\n✅ Saved processed dataset: {output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Input entries: {len(df)}")
    print(f"Output entries: {len(df_new)}")
    print(f"Processing mode: {args.mode}")
    print(f"Properties: {property_columns}")
    print(f"Sample entries:")
    print(df_new.head(5))
    
    print('\nDone!')

if __name__ == "__main__":
    # If run without arguments, use default behavior for backward compatibility
    if len(sys.argv) == 1:
        print("Running in legacy mode (enumeration2 with default settings)")
        print("For flexible usage, run with --help to see all options")
        
        # Legacy behavior - run enumeration2 mode with default settings
        df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop.csv')
        
        all_poly_inputs = []
        all_labels1 = []
        all_labels2 = []
        all_mono_combs = []
        
        for i in range(len(df.loc[:, 'poly_chemprop_input'])):
            poly_input = df.loc[i, 'poly_chemprop_input']
            poly_label1 = df.loc[i, 'EA vs SHE (eV)']
            poly_label2 = df.loc[i, 'IP vs SHE (eV)']
            all_labels1.append(poly_label1)
            all_labels2.append(poly_label2)
            all_poly_inputs.append(poly_input)
            all_mono_combs.append(poly_input.split("|")[0])
        
        # Enumeration2 logic (from original script)
        nr_enumerations = 1
        sm_en = SmilesEnumCanon()
        replacement_mon_comb = {}
        all_mono_combs_unique = list(set(all_mono_combs))
        
        for c in all_mono_combs_unique:
            monA = c.split('.')[0]
            monB = c.split('.')[1]
            monA_en = sm_en.randomize_smiles(monA, nr_enum=1, renumber_poly_position=True, stoich_con_info=False)[0]
            monB_en = sm_en.randomize_smiles(monB, nr_enum=1, renumber_poly_position=True, stoich_con_info=False)[0]
            replacement_mon_comb[c] = '.'.join([monA_en, monB_en])
        
        all_mono_combs_en = [replacement_mon_comb[item] for item in all_mono_combs]
        new_polys = ["|".join([all_mono_combs_en[i], item.split("|", 1)[1]]) for i, item in enumerate(all_poly_inputs)]
        
        new_entries = []
        for i, new_poly in enumerate(new_polys):
            new_entries.append({
                'poly_chemprop_input': new_poly,
                'EA vs SHE (eV)': all_labels1[i],
                'IP vs SHE (eV)': all_labels2[i],
                "poly_chemprop_input_nocan": df.loc[i, 'poly_chemprop_input']
            })
        
        df_new = pd.DataFrame(new_entries)
        df_new.to_csv(main_dir_path+'/data/dataset-combined-enumerated2-poly_chemprop.csv', index=False)
        print(df_new.head(20))
        print('Done')
    else:
        main()
