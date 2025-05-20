import pandas as pd
from rdkit import Chem
import argparse
import os

def canonicalize_smiles(smiles):
    """Canonicalize SMILES and convert attachment points to numbered format"""
    try:
        # Convert [*] to [*:1], [*:2] etc.
        numbered_smiles = smiles
        counter = 1
        while '[*]' in numbered_smiles:
            numbered_smiles = numbered_smiles.replace('[*]', f'[*:{counter}]', 1)
            counter += 1
        
        # Canonicalize with RDKit
        mol = Chem.MolFromSmiles(numbered_smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def make_poly_chemprop_input(mona, monb, stoich):
    """Create poly_chemprop_input format string"""
    can_mona = canonicalize_smiles(mona)
    can_monb = canonicalize_smiles(monb) if monb != mona else can_mona
    
    if can_mona is None or can_monb is None:
        return None

    # Standard connectivity with correct format
    connectivity = "<1.3:0.5:0.5<1.4:0.5:0.5<2.3:0.5:0.5<2.4:0.5:0.5"
    
    # For homopolymers (same monomer)
    if can_mona == can_monb:
        return f"{can_mona}|{stoich}|{connectivity}"
    else:
        # For copolymers
        return f"{can_mona}.{can_monb}|{stoich}|{connectivity}"

def detect_target_columns(df, exclude_columns=None):
    """
    Automatically detect potential target columns (numeric columns)
    
    Args:
        df: DataFrame
        exclude_columns: List of column names to exclude from consideration
    
    Returns:
        List of potential target column names
    """
    if exclude_columns is None:
        exclude_columns = ['pol_id', 'hp_id', 'MonA', 'MonB', 'stoich']
    
    # Find numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    
    # Remove excluded columns
    potential_targets = [col for col in numeric_columns if col not in exclude_columns]
    
    return potential_targets

def preprocess_polymer_data(input_file, output_file, target_columns=None, target_suffix=""):
    """
    General preprocessing function for polymer data with flexible target handling
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file  
        target_columns: List of target column names, or None for auto-detection
        target_suffix: Suffix to add to target column names (e.g., '_eV')
    """
    # Load CSV
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records from {input_file}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Rename 'smiles' to 'MonA' to make it uniform
    if 'smiles' in df.columns:
        df.rename(columns={'smiles': 'MonA'}, inplace=True)
    elif 'MonA' not in df.columns:
        raise ValueError("No 'smiles' or 'MonA' column found!")
    
    # If MonB not provided, assume homopolymer (MonB = MonA)
    if 'MonB' not in df.columns:
        df['MonB'] = df['MonA']
    else:
        df['MonB'].fillna(df['MonA'], inplace=True)
    
    # If stoich not provided, default to 0.5|0.5
    if 'stoich' not in df.columns:
        df['stoich'] = '0.5|0.5'
    else:
        df['stoich'].fillna('0.5|0.5', inplace=True)
    
    # Handle target columns
    if target_columns is None:
        # Auto-detect target columns
        potential_targets = detect_target_columns(df)
        print(f"Auto-detected potential target columns: {potential_targets}")
        
        if len(potential_targets) == 0:
            raise ValueError("No suitable target columns found!")
        
        # Use all detected targets or prompt user
        target_columns = potential_targets
        print(f"Using all detected target columns: {target_columns}")
    
    # Validate target columns exist
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found: {missing_targets}")
    
    # Rename target columns with suffix if provided
    target_mapping = {}
    final_target_columns = []
    for target_col in target_columns:
        if target_suffix and not target_col.endswith(target_suffix):
            new_name = f"{target_col}{target_suffix}"
            df.rename(columns={target_col: new_name}, inplace=True)
            target_mapping[target_col] = new_name
            final_target_columns.append(new_name)
        else:
            final_target_columns.append(target_col)
    
    print(f"Final target columns: {final_target_columns}")
    
    # Apply poly_chemprop_input formatting
    print("Creating poly_chemprop_input...")
    df['poly_chemprop_input'] = df.apply(
        lambda row: make_poly_chemprop_input(row['MonA'], row['MonB'], row['stoich']), 
        axis=1
    )
    
    # Remove rows where canonicalization failed
    initial_count = len(df)
    df = df[df['poly_chemprop_input'].notnull()]
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows due to canonicalization failures")
    
    # Remove rows with NaN target values
    df = df.dropna(subset=final_target_columns)
    
    # Final selection - keep poly_chemprop_input and all target columns
    columns_to_keep = ['poly_chemprop_input'] + final_target_columns
    df_final = df[columns_to_keep].copy()
    
    # Save to output file
    df_final.to_csv(output_file, index=False)
    
    print(f"Saved {len(df_final)} processed records to {output_file}")
    print(f"Columns: {df_final.columns.tolist()}")
    
    # Show statistics for each target column
    print(f"\nTarget property statistics:")
    for target_col in final_target_columns:
        print(f"\n{target_col}:")
        print(f"  Min:  {df_final[target_col].min():.4f}")
        print(f"  Max:  {df_final[target_col].max():.4f}")
        print(f"  Mean: {df_final[target_col].mean():.4f}")
        print(f"  Std:  {df_final[target_col].std():.4f}")
    
    # Show examples
    print(f"\nFirst 3 examples:")
    for i in range(min(3, len(df_final))):
        print(f"{i+1}. {df_final.iloc[i]['poly_chemprop_input']}")
        for target_col in final_target_columns:
            print(f"   {target_col}: {df_final.iloc[i][target_col]}")
    
    return df_final, final_target_columns

def main():
    parser = argparse.ArgumentParser(description='Preprocess polymer data for poly_chemprop with flexible target handling')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--targets', '-t', nargs='+', default=None,
                       help='Target column names (space-separated). If not provided, auto-detects all numeric columns')
    parser.add_argument('--target_suffix', '-s', default='',
                       help='Suffix to add to target column names (e.g., "_eV")')
    parser.add_argument('--list_columns', '-l', action='store_true',
                       help='List all columns in the input file and exit')
    
    args = parser.parse_args()
    
    # If user wants to see available columns
    if args.list_columns:
        df = pd.read_csv(args.input)
        print(f"Columns in {args.input}:")
        potential_targets = detect_target_columns(df)
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            is_target = " (potential target)" if col in potential_targets else ""
            print(f"  {col}: {col_type}{is_target}")
        
        print(f"\nAuto-detected target columns: {potential_targets}")
        return
    
    # Process the data
    df_final, final_targets = preprocess_polymer_data(
        args.input, 
        args.output, 
        args.targets, 
        args.target_suffix
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print(f"Output file: {args.output}")
    print(f"Target properties: {final_targets}")
    print("Ready for transform_batch_data.py")
    print("\nFor transform_batch_data.py, use:")
    target_names = [col.replace('_eV', '').replace('_', '') for col in final_targets]
    print(f"--property_columns {' '.join([f'\"{col}\"' for col in final_targets])}")
    print(f"--property_names {' '.join(target_names)}")
    print("="*60)

if __name__ == "__main__":
    # Example usage when run without arguments
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("# Auto-detect all target columns")
        print("python preprocess_polymer_data.py -i band_gap_chain.csv -o output.csv")
        print("")
        print("# Specify target columns")
        print("python preprocess_polymer_data.py -i data.csv -o output.csv -t value band_gap EA IP")
        print("")
        print("# Add suffix to target columns")
        print("python preprocess_polymer_data.py -i data.csv -o output.csv -s _eV")
        print("")
        print("# List available columns")
        print("python preprocess_polymer_data.py -i data.csv -l")
    else:
        main()
