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
        exclude_columns = ['pol_id', 'hp_id', 'MonA', 'MonB', 'stoich', 'smiles', 'canonical_smiles']
    
    # Find numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    
    # Remove excluded columns
    potential_targets = [col for col in numeric_columns if col not in exclude_columns]
    
    return potential_targets

def interactive_column_selection(df, interactive=True):
    """
    Interactively select target columns and their new names
    
    Args:
        df: DataFrame
        interactive: If True, ask user for input. If False, auto-detect.
    
    Returns:
        tuple: (selected_columns, column_mapping)
    """
    if not interactive:
        # Auto-detect mode
        potential_targets = detect_target_columns(df)
        if not potential_targets:
            raise ValueError("No suitable target columns found!")
        
        # Create default mapping (keep original names)
        column_mapping = {col: col for col in potential_targets}
        return potential_targets, column_mapping
    
    # Interactive mode
    print("\nAvailable columns in your data:")
    print("-" * 50)
    all_columns = df.columns.tolist()
    potential_targets = detect_target_columns(df)
    
    for i, col in enumerate(all_columns):
        col_type = str(df[col].dtype)
        is_numeric = "(numeric" in col_type or "int" in col_type or "float" in col_type
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
        while True:
            try:
                names = input("Column names: ").strip().split(',')
                selected_columns = []
                for name in names:
                    name = name.strip()
                    if name in all_columns:
                        selected_columns.append(name)
                    else:
                        print(f"Column '{name}' not found!")
                        selected_columns = []
                        break
                if selected_columns:
                    break
            except:
                print("Please enter valid column names separated by commas")
    
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

def preprocess_polymer_data(input_file, output_file, target_columns=None, column_mapping=None, interactive=True):
    """
    General preprocessing function for polymer data with flexible target handling
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file  
        target_columns: List of target column names, or None for interactive selection
        column_mapping: Dict mapping old names to new names
        interactive: If True, use interactive selection
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
    if target_columns is None or column_mapping is None:
        target_columns, column_mapping = interactive_column_selection(df, interactive=interactive)
    
    # Validate target columns exist
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found: {missing_targets}")
    
    # Rename target columns according to mapping
    final_target_columns = []
    for old_name in target_columns:
        new_name = column_mapping[old_name]
        if old_name != new_name:
            df.rename(columns={old_name: new_name}, inplace=True)
        final_target_columns.append(new_name)
    
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
    
    return df_final, final_target_columns, column_mapping

def main():
    parser = argparse.ArgumentParser(description='Preprocess polymer data for poly_chemprop with interactive target handling')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--targets', '-t', nargs='+', default=None,
                       help='Target column names (space-separated). If not provided, uses interactive selection')
    parser.add_argument('--target_names', '-n', nargs='+', default=None,
                       help='New names for target columns (must match --targets length)')
    parser.add_argument('--non_interactive', action='store_true',
                       help='Skip interactive selection and auto-detect all numeric columns')
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
    
    # Prepare column mapping if provided via command line
    target_columns = args.targets
    column_mapping = None
    
    if target_columns and args.target_names:
        if len(target_columns) != len(args.target_names):
            raise ValueError("Number of target columns must match number of target names")
        column_mapping = dict(zip(target_columns, args.target_names))
    elif target_columns:
        # Keep original names
        column_mapping = {col: col for col in target_columns}
    
    # Process the data
    df_final, final_targets, final_mapping = preprocess_polymer_data(
        args.input, 
        args.output, 
        target_columns, 
        column_mapping,
        interactive=not args.non_interactive
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print(f"Output file: {args.output}")
    print(f"Target properties: {final_targets}")
    print("Ready for transform_batch_data.py")
    print("\nFor transform_batch_data.py, use:")
    
    # Create property names for transform_batch_data.py (simplified from target names)
    property_names = []
    for target in final_targets:
        # Remove common suffixes and make lowercase
        clean_name = target.lower()
        for suffix in ['_ev', '_value', '_property', '_target']:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        clean_name = clean_name.replace('_', '').replace('-', '').replace(' ', '')
        property_names.append(clean_name)
    
    property_columns_str = ' '.join([f'"{col}"' for col in final_targets])
    property_names_str = ' '.join(property_names)
    
    print(f"--property_columns {property_columns_str}")
    print(f"--property_names {property_names_str}")
    print("="*60)

if __name__ == "__main__":
    # Example usage when run without arguments
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("# Interactive mode (default)")
        print("python data_processing_for_new_datasets.py -i input.csv -o output.csv")
        print("")
        print("# Non-interactive mode (auto-detect all)")
        print("python data_processing_for_new_datasets.py -i input.csv -o output.csv --non_interactive")
        print("")
        print("# Specify columns and names via command line")
        print("python data_processing_for_new_datasets.py -i input.csv -o output.csv -t value band_gap -n band_gap_eV conductivity")
        print("")
        print("# List available columns")
        print("python data_processing_for_new_datasets.py -i input.csv -l")
    else:
        main()
