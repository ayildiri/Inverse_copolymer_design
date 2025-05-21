import pandas as pd
from rdkit import Chem
import argparse
import os
import re

def canonicalize_smiles(smiles):
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

def check_polymer_validity(mona_smiles, monb_smiles):
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

def create_polymer_attachment_scheme(is_homopolymer, mona_pts, monb_pts):
    """
    Create a proper attachment scheme based on monomer structures
    
    Args:
        is_homopolymer: Whether monomers are identical
        mona_pts: List of attachment point numbers in MonA
        monb_pts: List of attachment point numbers in MonB
    
    Returns:
        Connectivity string appropriate for the structure
    """
    connectivity = ""
    
    # For homopolymers, connect same positions between units
    if is_homopolymer:
        for a_pt in mona_pts:
            for b_pt in monb_pts:
                connectivity += f"<{a_pt}-{b_pt}:0.5:0.5"
    else:
        # For copolymers, connect all available positions
        for a_pt in mona_pts:
            for b_pt in monb_pts:
                connectivity += f"<{a_pt}-{b_pt}:0.5:0.5"
    
    return connectivity

def prepare_homopolymer(monomer_smiles):
    """
    Properly prepare a homopolymer by creating consistent attachment points
    
    Args:
        monomer_smiles: Canonicalized SMILES of the monomer
    
    Returns:
        Tuple of (monA, monB) with proper attachment point numbering
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

def make_poly_chemprop_input(mona, monb, stoich, connectivity=None):
    """
    Create properly formatted poly_chemprop_input string with chemical validity checks
    """
    can_mona = canonicalize_smiles(mona)
    
    # Determine if this is a homopolymer
    is_homopolymer = (monb == mona)
    
    if is_homopolymer:
        # Handle homopolymer case
        can_mona, can_monb = prepare_homopolymer(can_mona)
    else:
        # Different monomers - just canonicalize
        can_monb = canonicalize_smiles(monb)
    
    if can_mona is None or can_monb is None:
        return None
    
    # Check if polymer structure is valid
    is_valid, _ = check_polymer_validity(can_mona, can_monb)
    if not is_valid:
        return None
    
    # Extract attachment points
    mona_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', can_mona)])
    monb_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', can_monb)])
    
    # Use provided connectivity or generate appropriate scheme
    if connectivity is None:
        connectivity = create_polymer_attachment_scheme(is_homopolymer, mona_points, monb_points)
    
    # Always ensure connectivity uses hyphens, not dots
    connectivity = connectivity.replace('.', '-')
    
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
        exclude_columns = ['pol_id', 'hp_id', 'MonA', 'MonB', 'stoich', 'connectivity', 'smiles', 'canonical_smiles']
    
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
    
    # For choice == '3' (manual column entry), replace the existing code with:
    else:  # choice == '3'
        try:
            # Try to use ipywidgets for larger input box
            from ipywidgets import widgets
            from IPython.display import display
            
            print("\nEnter column names (comma-separated):")
            print("Example: value,band_gap,property1")
            
            # Create larger text area widget
            textarea = widgets.Textarea(
                value='',
                placeholder='Enter column names separated by commas',
                description='Columns:',
                rows=5,
                width='100%'
            )
            
            display(textarea)
            
            # Wait for input (this is a bit hacky)
            import time
            max_wait = 60  # seconds
            last_val = textarea.value
            for _ in range(max_wait):
                time.sleep(1)
                if textarea.value != last_val and textarea.value.strip():
                    break
                last_val = textarea.value
            
            names = textarea.value.strip().split(',')
            
            # Process names as before
            selected_columns = []
            for name in names:
                name = name.strip()
                if name in all_columns:
                    selected_columns.append(name)
                else:
                    print(f"Column '{name}' not found!")
            
        except (ImportError, Exception):
            # Fallback to standard input if widgets not available
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

def preprocess_polymer_data(input_file, output_file, target_columns=None, column_mapping=None, interactive=True, stoichiometry="0.5|0.5", default_connectivity=None):
    """
    General preprocessing function for polymer data with flexible target handling
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file  
        target_columns: List of target column names, or None for interactive selection
        column_mapping: Dict mapping old names to new names
        interactive: If True, use interactive selection
        stoichiometry: Default stoichiometry to use (format: "fraction_A|fraction_B")
        default_connectivity: Default connectivity pattern for polymers (if None, will be derived from structure)
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
    
    # If stoich not provided, use the provided stoichiometry parameter
    if 'stoich' not in df.columns:
        df['stoich'] = stoichiometry
    else:
        df['stoich'].fillna(stoichiometry, inplace=True)
    
    # Handle connectivity if provided in the CSV
    if 'connectivity' not in df.columns:
        df['connectivity'] = None
    else:
        # Replace null values with None rather than a string
        df['connectivity'] = df['connectivity'].where(pd.notnull(df['connectivity']), None)
    
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
        lambda row: make_poly_chemprop_input(row['MonA'], row['MonB'], row['stoich'], row['connectivity']), 
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
        
        # Parse and show attachment points in the example
        smiles_part = df_final.iloc[i]['poly_chemprop_input'].split('|')[0]
        attachment_points = sorted([int(m.group(1)) for m in re.finditer(r'\[\*:(\d+)\]', smiles_part)])
        print(f"   Attachment points: {attachment_points}")
    
    return df_final, final_target_columns, column_mapping

def main():
    parser = argparse.ArgumentParser(description='Preprocess polymer data for poly_chemprop with interactive target handling')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--targets', '-t', nargs='+', default=None,
                        help='Target column names (space-separated). If not provided, uses interactive selection')
    parser.add_argument('--target_names', '-n', nargs='+', default=None,
                        help='New names for target columns (must match --targets length)')
    parser.add_argument('--stoichiometry', '-s', default="0.5|0.5",
                        help='Default stoichiometry for homopolymers (format: fraction_A|fraction_B)')
    parser.add_argument('--connectivity', '-c', default=None,
                        help='Default connectivity pattern for polymers (if None, derived from structure)')
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
        interactive=not args.non_interactive,
        stoichiometry=args.stoichiometry,
        default_connectivity=args.connectivity
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
        print("# With custom stoichiometry")
        print("python data_processing_for_new_datasets.py -i input.csv -o output.csv -s \"0.7|0.3\"")
        print("")
        print("# List available columns")
        print("python data_processing_for_new_datasets.py -i input.csv -l")
    else:
        main()
