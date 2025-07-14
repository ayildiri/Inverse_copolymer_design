# %% Packages
import os, sys
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import numpy as np
import torch
from torch.utils.data import Dataset
from data_processing.data_utils import *
import pandas as pd
import networkx as nx
from torch_geometric.utils import to_networkx
import random
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from data_processing.Function_Featurization_Own import poly_smiles_to_graph_flexible
import argparse
import shutil

# %% Hyperparameters
device = 'cpu'
# %% Call data
parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--batch_size", type=int, default=64)
# Add flexible property arguments
parser.add_argument("--property_columns", type=str, nargs='+', default=["EA vs SHE (eV)", "IP vs SHE (eV)"],
                    help="Names of the property columns in the CSV file")
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Short names for the properties (for file naming and processing)")
# Add custom dataset path argument
parser.add_argument("--input_file", type=str, default=None,
                    help="Path to custom input CSV file containing polymer data")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save output files (defaults to main_dir_path/data)")
parser.add_argument("--semi_supervised", action="store_true", default=False,
                    help="Enable semi-supervised mode for Stage 1 (includes unlabeled data)")
parser.add_argument("--exclude_properties", type=str, nargs='*', default=[],
                    help="Property columns to exclude in semi-supervised mode (e.g., bandgap_chain)")
# LONG-TERM FIX: Add vocab_file argument
parser.add_argument("--vocab_file", type=str, default=None,
                    help="Path to existing vocabulary file to use (ensures consistency across stages)")
parser.add_argument("--create_master_vocab", action="store_true", default=False,
                    help="Create a comprehensive vocabulary from all data (use this once before all stages)")

args = parser.parse_args()

augment = args.augment
batch_size = args.batch_size
tokenization = "RT_tokenized" # oldtok is the old tokenization scheme without numerical tokens
string_format = "poly_chemprop" # "poly_chemprop" or "gbigsmileslike"
smiles_enumeration = True

# Get property configuration
property_columns = args.property_columns
property_names = args.property_names
property_count = len(property_columns)

# Set output directory
output_dir = args.output_dir if args.output_dir else os.path.join(main_dir_path, 'data')
os.makedirs(output_dir, exist_ok=True)
print(f"Output files will be saved to: {output_dir}")

# Validate property arguments
if len(property_columns) != len(property_names):
    raise ValueError(f"Number of property columns ({len(property_columns)}) must match number of property names ({len(property_names)})")

print(f"Processing {property_count} properties: {property_names}")
print(f"Property columns in CSV: {property_columns}")

# LONG-TERM FIX: Master vocabulary creation function
def create_master_vocabulary(csv_path, output_path, tokenization='RT_tokenized', augment='augmented'):
    """Create a comprehensive master vocabulary from the entire dataset"""
    from collections import Counter
    
    print("\n🔨 CREATING MASTER VOCABULARY")
    print("="*60)
    
    # Load the full dataset
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} molecules from dataset")
    
    # Collect all SMILES strings
    all_poly_inputs = []
    
    # Get all poly_chemprop_input values
    if 'poly_chemprop_input' in df.columns:
        all_poly_inputs.extend(df['poly_chemprop_input'].dropna().tolist())
    
    # Also get nocan versions if available
    if 'poly_chemprop_input_nocan' in df.columns:
        all_poly_inputs.extend(df['poly_chemprop_input_nocan'].dropna().tolist())
    
    print(f"📝 Collected {len(all_poly_inputs)} polymer strings to tokenize")
    
    # Tokenize all strings
    token_counter = Counter()
    for i, poly_input in enumerate(all_poly_inputs):
        if tokenization == "oldtok":
            tokens = tokenize_poly_input(poly_input=poly_input)
        elif tokenization == "RT_tokenized":
            tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
        
        token_counter.update(tokens)
        
        if i % 1000 == 0:
            print(f"   Tokenized {i}/{len(all_poly_inputs)} polymer strings...")
    
    # Create vocabulary with special tokens first
    special_tokens = ['_PAD', '_SOS', '_EOS', '_UNK']
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Add all tokens sorted by frequency (most common first) for consistency
    for token, count in token_counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    
    # Save vocabulary
    with open(output_path, 'w') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            # Save without frequency counts for consistency
            f.write(f"{token}\t{idx}\n")
    
    print(f"\n✅ Created master vocabulary with {len(vocab)} unique tokens")
    print(f"📁 Saved to: {output_path}")
    print(f"🔤 Token examples: {list(token_counter.most_common(10))}")
    
    return vocab

# LONG-TERM FIX: Handle master vocabulary creation
if args.create_master_vocab:
    # Special mode to create master vocabulary
    if not args.input_file:
        raise ValueError("--input_file required when creating master vocabulary")
    
    master_vocab_path = os.path.join(
        os.path.dirname(args.output_dir) if args.output_dir else main_dir_path,
        f'master_vocab_{augment}_{tokenization}.txt'
    )
    
    create_master_vocabulary(
        args.input_file,
        master_vocab_path,
        tokenization=tokenization,
        augment=augment
    )
    
    print("\n🎯 MASTER VOCABULARY CREATED!")
    print("📋 Next steps:")
    print("1. Use this vocabulary for all stages by adding: --vocab_file " + master_vocab_path)
    print("2. This ensures all stages use identical vocabulary")
    sys.exit(0)  # Exit after creating master vocab

# Load input file based on command line argument or fall back to default paths
if args.input_file:
    # Use the custom input file path
    df = pd.read_csv(args.input_file)
    print(f"Loading custom dataset from: {args.input_file}")
    # Use augment value for consistency in file naming
    file_prefix = augment  # ← CHANGED THIS LINE
    print(f"Using file prefix for outputs: {file_prefix}")
else:
    # Use the default paths based on augment value
    if augment == "original":
        df = pd.read_csv(os.path.join(main_dir_path, 'data', 'dataset-poly_chemprop.csv'))
        file_prefix = "original"
    elif augment == "augmented":
        df = pd.read_csv(os.path.join(main_dir_path, 'data', 'dataset-combined-poly_chemprop.csv'))
        file_prefix = "augmented"

# Verify that all property columns exist in the dataframe
missing_columns = [col for col in property_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing property columns in CSV: {missing_columns}")

# Semi-supervised filtering for Stage 1
if args.semi_supervised:
    print("\n🔄 SEMI-SUPERVISED MODE ENABLED")
    print("="*60)
    
    original_size = len(df)
    
    # Create mask for molecules with the target properties
    has_target_properties = pd.Series([True] * len(df))
    for col in property_columns:
        has_target_properties &= df[col].notna()
    
    # Create mask for molecules with ONLY excluded properties
    has_only_excluded = pd.Series([False] * len(df))
    if args.exclude_properties:
        # Check if molecule has any excluded property but none of the target properties
        has_excluded = pd.Series([False] * len(df))
        for excl_col in args.exclude_properties:
            if excl_col in df.columns:
                has_excluded |= df[excl_col].notna()
        
        all_target_missing = pd.Series([True] * len(df))
        for col in property_columns:
            all_target_missing &= df[col].isna()
        
        has_only_excluded = has_excluded & all_target_missing
    
    # Create mask for completely unlabeled molecules
    is_unlabeled = pd.Series([True] * len(df))
    # Check all property columns (target + excluded)
    all_prop_cols = list(property_columns) + (args.exclude_properties if args.exclude_properties else [])
    for col in all_prop_cols:
        if col in df.columns:
            is_unlabeled &= df[col].isna()
    
    # Stage 1 includes: target property labeled + completely unlabeled
    # Excludes: molecules with ONLY excluded properties
    stage1_mask = has_target_properties | is_unlabeled
    
    # Apply the mask
    df_filtered = df[stage1_mask].copy()
    
    print(f"📊 Data composition:")
    print(f"   Original dataset size: {original_size:,}")
    print(f"   With {'/'.join(property_names)} labels: {has_target_properties.sum():,}")
    print(f"   Completely unlabeled: {is_unlabeled.sum():,}")
    if args.exclude_properties:
        print(f"   With only {'/'.join(args.exclude_properties)} (excluded): {has_only_excluded.sum():,}")
    print(f"   ✅ Filtered dataset size: {len(df_filtered):,}")
    print("="*60)
    
    df = df_filtered

# Supervised filtering - exclude unlabeled molecules for supervised learning
if not args.semi_supervised:
    print(f"\n🎯 SUPERVISED MODE: Filtering for {'/'.join(property_names)}-labeled molecules only")
    print("="*60)
    
    original_size = len(df)
    
    # Only include molecules that have ALL requested property labels
    has_all_labels = pd.Series([True] * len(df))
    for prop_col in property_columns:
        has_all_labels &= df[prop_col].notna()
    
    # Apply the mask
    df = df[has_all_labels].copy()
    
    print(f"📊 Data composition:")
    print(f"   Original dataset size: {original_size:,}")
    print(f"   With all {'/'.join(property_names)} labels: {has_all_labels.sum():,}")
    print(f"   ✅ Filtered dataset size: {len(df):,}")
    print("="*60)

# %% Lets create PyG data objects

# uncomment if graphs_list.pt does not exist
# Here we turn all smiles tring and featurize them into graphs and put them in a list: graphs_list
# additionally we add the target token ids of the target string as graph attributes 

Graphs_list = []
target_tokens_list = []
target_tokens_ids_list = []
target_tokens_lens_list = []
failed_molecules = 0  # ← ADD THIS

for i in range(len(df)):  # ← CHANGED: use len(df) instead
    try:  # ← ADD TRY BLOCK
        poly_input = df.iloc[i]['poly_chemprop_input']  # ← CHANGED to iloc
        try: 
            poly_input_nocan = df.iloc[i]['poly_chemprop_input_nocan']  # ← CHANGED to iloc
        except: 
            poly_input_nocan = None
        
        # Extract property values dynamically based on property_columns
        property_values = []
        for prop_col in property_columns:
            prop_value = df.iloc[i][prop_col]  # ← CHANGED to iloc
            property_values.append(prop_value)
        
        # Import the flexible function if not already imported
        from data_processing.Function_Featurization_Own import poly_smiles_to_graph_flexible
        
        # Use the flexible function for all cases
        graphs = poly_smiles_to_graph_flexible(poly_input, property_values, poly_input_nocan)
    
        if tokenization=="oldtok":
                target_tokens = tokenize_poly_input(poly_input=poly_input)
        elif tokenization=="RT_tokenized":
            target_tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
        
        Graphs_list.append(graphs)
        target_tokens_list.append(target_tokens)
        
        if i % 100 == 0:
            print(f"[{i} / {len(df)}] - Successfully processed")  # ← CHANGED message
            
    except Exception as e:  # ← ADD EXCEPTION HANDLING
        failed_molecules += 1
        print(f"ERROR processing molecule {i}: {str(e)}")
        if failed_molecules > 10:  # Stop if too many failures
            print(f"Too many failures ({failed_molecules}). Stopping.")
            raise

print(f"\n✅ Successfully processed {len(Graphs_list)} molecules")
print(f"❌ Failed to process {failed_molecules} molecules")

if len(Graphs_list) == 0:
    raise ValueError("No molecules were successfully processed!")

# Create flexible file names that include property information
property_suffix = "_".join(property_names)
vocab_filename = f'poly_smiles_vocab_{file_prefix}_{tokenization}_{property_suffix}.txt'
graphs_filename = f'Graphs_list_{file_prefix}_{tokenization}_{property_suffix}.pt'

# LONG-TERM FIX: Handle vocabulary file usage/creation
vocab_path = os.path.join(output_dir, vocab_filename)

if args.vocab_file:
    # Use existing vocabulary file
    print(f"\n📚 USING EXISTING VOCABULARY: {args.vocab_file}")
    
    # Copy the vocab file to output directory with appropriate name
    shutil.copy(args.vocab_file, vocab_path)
    print(f"✅ Copied vocabulary to: {vocab_path}")
    
    # Load the vocabulary
    vocab = load_vocab(vocab_file=args.vocab_file)
    print(f"🔤 Loaded vocabulary with {len(vocab)} tokens")
    
else:
    # Create new vocabulary from current data
    print(f"\n📝 CREATING NEW VOCABULARY from current data")
    print("⚠️  WARNING: This may create inconsistencies between stages!")
    print("💡 RECOMMENDATION: Create a master vocabulary first using --create_master_vocab")
    
    # Create vocab file with property suffix
    make_vocab(target_tokens_list=target_tokens_list, vocab_file=vocab_path)
    
    # Load the created vocabulary
    vocab = load_vocab(vocab_file=vocab_path)
    print(f"🔤 Created vocabulary with {len(vocab)} tokens")

# LONG-TERM FIX: Save vocab info for verification
vocab_info_path = os.path.join(output_dir, f'vocab_info_{property_suffix}.txt')
with open(vocab_info_path, 'w') as f:
    f.write(f"Vocabulary size: {len(vocab)}\n")
    f.write(f"Source: {'External' if args.vocab_file else 'Created from data'}\n")
    if args.vocab_file:
        f.write(f"External vocab path: {args.vocab_file}\n")
    f.write(f"Property suffix: {property_suffix}\n")
    f.write(f"Tokenization: {tokenization}\n")
    f.write(f"Dataset: {file_prefix}\n")
print(f"📋 Saved vocabulary info to: {vocab_info_path}")

# convert the target_tokens_list to target_token_ids_list using the vocab file
max_tgt_token_length = len(max(target_tokens_list, key=len))
for tgt_tokens in target_tokens_list:
    tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=tgt_tokens, vocab=vocab, max_tgt_len=max_tgt_token_length)
    target_tokens_ids_list.append(tgt_token_ids)
    target_tokens_lens_list.append(tgt_lens)

# Add the tgt_token_ids as additional attributes in graph
for sample_idx, g in enumerate(Graphs_list): 
    g.tgt_token_ids = target_tokens_ids_list[sample_idx]
    g.tgt_token_lens = target_tokens_lens_list[sample_idx]


# Save graphs (and tgt token) data with property suffix
graphs_path = os.path.join(output_dir, graphs_filename)
torch.save(Graphs_list, graphs_path)

# ← ADD THESE VERIFICATION LINES
# Verify the file was saved
if os.path.exists(graphs_path):
    file_size = os.path.getsize(graphs_path) / (1024 * 1024)  # Size in MB
    print(f"✅ Successfully saved {graphs_filename} ({file_size:.2f} MB)")
else:
    raise IOError(f"Failed to save {graphs_filename}")

# Create training, self supervised and test sets

# shuffle graphs
# we first take out the validation and test set from 
random.seed(12345)
if not args.input_file and augment == "original":
    # new improved test set: exclude monomer combinations completely
    mon_combs = []
    monB_list = []
    stoichiometry_connectivity_combs = []
    for i in range(len(df)):
        poly_input = df.iloc[i]['poly_chemprop_input']
        
        # Property extraction is no longer needed here as we're just doing train/test split
        mon_combs.append(".".join(poly_input.split("|")[0].split('.')))

    mon_combs= list(set(mon_combs))
    mon_combs_shuffle = random.sample(mon_combs, len(mon_combs))
    # take 80-20 split for trainig -  test data
    train_mon_combs = mon_combs_shuffle[:int(0.8*len(mon_combs_shuffle))]
    val_mon_combs = mon_combs_shuffle[int(0.8*len(mon_combs_shuffle)):int(0.9*len(mon_combs_shuffle))]
    test_mon_combs = mon_combs_shuffle[int(0.9*len(mon_combs_shuffle)):]

    
    #Go through graphs list and assign 
    Graphs_list = torch.load(os.path.join(output_dir, graphs_filename))
    data_list_shuffle = random.sample(Graphs_list, len(Graphs_list))
    train_datalist=[]
    val_datalist=[]
    test_datalist=[]
    for graph in Graphs_list: 
        if ".".join(graph.monomer_smiles) in train_mon_combs:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in val_mon_combs:
            val_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in test_mon_combs:
            test_datalist.append(graph)

elif not args.input_file and augment == "augmented":
    # Load original data with property suffix for consistency
    original_graphs_file = f'Graphs_list_original_{tokenization}_{property_suffix}.pt'
    try:
        Graphs_list = torch.load(os.path.join(output_dir, original_graphs_file))
    except FileNotFoundError:
        # Fallback to old naming convention if new file doesn't exist
        print(f"Warning: Could not find {original_graphs_file}, falling back to old naming convention")
        Graphs_list = torch.load(os.path.join(output_dir, f'Graphs_list_original_{tokenization}.pt'))
    
    Graphs_list_combined = torch.load(os.path.join(output_dir, graphs_filename))
    org_polymers = Graphs_list_combined[:len(Graphs_list)]
    augm_polymers = Graphs_list_combined[len(Graphs_list):] 
    mon_combs=[]
    # go through original data
    for graph in org_polymers:
        mon_combs.append(".".join(graph.monomer_smiles))

    mon_combs= list(set(mon_combs))
    mon_combs_shuffle = random.sample(mon_combs, len(mon_combs))
    # take 80-20 split for trainig -  test data
    # Split not the data randomly but monomer combinations randomly, so same monomer combinations are not in train and testset

    train_mon_combs = mon_combs_shuffle[:int(0.8*len(mon_combs_shuffle))]
    val_mon_combs = mon_combs_shuffle[int(0.8*len(mon_combs_shuffle)):int(0.9*len(mon_combs_shuffle))]
    test_mon_combs = mon_combs_shuffle[int(0.9*len(mon_combs_shuffle)):]

    train_datalist=[]
    val_datalist=[]
    test_datalist=[]
    for graph in org_polymers: 
        if ".".join(graph.monomer_smiles) in train_mon_combs:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in val_mon_combs:
            val_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in test_mon_combs:
            test_datalist.append(graph)

    # go through the augmented data 
    mon_combs_new=[]
    for graph in augm_polymers:
        if not ".".join(graph.monomer_smiles) in mon_combs:
            # only monomer combinations that have not been seen in the original dataset
            mon_combs_new.append(".".join(graph.monomer_smiles))

    mon_combs_augm= list(set(mon_combs_new))
    mon_combs_augm_shuffle = random.sample(mon_combs_augm, len(mon_combs_augm))
    train_mon_combs_augm = mon_combs_augm_shuffle[:int(0.9*len(mon_combs_augm_shuffle))]
    val_mon_combs_augm = mon_combs_augm_shuffle[int(0.9*len(mon_combs_augm_shuffle)):int(0.95*len(mon_combs_augm_shuffle))]
    test_mon_combs_augm = mon_combs_augm_shuffle[int(0.95*len(mon_combs_augm_shuffle)):]

    for graph in augm_polymers: 
        if ".".join(graph.monomer_smiles) in train_mon_combs_augm:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in val_mon_combs_augm:
            val_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in test_mon_combs_augm:
            test_datalist.append(graph)

else:
    # For custom input files, use enhanced splitting for augmented datasets
    print("Using enhanced train/validation/test split for custom input file")
    
    if augment == "augmented":
        print("🎯 AUGMENTED DATASET: Segregating labeled vs unlabeled data for optimal evaluation")
        
        # Separate labeled and unlabeled molecules
        labeled_graphs = []
        unlabeled_graphs = []
        
        for graph in Graphs_list:
            # Check if this graph has valid property labels
            is_labeled = True
            for i in range(property_count):
                prop_attr = f'y{i+1}'
                if hasattr(graph, prop_attr):
                    prop_value = getattr(graph, prop_attr)
                    if torch.is_tensor(prop_value):
                        if torch.isnan(prop_value).any():
                            is_labeled = False
                            break
                    elif pd.isna(prop_value):
                        is_labeled = False
                        break
                else:
                    is_labeled = False
                    break
            
            if is_labeled:
                labeled_graphs.append(graph)
            else:
                unlabeled_graphs.append(graph)
        
        print(f"   📊 Data composition:")
        print(f"      Labeled molecules: {len(labeled_graphs):,}")
        print(f"      Unlabeled molecules: {len(unlabeled_graphs):,}")
        
        # Extract monomer combinations from LABELED data only for fair splitting
        labeled_mon_combs = []
        for graph in labeled_graphs:
            labeled_mon_combs.append(".".join(graph.monomer_smiles))
        
        labeled_mon_combs = list(set(labeled_mon_combs))
        labeled_mon_combs_shuffle = random.sample(labeled_mon_combs, len(labeled_mon_combs))
        
        print(f"   🧬 Unique labeled monomer combinations: {len(labeled_mon_combs)}")
        
        # Split labeled monomer combinations (80/10/10)
        train_labeled_combs = labeled_mon_combs_shuffle[:int(0.8*len(labeled_mon_combs_shuffle))]
        val_labeled_combs = labeled_mon_combs_shuffle[int(0.8*len(labeled_mon_combs_shuffle)):int(0.9*len(labeled_mon_combs_shuffle))]
        test_labeled_combs = labeled_mon_combs_shuffle[int(0.9*len(labeled_mon_combs_shuffle)):]
        
        # Initialize split lists
        train_datalist = []
        val_datalist = []
        test_datalist = []
        
        # Distribute labeled molecules based on monomer combinations
        for graph in labeled_graphs:
            monomer_combo = ".".join(graph.monomer_smiles)
            if monomer_combo in train_labeled_combs:
                train_datalist.append(graph)
            elif monomer_combo in val_labeled_combs:
                val_datalist.append(graph)
            elif monomer_combo in test_labeled_combs:
                test_datalist.append(graph)
        
        # Add ALL unlabeled molecules to training set only
        train_datalist.extend(unlabeled_graphs)
        
        print(f"   ✅ Augmented split strategy:")
        print(f"      Training: {len(train_datalist):,} total ({len(train_datalist)-len(unlabeled_graphs):,} labeled + {len(unlabeled_graphs):,} unlabeled)")
        print(f"      Validation: {len(val_datalist):,} total ({len(val_datalist):,} labeled + 0 unlabeled)")  
        print(f"      Test: {len(test_datalist):,} total ({len(test_datalist):,} labeled + 0 unlabeled)")
        
    else:
        # Original dataset - standard splitting since all data is labeled
        print("📋 ORIGINAL DATASET: Standard splitting (all data is labeled)")
        
        # Extract monomer combinations to ensure similar monomers don't appear in both train and test
        mon_combs = []
        for i in range(len(df)):
            poly_input = df.iloc[i]['poly_chemprop_input']
            mon_combs.append(".".join(poly_input.split("|")[0].split('.')))

        mon_combs = list(set(mon_combs))
        mon_combs_shuffle = random.sample(mon_combs, len(mon_combs))
        
        # Standard 80/10/10 split
        train_mon_combs = mon_combs_shuffle[:int(0.8*len(mon_combs_shuffle))]
        val_mon_combs = mon_combs_shuffle[int(0.8*len(mon_combs_shuffle)):int(0.9*len(mon_combs_shuffle))]
        test_mon_combs = mon_combs_shuffle[int(0.9*len(mon_combs_shuffle)):]
        
        # Assign graphs to the appropriate split
        train_datalist = []
        val_datalist = []
        test_datalist = []
        
        for graph in Graphs_list:
            if ".".join(graph.monomer_smiles) in train_mon_combs:
                train_datalist.append(graph)
            elif ".".join(graph.monomer_smiles) in val_mon_combs:
                val_datalist.append(graph)
            elif ".".join(graph.monomer_smiles) in test_mon_combs:
                test_datalist.append(graph)

# =============================================================================
# ROBUST DATA SPLITTING VALIDATION AND REDISTRIBUTION
# =============================================================================

def validate_and_fix_splits(train_datalist, val_datalist, test_datalist, property_count, min_valid_per_split=50):
    """
    Validate that each split has sufficient molecules with valid property labels.
    Redistribute if necessary while maintaining monomer-based separation.
    """
    
    print(f"\n🔍 VALIDATING DATA SPLITS FOR {property_count} PROPERTIES")
    print("="*60)
    
    def count_valid_molecules(datalist, split_name):
        """Count molecules with valid (non-NaN) property labels"""
        if not datalist:
            return 0, 0, []
            
        valid_count = 0
        total_count = len(datalist)
        invalid_monomers = []
        
        for graph in datalist:
            # Check if all properties are valid (non-NaN)
            all_valid = True
            for i in range(property_count):
                prop_attr = f'y{i+1}'
                if hasattr(graph, prop_attr):
                    prop_value = getattr(graph, prop_attr)
                    if torch.is_tensor(prop_value):
                        if torch.isnan(prop_value).any():
                            all_valid = False
                            break
                    elif pd.isna(prop_value):
                        all_valid = False
                        break
                else:
                    all_valid = False
                    break
            
            if all_valid:
                valid_count += 1
            else:
                # Track which monomer combinations have invalid data
                monomer_combo = ".".join(graph.monomer_smiles)
                if monomer_combo not in invalid_monomers:
                    invalid_monomers.append(monomer_combo)
        
        validity_rate = valid_count / total_count if total_count > 0 else 0
        print(f"   {split_name:<12}: {valid_count:>5}/{total_count:<5} valid ({validity_rate:.1%})")
        
        if valid_count < min_valid_per_split:
            print(f"   ⚠️  {split_name} has only {valid_count} valid molecules (minimum: {min_valid_per_split})")
        
        return valid_count, total_count, invalid_monomers
    
    # Count valid molecules in each split
    train_valid, train_total, train_invalid_monomers = count_valid_molecules(train_datalist, "Train")
    val_valid, val_total, val_invalid_monomers = count_valid_molecules(val_datalist, "Validation")
    test_valid, test_total, test_invalid_monomers = count_valid_molecules(test_datalist, "Test")
    
    # Check if any split is critically low (especially test and validation for evaluation)
    critical_splits = []
    if val_valid < min_valid_per_split // 2:  # More lenient for validation
        critical_splits.append(("val", val_valid))
    if test_valid < min_valid_per_split // 2:  # More lenient for test
        critical_splits.append(("test", test_valid))
    
    if critical_splits:
        print(f"\n⚠️  CRITICAL: Some evaluation splits have insufficient valid data!")
        for split_name, count in critical_splits:
            print(f"   {split_name}: only {count} valid molecules")
        
        # Strategy: Redistribute while preserving monomer-based separation
        print(f"\n🔄 ATTEMPTING TO REDISTRIBUTE FOR BETTER EVALUATION DATA...")
        
        # Collect all molecules organized by monomer combinations and validity
        all_monomer_combos = {}
        valid_monomer_combos = {}
        
        for graph in train_datalist + val_datalist + test_datalist:
            monomer_combo = ".".join(graph.monomer_smiles)
            
            # Add to overall collection
            if monomer_combo not in all_monomer_combos:
                all_monomer_combos[monomer_combo] = []
            all_monomer_combos[monomer_combo].append(graph)
            
            # Check if this graph has valid properties
            is_valid = True
            for i in range(property_count):
                prop_attr = f'y{i+1}'
                if hasattr(graph, prop_attr):
                    prop_value = getattr(graph, prop_attr)
                    if torch.is_tensor(prop_value):
                        if torch.isnan(prop_value).any():
                            is_valid = False
                            break
                    elif pd.isna(prop_value):
                        is_valid = False
                        break
                else:
                    is_valid = False
                    break
            
            if is_valid:
                if monomer_combo not in valid_monomer_combos:
                    valid_monomer_combos[monomer_combo] = []
                valid_monomer_combos[monomer_combo].append(graph)
        
        # Prioritize monomer combinations with valid data for evaluation splits
        valid_combo_list = list(valid_monomer_combos.keys())
        all_combo_list = list(all_monomer_combos.keys())
        
        random.shuffle(valid_combo_list)
        random.shuffle(all_combo_list)
        
        # Reserve some valid combinations for test and validation
        min_test_combos = max(5, len(valid_combo_list) // 20)  # At least 5% for test
        min_val_combos = max(5, len(valid_combo_list) // 20)   # At least 5% for val
        
        # Assign combinations
        new_train_list = []
        new_val_list = []
        new_test_list = []
        
        # First, ensure test gets some valid combinations
        test_combos = valid_combo_list[:min_test_combos]
        remaining_valid_combos = valid_combo_list[min_test_combos:]
        
        # Then, ensure validation gets some valid combinations
        val_combos = remaining_valid_combos[:min_val_combos]
        remaining_valid_combos = remaining_valid_combos[min_val_combos:]
        
        # Rest goes to training (valid + any remaining)
        train_combos = remaining_valid_combos
        
        # Add remaining combinations (including invalid ones) to training
        remaining_all_combos = [combo for combo in all_combo_list if combo not in test_combos and combo not in val_combos and combo not in train_combos]
        train_combos.extend(remaining_all_combos)
        
        # Create new splits
        for combo in train_combos:
            new_train_list.extend(all_monomer_combos[combo])
        
        for combo in val_combos:
            new_val_list.extend(all_monomer_combos[combo])
            
        for combo in test_combos:
            new_test_list.extend(all_monomer_combos[combo])
        
        # Count valid molecules in new splits
        def count_valid_in_list(datalist):
            valid_count = 0
            for graph in datalist:
                all_valid = True
                for i in range(property_count):
                    prop_attr = f'y{i+1}'
                    if hasattr(graph, prop_attr):
                        prop_value = getattr(graph, prop_attr)
                        if torch.is_tensor(prop_value):
                            if torch.isnan(prop_value).any():
                                all_valid = False
                                break
                        elif pd.isna(prop_value):
                            all_valid = False
                            break
                    else:
                        all_valid = False
                        break
                if all_valid:
                    valid_count += 1
            return valid_count
        
        train_valid_new = count_valid_in_list(new_train_list)
        val_valid_new = count_valid_in_list(new_val_list)
        test_valid_new = count_valid_in_list(new_test_list)
        
        print(f"   ✅ Redistributed to prioritize evaluation data:")
        print(f"      Train: {len(new_train_list)} total ({train_valid_new} valid)")
        print(f"      Val:   {len(new_val_list)} total ({val_valid_new} valid)")
        print(f"      Test:  {len(new_test_list)} total ({test_valid_new} valid)")
        
        # Only use new splits if they improve evaluation data
        if test_valid_new > test_valid or val_valid_new > val_valid:
            print(f"   🎯 Using redistributed splits (better evaluation data)")
            return new_train_list, new_val_list, new_test_list
        else:
            print(f"   📋 Keeping original splits (redistribution didn't help)")
            return train_datalist, val_datalist, test_datalist
    
    else:
        print(f"\n✅ ALL SPLITS HAVE SUFFICIENT VALID DATA FOR EVALUATION")
        return train_datalist, val_datalist, test_datalist

# Apply robust validation and redistribution
print(f"\n🔄 Validating splits for {property_count} properties: {property_names}")

# Validate and fix splits if necessary
train_datalist, val_datalist, test_datalist = validate_and_fix_splits(
    train_datalist, val_datalist, test_datalist, 
    property_count=property_count,
    min_valid_per_split=50  # Minimum valid molecules per split
)

# =============================================================================
# FINAL VALIDATION AND STATISTICS
# =============================================================================

# Check if there are any graphs in the datalists
if len(train_datalist) == 0:
    raise ValueError("No training data found. Check your input file and property columns.")
if len(val_datalist) == 0:
    print("Warning: No validation data found. Using a portion of training data for validation.")
    # Use 10% of training data for validation if none exists
    train_datalist, val_datalist = train_test_split(train_datalist, test_size=0.1, random_state=42)
if len(test_datalist) == 0:
    print("Warning: No test data found. Using a portion of training data for testing.")
    # Use 10% of training data for testing if none exists
    train_datalist, test_datalist = train_test_split(train_datalist, test_size=0.1, random_state=42)

# Print final statistics
print(f'\n📊 FINAL ROBUST SPLITS:')
print(f'Number of training graphs: {len(train_datalist)}')
print(f'Number of validation graphs: {len(val_datalist)}')
print(f'Number of test graphs: {len(test_datalist)}')

# Final validation check with detailed reporting
def final_validation_check(datalist, split_name):
    """Final check that split has adequate valid data for evaluation"""
    if not datalist:
        return 0
    
    valid_count = 0
    labeled_count = 0  # Count molecules with any label (even if some NaN)
    unlabeled_count = 0  # Count completely unlabeled molecules
    
    for graph in datalist:
        has_any_label = False
        all_valid = True
        
        for i in range(property_count):
            prop_attr = f'y{i+1}'
            if hasattr(graph, prop_attr):
                prop_value = getattr(graph, prop_attr)
                if torch.is_tensor(prop_value):
                    if not torch.isnan(prop_value).any():
                        has_any_label = True
                    else:
                        all_valid = False
                elif not pd.isna(prop_value):
                    has_any_label = True
                else:
                    all_valid = False
            else:
                all_valid = False
        
        if has_any_label:
            labeled_count += 1
            if all_valid:
                valid_count += 1
        else:
            unlabeled_count += 1
    
    total_count = len(datalist)
    validity_rate = valid_count / total_count if total_count > 0 else 0
    labeled_rate = labeled_count / total_count if total_count > 0 else 0
    
    print(f"✅ {split_name} final composition:")
    print(f"   Total: {total_count} molecules")
    print(f"   Fully labeled: {valid_count} ({validity_rate:.1%})")
    print(f"   Partially labeled: {labeled_count - valid_count}")  
    print(f"   Unlabeled: {unlabeled_count} ({unlabeled_count/total_count:.1%})")
    
    return valid_count

train_final_valid = final_validation_check(train_datalist, "Train")
val_final_valid = final_validation_check(val_datalist, "Validation") 
test_final_valid = final_validation_check(test_datalist, "Test")

# Warning if test set has very few valid labels for evaluation
if test_final_valid < 10:
    print(f"\n⚠️  WARNING: Test set has only {test_final_valid} fully labeled molecules for evaluation")
    print(f"   Consider using validation set for evaluation, or re-running with different seed")
elif test_final_valid < 50:
    print(f"\n💡 INFO: Test set has {test_final_valid} fully labeled molecules (sufficient for evaluation)")
else:
    print(f"\n✅ EXCELLENT: Test set has {test_final_valid} fully labeled molecules for robust evaluation")

num_node_features = train_datalist[0].num_node_features
num_edge_features = train_datalist[0].num_edge_features
print(f'Number of node features: {num_node_features}')
print(f'Number of edge features: {num_edge_features}')

# %%batch them
train_loader = DataLoader(dataset=train_datalist,
                          batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(dataset=val_datalist,
                         batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_datalist,
                         batch_size=batch_size, shuffle=False)

# check that it works, each batch has one big graph
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}', '\n')
    print(data)
    print()
    if step == 1:
        break

# %% Create Matrices needed for Message Passing

# %% Create dictionary with bathed graphs and message passing matrices for supervised train set

dict_train_loader = MP_Matrix_Creator(train_loader, device)
train_path = os.path.join(output_dir, f'dict_train_loader_{file_prefix}_{tokenization}_{property_suffix}.pt')
torch.save(dict_train_loader, train_path)
if os.path.exists(train_path):  # ← ADD VERIFICATION
    print(f"✅ Saved training loader: {os.path.basename(train_path)}")
else:
    raise IOError(f"Failed to save training loader")

dict_val_loader = MP_Matrix_Creator(val_loader, device)
val_path = os.path.join(output_dir, f'dict_val_loader_{file_prefix}_{tokenization}_{property_suffix}.pt')
torch.save(dict_val_loader, val_path)
if os.path.exists(val_path):  # ← ADD VERIFICATION
    print(f"✅ Saved validation loader: {os.path.basename(val_path)}")
else:
    raise IOError(f"Failed to save validation loader")

dict_test_loader = MP_Matrix_Creator(test_loader, device)
test_path = os.path.join(output_dir, f'dict_test_loader_{file_prefix}_{tokenization}_{property_suffix}.pt')
torch.save(dict_test_loader, test_path)
if os.path.exists(test_path):  # ← ADD VERIFICATION
    print(f"✅ Saved test loader: {os.path.basename(test_path)}")
else:
    raise IOError(f"Failed to save test loader")

# LONG-TERM FIX: Print summary of what to use for consistent training
print("\n" + "="*60)
print("🎯 VOCABULARY CONSISTENCY SUMMARY")
print("="*60)

if args.vocab_file:
    print(f"✅ Used external vocabulary: {args.vocab_file}")
    print(f"   All stages should use the SAME vocabulary file!")
else:
    print(f"⚠️  Created new vocabulary from data")
    print(f"   This may cause inconsistencies between stages!")
    print(f"\n💡 RECOMMENDATION FOR TRANSFER LEARNING:")
    print(f"   1. Create master vocabulary first:")
    print(f"      python Transform_Batch_Data.py --create_master_vocab --input_file your_data.csv")
    print(f"   2. Use it for all stages:")
    print(f"      python Transform_Batch_Data.py --vocab_file master_vocab_{augment}_{tokenization}.txt ...")

print(f"\n📁 Output files saved with:")
print(f'   Prefix: {file_prefix}')
print(f'   Property suffix: {property_suffix}')
print(f'   Vocabulary: {len(vocab)} tokens')
print('Done')
print(f"\n🎉 ROBUST DATA SPLITTING COMPLETED SUCCESSFULLY!")
print(f"✅ Evaluation sets guaranteed to have labeled data for performance measurement")
