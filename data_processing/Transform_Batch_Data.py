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

# Load input file based on command line argument or fall back to default paths
if args.input_file:
    # Use the custom input file path
    df = pd.read_csv(args.input_file)
    print(f"Loading custom dataset from: {args.input_file}")
    # Override the augment value to ensure correct file naming
    file_prefix = os.path.basename(args.input_file).split('.')[0]
    print(f"Using file prefix for outputs: {file_prefix}")
else:
    # Use the default paths based on augment value
    if augment == "original":
        df = pd.read_csv(os.path.join(main_dir_path, 'data', 'dataset-poly_chemprop.csv'))
        file_prefix = "original"
    elif augment == "augmented":
        df = pd.read_csv(os.path.join(main_dir_path, 'data', 'dataset-combined-poly_chemprop.csv'))
        file_prefix = "augmented"
    print(f"Loading default dataset based on augment={augment}")

# Verify that all property columns exist in the dataframe
missing_columns = [col for col in property_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing property columns in CSV: {missing_columns}")

# %% Lets create PyG data objects

# uncomment if graphs_list.pt does not exist
# Here we turn all smiles tring and featurize them into graphs and put them in a list: graphs_list
# additionally we add the target token ids of the target string as graph attributes 

Graphs_list = []
target_tokens_list = []
target_tokens_ids_list = []
target_tokens_lens_list = []
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    try: poly_input_nocan = df.loc[i, 'poly_chemprop_input_nocan']
    except: poly_input_nocan=None
    
    # Extract property values dynamically based on property_columns
    property_values = []
    for prop_col in property_columns:
        prop_value = df.loc[i, prop_col]
        property_values.append(prop_value)
    
    # Create graph with flexible property values using the updated Function_Featurization_Own.py
    # Import the flexible function if not already imported
    from data_processing.Function_Featurization_Own import poly_smiles_to_graph_flexible
    
    # Use the flexible function for all cases
    graphs = poly_smiles_to_graph_flexible(poly_input, property_values, poly_input_nocan)
    
    #if string_format == "gbigsmileslike":
    #    poly_input_gbigsmileslike = df.loc[i, 'poly_chemprop_input_GbigSMILESlike']
    #    target_tokens = tokenize_poly_input_new(poly_input=poly_input_gbigsmileslike, tokenization=tokenization)
    #elif string_format=="poly_chemprop":
    if tokenization=="oldtok":
        target_tokens = tokenize_poly_input(poly_input=poly_input)
    elif tokenization=="RT_tokenized":
        target_tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
    Graphs_list.append(graphs)
    target_tokens_list.append(target_tokens)
    if i % 100 == 0:
        print(f"[{i} / {len(df.loc[:, 'poly_chemprop_input'])}]")

# Create flexible file names that include property information
property_suffix = "_".join(property_names)
vocab_filename = f'poly_smiles_vocab_{file_prefix}_{tokenization}_{property_suffix}.txt'
graphs_filename = f'Graphs_list_{file_prefix}_{tokenization}_{property_suffix}.pt'

# Create vocab file with property suffix
make_vocab(target_tokens_list=target_tokens_list, vocab_file=os.path.join(output_dir, vocab_filename))

# convert the target_tokens_list to target_token_ids_list using the vocab file
# load vocab dict (token:id)
vocab = load_vocab(vocab_file=os.path.join(output_dir, vocab_filename))
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
torch.save(Graphs_list, os.path.join(output_dir, graphs_filename))

# Create training, self supervised and test sets

# shuffle graphs
# we first take out the validation and test set from 
random.seed(12345)
if not args.input_file and augment == "original":
    # new improved test set: exclude monomer combinations completely
    mon_combs = []
    monB_list = []
    stoichiometry_connectivity_combs = []
    for i in range(len(df.loc[:, 'poly_chemprop_input'])):
        poly_input = df.loc[i, 'poly_chemprop_input']
        
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

    # print some statistics
    print(f'Number of training graphs: {len(train_datalist)}')
    print(f'Number of validation graphs: {len(val_datalist)}')
    print(f'Number of test graphs: {len(test_datalist)}')

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

    # print some statistics
    print(f'Number of training graphs: {len(train_datalist)}')
    print(f'Number of validation graphs: {len(val_datalist)}')
    print(f'Number of test graphs: {len(test_datalist)}')

else:
    # For custom input files, use a standard split
    print("Using standard train/validation/test split for custom input file")
    # Extract monomer combinations to ensure similar monomers don't appear in both train and test
    mon_combs = []
    for i in range(len(df.loc[:, 'poly_chemprop_input'])):
        poly_input = df.loc[i, 'poly_chemprop_input']
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
    
    # Print statistics
    print(f'Number of training graphs: {len(train_datalist)}')
    print(f'Number of validation graphs: {len(val_datalist)}')
    print(f'Number of test graphs: {len(test_datalist)}')

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

# Save with property suffix for consistency
dict_train_loader = MP_Matrix_Creator(train_loader, device)
torch.save(dict_train_loader, os.path.join(output_dir, f'dict_train_loader_{file_prefix}_{tokenization}_{property_suffix}.pt'))
dict_val_loader = MP_Matrix_Creator(val_loader, device)
torch.save(dict_val_loader, os.path.join(output_dir, f'dict_val_loader_{file_prefix}_{tokenization}_{property_suffix}.pt'))
dict_test_loader = MP_Matrix_Creator(test_loader, device)
torch.save(dict_test_loader, os.path.join(output_dir, f'dict_test_loader_{file_prefix}_{tokenization}_{property_suffix}.pt'))

print('Done')
print(f'Saved data files with prefix: {file_prefix} and property suffix: {property_suffix}')
print(f'All files saved to: {output_dir}')
