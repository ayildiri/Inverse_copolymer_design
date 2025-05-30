# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from model.G2S_clean import *

import time
from datetime import datetime
import random
# deep learning packages
import torch
from statistics import mean
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import umap
from mpl_toolkits.mplot3d import Axes3D
import argparse


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--alpha", default="fixed", choices=["fixed","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initialization", default="random", choices=["random"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.1)
parser.add_argument("--max_alpha", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--save_dir", type=str, required=True, help="Path to load model results from and save plots to")
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Path to custom dataset files (will use default naming pattern if not specified)")
parser.add_argument("--dim_red_type", default="pca", choices=["umap", "pca", "tsne"])

# Add flexible property arguments
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties to visualize")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")

args = parser.parse_args()

# Handle property configuration
property_names = args.property_names
if args.property_count is not None:
    property_count = args.property_count
else:
    property_count = len(property_names)

# Validate that property count matches property names
if len(property_names) != property_count:
    raise ValueError(f"Number of property names ({len(property_names)}) must match property count ({property_count})")

print(f"Creating plots for {property_count} properties: {property_names}")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
data_augment = "old"  # Fixed value for consistency

if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

# Handle vocabulary path flexibility
if args.dataset_path:
    # Use custom dataset path
    vocab_file_path = os.path.join(args.dataset_path, f'poly_smiles_vocab_{augment}_{tokenization}.txt')
else:
    # Use default path
    vocab_file_path = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'

print(f"Loading vocabulary from: {vocab_file_path}")
vocab = load_vocab(vocab_file=vocab_file_path)

# Include property info in model name for consistency
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+f'{data_augment}data_DecL={args.dec_layers}_beta={args.beta}_alpha={args.alpha}_maxbeta={args.max_beta}_maxalpha={args.max_alpha}eps={args.epsilon}_loss={args.loss}_augment={args.augment}_tokenization={args.tokenization}_AE_warmup={args.AE_Warmup}_init={args.initialization}_seed={args.seed}_add_latent={add_latent}_pp-guided={args.ppguided}_props={property_str}/'

# Determine save directory
dir_name = os.path.join(args.save_dir, model_name)

print(f"Model directory: {dir_name}")
print(f"Dimensionality reduction method: {args.dim_red_type}")
print(f"Properties to visualize: {property_names}")

if not os.path.exists(dir_name):
    print(f"Error: Model directory does not exist: {dir_name}")
    print("Available directories:")
    parent_dir = os.path.dirname(dir_name)
    if os.path.exists(parent_dir):
        for item in os.listdir(parent_dir):
            if os.path.isdir(os.path.join(parent_dir, item)):
                print(f"  {item}")
    exit(1)

def load_property_data(dataset_type, dir_name, property_count):
    """Load property data dynamically based on property count."""
    print(f"\n📊 Loading {dataset_type} dataset...")
    
    # Load latent space
    latent_file = os.path.join(dir_name, f'latent_space_{dataset_type}.npy')
    if not os.path.exists(latent_file):
        print(f"❌ Error: Latent space file not found: {latent_file}")
        return None, None, None, None, None, None
        
    with open(latent_file, 'rb') as f:
        latent_space = np.load(f)
        print(f"✅ Loaded latent space: {latent_space.shape}")

    # Load real property values (y1_all, y2_all, ...)
    properties_real = []
    for i in range(property_count):
        y_file = os.path.join(dir_name, f'y{i+1}_all_{dataset_type}.npy')
        if os.path.exists(y_file):
            with open(y_file, 'rb') as f:
                prop_data = np.load(f)
                properties_real.append(prop_data)
            print(f"✅ Loaded {property_names[i]} real values: {prop_data.shape}")
        else:
            print(f"⚠️ Warning: Real property file not found: {y_file}")
            properties_real.append(np.array([]))

    # Load predicted property values (yp_all)
    yp_file = os.path.join(dir_name, f'yp_all_{dataset_type}.npy')
    if os.path.exists(yp_file):
        with open(yp_file, 'rb') as f:
            yp_all = np.load(f)
        print(f"✅ Loaded predicted values: {yp_all.shape}")
        
        # Extract individual property predictions
        properties_pred = []
        for i in range(property_count):
            if yp_all.shape[1] > i:
                properties_pred.append(yp_all[:, i])  # Extract column i for property i
            else:
                print(f"⚠️ Warning: Not enough properties in predicted data for {property_names[i]}")
                properties_pred.append(np.array([]))
    else:
        print(f"⚠️ Warning: Predicted property file not found: {yp_file}")
        properties_pred = [np.array([]) for _ in range(property_count)]

    # Load other data with error handling
    try:
        with open(os.path.join(dir_name, f'monomers_{dataset_type}'), "rb") as f:
            monomers = pickle.load(f)
        print(f"✅ Loaded monomers data")
    except FileNotFoundError:
        print(f"⚠️ Warning: Monomers file not found")
        monomers = []
        
    try:
        with open(os.path.join(dir_name, f'stoichiometry_{dataset_type}'), "rb") as f:
            stoichiometry = pickle.load(f)
        print(f"✅ Loaded stoichiometry data")
    except FileNotFoundError:
        print(f"⚠️ Warning: Stoichiometry file not found")
        stoichiometry = []
        
    try:
        with open(os.path.join(dir_name, f'connectivity_{dataset_type}'), "rb") as f:
            connectivity_pattern = pickle.load(f)
        print(f"✅ Loaded connectivity data")
    except FileNotFoundError:
        print(f"⚠️ Warning: Connectivity file not found")
        connectivity_pattern = []

    return latent_space, properties_real, properties_pred, monomers, stoichiometry, connectivity_pattern

def perform_dimensionality_reduction(latent_space, dim_red_type, dataset_type, dir_name):
    """Perform dimensionality reduction and save/load reducer."""
    print(f"🔍 Performing {dim_red_type.upper()} dimensionality reduction...")
    
    if dim_red_type == "pca":
        if dataset_type == "train":
            reducer = PCA(n_components=2).fit(latent_space)
            reducer_file = os.path.join(dir_name, f'{dim_red_type}_fitted_{dataset_type}')
            with open(reducer_file, 'wb') as f:
                pickle.dump(reducer, f)
            z_embedded = reducer.transform(latent_space)
            print(f"✅ Fitted PCA and saved to {reducer_file}")
        else:
            reducer_file = os.path.join(dir_name, f'{dim_red_type}_fitted_train')
            try:
                with open(reducer_file, 'rb') as f:
                    reducer = pickle.load(f)
                z_embedded = reducer.transform(latent_space)
                print(f"✅ Loaded PCA from {reducer_file}")
            except FileNotFoundError:
                print(f"❌ Error: PCA reducer not found at {reducer_file}")
                print("You need to run with train dataset first!")
                return None
    
    elif dim_red_type == "umap":
        if dataset_type == "train":
            reducer = umap.UMAP(n_components=2, min_dist=0.5).fit(latent_space)
            reducer_file = os.path.join(dir_name, f'umap_fitted_train_{dataset_type}')
            with open(reducer_file, 'wb') as f:
                pickle.dump(reducer, f)
            z_embedded = reducer.embedding_
            print(f"✅ Fitted UMAP and saved to {reducer_file}")
        else:
            reducer_file = os.path.join(dir_name, f'umap_fitted_train_train')
            try:
                with open(reducer_file, 'rb') as f:
                    reducer = pickle.load(f)
                z_embedded = reducer.transform(latent_space)
                print(f"✅ Loaded UMAP from {reducer_file}")
            except FileNotFoundError:
                print(f"❌ Error: UMAP reducer not found at {reducer_file}")
                print("You need to run with train dataset first!")
                return None
    
    elif dim_red_type == "tsne":
        if dataset_type == "train":
            reducer = TSNE(n_components=2, random_state=42)
            z_embedded = reducer.fit_transform(latent_space)
            # Note: t-SNE doesn't have a transform method, so we save the embedding directly
            embedding_file = os.path.join(dir_name, f'{dim_red_type}_embedding_{dataset_type}.npy')
            np.save(embedding_file, z_embedded)
            print(f"✅ Computed t-SNE and saved embedding to {embedding_file}")
        else:
            # For test set, we need to recompute t-SNE (it's not transformable)
            print("⚠️ Warning: t-SNE doesn't support transform - computing fresh embedding for test set")
            reducer = TSNE(n_components=2, random_state=42)
            z_embedded = reducer.fit_transform(latent_space)
    
    return z_embedded

def create_property_plots(z_embedded, properties_real, properties_pred, dataset_type, dim_red_type, dir_name, figure_offset=0):
    """Create plots for all properties (both real and predicted)."""
    plt.rcParams.update({'font.size': 18})
    
    print(f"🎨 Creating property plots for {dataset_type} dataset...")
    
    # Create plots for real property values
    for i, (prop_name, prop_values) in enumerate(zip(property_names, properties_real)):
        if len(prop_values) == 0:
            print(f"⚠️ Skipping real property plot for {prop_name} - no data")
            continue
            
        fig_num = figure_offset + i * 2
        plt.figure(fig_num, figsize=(10, 8))
        scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=prop_values, cmap='viridis')
        clb = plt.colorbar(scatter)
        clb.ax.set_title(f'{prop_name} (Real)')
        plt.xlabel(f"{dim_red_type.upper()} 1")
        plt.ylabel(f"{dim_red_type.upper()} 2")
        plt.title(f'{dataset_type.capitalize()} - {prop_name} Real Values')
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        save_path = os.path.join(dir_name, f'{dataset_type}_latent_{prop_name}_real_{dim_red_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    
    # Create plots for predicted property values
    for i, (prop_name, prop_values) in enumerate(zip(property_names, properties_pred)):
        if len(prop_values) == 0:
            print(f"⚠️ Skipping predicted property plot for {prop_name} - no data")
            continue
            
        fig_num = figure_offset + i * 2 + 1
        plt.figure(fig_num, figsize=(10, 8))
        scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=prop_values, cmap='viridis')
        clb = plt.colorbar(scatter)
        clb.ax.set_title(f'{prop_name} (Predicted)')
        plt.xlabel(f"{dim_red_type.upper()} 1")
        plt.ylabel(f"{dim_red_type.upper()} 2")
        plt.title(f'{dataset_type.capitalize()} - {prop_name} Predicted Values')
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        save_path = os.path.join(dir_name, f'{dataset_type}_latent_{prop_name}_pred_{dim_red_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

def create_monomer_plots(z_embedded, monomers, dataset_type, dim_red_type, dir_name, figure_offset):
    """Create monomer type plots."""
    if not monomers:
        print(f"⚠️ Skipping monomer plots - no monomer data")
        return
        
    print(f"🧬 Creating monomer plots...")
    
    # Get all the A monomers and create monomer label list samples x 1
    monomersA = {}
    sample_idx = 0 
    for b in monomers:
        for sample in b: 
            monomersA[sample_idx] = sample[0]
            sample_idx += 1

    unique_A_monomers = list(set(monomersA.values()))
    print(f"Unique A monomers: {len(unique_A_monomers)}")

    # Color information; create custom colormap
    amons = ['[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2','[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12','[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2','[*:1]c1cc(F)c([*:2])cc1F','[*:1]c1ccc([*:2])cc1','[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34', '[*:1]c1ccc(-c2ccc([*:2])s2)s1','[*:1]c1cc2cc3sc([*:2])cc3cc2s1','[*:1]c1ccc([*:2])c2nsnc12', 'no_A_monomer']
    cols = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc',"#6574cd" ,'#9561e2','#f66d9b','#808080']
    
    # Adjust for dataset_type
    if dataset_type == 'test':
        label_color_dict = {amons[i]:cols[i] for i in range(len(amons)-1)}
    else:
        label_color_dict = {amons[i]:cols[i] for i in range(len(amons))}
    
    all_labels = list(label_color_dict.keys())
    all_colors = list(label_color_dict.values())
    n_colors = len(all_colors)
    cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)

    # Get indices from color list for given labels
    def assign_color(mon):
        try: 
            alpha = 1.0
            color = all_colors.index(label_color_dict[mon]) 
            return color, alpha
        except:
            alpha = 0.1
            color = all_colors.index(label_color_dict['no_A_monomer']) if 'no_A_monomer' in label_color_dict else 0
            return color, alpha

    color_idx = [assign_color(monomerA)[0] for key, monomerA in monomersA.items()]
    alphas = [assign_color(monomerA)[1] for key, monomerA in monomersA.items()]

    # Create plot
    plt.figure(figure_offset, figsize=(12, 8))
    sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=2, c=color_idx, cmap=cm, alpha=alphas)
    c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (2 / n_colors)
    cbar = plt.colorbar(sc, ticks=c_ticks)
    cbar.ax.set_yticklabels(all_labels)
    cbar.ax.set_title('Monomer A type')
    plt.xlabel(f"{dim_red_type.upper()} 1")
    plt.ylabel(f"{dim_red_type.upper()} 2")
    plt.title(f'{dataset_type.capitalize()} - Monomer A Types')
    save_path = os.path.join(dir_name, f'{dataset_type}_latent_Amonomers_{dim_red_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def create_stoichiometry_plots(z_embedded, stoichiometry, dataset_type, dim_red_type, dir_name, figure_offset):
    """Create stoichiometry plots."""
    if not stoichiometry:
        print(f"⚠️ Skipping stoichiometry plots - no stoichiometry data")
        return
        
    print(f"⚖️ Creating stoichiometry plots...")
    
    label_color_dict = {'0.5|0.5': '#e3342f',
                        '0.25|0.75': '#f6993f',
                        '0.75|0.25': '#ffed4a'}
    all_labels = list(label_color_dict.keys())
    all_colors = list(label_color_dict.values())
    n_colors = len(all_colors)
    cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)
    labels = {'0.5|0.5':'1:1','0.25|0.75':'1:3','0.75|0.25':'3:1'}
    all_labels = [labels[x] for x in all_labels]

    # Get indices from color list for given labels
    color_idx = [all_colors.index(label_color_dict[st]) for st in stoichiometry]

    plt.figure(figure_offset, figsize=(10, 8))
    sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=2, c=color_idx, cmap=cm)
    c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (1 / (n_colors+1))
    cbar = plt.colorbar(sc, ticks=c_ticks)
    cbar.ax.set_yticklabels(all_labels)
    cbar.ax.set_title('Stoichiometric ratio')
    plt.xlabel(f"{dim_red_type.upper()} 1")
    plt.ylabel(f"{dim_red_type.upper()} 2")
    plt.title(f'{dataset_type.capitalize()} - Stoichiometry')
    save_path = os.path.join(dir_name, f'{dataset_type}_latent_stoichiometry_{dim_red_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def create_connectivity_plots(z_embedded, connectivity_pattern, dataset_type, dim_red_type, dir_name, figure_offset):
    """Create connectivity plots."""
    if not connectivity_pattern:
        print(f"⚠️ Skipping connectivity plots - no connectivity data")
        return
        
    print(f"🔗 Creating connectivity plots...")
    
    label_color_dict = {'0.5': '#e3342f',
                        '0.375': '#f6993f',
                        '0.25': '#ffed4a'}
    all_labels = list(label_color_dict.keys())
    all_colors = list(label_color_dict.values())
    n_colors = len(all_colors)
    cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)
    labels = {'0.5':'Alternating','0.25':'Random','0.375':'Block'}
    all_labels = [labels[x] for x in all_labels]

    # Get indices from color list for given labels
    color_idx = [all_colors.index(label_color_dict[st]) for st in connectivity_pattern]

    plt.figure(figure_offset, figsize=(10, 8))
    sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=2, c=color_idx, cmap=cm)
    c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (1 / (n_colors+1))
    cbar = plt.colorbar(sc, ticks=c_ticks)
    cbar.ax.set_yticklabels(all_labels)
    cbar.ax.set_title('Chain architecture')
    plt.xlabel(f"{dim_red_type.upper()} 1")
    plt.ylabel(f"{dim_red_type.upper()} 2")
    plt.title(f'{dataset_type.capitalize()} - Connectivity')
    save_path = os.path.join(dir_name, f'{dataset_type}_latent_connectivity_{dim_red_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def process_dataset(dataset_type, dir_name, dim_red_type, figure_offset=0):
    """Process one dataset (train or test)."""
    print(f'\n{"="*60}')
    print(f'PROCESSING {dataset_type.upper()} DATASET')
    print(f'{"="*60}')
    
    # Load data
    result = load_property_data(dataset_type, dir_name, property_count)
    if result[0] is None:  # Check if loading failed
        print(f"❌ Failed to load {dataset_type} data")
        return figure_offset
        
    latent_space, properties_real, properties_pred, monomers, stoichiometry, connectivity_pattern = result
    
    # Perform dimensionality reduction
    z_embedded = perform_dimensionality_reduction(latent_space, dim_red_type, dataset_type, dir_name)
    if z_embedded is None:
        print(f"❌ Failed dimensionality reduction for {dataset_type}")
        return figure_offset
        
    print(f"✅ Dimensionality reduction completed: {z_embedded.shape}")
    
    # Create property plots
    prop_figure_offset = figure_offset
    create_property_plots(z_embedded, properties_real, properties_pred, dataset_type, dim_red_type, dir_name, prop_figure_offset)
    
    # Create structural plots
    struct_figure_offset = figure_offset + property_count * 2
    create_monomer_plots(z_embedded, monomers, dataset_type, dim_red_type, dir_name, struct_figure_offset)
    create_stoichiometry_plots(z_embedded, stoichiometry, dataset_type, dim_red_type, dir_name, struct_figure_offset + 1)
    create_connectivity_plots(z_embedded, connectivity_pattern, dataset_type, dim_red_type, dir_name, struct_figure_offset + 2)
    
    print(f"🎨 Completed plotting for {dataset_type} dataset")
    return struct_figure_offset + 3

# Main execution
print(f"🎯 STARTING LATENT SPACE VISUALIZATION")
print(f"Model directory: {dir_name}")
print(f"Dimensionality reduction: {args.dim_red_type}")
print(f"Properties to visualize: {property_names}")

# Process train dataset
figure_offset = process_dataset("train", dir_name, args.dim_red_type, 0)

# Process test dataset
print('\n🔄 Now processing test set!')
figure_offset = process_dataset("test", dir_name, args.dim_red_type, figure_offset)

print('\n🎉 VISUALIZATION COMPLETED!')
print(f"📊 All plots saved to: {dir_name}")
print(f"📈 Created visualizations for {property_count} properties: {property_names}")
print(f"🗂️ Saved plots for both train and test datasets")
print(f"🎨 Used {args.dim_red_type.upper()} for dimensionality reduction")
