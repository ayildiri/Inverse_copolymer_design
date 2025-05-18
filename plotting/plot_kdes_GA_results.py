import pickle

import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
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

# Add flexible property arguments
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties used in the model")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
parser.add_argument("--objective_type", type=str, default="EAmin",
                    choices=["EAmin", "mimick_peak", "mimick_best", "max_gap", "custom"],
                    help="Type of objective function used in GA optimization")

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

print(f"Plotting results for {property_count} properties: {property_names}")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "train"
data_augment = "old" # new or old

# Load training data
dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

# Create model name with property information
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+property_str+'/'

filepath = os.path.join(main_dir_path,'Checkpoints/', model_name,"model_best_loss.pt")
dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)

if not os.path.exists(dir_name):
    print(f"Error: Model directory does not exist: {dir_name}")
    exit(1)

# Load GA optimization results
objective_type = args.objective_type
results_file = f'optimization_results_custom_GA_correct_{objective_type}.pkl'
results_path = os.path.join(dir_name, results_file)

if not os.path.exists(results_path):
    print(f"Error: GA results file does not exist: {results_path}")
    exit(1)

with open(results_path, 'rb') as f:
    results_custom = pickle.load(f)

# Extract data from GA results
Latents_RE = []
pred_RE = []
decoded_mols = []
pred_RE_corrected = []

for idx, (pop, res) in enumerate(list(results_custom.items())):
    population = int(pop)
    # loop through population
    pop_size = len(list(res["objective"]))
    for point in range(pop_size):
        L_re = res["latents_reencoded"][point]
        Latents_RE.append(L_re)
        pred_RE.append(res["predictions"][point])
        pred_RE_corrected.append(res["predictions_doublecorrect"][point])
        decoded_mols.append(res["string_decoded"][point])

# Extract properties dynamically based on property count
property_data_RE = {}
property_data_RE_corrected = {}

for i in range(property_count):
    prop_name = property_names[i]
    property_data_RE[prop_name] = [arr[i].cpu() if hasattr(arr[i], 'cpu') else arr[i] for arr in pred_RE]
    property_data_RE_corrected[prop_name] = [arr[i] for arr in pred_RE_corrected]

# Legacy support - keep EA_re and IP_re for backward compatibility with existing objective functions
if "EA" in property_names and "IP" in property_names:
    EA_re = property_data_RE["EA"]
    IP_re = property_data_RE["IP"]
    EA_re_c = property_data_RE_corrected["EA"]
    IP_re_c = property_data_RE_corrected["IP"]

def top_n_molecule_indices(objective_values, decoded_mols, n_idx=10):
    """Get the indices of n molecules with the best objective values."""
    # Filter out NaN values and keep track of original indices
    filtered_indexed_values = [(index, value) for index, value in enumerate(objective_values) if not math.isnan(value)]
    # Sort the indexed values by the value in ascending order and take n_idx best ones
    sorted_filtered_indexed_values = sorted(filtered_indexed_values, key=lambda x: x[1], reverse=False)
    _best_mols = []
    best_mols_count = {}
    top_idxs = []
    for index, value in sorted_filtered_indexed_values: 
        if not decoded_mols[index] in _best_mols: 
            top_idxs.append(index)
            best_mols_count[decoded_mols[index]] = 1
            _best_mols.append(decoded_mols[index])
        else:
            best_mols_count[decoded_mols[index]] += 1
        if len(top_idxs) == min(n_idx, 20):  # Limit to max 20 for consistency
            break

    return top_idxs, best_mols_count

def calculate_objective_values(objective_type, pred_RE, pred_RE_corrected):
    """Calculate objective values based on objective type and properties."""
    if objective_type == 'mimick_peak' and "EA" in property_names and "IP" in property_names:
        ea_idx, ip_idx = property_names.index("EA"), property_names.index("IP")
        objective_values = [(np.abs(arr.cpu()[ea_idx]+2)+np.abs(arr.cpu()[ip_idx]-1.2)) for arr in pred_RE]
        objective_values_c = [(np.abs(arr[ea_idx]+2)+np.abs(arr[ip_idx]-1.2)) for arr in pred_RE_corrected]
    elif objective_type == 'mimick_best' and "EA" in property_names and "IP" in property_names:
        ea_idx, ip_idx = property_names.index("EA"), property_names.index("IP")
        objective_values = [(np.abs(arr.cpu()[ea_idx]+2.64)+np.abs(arr.cpu()[ip_idx]-1.61)) for arr in pred_RE]
        objective_values_c = [(np.abs(arr[ea_idx]+2.64)+np.abs(arr[ip_idx]-1.61)) for arr in pred_RE_corrected]
    elif objective_type == 'EAmin' and "EA" in property_names and "IP" in property_names:
        ea_idx, ip_idx = property_names.index("EA"), property_names.index("IP")
        objective_values = [arr.cpu()[ea_idx]+np.abs(arr.cpu()[ip_idx]-1) for arr in pred_RE]
        objective_values_c = [arr[ea_idx]+np.abs(arr[ip_idx]-1) for arr in pred_RE_corrected]
    elif objective_type == 'max_gap' and "EA" in property_names and "IP" in property_names:
        ea_idx, ip_idx = property_names.index("EA"), property_names.index("IP")
        objective_values = [arr.cpu()[ea_idx]-arr.cpu()[ip_idx] for arr in pred_RE]
        objective_values_c = [arr[ea_idx]-arr[ip_idx] for arr in pred_RE_corrected]
    else:
        # For custom objectives or when properties don't match legacy objectives,
        # calculate a simple sum of all properties as placeholder
        print(f"Warning: Objective type '{objective_type}' not supported for properties {property_names}")
        print("Using sum of all properties as default objective")
        objective_values = [sum(arr.cpu() if hasattr(arr.cpu(), '__iter__') else [arr.cpu()]) for arr in pred_RE]
        objective_values_c = [sum(arr if hasattr(arr, '__iter__') else [arr]) for arr in pred_RE_corrected]
    
    return objective_values, objective_values_c

# Calculate objective values
objective_values, objective_values_c = calculate_objective_values(objective_type, pred_RE, pred_RE_corrected)

# Get top molecules
top_20_indices, top_20_mols = top_n_molecule_indices(objective_values, decoded_mols, n_idx=500)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}
best_objs_t20 = {i+1: objective_values[i] for i in top_20_indices}

# Extract best property values for all properties
best_property_values = {}
for i, prop_name in enumerate(property_names):
    best_property_values[prop_name] = [property_data_RE[prop_name][idx] for idx in top_20_indices]

from sklearn.neighbors import KernelDensity

# Load properties from dataset (real and augmented predictions)
property_files = {}
for i, prop_name in enumerate(property_names):
    y_file = os.path.join(dir_name, f'y{i+1}_all_{dataset_type}.npy')
    if os.path.exists(y_file):
        with open(y_file, 'rb') as f:
            property_files[f'y{i+1}_all'] = np.load(f)
    else:
        print(f"Warning: Property file {y_file} not found")

# Load predicted properties
yp_file = os.path.join(dir_name, f'yp_all_{dataset_type}.npy')
if os.path.exists(yp_file):
    with open(yp_file, 'rb') as f:
        yp_all = np.load(f)
else:
    print(f"Warning: Predicted properties file {yp_file} not found")
    yp_all = []

# Create KDE plots for each property
def create_kde_plot(prop_idx, prop_name, save_name):
    """Create and save KDE plot for a specific property."""
    plt.figure(figsize=(10, 8))
    
    # Get data for this property
    real_key = f'y{prop_idx+1}_all'
    if real_key in property_files:
        real_data = [r for r in property_files[real_key] if not np.isnan(r)]
    else:
        real_data = []
    
    if len(yp_all) > 0 and yp_all.shape[1] > prop_idx:
        augmented_data = [p[prop_idx] for p in yp_all]
    else:
        augmented_data = []
    
    ga_data = best_property_values[prop_name] if prop_name in best_property_values else []
    
    # Check if we have enough data
    if len(real_data) == 0 and len(augmented_data) == 0 and len(ga_data) == 0:
        print(f"Warning: No data available for property {prop_name}, skipping plot")
        return
    
    # Convert to numpy arrays and reshape
    distributions = {}
    if len(real_data) > 0:
        distributions['Real Data'] = np.array(real_data).reshape(-1, 1)
    if len(augmented_data) > 0:
        distributions['Augmented Data'] = np.array(augmented_data).reshape(-1, 1)
    if len(ga_data) > 0:
        distributions['Best 500 molecules GA'] = np.array(ga_data).reshape(-1, 1)
    
    # Define bandwidth
    bandwidth = 0.1
    
    # Create x-axis range
    all_values = []
    for data in distributions.values():
        all_values.extend(data.flatten())
    
    if len(all_values) == 0:
        print(f"Warning: No valid data for property {prop_name}")
        return
    
    padding = 0.5
    x_min = min(all_values) - padding
    x_max = max(all_values) + padding
    x_values = np.linspace(x_min, x_max, 1000)
    
    # Set font parameters
    plt.rcParams.update({'font.size': 18, 'fontname':'Droid Sans'})
    
    # Create KDE and plot for each distribution
    colors = ['blue', 'orange', 'green']
    for i, (label, data) in enumerate(distributions.items()):
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(data)
        density = np.exp(kde.score_samples(x_values.reshape(-1, 1)))
        
        plt.plot(x_values, density, label=label, color=colors[i % len(colors)])
        plt.fill_between(x_values, density, alpha=0.5, color=colors[i % len(colors)])
    
    # Customize plot
    plt.xlabel(f'{prop_name}')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(dir_name, f'KDE_{save_name}_GA.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved KDE plot for {prop_name}: {save_path}")
    
    plt.close()  # Close to free memory

# Create KDE plots for all properties
for i, prop_name in enumerate(property_names):
    create_kde_plot(i, prop_name, prop_name)

# Create histogram for the first property (for backward compatibility)
if len(property_names) > 0:
    prop_name = property_names[0]
    prop_idx = 0
    
    plt.figure(figsize=(10, 6))
    
    # Get data
    real_key = f'y{prop_idx+1}_all'
    if real_key in property_files:
        real_distribution = np.array([r for r in property_files[real_key] if not np.isnan(r)])
    else:
        real_distribution = np.array([])
    
    if len(yp_all) > 0 and yp_all.shape[1] > prop_idx:
        augmented_distribution = np.array([p[prop_idx] for p in yp_all])
    else:
        augmented_distribution = np.array([])
    
    GA_distribution = np.array(best_property_values[prop_name] if prop_name in best_property_values else [])
    
    # Create histograms
    if len(real_distribution) > 0:
        plt.hist(real_distribution, bins=30, alpha=0.5, label='Real Data', density=True, edgecolor='k')
    if len(augmented_distribution) > 0:
        plt.hist(augmented_distribution, bins=30, alpha=0.5, label='Augmented Data', density=True, edgecolor='k')
    if len(GA_distribution) > 0:
        plt.hist(GA_distribution, bins=30, alpha=0.5, label='Best 500 molecules GA', density=True, edgecolor='k')
    
    plt.xlabel(f'{prop_name}')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(dir_name, f'Histogram_{prop_name}_GA.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved histogram for {prop_name}: {save_path}")
    
    plt.close()

print(f"Plotting completed for {property_count} properties: {property_names}")
print(f"All plots saved to: {dir_name}")

# Save summary of best molecules
summary_file = os.path.join(dir_name, f'best_molecules_summary_{objective_type}.txt')
with open(summary_file, 'w') as f:
    f.write(f"Best molecules for objective type: {objective_type}\n")
    f.write(f"Properties: {property_names}\n")
    f.write(f"Top {len(top_20_indices)} molecules:\n\n")
    
    for i, idx in enumerate(top_20_indices):
        f.write(f"Rank {i+1}:\n")
        f.write(f"  Molecule: {decoded_mols[idx]}\n")
        f.write(f"  Objective value: {objective_values[idx]:.4f}\n")
        f.write("  Properties:\n")
        for prop_name in property_names:
            prop_value = property_data_RE[prop_name][idx]
            f.write(f"    {prop_name}: {prop_value:.4f}\n")
        f.write("\n")

print(f"Saved best molecules summary: {summary_file}")
