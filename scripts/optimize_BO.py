import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import numpy as np
from optimization_custom.bayesian_optimization import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt.event import DEFAULT_EVENTS, Events
from sklearn.neighbors import KernelDensity

import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import pickle
import json
import pandas as pd
import math
import glob
from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from data_processing.Smiles_enum_canon import SmilesEnumCanon

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.rdkit_poly import make_polymer_mol

import time
from datetime import datetime
import shutil  # for backup operations
import re

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

def detect_polymer_type(smiles_part):
    """
    Detect if this is a homopolymer or copolymer
    """
    monomers = smiles_part.split(".")
    if len(monomers) == 1:
        return "homopolymer", monomers
    elif len(monomers) == 2:
        # Check if both monomers are identical (homopolymer written as A.A)
        if monomers[0] == monomers[1]:
            return "homopolymer", [monomers[0]]
        else:
            return "copolymer", monomers
    else:
        return "multipolymer", monomers

def intelligent_rgroup_mapping(smiles_part, connectivity_part):
    """
    Intelligently map R-groups based on polymer type and connectivity patterns
    """
    import re
    
    # Extract R-groups from SMILES and connectivity
    r_groups_in_smiles = set(re.findall(r'\[\*:(\d+)\]', smiles_part))
    connectivity_pairs = re.findall(r'(\d+)-(\d+):', connectivity_part)
    
    # Get all R-groups mentioned in connectivity
    conn_r_groups = set()
    for pair in connectivity_pairs:
        conn_r_groups.update(pair)
    
    polymer_type, monomers = detect_polymer_type(smiles_part)
    
    print(f"Polymer type: {polymer_type}")
    print(f"R-groups in SMILES: {r_groups_in_smiles}")
    print(f"R-groups in connectivity: {conn_r_groups}")
    
    # If R-groups already match, no need to change
    if r_groups_in_smiles == conn_r_groups:
        return smiles_part, connectivity_part
    
    new_smiles = smiles_part
    new_connectivity = connectivity_part
    
    if polymer_type == "homopolymer":
        # For homopolymers, map connectivity R-groups to available SMILES R-groups
        available_r_groups = sorted(r_groups_in_smiles, key=int)
        needed_r_groups = sorted(conn_r_groups, key=int)
        
        # Create mapping: connectivity R-groups → SMILES R-groups
        mapping = {}
        for i, needed in enumerate(needed_r_groups):
            if i < len(available_r_groups):
                mapping[needed] = available_r_groups[i]
            else:
                # If we need more R-groups than available, cycle through
                mapping[needed] = available_r_groups[i % len(available_r_groups)]
        
        # Apply mapping to connectivity
        for old, new in mapping.items():
            new_connectivity = new_connectivity.replace(f'{old}-', f'{new}-')
            new_connectivity = new_connectivity.replace(f'-{old}:', f'-{new}:')
            
        print(f"Homopolymer mapping: {mapping}")
        
    elif polymer_type == "copolymer":
        # For copolymers, we expect connectivity to reference higher numbered R-groups
        # Map them to available R-groups in SMILES
        available_r_groups = sorted(r_groups_in_smiles, key=int)
        needed_r_groups = sorted(set(conn_r_groups), key=int)
        
        if len(available_r_groups) >= 2 and len(needed_r_groups) >= 2:
            # Standard copolymer mapping
            mapping = {}
            for i, needed in enumerate(needed_r_groups):
                if i < len(available_r_groups):
                    mapping[needed] = available_r_groups[i]
                else:
                    # Fallback to cycling
                    mapping[needed] = available_r_groups[i % len(available_r_groups)]
            
            # Apply mapping to connectivity
            for old, new in mapping.items():
                new_connectivity = new_connectivity.replace(f'{old}-', f'{new}-')
                new_connectivity = new_connectivity.replace(f'-{old}:', f'-{new}:')
                
            print(f"Copolymer mapping: {mapping}")
        else:
            # Fallback: renumber SMILES to match connectivity expectations
            if len(needed_r_groups) <= 4:  # Common case
                new_smiles = smiles_part
                for i, old_r in enumerate(available_r_groups):
                    new_r = str(i + 1)
                    new_smiles = new_smiles.replace(f'[*:{old_r}]', f'[*:{new_r}]')
                print(f"Renumbered SMILES R-groups to start from 1")
                return new_smiles, connectivity_part
    
    else:  # multipolymer
        # For complex polymers, use sequential mapping
        available_r_groups = sorted(r_groups_in_smiles, key=int)
        needed_r_groups = sorted(set(conn_r_groups), key=int)
        
        mapping = {}
        for i, needed in enumerate(needed_r_groups):
            if i < len(available_r_groups):
                mapping[needed] = available_r_groups[i]
        
        # Apply mapping to connectivity
        for old, new in mapping.items():
            new_connectivity = new_connectivity.replace(f'{old}-', f'{new}-')
            new_connectivity = new_connectivity.replace(f'-{old}:', f'-{new}:')
            
        print(f"Multipolymer mapping: {mapping}")
    
    return new_smiles, new_connectivity

def validate_and_fix_polymer_format(poly_input):
    """
    Enhanced validation with intelligent R-group mapping for all polymer types
    """
    try:
        parts = poly_input.split("|")
        
        # Check if we have enough parts
        if len(parts) < 3:
            print(f"Warning: Insufficient parts in polymer input: {poly_input}")
            return None
            
        smiles_part = parts[0]
        stoich_parts = parts[1:-1]
        connectivity_part = parts[-1]
        
        # Count monomers
        monomers = smiles_part.split(".")
        monomer_count = len(monomers)
        
        # Intelligent R-group mapping
        fixed_smiles, fixed_connectivity = intelligent_rgroup_mapping(smiles_part, connectivity_part)
        
        # Check stoichiometry match
        if len(stoich_parts) != monomer_count:
            print(f"Fixing stoichiometry: {monomer_count} monomers, {len(stoich_parts)} weights")
            
            if monomer_count == 1:
                stoich_parts = ["1.0"]
            elif monomer_count == 2:
                stoich_parts = ["0.5", "0.5"]
            else:
                weight = 1.0 / monomer_count
                stoich_parts = [str(weight)] * monomer_count
        
        # Reconstruct the polymer string
        fixed_poly = f"{fixed_smiles}|{('|'.join(stoich_parts))}|{fixed_connectivity}"
        print(f"Final fixed polymer: {fixed_poly}")
        return fixed_poly
        
    except Exception as e:
        print(f"Error in validation: {e}")
        return None

def adaptive_sampling_around_seed(model, seed_z, vocab, tokenization, prop_predictor, args, device, model_name):
    """
    Adaptive sampling with multiple strategies
    """
    all_prediction_strings = []
    all_reconstruction_strings = []
    all_y_p = []
    
    print("Starting adaptive sampling around best solution...")
    
    # Strategy 1: Small noise sampling
    for noise_level in [0.01, 0.05, 0.1]:  # Multiple noise levels
        print(f"Trying noise level: {noise_level}")
        
        for r in range(3):  # Fewer samples per noise level
            try:
                # Create noise
                noise = torch.tensor(np.random.normal(0, noise_level, size=seed_z.size()), 
                                   dtype=torch.float, device=device)
                seed_z_noise = seed_z + noise
                
                # Ensure we stay within bounds
                seed_z_noise = torch.clamp(seed_z_noise, -3, 3)  # Reasonable bounds
                
                with torch.no_grad():
                    predictions, _, _, _, y = model.inference(data=seed_z_noise, device=device, 
                                                            sample=False, log_var=None)
                    
                    prediction_strings, validity = prop_predictor._calc_validity(predictions)
                    predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
                    prediction_strings_valid = [j for j, valid in zip(prediction_strings, validity) if valid]
                    
                    if predictions_valid:  # Only process if we have valid predictions
                        y_p_valid, z_p_valid, reconstructions_valid, _ = prop_predictor._encode_and_predict_decode_molecules(predictions_valid)
                        all_prediction_strings.extend(prediction_strings_valid)
                        all_reconstruction_strings.extend(reconstructions_valid)
                        all_y_p.extend(y_p_valid)
                        
                        print(f"  Generated {len(predictions_valid)} valid molecules with noise {noise_level}")
                        
                        # If we got good results, continue with this noise level
                        if len(predictions_valid) >= 8:  # Good success rate
                            break
                    
            except Exception as e:
                print(f"  Error with noise level {noise_level}, attempt {r}: {e}")
                continue
        
        # If we have enough samples, break
        if len(all_prediction_strings) >= 20:
            break
    
    # Strategy 2: If still no results, try interpolation with training data
    if not all_prediction_strings:
        print("Trying interpolation strategy...")
        try:
            # Load some training latents for interpolation
            with open(os.path.join(args.save_dir, model_name, 'latent_space_train.npy'), 'rb') as f:
                train_latents = np.load(f)
            
            # Sample random training points
            random_indices = np.random.choice(len(train_latents), size=5, replace=False)
            
            for idx in random_indices:
                train_point = torch.tensor(train_latents[idx], dtype=torch.float, device=device).unsqueeze(0)
                
                # Interpolate between seed and training point
                for alpha in [0.1, 0.3, 0.5]:
                    interpolated = alpha * seed_z[0:1] + (1 - alpha) * train_point
                    interpolated = interpolated.repeat(16, 1)  # Batch size
                    
                    try:
                        with torch.no_grad():
                            predictions, _, _, _, y = model.inference(data=interpolated, device=device, 
                                                                    sample=False, log_var=None)
                            
                            prediction_strings, validity = prop_predictor._calc_validity(predictions)
                            predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
                            prediction_strings_valid = [j for j, valid in zip(prediction_strings, validity) if valid]
                            
                            if predictions_valid:
                                y_p_valid, z_p_valid, reconstructions_valid, _ = prop_predictor._encode_and_predict_decode_molecules(predictions_valid)
                                all_prediction_strings.extend(prediction_strings_valid[:5])  # Limit samples
                                all_reconstruction_strings.extend(reconstructions_valid[:5])
                                all_y_p.extend(y_p_valid[:5])
                                
                                print(f"  Interpolation generated {len(predictions_valid)} valid molecules")
                    except Exception as e:
                        print(f"  Interpolation error: {e}")
                        continue
                
                if len(all_prediction_strings) >= 15:
                    break
                    
        except Exception as e:
            print(f"Interpolation strategy failed: {e}")
    
    print(f"Total valid molecules generated: {len(all_prediction_strings)}")
    return all_prediction_strings, all_reconstruction_strings, all_y_p

def auto_detect_property_count_and_names(model, device):
    """
    Automatically detect the number of properties the model predicts
    """
    try:
        # Create a random latent vector and get prediction
        with torch.no_grad():
            test_z = torch.randn(1, 32).to(device)
            _, _, _, _, test_y = model.inference(data=test_z, device=device, sample=False, log_var=None)
            
            if torch.is_tensor(test_y):
                property_count = test_y.shape[-1] if test_y.dim() > 1 else 1
            else:
                property_count = len(test_y) if hasattr(test_y, '__len__') else 1
                
        print(f"Auto-detected {property_count} properties from model")
        
        # Generate default property names
        if property_count == 1:
            property_names = ["property1"]
        elif property_count == 2:
            property_names = ["property1", "property2"]  # Could be EA, IP or bandgap, something else
        else:
            property_names = [f"property{i+1}" for i in range(property_count)]
            
        return property_count, property_names
        
    except Exception as e:
        print(f"Error in auto-detection: {e}")
        return 1, ["property1"]  # Safe fallback

parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--alpha", default="fixed", choices=["fixed","schedule"])  # Added alpha parameter
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="RT_tokenized", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32) # NOTE: Ignored — loaded from checkpoint['model_config']['embedding_dim']
parser.add_argument("--beta", default="schedule", help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="wce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--initialization", default="random", choices=["random"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=1)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.0004)
parser.add_argument("--max_alpha", type=float, default=0.2)
parser.add_argument("--epsilon", type=float, default=0.05)  # Reduced default
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--max_time", type=int, default=3600)
parser.add_argument("--stopping_type", type=str, default="iter", choices=["iter","time"])
parser.add_argument("--opt_run", type=int, default=1)
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load model checkpoints from and save results to")
parser.add_argument("--checkpoint_every", type=int, default=200, help="Save checkpoint every N iterations")
parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint file to resume from")
parser.add_argument("--monitor_every", type=int, default=50, help="Print progress every N iterations")

# Add flexible property arguments
parser.add_argument("--property_names", type=str, nargs='+', default=None,
                    help="Names of the properties to optimize (auto-detected if not specified)")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected if not specified)")
parser.add_argument("--property_targets", type=float, nargs='+', default=None,
                    help="Target values for each property (one per property)")
parser.add_argument("--property_weights", type=float, nargs='+', default=None,
                    help="Weights for each property in the objective function (one per property)")
parser.add_argument("--property_objectives", type=str, nargs='+', default=None, 
                    choices=["minimize", "maximize", "target"],
                    help="Objective for each property: minimize, maximize, or target a specific value")
parser.add_argument("--objective_type", type=str, default="custom",
                    choices=["EAmin", "mimick_peak", "mimick_best", "max_gap", "custom"],
                    help="Type of objective function to use (legacy option)")
parser.add_argument("--custom_equation", type=str, default=None,
                    help="Custom equation for objective function. Use 'p[i]' to reference property i. Example: '(1 + p[0] - p[1])*2'")
parser.add_argument("--maximize_equation", action="store_true", 
                    help="Maximize the custom equation instead of minimizing it")

# Add CSV dataset argument
parser.add_argument("--dataset_csv", type=str, default=None,
                    help="Path to the CSV file containing polymer data for novelty analysis. Should have 'poly_chemprop_input' column.")
parser.add_argument("--polymer_column", type=str, default="poly_chemprop_input",
                    help="Name of the column containing polymer SMILES in the CSV file (default: poly_chemprop_input)")

args = parser.parse_args()

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "train" 
data_augment = "old" # new or old
dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')

num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

# Load model first to auto-detect properties
# Create an instance of the G2S model from checkpoint
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='
# We'll add the property string after we determine the properties

# Load model without property string first
temp_model_name = model_name + 'temp/'
model_files = glob.glob(os.path.join(args.save_dir, 'Model_*data_DecL=*', "model_best_loss.pt"))

if not model_files:
    raise FileNotFoundError("No model files found. Please check the save directory.")

# Take the first matching model file
filepath = model_files[0]
print(f"Loading model from: {filepath}")

if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_VAE_PPguided
    else: 
        model_type = G2S_VAE_PPguideddisabled
        
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    
    batch_size = model_config.get('batch_size', 64)
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    model_config["max_alpha"]=args.max_alpha
    vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    vocab = load_vocab(vocab_file=vocab_file)
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

# Now auto-detect properties from the loaded model
auto_property_count, auto_property_names = auto_detect_property_count_and_names(model, device)

# Handle property configuration with auto-detection
if args.property_names is not None:
    property_names = args.property_names
    property_count = len(property_names)
else:
    property_names = auto_property_names
    property_count = auto_property_count

if args.property_count is not None:
    if args.property_count != property_count:
        print(f"Warning: Specified property count ({args.property_count}) doesn't match auto-detected count ({property_count})")
        print(f"Using auto-detected count: {property_count}")

# Validate and adjust user inputs to match detected property count
def adjust_property_config(user_list, default_value, property_count, config_name):
    """Adjust property configuration lists to match the detected property count"""
    if user_list is None:
        return [default_value] * property_count
    elif len(user_list) < property_count:
        print(f"Warning: {config_name} has {len(user_list)} values but model has {property_count} properties")
        print(f"Padding with default value: {default_value}")
        return user_list + [default_value] * (property_count - len(user_list))
    elif len(user_list) > property_count:
        print(f"Warning: {config_name} has {len(user_list)} values but model has {property_count} properties")
        print(f"Truncating to first {property_count} values")
        return user_list[:property_count]
    else:
        return user_list

# Adjust property configurations
property_weights = adjust_property_config(args.property_weights, 1.0, property_count, "property_weights")
property_objectives = adjust_property_config(args.property_objectives, "minimize", property_count, "property_objectives")
property_targets = adjust_property_config(args.property_targets, 0.0, property_count, "property_targets")

print(f"Final configuration:")
print(f"Properties ({property_count}): {property_names}")
print(f"Objectives: {property_objectives}")
print(f"Weights: {property_weights}")
print(f"Targets: {property_targets}")

# Now we can properly set the model name with the correct property string
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = model_name + property_str + '/'

# Update the directory name
dir_name = os.path.dirname(filepath).replace(os.path.basename(os.path.dirname(filepath)), os.path.basename(model_name.rstrip('/')))

# PropertyPrediction class - enhanced for flexible properties
class PropertyPrediction():
    def __init__(self, model, nr_vars, objective_type="custom", property_names=None, 
                 property_targets=None, property_weights=None, property_objectives=None,
                 custom_equation=None, maximize_equation=False, property_count=None):
        self.model_predictor = model
        self.weight_electron_affinity = 1  # Legacy - weight for electron affinity
        self.weight_ionization_potential = 1  # Legacy - weight for ionization potential
        self.weight_z_distances = 5  # Adjust the weight for distance between GA chosen z and reencoded z
        self.penalty_value = -10  # Adjust the weight for penalty of validity
        self.results_custom = {}
        self.nr_vars = nr_vars
        self.eval_calls = 0
        self.objective_type = objective_type
        self.custom_equation = custom_equation
        self.maximize_equation = maximize_equation
        
        # Flexible property configuration
        self.property_names = property_names if property_names else ["property1"]
        self.property_count = property_count if property_count else len(self.property_names)
        self.property_weights = property_weights if property_weights else [1.0] * self.property_count
        self.property_targets = property_targets if property_targets else [0.0] * self.property_count
        self.property_objectives = property_objectives if property_objectives else ["minimize"] * self.property_count
        
        # Validate inputs
        if len(self.property_weights) != self.property_count:
            print(f"Adjusting property_weights to match property_count ({self.property_count})")
            self.property_weights = (self.property_weights * self.property_count)[:self.property_count]
            
        if len(self.property_targets) != self.property_count:
            print(f"Adjusting property_targets to match property_count ({self.property_count})")
            self.property_targets = (self.property_targets * self.property_count)[:self.property_count]
            
        if len(self.property_objectives) != self.property_count:
            print(f"Adjusting property_objectives to match property_count ({self.property_count})")
            self.property_objectives = (self.property_objectives * self.property_count)[:self.property_count]
            
        # Check that targets are provided for 'target' objectives
        for i, obj in enumerate(self.property_objectives):
            if obj == "target" and self.property_targets[i] is None:
                print(f"Warning: Target value not provided for property {self.property_names[i]}, using 0.0")
                self.property_targets[i] = 0.0
                    
        # If using a custom equation, we need at least one property defined
        if self.custom_equation and not property_names:
            raise ValueError("Property names must be provided when using a custom equation")

    def _normalize_property_predictions(self, predictions):
        """
        Normalize property predictions to always be a consistent format
        """
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        
        predictions = np.array(predictions)
        
        # Ensure we have the right shape
        if predictions.ndim == 1 and self.property_count == 1:
            # Single property, single prediction
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim == 1 and self.property_count > 1:
            # Multiple properties, single prediction
            predictions = predictions.reshape(1, -1)
        elif predictions.ndim == 2:
            # Already in correct format
            pass
        else:
            # Fallback: flatten and reshape
            predictions = predictions.flatten().reshape(-1, self.property_count)
        
        # Pad or truncate to match expected property count
        if predictions.shape[1] < self.property_count:
            # Pad with NaN
            padding = np.full((predictions.shape[0], self.property_count - predictions.shape[1]), np.nan)
            predictions = np.concatenate([predictions, padding], axis=1)
        elif predictions.shape[1] > self.property_count:
            # Truncate
            predictions = predictions[:, :self.property_count]
            
        return predictions
    
    def evaluate(self, **params):
        self.eval_calls += 1
        print(f"Evaluation {self.eval_calls}: {params}")
        _vector = [params[f'x{i}'] for i in range(self.nr_vars)]
        print(f"Parameter vector: {_vector}")
        
        x = torch.from_numpy(np.array(_vector)).to(device).to(torch.float32)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=x, device=device, sample=False, log_var=None)
        
        print(f"Step 1: Generated {len(predictions)} predictions")
        
        # Normalize property predictions
        y_normalized = self._normalize_property_predictions(y)
        print(f"Model output shape: {y_normalized.shape}, Properties expected: {self.property_count}")
        
        # Validity check of the decoded molecule + penalize invalid molecules
        prediction_strings, validity = self._calc_validity(predictions)
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        prediction_strings_valid = [j for j, valid in zip(prediction_strings, validity) if valid]
        
        print(f"Step 2: Validity check - {sum(validity)}/{len(validity)} valid")
        
        # Early return if no valid molecules
        if not any(validity):
            print("Warning: No valid molecules generated in this iteration")
            results_dict = {
                "objective": self.penalty_value,
                "latents_BO": x,
                "latents_reencoded": [np.zeros(32)], 
                "predictions_BO": y_normalized,
                "predictions_reencoded": [[np.nan] * self.property_count],
                "string_decoded": prediction_strings, 
                "string_reconstructed": [],
            }
            self.results_custom[str(self.eval_calls)] = results_dict
            return self.penalty_value
        
        y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _ = self._encode_and_predict_decode_molecules(predictions_valid)
        
        print(f"Step 3: Reencoding - {len(y_p_after_encoding_valid)} successful")
        
        # Handle case where reencoding fails completely
        if not y_p_after_encoding_valid:
            print("Warning: All reencoding attempts failed")
            results_dict = {
                "objective": self.penalty_value,
                "latents_BO": x,
                "latents_reencoded": [np.zeros(32)], 
                "predictions_BO": y_normalized,
                "predictions_reencoded": [[np.nan] * self.property_count],
                "string_decoded": prediction_strings, 
                "string_reconstructed": [],
            }
            self.results_custom[str(self.eval_calls)] = results_dict
            return self.penalty_value
        
        # Build expanded arrays more carefully
        expanded_y_p = []
        expanded_z_p = []
        valid_idx = 0
        
        for val in validity:
            if val == 1:
                if valid_idx < len(y_p_after_encoding_valid):
                    # Normalize the reencoded predictions
                    reencoded_props = y_p_after_encoding_valid[valid_idx]
                    if len(reencoded_props) < self.property_count:
                        # Pad with NaN
                        reencoded_props = reencoded_props + [np.nan] * (self.property_count - len(reencoded_props))
                    elif len(reencoded_props) > self.property_count:
                        # Truncate
                        reencoded_props = reencoded_props[:self.property_count]
                    
                    expanded_y_p.append(reencoded_props)
                    expanded_z_p.append(z_p_after_encoding_valid[valid_idx])
                    valid_idx += 1
                else:
                    # Fallback if we run out of valid results
                    expanded_y_p.append([np.nan] * self.property_count)
                    expanded_z_p.append([0] * 32)
            else:
                expanded_y_p.append([np.nan] * self.property_count)
                expanded_z_p.append([0] * 32)
        
        expanded_y_p = np.array(expanded_y_p)
        expanded_z_p = np.array(expanded_z_p)
        
        # Calculate objective based on the first valid molecule
        first_valid_idx = np.where(validity)[0]
        if len(first_valid_idx) == 0:
            aggr_obj = self.penalty_value
        else:
            first_valid = first_valid_idx[0]
            property_values = expanded_y_p[first_valid]
            
            # Check for NaN values
            if np.isnan(property_values).all():
                print("Warning: All property values are NaN")
                aggr_obj = self.penalty_value
            else:
                aggr_obj = self._calculate_objective(property_values)
        
        print(f"Step 4: Objective calculated - {aggr_obj}")
        
        # results
        results_dict = {
        "objective":aggr_obj,
        "latents_BO": x,
        "latents_reencoded": expanded_z_p, 
        "predictions_BO": y_normalized,
        "predictions_reencoded": expanded_y_p,
        "string_decoded": prediction_strings, 
        "string_reconstructed": all_reconstructions_valid,
        }
        self.results_custom[str(self.eval_calls)] = results_dict
        
        return aggr_obj

    def _calculate_objective(self, property_values):
        """
        Calculate objective value based on property values and configuration
        """
        # Replace NaN values with large penalty
        clean_values = []
        for i, val in enumerate(property_values):
            if np.isnan(val):
                clean_values.append(1000.0)  # Large penalty for NaN
            else:
                clean_values.append(val)
        
        # If using a custom equation
        if self.objective_type == "custom" and self.custom_equation:
            # Create a safe evaluation environment
            eval_locals = {'p': clean_values, 'abs': abs, 'np': np, 'math': math}
            
            try:
                # Evaluate the custom equation with the property values
                equation_result = eval(self.custom_equation, {"__builtins__": {}}, eval_locals)
                
                # Apply maximization if requested
                if self.maximize_equation:
                    equation_result = -equation_result
                    
                return -equation_result  # The negative sign is because BO maximizes by default
            except Exception as e:
                print(f"Error evaluating custom equation: {e}")
                return self.penalty_value
        
        # Legacy objectives (only for 2-property models)
        elif self.objective_type in ['EAmin', 'mimick_peak', 'mimick_best', 'max_gap'] and len(clean_values) >= 2:
            if self.objective_type=='EAmin':
                obj1 = self.weight_electron_affinity * clean_values[0]
                obj2 = self.weight_ionization_potential * np.abs(clean_values[1] - 1)
            elif self.objective_type=='mimick_peak':
                obj1 = self.weight_electron_affinity * np.abs(clean_values[0] + 2)
                obj2 = self.weight_ionization_potential * np.abs(clean_values[1] - 1.2)
            elif self.objective_type=='mimick_best':
                obj1 = self.weight_electron_affinity * np.abs(clean_values[0] + 2.64)
                obj2 = self.weight_ionization_potential * np.abs(clean_values[1] - 1.61)
            elif self.objective_type=='max_gap':
                obj1 = self.weight_electron_affinity * clean_values[0]
                obj2 = - self.weight_ionization_potential * clean_values[1]
            
            return -(obj1 + obj2)
        
        # Flexible property handling (default)
        else:
            obj_values = []
            
            for i in range(min(len(clean_values), self.property_count)):
                if self.property_objectives[i] == "minimize":
                    # For minimization, use the raw value
                    obj_value = clean_values[i]
                elif self.property_objectives[i] == "maximize":
                    # For maximization, negate the value
                    obj_value = -clean_values[i]
                elif self.property_objectives[i] == "target":
                    # For targeting a specific value, use absolute difference
                    obj_value = np.abs(clean_values[i] - self.property_targets[i])
                
                # Apply the weight
                obj_value *= self.property_weights[i]
                obj_values.append(obj_value)
            
            # Sum up all objective components
            return -sum(obj_values)

    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _calc_validity(self, predictions):
        """
        Enhanced validity calculation with intelligent R-group fixing
        """
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), 
                                           tokenization=tokenization) for sample in range(len(predictions))]
        mols_valid = []
        fixed_strings = []
        
        for _s in prediction_strings:
            poly_input = _s[:-1]  # Remove last character
            
            # Try to validate and fix format
            fixed_poly = validate_and_fix_polymer_format(poly_input)
            
            if fixed_poly is None:
                mols_valid.append(0)
                fixed_strings.append(poly_input)
                continue
                
            try:
                poly_graph = poly_smiles_to_graph(fixed_poly, np.nan, np.nan, None)
                mols_valid.append(1)
                fixed_strings.append(fixed_poly)
            except Exception as e:
                print(f"Graph creation failed even after fixing: {e}")
                mols_valid.append(0)
                fixed_strings.append(poly_input)
        
        return fixed_strings, np.array(mols_valid)
    
    def _encode_and_predict_decode_molecules(self, predictions):
        """
        More robust encoding with better error handling and flexible property support
        """
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), 
                                           tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
        valid_indices = []
        
        print(f"Processing {len(prediction_strings)} prediction strings...")
        
        for i, s in enumerate(prediction_strings):
            try:
                poly_input = s[:-1]  # Remove last character
                
                # Validate format first
                fixed_poly = validate_and_fix_polymer_format(poly_input)
                if fixed_poly is None:
                    print(f"Skipping invalid polymer {i}: {poly_input}")
                    continue
                
                # Try to create graph with advanced error handling
                try:
                    g = poly_smiles_to_graph(fixed_poly, np.nan, np.nan, None)
                except Exception as graph_error:
                    print(f"Graph creation failed for polymer {i}: {graph_error}")
                    continue
                
                # Tokenize
                if tokenization == "oldtok":
                    target_tokens = tokenize_poly_input(poly_input=fixed_poly)
                elif tokenization == "RT_tokenized":
                    target_tokens = tokenize_poly_input_RTlike(poly_input=fixed_poly)
                
                tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=target_tokens, vocab=vocab)
                g.tgt_token_ids = tgt_token_ids
                g.tgt_lens = tgt_lens
                g.to(device)
                
                data_list.append(g)
                valid_indices.append(i)
                
            except Exception as e:
                print(f"Error processing polymer {i}: {e}")
                continue
        
        if not data_list:
            print("No valid polymers to encode!")
            return [], [], [], None
        
        print(f"Successfully processed {len(data_list)} valid polymers")
        
        # Continue with encoding...
        data_loader = DataLoader(dataset=data_list, batch_size=min(64, len(data_list)), shuffle=False)
        dict_data_loader = MP_Matrix_Creator(data_loader, device)
        
        y_p = []
        z_p = []
        all_reconstructions = []
        
        with torch.no_grad():
            for i, batch in enumerate(range(len(dict_data_loader))):
                try:
                    data = dict_data_loader[str(batch)][0]
                    data.to(device)
                    dest_is_origin_matrix = dict_data_loader[str(batch)][1]
                    dest_is_origin_matrix.to(device)
                    inc_edges_to_atom_matrix = dict_data_loader[str(batch)][2]
                    inc_edges_to_atom_matrix.to(device)

                    # Forward pass
                    reconstruction, _, _, z, y = model.inference(
                        data=data, device=device, 
                        dest_is_origin_matrix=dest_is_origin_matrix, 
                        inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, 
                        sample=False, log_var=None
                    )
                    
                    # Normalize property predictions
                    y_normalized = self._normalize_property_predictions(y)
                    
                    # Check for NaN values
                    if np.isnan(y_normalized).any():
                        print(f"Warning: NaN detected in predictions for batch {i}")
                        y_normalized = np.nan_to_num(y_normalized, nan=0.0)
                    
                    y_p.append(y_normalized)
                    z_p.append(z.cpu().numpy())
                    
                    reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), 
                                                           tokenization=tokenization) for sample in range(len(reconstruction))]
                    all_reconstructions.extend(reconstruction_strings)
                    
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    continue
        
        # Flatten results with proper property handling
        y_p_flat = []
        for array_ in y_p:
            for sublist in array_:
                if isinstance(sublist, np.ndarray):
                    y_p_flat.append(sublist.tolist())
                else:
                    y_p_flat.append(sublist)
                    
        z_p_flat = [sublist.tolist() for array_ in z_p for sublist in array_]
        self.modified_solution = z_p_flat

        return y_p_flat, z_p_flat, all_reconstructions, dict_data_loader

def save_checkpoint(optimizer, prop_predictor, iteration, checkpoint_dir, opt_run):
    """Save optimization checkpoint"""
    checkpoint_data = {
        'iteration': iteration,
        'optimizer_state': {
            'res': optimizer.res,
            'space': optimizer.space,
            'random_state': optimizer._random_state,  
            'gp': {
                'X_': optimizer._gp.X_.tolist() if hasattr(optimizer._gp, 'X_') else [],
                'y_': optimizer._gp.y_.tolist() if hasattr(optimizer._gp, 'y_') else [],
                'kernel_': str(optimizer._gp.kernel_) if hasattr(optimizer._gp, 'kernel_') else ""
            }
        },
        'custom_results': prop_predictor.results_custom,
        'eval_calls': prop_predictor.eval_calls,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}_run{opt_run}.pkl')
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Keep backup of last checkpoint
    if iteration > 200:
        prev_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration-200}_run{opt_run}.pkl')
        if os.path.exists(prev_checkpoint):
            backup_file = os.path.join(checkpoint_dir, f'backup_iter_{iteration-200}_run{opt_run}.pkl')
            shutil.copy2(prev_checkpoint, backup_file)
    
    return checkpoint_file

def load_checkpoint(checkpoint_file):
    """Load optimization checkpoint"""
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    return checkpoint_data

def log_progress(message, log_file):
    """Log progress to file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(log_file, 'a') as f:
        f.write(log_message + '\n')

def calculate_current_validity_rate(results_custom):
    """Calculate current validity rate from optimization results"""
    total = 0
    valid = 0
    
    for eval_call, results in results_custom.items():
        total += 1
        # Check if objective is not penalty value
        if results['objective'] != -10:
            valid += 1
    
    if total > 0:
        validity_rate = (valid / total) * 100
        return validity_rate, valid, total
    return 0, 0, 0

# Determine the boundaries for the latent dimensions from training dataset
dir_name = os.path.join(args.save_dir, model_name)

# Create checkpoint directory
checkpoint_dir = os.path.join(dir_name, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Setup log file for monitoring
log_file = os.path.join(dir_name, f'optimization_log_{args.opt_run}.txt')

with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space = np.load(f)
min_values = np.amin(latent_space, axis=0).tolist()
max_values = np.amax(latent_space, axis=0).tolist()

cutoff=-0.1  # Slightly wider bounds than original

if not cutoff==0.0:
    transformed_min_values = []
    transformed_max_values = []
    for min_val, max_val in zip(min_values, max_values):
        #bounds are larger than in training set if cutoff value is negative
        # Calculate amount to cut off from each end (5%)
        cutoff_amount = cutoff * abs(max_val - min_val)
        # Adjust min and max values
        transformed_min = min_val + cutoff_amount
        transformed_max = max_val - cutoff_amount
        transformed_min_values.append(transformed_min)
        transformed_max_values.append(transformed_max)
    bounds = {'x{}'.format(i): (j, k) for i,(j,k) in enumerate(zip(transformed_min_values,transformed_max_values))}
elif cutoff==0: 
    bounds = {'x{}'.format(i): (j, k) for i,(j,k) in enumerate(zip(min_values,max_values))}

opt_run = args.opt_run

# Instantiating the PropertyPrediction class with flexible property support
nr_vars = 32

# Get property configuration from arguments
property_targets = property_targets
property_weights = property_weights
property_objectives = property_objectives
objective_type = args.objective_type
custom_equation = args.custom_equation
maximize_equation = args.maximize_equation

prop_predictor = PropertyPrediction(
    model, 
    nr_vars, 
    objective_type=objective_type,
    property_names=property_names,
    property_targets=property_targets,
    property_weights=property_weights,
    property_objectives=property_objectives,
    custom_equation=custom_equation,
    maximize_equation=maximize_equation,
    property_count=property_count
)

# Check if resuming from checkpoint
start_iteration = 0
if args.resume_from:
    log_progress(f"Resuming from checkpoint: {args.resume_from}", log_file)
    checkpoint_data = load_checkpoint(args.resume_from)
    
    # Initialize optimizer
    optimizer = BayesianOptimization(f=prop_predictor.evaluate, pbounds=bounds, random_state=opt_run)
    
    # Restore optimizer state
    optimizer.res = checkpoint_data['optimizer_state']['res']
    prop_predictor.results_custom = checkpoint_data['custom_results']
    prop_predictor.eval_calls = checkpoint_data['eval_calls']
    start_iteration = checkpoint_data['iteration']
    
    # Check validity rate at checkpoint
    validity_rate, valid_count, total_count = calculate_current_validity_rate(prop_predictor.results_custom)
    
    log_progress(f"Restored state from iteration {start_iteration} - Checkpoint validity: {valid_count}/{total_count} ({validity_rate:.1f}%)", log_file)
else:
    # Initialize new optimization
    optimizer = BayesianOptimization(f=prop_predictor.evaluate, pbounds=bounds, random_state=opt_run)
    log_progress("Starting new optimization", log_file)

# Perform optimization
# Use EI instead of UCB for potentially better exploration
utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1)

# Define the time limit in seconds
stopping_type = args.stopping_type # time or iter
max_time = args.max_time  # Set to 600 seconds, for example
max_iter = args.max_iter # Set to a maximum number of iterations 
if stopping_type == "time":
    stopping_criterion = stopping_type+"_"+str(max_time)
elif stopping_type == "iter":
    stopping_criterion = stopping_type+"_"+str(max_iter)
    max_time = float('inf')  

# Custom optimization loop with checkpointing
start_time = time.time()
log_progress(f"Starting optimization with {max_iter} iterations", log_file)

# Initial exploration if not resuming
if not args.resume_from:
    init_points = 30  # Increased from 20
    print("Performing initial exploration...")
    optimizer.maximize(init_points=init_points, n_iter=0, acquisition_function=utility)
else:
    init_points = 0  # Safe fallback

# Modified optimization loop
total_iterations = max_iter
checkpoint_every = args.checkpoint_every
monitor_every = args.monitor_every
    
# Adjust start_iteration if we just did init_points
if not args.resume_from and init_points > 0:
    start_iteration = init_points

# Run the main optimization with proper iteration tracking
for iter_num in range(start_iteration, total_iterations):
    try:      
        # Run one iteration (skip if we're still in init phase)
        if iter_num >= init_points:
            optimizer.maximize(init_points=0, n_iter=1, acquisition_function=utility)
        
        # Check for summary display
        if len(optimizer.res) % monitor_every == 0 and len(optimizer.res) > 0:
            print("*** ENTERING SUMMARY BLOCK ***")
            elapsed = time.time() - start_time
            validity_rate, valid_count, total_count = calculate_current_validity_rate(prop_predictor.results_custom)
            
            # Add our custom summary
            print("\n" + "="*80)
            print(f"SUMMARY AT ITERATION {len(optimizer.res)}")
            print(f"Elapsed Time: {elapsed:.1f}s")
            print(f"Validity Rate: {valid_count}/{total_count} ({validity_rate:.1f}%)")
            print("="*80 + "\n")
            
            # Log to file
            file_message = f"Iteration {iter_num}/{total_iterations} - Best objective: {optimizer.max['target']:.4f} - Elapsed: {elapsed:.1f}s - Validity: {valid_count}/{total_count} ({validity_rate:.1f}%)"
            log_progress(file_message, log_file)
            
            with open(log_file, 'a') as f:
                f.write('\n')
        
        # Save checkpoint
        if iter_num % checkpoint_every == 0 and iter_num > 0:
            checkpoint_file = save_checkpoint(optimizer, prop_predictor, iter_num, checkpoint_dir, args.opt_run)
            log_progress(f"Checkpoint saved at iteration {iter_num}: {checkpoint_file}", log_file)
        
        # Check time limit if specified
        if max_time != float('inf') and (time.time() - start_time) > max_time:
            log_progress(f"Time limit reached. Stopping at iteration {iter_num}", log_file)
            break
            
    except Exception as e:
        log_progress(f"Error at iteration {iter_num}: {str(e)}", log_file)
        # Save emergency checkpoint
        emergency_file = save_checkpoint(optimizer, prop_predictor, iter_num, checkpoint_dir, args.opt_run)
        log_progress(f"Emergency checkpoint saved: {emergency_file}", log_file)
        raise e

elapsed_time = time.time() - start_time
log_progress(f"Optimization completed. Total time: {elapsed_time:.2f} seconds", log_file)

# Save final checkpoint
final_checkpoint = save_checkpoint(optimizer, prop_predictor, total_iterations, checkpoint_dir, args.opt_run)
log_progress(f"Final checkpoint saved: {final_checkpoint}", log_file)

# List all checkpoints
def list_checkpoints(checkpoint_dir, opt_run):
    """List all checkpoints for a given run"""
    pattern = os.path.join(checkpoint_dir, f'checkpoint_iter_*_run{opt_run}.pkl')
    checkpoints = glob.glob(pattern)
    checkpoints.sort()
    return checkpoints

# Save checkpoint summary
checkpoint_summary = {
    'total_iterations': total_iterations,
    'final_objective': optimizer.max['target'],
    'final_params': optimizer.max['params'],
    'checkpoints': list_checkpoints(checkpoint_dir, args.opt_run),
    'log_file': log_file
}

summary_file = os.path.join(checkpoint_dir, f'checkpoint_summary_run{args.opt_run}.json')
with open(summary_file, 'w') as f:
    json.dump(checkpoint_summary, f, indent=2, default=str)

log_progress(f"Checkpoint summary saved: {summary_file}", log_file)

results = optimizer.res
results_custom = prop_predictor.results_custom

with open(dir_name+'optimization_results_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'wb') as f:
    pickle.dump(results, f)
with open(dir_name+'optimization_results_custom_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'wb') as f:
    pickle.dump(results_custom, f)

# Get the best parameters found
best_params = optimizer.max['params']
best_objective = optimizer.max['target']

print("Best Parameters:", best_params)
print("Best Objective Value:", best_objective)

with open(dir_name+'optimization_results_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'rb') as f:
    results = pickle.load(f)
with open(dir_name+'optimization_results_custom_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'rb') as f:
    results_custom = pickle.load(f)

with open(dir_name+'optimization_results_custom_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
     print(results_custom, file=fl)

# Calculate distances between the BO and reencoded latents with flexible property support
Latents_BO = []
Latents_RE = []
latent_inconsistencies = []
pred_BO = []
pred_RE = []
decoded_mols= []
rec_mols=[]

# Where you're collecting results from optimization
for eval, res in results_custom.items():
    eval_int = int(eval)
    L_bo = res["latents_BO"].detach().cpu().numpy()
    L_re = res["latents_reencoded"][0]
    latent_inconsistency = np.linalg.norm(L_bo-L_re)
    latent_inconsistencies.append(latent_inconsistency)
    Latents_BO.append(L_bo)
    Latents_RE.append(L_re)
    
    # Store the prediction values - make sure to detach and move to CPU if they're tensors
    bo_pred = res["predictions_BO"]
    if torch.is_tensor(bo_pred):
        bo_pred = bo_pred.detach().cpu().numpy()
    pred_BO.append(bo_pred)
    
    re_pred = res["predictions_reencoded"][0]
    pred_RE.append(re_pred)
    
    decoded_mols.append(res["string_decoded"][0])
    if not len(res["string_reconstructed"])==0:
        rec_mols.append(res["string_reconstructed"][0])
    else: rec_mols.append("Invalid decoded molecule")

# Add debug print statements here:
print("Debug information for pred_BO and pred_RE:")
for i, (bo_pred, re_pred) in enumerate(zip(pred_BO, pred_RE)):
    print(f"Item {i}:")
    print(f"  pred_BO: type={type(bo_pred)}, ", end="")
    if torch.is_tensor(bo_pred):
        print(f"shape={bo_pred.shape}, device={bo_pred.device}, value={bo_pred}")
    elif hasattr(bo_pred, 'shape'):  # numpy arrays
        print(f"shape={bo_pred.shape}, value={bo_pred}")
    else:
        print(f"value={bo_pred}")
        
    print(f"  pred_RE: type={type(re_pred)}, ", end="")
    if hasattr(re_pred, 'shape'):  # numpy arrays
        print(f"shape={re_pred.shape}, value={re_pred}")
    else:
        print(f"value={re_pred}")

def distance_matrix(arrays):
    num_arrays = len(arrays)
    dist_matrix = np.zeros((num_arrays, num_arrays))

    for i in range(num_arrays):
        for j in range(num_arrays):
            dist_matrix[i, j] = np.linalg.norm(arrays[i] - arrays[j])
    # Flatten upper triangular part of the matrix (excluding diagonal)
    flattened_upper_triangular = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]

    # Calculate mean and standard deviation
    mean_distance = np.mean(flattened_upper_triangular)
    std_distance = np.std(flattened_upper_triangular)

    return dist_matrix, mean_distance, std_distance

dist_matrix_zBO, mBO, sBO=distance_matrix(Latents_BO)
dist_matrix_zRE, mRE, sRE=distance_matrix(Latents_RE)
print(mBO, sBO)
print(mRE, sRE)
print(np.mean(latent_inconsistencies), np.std(latent_inconsistencies))

import matplotlib.pyplot as plt

# Extract data for the curves with flexible property support
iterations = range(len(pred_BO))

# Handle different possible formats safely with proper error checking
property_values_bo = [[] for _ in range(property_count)]  # Create lists for each property
for x in pred_BO:
    # First ensure x is a numpy array that we can work with
    if torch.is_tensor(x):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x) if not isinstance(x, np.ndarray) else x
    
    # Access elements safely - handle different array shapes
    for prop_idx in range(property_count):
        if x_np.ndim > 1 and x_np.shape[1] > prop_idx:  # Handle (1,N) shaped arrays
            property_values_bo[prop_idx].append(x_np[0, prop_idx])
        elif x_np.size > prop_idx:  # Check if flat array has enough elements
            property_values_bo[prop_idx].append(x_np.flat[prop_idx])
        else:  # Handle arrays without enough elements
            property_values_bo[prop_idx].append(float('nan'))

# Similarly handle pred_RE with proper error checking
property_values_re = [[] for _ in range(property_count)]  # Create lists for each property
for x in pred_RE:
    for prop_idx in range(property_count):
        if isinstance(x, np.ndarray):
            if x.ndim > 1 and x.shape[1] > prop_idx:  # Handle (1,N) shaped arrays
                property_values_re[prop_idx].append(x[0, prop_idx])
            elif x.size > prop_idx:  # Check if flat array has enough elements
                property_values_re[prop_idx].append(x.flat[prop_idx])
            else:
                property_values_re[prop_idx].append(float('nan'))
        elif isinstance(x, list) and len(x) > prop_idx:
            property_values_re[prop_idx].append(x[prop_idx])
        else:
            property_values_re[prop_idx].append(float('nan'))

# Create plot with dynamic labels for all properties
plt.figure(0, figsize=(12, 8))

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

for prop_idx in range(property_count):
    prop_name = property_names[prop_idx]
    color = colors[prop_idx % len(colors)]
    
    plt.plot(iterations, property_values_bo[prop_idx], label=f'{prop_name} (BO)', 
             color=color, linestyle='-', alpha=0.7)
    plt.plot(iterations, property_values_re[prop_idx], label=f'{prop_name} (RE)', 
             color=color, linestyle='--', alpha=0.7)

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.title(f'Property Optimization Progress ({property_count} properties)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(dir_name+'BO_objectives_'+str(cutoff)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png',  dpi=300)
plt.close()

# Plot the training data pca and optimization points
# Dimensionality reduction
# 1. Load trained latent space
dataset_type = "train"
with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space_train = np.load(f)
with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all_train = np.load(f)
#2. load fitted pca
dim_red_type="pca"
with open(dir_name+dim_red_type+'_fitted_train', 'rb') as f:
    reducer = pickle.load(f)

# 3. Transform train LS   
z_embedded_train = reducer.transform(latent_space_train)
# 4. Transform points of Optimization
latents_BO_np = np.stack(Latents_BO)
z_embedded_BO = reducer.transform(latents_BO_np)

latents_RE_np = np.stack(Latents_RE)
z_embedded_RE = reducer.transform(latents_RE_np)
plt.figure(1)

# PCA projection colored by first property
plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title(f'{property_names[0]}' if property_count >= 1 else 'Property 1')
plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=2, c='black')
plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=2, c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA Projection Colored by {property_names[0]}')
plt.savefig(dir_name+'BO_projected_to_pca_'+str(cutoff)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png',  dpi=300)
plt.close()

# PCA projection colored by other properties (if they exist)
for prop_idx in range(1, min(property_count, 3)):  # Limit to first 3 properties for visualization
    try:
        with open(dir_name+f'y{prop_idx+1}_all_'+dataset_type+'.npy', 'rb') as f:
            y_prop_all_train = np.load(f)
        plt.figure(1)
        plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y_prop_all_train, cmap='plasma')
        clb = plt.colorbar()
        clb.ax.set_title(f'{property_names[prop_idx]}')
        plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=2, c='black')
        plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=2, c='red')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'PCA Projection Colored by {property_names[prop_idx]}')
        plt.savefig(dir_name + 'BO_projected_to_pca_' + property_names[prop_idx] + '_' + str(cutoff) + '_' + str(stopping_criterion) + '_run' + str(opt_run) + '.png', dpi=300)
        plt.close()
    except FileNotFoundError:
        print(f"Warning: y{prop_idx+1}_all_{dataset_type}.npy not found, skipping PCA plot for {property_names[prop_idx]}")

### Do the same but only for improved points with flexible property support
def indices_of_improvement(values):
    indices_of_increases = []

    # Initialize the highest value and its index
    highest_value = values[0]
    highest_index = 0

    # Iterate through the values
    for i, value in enumerate(values):
        # If the current value is greater than the previous highest value
        if value > highest_value:
            highest_value = value  # Update the highest value
            highest_index = i      # Update the index of the highest value
            indices_of_increases.append(i)  # Save the index of increase

    return indices_of_increases

def top_n_molecule_indices(objective_values, n_idx=10):
    # Get the indices of molecules with the highest objective values
    # Filter out NaN values and keep track of original indices
    filtered_indexed_values = [(index, value) for index, value in enumerate(objective_values) if not math.isnan(value)]
    # Sort the indexed values by the value in descending order and take n_idx best ones
    sorted_filtered_indexed_values = sorted(filtered_indexed_values, key=lambda x: x[1], reverse=True)
    _best_mols = []
    best_mols_count = {}
    top_idxs = []
    for index, value in sorted_filtered_indexed_values: 
        if not decoded_mols[index] in _best_mols: 
            top_idxs.append(index)
            best_mols_count[decoded_mols[index]]=1
            _best_mols.append(decoded_mols[index])
        else:
            best_mols_count[decoded_mols[index]]+=1
        if len(top_idxs)==n_idx:
            break

    return top_idxs, best_mols_count

# Calculate objective values with flexible property support
if objective_type in ['mimick_peak', 'mimick_best', 'EAmin', 'max_gap'] and property_count >= 2:
    # Legacy objectives for 2+ property models
    if objective_type=='mimick_peak':
        objective_values = [-(np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) if len(arr) >= 2 and not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]
    elif objective_type=='mimick_best':
        objective_values = [-((np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61))) if len(arr) >= 2 and not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]
    elif objective_type=='EAmin': 
        objective_values = [-(arr[0]+np.abs(arr[1]-1)) if len(arr) >= 2 and not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]
    elif objective_type =='max_gap':
        objective_values = [-(arr[0]-arr[1]) if len(arr) >= 2 and not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]
else:
    # For flexible property models, use the primary property
    if property_objectives[0] == "target":
        objective_values = [-np.abs(arr[0] - property_targets[0]) if len(arr) >= 1 and not np.isnan(arr[0]) else float('-inf') for arr in pred_RE]
    elif property_objectives[0] == "maximize":
        objective_values = [arr[0] if len(arr) >= 1 and not np.isnan(arr[0]) else float('-inf') for arr in pred_RE]
    else:  # minimize
        objective_values = [-arr[0] if len(arr) >= 1 and not np.isnan(arr[0]) else float('-inf') for arr in pred_RE]

# Find valid indices (where we have non-NaN values)
valid_indices = [i for i, val in enumerate(objective_values) if val != float('-inf')]

# If we have valid indices, use the normal improvement function
if valid_indices:
    indices_of_increases = indices_of_improvement(objective_values)
# If no valid improvements are found, just use the index of the one valid result
if not valid_indices or not indices_of_increases:
    # Find the index of the single valid entry or valid entry with max value
    try:
        # First try to find the best valid value
        max_idx = max((i for i, val in enumerate(objective_values) if val != float('-inf')), 
                      key=lambda i: objective_values[i])
        indices_of_increases = [max_idx]
    except ValueError:
        # If there are no valid values, just use index 0
        indices_of_increases = [0]

# Now proceed with using the indices - handle variable number of properties
property_values_bo_imp = [[property_values_bo[prop_idx][i] for i in indices_of_increases] for prop_idx in range(property_count)]
property_values_re_imp = [[property_values_re[prop_idx][i] for i in indices_of_increases] for prop_idx in range(property_count)]

best_z_re = [Latents_RE[i] for i in indices_of_increases]
best_mols = {i+1: decoded_mols[i] for i in indices_of_increases}

# Build best_props with correct number of properties
best_props = {}
for i, idx in enumerate(indices_of_increases):
    prop_values = []
    # Add RE values for each property
    for prop_idx in range(property_count):
        if prop_idx < len(property_values_re[idx]) if property_values_re and len(property_values_re) > idx else False:
            prop_values.append(property_values_re[prop_idx][idx])
        else:
            prop_values.append(float('nan'))
    # Add BO values for each property  
    for prop_idx in range(property_count):
        if prop_idx < len(property_values_bo[idx]) if property_values_bo and len(property_values_bo) > idx else False:
            prop_values.append(property_values_bo[prop_idx][idx])
        else:
            prop_values.append(float('nan'))
    best_props[i+1] = prop_values

best_mols_rec = {i+1: rec_mols[i] for i in indices_of_increases}

with open(dir_name+'best_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols, file=fl)
    print(best_props, file=fl)
with open(dir_name+'best_recon_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols_rec, file=fl)

top_20_indices, top_20_mols = top_n_molecule_indices(objective_values, n_idx=20)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}

# Build best_props_t20 with correct number of properties
best_props_t20 = {}
for i, idx in enumerate(top_20_indices):
    prop_values = []
    # Add RE values for each property
    for prop_idx in range(property_count):
        if prop_idx < len(property_values_re[idx]) if property_values_re and len(property_values_re) > idx else False:
            prop_values.append(property_values_re[prop_idx][idx])
        else:
            prop_values.append(float('nan'))
    # Add BO values for each property  
    for prop_idx in range(property_count):
        if prop_idx < len(property_values_bo[idx]) if property_values_bo and len(property_values_bo) > idx else False:
            prop_values.append(property_values_bo[prop_idx][idx])
        else:
            prop_values.append(float('nan'))
    best_props_t20[i+1] = prop_values
    
with open(dir_name+'top20_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols_t20, file=fl)
    print(best_props_t20, file=fl)

print(objective_values)
print(indices_of_increases)
latents_BO_np_imp = np.stack([Latents_BO[i] for i in indices_of_increases])
z_embedded_BO_imp = reducer.transform(latents_BO_np_imp)

latents_RE_np_imp = np.stack([Latents_RE[i] for i in indices_of_increases])
z_embedded_RE_imp = reducer.transform(latents_RE_np_imp)

plt.figure(2, figsize=(10, 8))

plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis', alpha=0.2)
clb = plt.colorbar()
clb.ax.set_title(f'{property_names[0]}' if property_count >= 1 else 'Property 1')

# Real latent space (reencoded)
for i, (x, y) in enumerate(z_embedded_RE_imp):
    it=indices_of_increases[i]
    plt.scatter(x, y, color='red', s=3,  marker="2")  # Plot points
    plt.text(x, y+0.2, f'{i+1}({it+1})', fontsize=6, color="red", ha='center', va='center')  # Annotate with labels

# Connect points with arrows
for i in range(len(z_embedded_RE_imp) - 1):
    x_start, y_start = z_embedded_RE_imp[i]
    x_end, y_end = z_embedded_RE_imp[i + 1]
    plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
              shape='full', lw=0.05, length_includes_head=True, head_width=0.1, color='red')

for i, (x, y) in enumerate(z_embedded_BO_imp):
    it=indices_of_increases[i]
    plt.scatter(x, y, color='black', s=2,  marker="1")  # Plot points
    plt.text(x, y+0.2, f'{i+1}', fontsize=6, color="black", ha='center', va='center')  # Annotate with labels

# Connect points with arrows
for i in range(len(z_embedded_BO_imp) - 1):
    x_start, y_start = z_embedded_BO_imp[i]
    x_end, y_end = z_embedded_BO_imp[i + 1]
    plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
              shape='full', lw=0.05, length_includes_head=True, head_width=0.1, color='black')

# Set plot labels and title
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title('Optimization in latent space')

plt.savefig(dir_name+'BO_imp_projected_to_pca_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png',  dpi=300)

plt.figure(3, figsize=(10, 8))

plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis', alpha=0.2)
clb = plt.colorbar()
clb.ax.set_title(f'{property_names[0]}' if property_count >= 1 else 'Property 1')

# Real latent space (reencoded)
for i, (x, y) in enumerate(z_embedded_RE_imp):
    it=indices_of_increases[i]
    plt.scatter(x, y, color='red', s=3,  marker="2")  # Plot points
    plt.text(x, y+0.2, f'{i+1}({it+1})', fontsize=6, color="red", ha='center', va='center')  # Annotate with labels

# Connect points with arrows
for i in range(len(z_embedded_RE_imp) - 1):
    x_start, y_start = z_embedded_RE_imp[i]
    x_end, y_end = z_embedded_RE_imp[i + 1]
    plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
              shape='full', lw=0.05, length_includes_head=True, head_width=0.1, color='red')

# Set plot labels and title
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title('Optimization in latent space')

plt.savefig(dir_name+'BO_imp_projected_to_pca_onlyred_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png',  dpi=300)

# === DEBUG CHECK: BEFORE SAMPLING ===
print(f"Length of decoded_mols: {len(decoded_mols)}")
print(f"Length of Latents_RE: {len(Latents_RE)}")
if len(decoded_mols) > 0:
    print(f"Example decoded mol: {decoded_mols[-1]}")
else:
    print("No decoded molecules available.")

if len(Latents_RE) > 0:
    print(f"Example latent RE: {Latents_RE[-1][:5]} ...")
else:
    print("No reencoded latent vectors available.")

# === TEST DECODABILITY OF RANDOM LATENT ===
print("\nTesting decoding from a random latent vector:")
with torch.no_grad():
    z_test = torch.randn(1, 32).to(device)
    pred_test, _, _, _, _ = model.inference(z_test, device=device, sample=False, log_var=None)
    try:
        decoded_test = combine_tokens(tokenids_to_vocab(pred_test[0][0].tolist(), vocab), tokenization=tokenization)
        print("Decoded string from random latent:", decoded_test)
    except Exception as e:
        print("Failed to decode random latent. Error:", e)

""" Sample around seed molecule - seed being the optimal solution found by optimizer """
# Sample around the optimal molecule and predict the property values

# Add these debug prints BEFORE the above line:
print(f"\nCHECKING BEST MOLECULE:")
print(f"best_z_re length: {len(best_z_re) if 'best_z_re' in locals() else 'not found'}")
print(f"best_z_re type: {type(best_z_re) if 'best_z_re' in locals() else 'not found'}")
if 'best_z_re' in locals() and len(best_z_re) > 0:
    print(f"Last best_z_re: {best_z_re[-1]}")
    
    seed_z = best_z_re[-1] # last reencoded best molecule
    
    # extract the predictions of best molecule together with the latents of best molecule(by encoding? or BO one)
    seed_z = torch.from_numpy(np.array(seed_z)).unsqueeze(0).repeat(64,1).to(device).to(torch.float32)
    print(seed_z)
    with torch.no_grad():
        predictions, _, _, _, y_seed = model.inference(data=seed_z, device=device, sample=False, log_var=None)
        seed_string = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
    
    # Use the new adaptive sampling function
    all_prediction_strings, all_reconstruction_strings, all_y_p = adaptive_sampling_around_seed(
        model, seed_z, vocab, tokenization, prop_predictor, args, device, model_name
    )
else:
    print("No valid best molecules found!")
    all_prediction_strings = []
    all_reconstruction_strings = []
    all_y_p = []
    seed_string = ["No seed generated"]
    y_seed = [np.array([np.nan] * property_count)]

# Add these debug prints:
print(f"\nSAMPLING RESULTS CHECK:")
print(f"all_prediction_strings length: {len(all_prediction_strings)}")
print(f"all_reconstruction_strings length: {len(all_reconstruction_strings)}")
print(f"all_y_p length: {len(all_y_p)}")
if all_y_p:
    print(f"First 3 y_p values: {all_y_p[:3]}")
    print(f"Types in all_y_p: {[type(x) for x in all_y_p[:3]]}")

# Check if sampling produced any valid results
if not all_prediction_strings:
    print("Warning: No valid molecules generated during sampling around seed")
    print("This may indicate issues with the latent space bounds or model validity")
    
    # Create results file with warning
    with open(dir_name+'results_around_BO_seed_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as f:
        f.write("WARNING: No valid molecules generated during sampling around seed\n")
        f.write("Seed string decoded: " + (seed_string[0] if 'seed_string' in locals() and seed_string else "No seed generated") + "\n")
        f.write("Prediction: " + str(y_seed[0] if 'y_seed' in locals() and len(y_seed) > 0 else "No predictions") + "\n")
        f.write("This may indicate:\n")
        f.write("1. Latent space bounds are too restrictive\n") 
        f.write("2. Model has difficulty generating valid molecules in this region\n")
        f.write("3. Epsilon value for noise may be too large\n")
    
    # Set empty lists for the rest of the analysis
    all_predictions = []
    all_predictions_can = []
    
else:
    print(f'Saving generated strings')

    i=0
    with open(dir_name+'results_around_BO_seed_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as f:
        f.write("Seed string decoded: " + seed_string[0] + "\n")
        f.write("Prediction: "+ str(y_seed[0]))
        f.write("The results, sampling around the best population are the following\n")
        for l1,l2 in zip(all_reconstruction_strings,all_prediction_strings):
            if l1 == l2:
                f.write("decoded molecule from GA selection is encoded and decoded to the same molecule\n")
                f.write(l1 + "\n")
                f.write("The predicted properties from z(reencoded) are: " + str(all_y_p[i]) + "\n")
            else:
                f.write("Sampled molecule around BO seed: ")
                f.write(l2 + "\n")
                f.write("Encoded and decoded sampled molecule: ")
                f.write(l1 + "\n")
                f.write("The predicted properties from z(reencoded) are: " + str(all_y_p[i]) + "\n")
            i+=1

""" Check the molecules around the optimized seed """

def poly_smiles_to_molecule(poly_input):
    '''
    Turns adjusted polymer smiles string into PyG data objects
    '''
    # Turn into RDKIT mol object
    mols = make_monomer_mols(poly_input.split("|")[0], 0, 0,  # smiles
                            fragment_weights=poly_input.split("|")[1:-1])
    
    return mols

def valid_scores(smiles):
    return np.array(list(map(make_polymer_mol, smiles)), dtype=np.float32)

dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
data_augment ="old"
vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
vocab = load_vocab(vocab_file=vocab_file)

# Handle the case where no predictions were generated
if all_prediction_strings:
    all_predictions=all_prediction_strings.copy()
    sm_can = SmilesEnumCanon()
    all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
else:
    print("No prediction strings to analyze - skipping detailed novelty analysis")
    all_predictions = []
    all_predictions_can = []
    sm_can = SmilesEnumCanon()

prediction_validityA= []
prediction_validityB =[]
data_dir = os.path.join(main_dir_path,'data/')

# Load dataset CSV file for novelty analysis
all_polymers_data = []
all_train_polymers = []

# Get training polymers from dict_train_loader
for batch, graphs in enumerate(dict_train_loader):
    data = dict_train_loader[str(batch)][0]
    train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
    all_train_polymers.extend(train_polymers_batch)

# Load dataset CSV for novelty comparison
if args.dataset_csv:
    try:
        print(f"Loading dataset from: {args.dataset_csv}")
        df = pd.read_csv(args.dataset_csv)
        
        # Check if the specified column exists
        if args.polymer_column in df.columns:
            for i in range(len(df.loc[:, args.polymer_column])):
                poly_input = df.loc[i, args.polymer_column]
                all_polymers_data.append(poly_input)
            print(f"Loaded {len(all_polymers_data)} polymers from {args.polymer_column} column")
        else:
            print(f"Warning: Column '{args.polymer_column}' not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            print("Using empty dataset for novelty analysis")
            all_polymers_data = []
            
    except FileNotFoundError:
        print(f"Error: Dataset CSV file not found: {args.dataset_csv}")
        print("Using empty dataset for novelty analysis")
        all_polymers_data = []
    except Exception as e:
        print(f"Error loading dataset CSV: {e}")
        print("Using empty dataset for novelty analysis")
        all_polymers_data = []
else:
    print("No dataset CSV provided (--dataset_csv). Skipping novelty analysis against external dataset.")
    all_polymers_data = []

# Continue with the rest of the novelty analysis only if we have data to analyze
if all_predictions_can and (all_polymers_data or all_train_polymers):
    all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
    all_train_can = list(map(sm_can.canonicalize, all_train_polymers))
    all_pols_data_can = list(map(sm_can.canonicalize, all_polymers_data)) if all_polymers_data else []
    
    monomers= [s.split("|")[0].split(".") for s in all_train_polymers]
    monomers_all=[mon for sub_list in monomers for mon in sub_list]
    all_mons_can = []
    for m in monomers_all:
        m_can = sm_can.canonicalize(m, monomer_only=True, stoich_con_info=False)
        modified_string = re.sub(r'\*\:\d+', '*', m_can)
        all_mons_can.append(modified_string)
    all_mons_can = list(set(all_mons_can))
    print(len(all_mons_can), all_mons_can[1:3] if len(all_mons_can) > 2 else all_mons_can)

    # Save canonicalized data for future use
    with open(data_dir+'all_train_pols_can'+'.pkl', 'wb') as f:
        pickle.dump(all_train_can, f)
    if all_pols_data_can:
        with open(data_dir+'all_pols_data_can'+'.pkl', 'wb') as f:
            pickle.dump(all_pols_data_can, f)
    with open(data_dir+'all_mons_train_can'+'.pkl', 'wb') as f:
        pickle.dump(all_mons_can, f)

    # Load back the data (optional, could skip this step)
    with open(data_dir+'all_train_pols_can'+'.pkl', 'rb') as f:
        all_train_can = pickle.load(f)
    if all_pols_data_can:
        with open(data_dir+'all_pols_data_can'+'.pkl', 'rb') as f:
            all_pols_data_can= pickle.load(f)
    with open(data_dir+'all_mons_train_can'+'.pkl', 'rb') as f:
        all_mons_can = pickle.load(f)

    # Calculate validity scores
    prediction_mols = list(map(poly_smiles_to_molecule, all_predictions))
    for mon in prediction_mols: 
        try: prediction_validityA.append(mon[0] is not None)
        except: prediction_validityA.append(False)
        try: prediction_validityB.append(mon[1] is not None)
        except: prediction_validityB.append(False)

    # Evaluation of validation set reconstruction accuracy (inference)
    monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
    monomer_comb_predicted = [poly_smiles.split("|")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
    monomer_comb_train = [poly_smiles.split("|")[0] for poly_smiles in all_train_can if poly_smiles]

    monA_pred = [mon[0] for mon in monomer_smiles_predicted if len(mon) > 0]
    monB_pred = [mon[1] for mon in monomer_smiles_predicted if len(mon) > 1]
    monA_pred_gen = []
    monB_pred_gen = []
    for m_c in monomer_smiles_predicted:
        if len(m_c) > 0:
            ma = m_c[0]
            ma_can = sm_can.canonicalize(ma, monomer_only=True, stoich_con_info=False)
            monA_pred_gen.append(re.sub(r'\*\:\d+', '*', ma_can))
        
        if len(m_c) > 1:
            mb = m_c[1]
            mb_can = sm_can.canonicalize(mb, monomer_only=True, stoich_con_info=False)
            monB_pred_gen.append(re.sub(r'\*\:\d+', '*', mb_can))

    monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
    monomer_con_predicted = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']

    validityA = sum(prediction_validityA)/len(prediction_validityA) if prediction_validityA else 0
    validityB = sum(prediction_validityB)/len(prediction_validityB) if prediction_validityB else 0

    # Novelty metrics
    novel = 0
    novel_pols=[]
    for pol in monomer_comb_predicted:
        if pol not in monomer_comb_train:
            novel+=1
            novel_pols.append(pol)
    novelty_mon_comb = novel/len(monomer_comb_predicted) if monomer_comb_predicted else 0
    
    novel = 0
    novel_pols_full=[]
    for pol in all_predictions_can:
        if pol not in all_train_can:
            novel+=1
            novel_pols_full.append(pol)
    novelty = novel/len(all_predictions_can) if all_predictions_can else 0
    
    # Novelty against external dataset (if provided)
    if all_pols_data_can:
        novel = 0
        for pol in all_predictions_can:
            if pol not in all_pols_data_can:
                novel+=1
        novelty_full_dataset = novel/len(all_predictions_can) if all_predictions_can else 0
    else:
        novelty_full_dataset = 0
        
    novelA = 0
    novelAs = []
    for monA in monA_pred_gen:
        if monA not in all_mons_can:
            novelA+=1
            novelAs.append(monA)
    print(novelAs, len(list(set(novelAs))))
    novelty_A = novelA/len(monA_pred_gen) if monA_pred_gen else 0
    
    novelB = 0
    novelBs = []
    for monB in monB_pred_gen:
        if monB not in all_mons_can:
            novelB+=1
            novelBs.append(monB)
    print(novelBs, len(list(set(novelBs))))

    novelty_B = novelB/len(monB_pred_gen) if monB_pred_gen else 0

    diversity = len(list(set(all_predictions_can)))/len(all_predictions_can) if all_predictions_can else 0
    diversity_novel = len(list(set(novel_pols_full)))/len(novel_pols_full) if novel_pols_full else 0

    classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
    classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
    whole_valid = len(monomer_smiles_predicted)
    validity = whole_valid/len(all_predictions) if all_predictions else 0
    
    with open(dir_name+'novelty_BO_seed_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as f:
        f.write("Gen Mon A validity: %.4f %% Gen Mon B validity: %.4f %% "% (100*validityA, 100*validityB,))
        f.write("Gen validity: %.4f %% "% (100*validity,))
        f.write("Novelty: %.4f %% "% (100*novelty,))
        f.write("Novelty (mon_comb): %.4f %% "% (100*novelty_mon_comb,))
        f.write("Novelty MonA full dataset: %.4f %% "% (100*novelty_A,))
        f.write("Novelty MonB full dataset: %.4f %% "% (100*novelty_B,))
        if all_pols_data_can:
            f.write("Novelty in external dataset: %.4f %% "% (100*novelty_full_dataset,))
        f.write("Diversity: %.4f %% "% (100*diversity,))
        f.write("Diversity (novel polymers): %.4f %% "% (100*diversity_novel,))

else:
    # Create a minimal novelty file when no analysis is possible
    with open(dir_name+'novelty_BO_seed_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as f:
        f.write("No valid molecules generated for novelty analysis\n")

""" Plot the kde of the properties of training data and sampled data """
from sklearn.neighbors import KernelDensity

# Load training data for all properties
training_property_data = []
for prop_idx in range(property_count):
    try:
        with open(dir_name+f'y{prop_idx+1}_all_'+dataset_type+'.npy', 'rb') as f:
            y_prop_all = np.load(f)
            training_property_data.append(list(y_prop_all))
    except FileNotFoundError:
        print(f"Warning: y{prop_idx+1}_all_{dataset_type}.npy not found, creating dummy data")
        # Create dummy data if file doesn't exist
        training_property_data.append([0.0] * 100)  # Dummy data

with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)

yp_all_list = [yp for yp in yp_all]

# debug prints:
print(f"\nBEFORE KDE PLOTTING:")
print(f"Length of all_y_p: {len(all_y_p)}")
print(f"Properties: {property_names}")
if all_y_p:
    print(f"First few y_p values: {all_y_p[:3]}")
    print(f"Types in all_y_p: {[type(x) for x in all_y_p[:3]]}")

# Create KDE data for each property
for prop_idx in range(property_count):
    if not all_y_p:  # Skip if no data
        continue
        
    prop_name = property_names[prop_idx]
    
    # Get training data for this property
    y_prop_all = training_property_data[prop_idx] if prop_idx < len(training_property_data) else [0.0] * 100
    
    # Get predicted data for this property
    yp_prop_all = [yp[prop_idx] for yp in yp_all_list if len(yp) > prop_idx]
    yp_prop_all_seed = [yp[prop_idx] for yp in all_y_p if len(yp) > prop_idx]
    
    if not yp_prop_all_seed:
        print(f"No data for property {prop_name}, skipping KDE plot")
        continue
    
    plt.figure(figsize=(10, 8))
    real_distribution = np.array([r for r in y_prop_all if not np.isnan(r)])
    augmented_distribution = np.array([p for p in yp_prop_all if not np.isnan(p)])
    seed_distribution = np.array([s for s in yp_prop_all_seed if not np.isnan(s)])

    if len(real_distribution) == 0 or len(augmented_distribution) == 0 or len(seed_distribution) == 0:
        print(f"Insufficient data for KDE plot of {prop_name}")
        plt.close()
        continue

    # Reshape the data
    real_distribution = real_distribution.reshape(-1, 1)
    augmented_distribution = augmented_distribution.reshape(-1, 1)
    seed_distribution = seed_distribution.reshape(-1, 1)

    # Define bandwidth
    bandwidth = 0.1

    try:
        # Fit kernel density estimators
        kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_real.fit(real_distribution)
        kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_augmented.fit(augmented_distribution)
        kde_sampled_seed = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_sampled_seed.fit(seed_distribution)

        # Create a range of values for the x-axis
        x_min = min(np.min(real_distribution), np.min(augmented_distribution), np.min(seed_distribution))
        x_max = max(np.max(real_distribution), np.max(augmented_distribution), np.max(seed_distribution))
        x_values = np.linspace(x_min, x_max, 1000)
        
        # Evaluate the KDE on the range of values
        real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
        augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
        seed_density = np.exp(kde_sampled_seed.score_samples(x_values.reshape(-1, 1)))

        # Plotting
        plt.plot(x_values, real_density, label='Real Data', linewidth=2)
        plt.plot(x_values, augmented_density, label='Augmented Data', linewidth=2)
        plt.plot(x_values, seed_density, label='Sampled around optimal molecule', linewidth=2)

        plt.xlabel(f'{prop_name}')
        plt.ylabel('Density')
        plt.title(f'Kernel Density Estimation ({prop_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(dir_name+f'KDE{prop_name}_BO_seed'+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"KDE plot saved for {prop_name}")
        
    except Exception as e:
        print(f"Error creating KDE plot for {prop_name}: {e}")
        plt.close()

# Add a final verification
print(f"\nKDE plot files created for {property_count} properties")
for prop_idx, prop_name in enumerate(property_names):
    kde_file = dir_name+f'KDE{prop_name}_BO_seed'+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png'
    if os.path.exists(kde_file):
        size = os.path.getsize(kde_file)
        print(f"KDE{prop_name} file: {size} bytes")
    else:
        print(f"KDE{prop_name} file: Not created")

print(f"\nOptimization completed successfully!")
print(f"Results saved to: {dir_name}")
print(f"Properties optimized: {property_names} ({property_count} total)")
print(f"Property objectives: {property_objectives}")
print(f"Property weights: {property_weights}")
if args.dataset_csv:
    print(f"Novelty analysis performed against: {args.dataset_csv}")
else:
    print("No external dataset provided for novelty analysis")
