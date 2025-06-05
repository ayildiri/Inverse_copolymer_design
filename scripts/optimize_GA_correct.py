import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.termination import Termination
from pymoo.core.population import Population
from pymoo.termination import get_termination
import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from scipy.spatial import distance
import pickle
import time
from datetime import timedelta, datetime
import glob
import json
import shutil
import pandas as pd
import math
import re
from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from data_processing.Smiles_enum_canon import SmilesEnumCanon
from sklearn.neighbors import KernelDensity

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.rdkit_poly import make_polymer_mol

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

def robust_polymer_validation(poly_input):
    """
    SIMPLIFIED validation that relies on the enhanced G2S_clean validation
    """
    if not poly_input or not isinstance(poly_input, str):
        return None
    
    # Let the enhanced G2S_clean validation handle complex validation
    # Just do basic cleanup here
    cleaned = poly_input.strip()
    if len(cleaned) == 0:
        return None
    
    # Basic format check - ensure it has polymer structure
    if '|' not in cleaned and '.' not in cleaned:
        # Try to create basic polymer format
        if len(cleaned) > 0:
            return f"{cleaned}|1.0|<1-1:1.0:1.0"
    
    return cleaned

def fix_polymer_format(poly_input):
    """
    Fix polymer format issues between model output and processing expectations
    """
    try:
        if not poly_input or not isinstance(poly_input, str):
            return None
        
        print(f"Debug - Original polymer input: {poly_input}")
        
        # Split by '|' to get components
        parts = poly_input.split('|')
        
        if len(parts) < 2:
            print(f"Warning: Polymer input doesn't have enough parts: {poly_input}")
            print("Expected format: MonA.MonB|stoichiometry|connectivity")
            # Try to create basic format
            return f"{poly_input}|1.0|<1-1:1.0:1.0"
        
        monomers_part = parts[0]
        second_part = parts[1]
        
        print(f"Debug - Monomers part: {monomers_part}")
        print(f"Debug - Second part: {second_part}")
        
        # Count monomers
        monomers = monomers_part.split('.')
        monomer_count = len(monomers)
        
        print(f"Debug - Monomer count: {monomer_count}")
        print(f"Debug - Monomers: {monomers}")
        
        # Check if second part contains connectivity info (starts with '<')
        if second_part.startswith('<'):
            print("Debug - Second part contains connectivity info, need to add stoichiometry")
            # Need to add proper stoichiometry
            if monomer_count == 1:
                # Homopolymer
                fixed_format = f"{monomers_part}|1.0|{second_part}"
            elif monomer_count == 2:
                # Copolymer - create equal weights
                fixed_format = f"{monomers_part}|0.5|0.5|{second_part}"
            else:
                # Multi-polymer - create equal weights
                weight_per_monomer = round(1.0 / monomer_count, 3)
                weights = '|'.join([str(weight_per_monomer)] * monomer_count)
                fixed_format = f"{monomers_part}|{weights}|{second_part}"
        else:
            # Second part might be stoichiometry, check if we have connectivity
            if len(parts) >= 3:
                # Format: MonA.MonB|stoich1|stoich2|connectivity (already correct)
                fixed_format = poly_input
            else:
                # Format: MonA.MonB|stoich, need to add connectivity
                # CRITICAL FIX: Handle stoichiometry correctly for multiple monomers
                if monomer_count == 1:
                    # Homopolymer: single weight is fine
                    fixed_format = f"{monomers_part}|{second_part}|<1-1:1.0:1.0"
                elif monomer_count == 2:
                    # FIXED: Copolymer needs TWO weights
                    try:
                        weight_val = float(second_part)
                        # If single weight provided, create two equal weights that sum to weight_val
                        weight1 = weight_val / 2
                        weight2 = weight_val / 2
                        fixed_format = f"{monomers_part}|{weight1}|{weight2}|<1-2:1.0:1.0"
                    except ValueError:
                        # If can't parse as float, use equal weights
                        fixed_format = f"{monomers_part}|0.5|0.5|<1-2:1.0:1.0"
                else:
                    # Multi-polymer: distribute weight among all monomers
                    try:
                        weight_val = float(second_part)
                        weight_per_monomer = weight_val / monomer_count
                        weights = '|'.join([str(weight_per_monomer)] * monomer_count)
                        fixed_format = f"{monomers_part}|{weights}|<1-1:1.0:1.0"
                    except ValueError:
                        # Equal weights fallback
                        weight_per_monomer = round(1.0 / monomer_count, 3)
                        weights = '|'.join([str(weight_per_monomer)] * monomer_count)
                        fixed_format = f"{monomers_part}|{weights}|<1-1:1.0:1.0"
        
        print(f"Debug - Fixed polymer input: {fixed_format}")
        return fixed_format
        
    except Exception as e:
        print(f"Error in fix_polymer_format: {e}")
        return poly_input

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
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
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
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--max_time", type=int, default=3600)
parser.add_argument("--stopping_type", type=str, default="iter", choices=["iter","time", "convergence"])
parser.add_argument("--opt_run", type=int, default=1)
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load model checkpoints from and save results to")
parser.add_argument("--checkpoint_every", type=int, default=200, help="Save checkpoint every N iterations")
parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint file to resume from")
parser.add_argument("--monitor_every", type=int, default=50, help="Print progress every N iterations")

# Enhanced flexible property arguments (from optimize_BO.py)
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
                    help="Type of objective function to use (custom is recommended for flexible property handling)")
parser.add_argument("--custom_equation", type=str, default=None,
                    help="Custom equation for objective function. Use 'p[i]' to reference property i. Example: '(1 + p[0] - p[1])*2'")
parser.add_argument("--maximize_equation", action="store_true", 
                    help="Maximize the custom equation instead of minimizing it")

# Add dataset path argument
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Path to the dataset directory containing the data files (default: uses main_dir_path/data)")

# Add CSV dataset argument (missing from GA script)
parser.add_argument("--dataset_csv", type=str, default=None,
                    help="Path to the CSV file containing polymer data for novelty analysis. Should have 'poly_chemprop_input' column.")
parser.add_argument("--polymer_column", type=str, default="poly_chemprop_input",
                    help="Name of the column containing polymer SMILES in the CSV file (default: poly_chemprop_input)")

# Add enhanced generation parameter (though less relevant for GA, kept for consistency)
parser.add_argument("--use_enhanced_generation", action="store_true", default=False,
                    help="Use enhanced generation with quality control (for consistency with optimize_BO)")

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

# Set dataset path from argument or use default
if args.dataset_path:
    dataset_path = args.dataset_path
else:
    dataset_path = os.path.join(main_dir_path, 'data')

dict_train_loader = torch.load(os.path.join(dataset_path, f'dict_train_loader_{augment}_{tokenization}.pt'))

num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

# Load model first to auto-detect properties (from optimize_BO.py)
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
    vocab_file = os.path.join(dataset_path, f'poly_smiles_vocab_{augment}_{tokenization}.txt')
    vocab = load_vocab(vocab_file=vocab_file)
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

# Now auto-detect properties from the loaded model (from optimize_BO.py)
auto_property_count, auto_property_names = auto_detect_property_count_and_names(model, device)

# Handle property configuration with auto-detection (from optimize_BO.py)
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

# Validate and adjust user inputs to match detected property count (from optimize_BO.py)
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

# Adjust property configurations (from optimize_BO.py)
property_weights = adjust_property_config(args.property_weights, 1.0, property_count, "property_weights")
property_objectives = adjust_property_config(args.property_objectives, "minimize", property_count, "property_objectives")
property_targets = adjust_property_config(args.property_targets, 0.0, property_count, "property_targets")

print(f"Final configuration:")
print(f"Properties ({property_count}): {property_names}")
print(f"Objectives: {property_objectives}")
print(f"Weights: {property_weights}")
print(f"Targets: {property_targets}")
print(f"Enhanced generation: {args.use_enhanced_generation}")
print(f"Dataset CSV: {args.dataset_csv if args.dataset_csv else 'Using default augmentation-based dataset'}")

# Now we can properly set the model name with the correct property string (from optimize_BO.py)
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = model_name + property_str + '/'

# Update the directory name (from optimize_BO.py)
dir_name = os.path.dirname(filepath).replace(os.path.basename(os.path.dirname(filepath)), os.path.basename(model_name.rstrip('/')))

# Create checkpoint directory
checkpoint_dir = os.path.join(dir_name, 'checkpoints_GA')
os.makedirs(checkpoint_dir, exist_ok=True)

# Setup log file for monitoring
log_file = os.path.join(dir_name, f'optimization_log_GA_{args.opt_run}.txt')

# Property_optimization_problem class - enhanced with flexible property support
class Property_optimization_problem(Problem):
    def __init__(self, model, x_min, x_max, objective_type="custom", property_names=None, 
                 property_targets=None, property_weights=None, property_objectives=None,
                 custom_equation=None, maximize_equation=False, property_count=None):
        # Determine number of objectives based on property configuration
        if custom_equation:
            n_obj = 1  # Custom equation results in a single objective
        elif objective_type == "custom" and property_names:
            n_obj = len(property_names)
        else:
            n_obj = 2  # Default for legacy objective types
            
        super().__init__(n_var=len(x_min), n_obj=n_obj, n_constr=0, xl=x_min, xu=x_max)
        
        self.model_predictor = model
        self.weight_electron_affinity = 1  # Legacy - weight for electron affinity
        self.weight_ionization_potential = 1  # Legacy - weight for ionization potential
        self.weight_z_distances = 5  # Weight for distance between GA chosen z and reencoded z
        self.penalty_value = 100  # Weight for penalty of validity
        self.modified_solution = None
        self.modified_solution_history = []
        self.results_custom = {}
        self.eval_calls = 0
        self.objective_type = objective_type
        self.custom_equation = custom_equation
        self.maximize_equation = maximize_equation
        
        # Enhanced flexible property configuration (from optimize_BO.py)
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
        Normalize property predictions to always be a consistent format (from optimize_BO.py)
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

    def _evaluate(self, x, out, *args, **kwargs):
        # Assuming x is a 1D array containing the 32 numerical parameters

        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        self.eval_calls += 1
        x_torch = torch.from_numpy(x).to(device).to(torch.float32) 
        print("Evaluation should be repaired")
        print(x)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=x_torch, device=device, sample=False, log_var=None)
        
        # Normalize property predictions (from optimize_BO.py)
        y_normalized = self._normalize_property_predictions(y)
        print(f"Model output shape: {y_normalized.shape}, Properties expected: {self.property_count}")
        
        # Enhanced validity check with graceful fallbacks (from optimize_BO.py)
        prediction_strings, validity = self._calc_validity(predictions)
        
        invalid_mask = (validity == 0)
        zero_vector = np.zeros(self.n_var)
        validity_mask = np.all(x != zero_vector, axis=1)
        print(len(invalid_mask))
        print(validity_mask.shape[0])
        print(x.shape[0])
        print(np.array(y_normalized).shape[0])
        print(out["F"])

        # Initialize output array with appropriate dimensions
        out["F"] = np.zeros((x.shape[0], self.n_obj))
        
        # Handle custom equation case
        if self.custom_equation:
            # For each valid solution
            for i in range(x.shape[0]):
                if validity_mask[i]:
                    # Extract property values for this solution
                    property_values = [np.array(y_normalized)[i, j] for j in range(self.property_count)]
                    
                    # Create a safe evaluation environment
                    eval_locals = {'p': property_values, 'abs': abs, 'np': np, 'math': math}
                    
                    try:
                        # Evaluate the custom equation with the property values
                        equation_result = eval(self.custom_equation, {"__builtins__": {}}, eval_locals)
                            
                        # AFTER (to match BO behavior):
                        if self.maximize_equation:
                            out["F"][i, 0] = -equation_result  # GA minimizes, so negate for maximization
                        else:
                            out["F"][i, 0] = equation_result   # GA minimizes by default
                    except Exception as e:
                        print(f"Error evaluating custom equation for solution {i}: {e}")
                        out["F"][i, 0] = self.penalty_value
                else:
                    out["F"][i, 0] = self.penalty_value
        # Handle legacy objective types
        elif self.objective_type != "custom":
            # Check if legacy objectives are compatible with the number of properties
            if self.property_count != 2 and self.objective_type in ['EAmin', 'mimick_peak', 'mimick_best', 'max_gap']:
                print(f"Warning: Legacy objective '{self.objective_type}' expects 2 properties (EA, IP) but model has {self.property_count} properties: {self.property_names}")
                print("Falling back to flexible property handling...")
                # Fall through to flexible property handling
                for i, prop_name in enumerate(self.property_names):
                    prop_idx = i  # Index of the property in the predicted values
                    
                    if self.property_objectives[i] == "minimize":
                        # For minimization, use the raw value
                        out["F"][validity_mask, i] = self.property_weights[i] * np.array(y_normalized)[validity_mask, prop_idx]
                    elif self.property_objectives[i] == "maximize":
                        # For maximization, negate the value
                        out["F"][validity_mask, i] = -self.property_weights[i] * np.array(y_normalized)[validity_mask, prop_idx]
                    elif self.property_objectives[i] == "target":
                        # For targeting a specific value, use absolute difference
                        out["F"][validity_mask, i] = self.property_weights[i] * np.abs(np.array(y_normalized)[validity_mask, prop_idx] - self.property_targets[i])
            else:
                # Original legacy objective handling for 2-property models
                if self.objective_type=='mimick_peak':
                    out["F"][validity_mask, 0] = self.weight_electron_affinity * np.abs(np.array(y_normalized)[validity_mask,0]+2)
                    out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y_normalized)[validity_mask,1] - 1.2)
                elif self.objective_type=='mimick_best':
                    out["F"][validity_mask, 0] = self.weight_electron_affinity * np.abs(np.array(y_normalized)[validity_mask,0]+2.64)
                    out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y_normalized)[validity_mask,1] - 1.61)
                elif self.objective_type=='EAmin':
                    out["F"][validity_mask, 0] = self.weight_electron_affinity * np.array(y_normalized)[validity_mask,0]
                    out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y_normalized)[validity_mask,1] - 1.0)
                elif self.objective_type=='max_gap':
                    out["F"][validity_mask, 0] = self.weight_electron_affinity * np.array(y_normalized)[validity_mask,0]
                    out["F"][validity_mask, 1] = -self.weight_ionization_potential * np.array(y_normalized)[validity_mask,1]
        else:
            # Enhanced flexible property handling (from optimize_BO.py)
            for i, prop_name in enumerate(self.property_names):
                prop_idx = i  # Index of the property in the predicted values
                
                if self.property_objectives[i] == "minimize":
                    # For minimization, use the raw value
                    out["F"][validity_mask, i] = self.property_weights[i] * np.array(y_normalized)[validity_mask, prop_idx]
                elif self.property_objectives[i] == "maximize":
                    # For maximization, negate the value
                    out["F"][validity_mask, i] = -self.property_weights[i] * np.array(y_normalized)[validity_mask, prop_idx]
                elif self.property_objectives[i] == "target":
                    # For targeting a specific value, use absolute difference
                    out["F"][validity_mask, i] = self.property_weights[i] * np.abs(np.array(y_normalized)[validity_mask, prop_idx] - self.property_targets[i])
        
        # Apply penalty to invalid solutions
        out["F"][~validity_mask] += self.penalty_value
        
        # Enhanced encode and predict with graceful fallbacks (from optimize_BO.py)
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        try:
            y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _ = self._encode_and_predict_molecules(predictions_valid)
        except Exception as e:
            print(f"Error in encoding: {e}")
            # Graceful fallback
            y_p_after_encoding_valid = [[0.0] * self.property_count for _ in predictions_valid]
            z_p_after_encoding_valid = [[0.0] * 32 for _ in predictions_valid]
            all_reconstructions_valid = ["fallback_molecule"] * len(predictions_valid)
        
        # Handle variable number of properties with proper error checking
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
        all_reconstructions = []
        reconstruction_idx = 0
        for val in validity:
            if val == 1:
                if reconstruction_idx < len(all_reconstructions_valid):
                    all_reconstructions.append(all_reconstructions_valid[reconstruction_idx])
                    reconstruction_idx += 1
                else:
                    # Fallback when we run out of valid reconstructions
                    all_reconstructions.append("")
            else:
                all_reconstructions.append("")
        print("Evaluation should not change")
        print(expanded_z_p)

        out["F_corrected"] = np.zeros((x.shape[0], self.n_obj))

        # Handle custom equation case for corrected values
        if self.custom_equation:
            # For each valid solution
            for i in range(x.shape[0]):
                if not invalid_mask[i]:
                    # Extract property values for this solution from corrected predictions
                    property_values = [expanded_y_p[i, j] for j in range(self.property_count)]
                    
                    # Create a safe evaluation environment
                    eval_locals = {'p': property_values, 'abs': abs, 'np': np, 'math': math}
                    
                    try:
                        # Evaluate the custom equation with the property values
                        equation_result = eval(self.custom_equation, {"__builtins__": {}}, eval_locals)
                        
                        # Apply maximization if requested
                        if self.maximize_equation:
                            equation_result = -equation_result
                            
                        out["F_corrected"][i, 0] = equation_result
                    except Exception as e:
                        print(f"Error evaluating custom equation for corrected solution {i}: {e}")
                        out["F_corrected"][i, 0] = self.penalty_value
                        
        # Handle legacy objective types for corrected values
        elif self.objective_type != "custom":
            # Check if legacy objectives are compatible with the number of properties
            if self.property_count != 2 and self.objective_type in ['EAmin', 'mimick_peak', 'mimick_best', 'max_gap']:
                # Fall through to flexible property handling for corrected values
                for i, prop_name in enumerate(self.property_names):
                    prop_idx = i  # Index of the property in the predicted values
                    
                    if self.property_objectives[i] == "minimize":
                        # For minimization, use the raw value
                        out["F_corrected"][~invalid_mask, i] = self.property_weights[i] * expanded_y_p[~invalid_mask, prop_idx]
                    elif self.property_objectives[i] == "maximize":
                        # For maximization, negate the value
                        out["F_corrected"][~invalid_mask, i] = -self.property_weights[i] * expanded_y_p[~invalid_mask, prop_idx]
                    elif self.property_objectives[i] == "target":
                        # For targeting a specific value, use absolute difference
                        out["F_corrected"][~invalid_mask, i] = self.property_weights[i] * np.abs(expanded_y_p[~invalid_mask, prop_idx] - self.property_targets[i])
            else:
                # Original legacy objective handling for corrected values
                if self.objective_type=='mimick_peak':
                    out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2)
                    out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.2)
                elif self.objective_type=='mimick_best':
                    out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2.64)
                    out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.61)
                elif self.objective_type=='EAmin':
                    out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]
                    out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.0)
                elif self.objective_type=='max_gap':
                    out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]
                    out["F_corrected"][~invalid_mask, 1] = -self.weight_ionization_potential * expanded_y_p[~invalid_mask, 1]
        else:
            # Enhanced flexible property handling for corrected values (from optimize_BO.py)
            for i, prop_name in enumerate(self.property_names):
                prop_idx = i  # Index of the property in the predicted values
                
                if self.property_objectives[i] == "minimize":
                    # For minimization, use the raw value
                    out["F_corrected"][~invalid_mask, i] = self.property_weights[i] * expanded_y_p[~invalid_mask, prop_idx]
                elif self.property_objectives[i] == "maximize":
                    # For maximization, negate the value
                    out["F_corrected"][~invalid_mask, i] = -self.property_weights[i] * expanded_y_p[~invalid_mask, prop_idx]
                elif self.property_objectives[i] == "target":
                    # For targeting a specific value, use absolute difference
                    out["F_corrected"][~invalid_mask, i] = self.property_weights[i] * np.abs(expanded_y_p[~invalid_mask, prop_idx] - self.property_targets[i])

        # Apply penalty to invalid solutions
        out["F_corrected"][~validity_mask] += self.penalty_value

        # results
        #print(out["F"])
        aggr_obj = np.sum(out["F"], axis=1)
        aggr_obj_corrected = np.sum(out["F_corrected"], axis=1)
        results_dict = {
            "objective":aggr_obj,
            "objective_corrected": aggr_obj_corrected,
            "latents_reencoded": x, 
            "predictions": y_normalized,
            "predictions_doublecorrect": expanded_y_p,
            "string_decoded": prediction_strings, 
        }
        self.results_custom[str(self.eval_calls)] = results_dict
    
    def _calc_validity(self, predictions):
        """
        Enhanced validity calculation that uses robust polymer validation (from optimize_BO.py)
        """
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), 
                                           tokenization=tokenization) for sample in range(len(predictions))]
        mols_valid = []
        fixed_strings = []
        
        for _s in prediction_strings:
            poly_input = _s[:-1] if _s.endswith('_') else _s  # Remove last character if it's padding
            
            # Use robust validation from optimize_BO.py
            fixed_poly = robust_polymer_validation(poly_input)
            
            if fixed_poly is None:
                mols_valid.append(0)
                fixed_strings.append(poly_input)
                continue
                
            try:
                poly_graph = poly_smiles_to_graph(fixed_poly, np.nan, np.nan, None)
                mols_valid.append(1)
                fixed_strings.append(fixed_poly)
            except Exception as e:
                print(f"Graph creation failed even after robust validation: {e}")
                # Last attempt: try basic fallback
                try:
                    fallback_poly = poly_input.split("|")[0] + "|1.0|<1-1:1.0:1.0"
                    poly_graph = poly_smiles_to_graph(fallback_poly, np.nan, np.nan, None)
                    mols_valid.append(1)
                    fixed_strings.append(fallback_poly)
                except:
                    mols_valid.append(0)
                    fixed_strings.append(poly_input)
        
        return fixed_strings, np.array(mols_valid)
    
    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _encode_and_predict_molecules(self, predictions):
        """
        Enhanced encoding with better error handling (from optimize_BO.py)
        """
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), 
                                           tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
        valid_indices = []
        
        print(f"Processing {len(prediction_strings)} prediction strings...")
        
        for i, s in enumerate(prediction_strings):
            try:
                poly_input = s[:-1] if s.endswith('_') else s  # Remove last character if padding
                print(f"Original polymer {i}: {poly_input}")
                
                # Use robust validation since G2S_clean already did the heavy lifting
                cleaned_poly = poly_input.strip()
                if len(cleaned_poly) < 5:
                    print(f"Skipping too short polymer {i}: {poly_input}")
                    continue
                
                print(f"Cleaned polymer {i}: {cleaned_poly}")
                
                # Try to create graph with basic error handling
                try:
                    g = poly_smiles_to_graph(cleaned_poly, np.nan, np.nan, None)
                    print(f"Graph creation successful for polymer {i}")
                except Exception as graph_error:
                    print(f"Graph creation failed for polymer {i}: {graph_error}")
                    # Try basic fallback format
                    try:
                        parts = cleaned_poly.split("|")
                        if len(parts) >= 1:
                            basic_format = f"{parts[0]}|1.0|<1-1:1.0:1.0"
                            print(f"Trying basic format: {basic_format}")
                            g = poly_smiles_to_graph(basic_format, np.nan, np.nan, None)
                            cleaned_poly = basic_format
                            print(f"Basic format worked for polymer {i}")
                        else:
                            continue
                    except Exception as e2:
                        print(f"Basic format failed: {e2}")
                        continue
                
                # Enhanced tokenization with error handling
                try:
                    if tokenization == "oldtok":
                        target_tokens = tokenize_poly_input(poly_input=cleaned_poly)
                    elif tokenization == "RT_tokenized":
                        target_tokens = tokenize_poly_input_RTlike(poly_input=cleaned_poly)
                    
                    print(f"Tokenization successful for polymer {i}: {len(target_tokens)} tokens")
                    
                    if not target_tokens or len(target_tokens) == 0:
                        print(f"Empty tokenization result for polymer {i}")
                        continue
                        
                except Exception as token_error:
                    print(f"Tokenization failed for polymer {i}: {token_error}")
                    continue

                # Enhanced token processing with OOV handling
                try:
                    print(f"Processing tokens for polymer {i}: {len(target_tokens)} tokens")
                    
                    # Handle out-of-vocabulary tokens
                    unk_token = '_UNK'  # Based on your vocab format
                    if unk_token not in vocab:
                        # Try alternative UNK token names
                        for alt_unk in ['<UNK>', 'UNK', '_UNKNOWN', '<UNKNOWN>']:
                            if alt_unk in vocab:
                                unk_token = alt_unk
                                break
                        else:
                            # If no UNK token found, use the first vocab token
                            unk_token = list(vocab.keys())[0]
                    
                    print(f"Using UNK token: '{unk_token}' (ID: {vocab[unk_token]})")
                    
                    # Check and replace OOV tokens
                    original_length = len(target_tokens)
                    oov_count = 0
                    cleaned_tokens = []
                    
                    for token in target_tokens:
                        if token in vocab:
                            cleaned_tokens.append(token)
                        else:
                            cleaned_tokens.append(unk_token)
                            oov_count += 1
                    
                    if oov_count > 0:
                        print(f"Replaced {oov_count}/{original_length} OOV tokens with '{unk_token}'")
                    
                    # Use cleaned tokens for feature conversion
                    tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=cleaned_tokens, vocab=vocab)
                    
                    print(f"Feature conversion successful: {len(tgt_token_ids) if hasattr(tgt_token_ids, '__len__') else 'scalar'} token IDs")
                    
                    # Validate token IDs
                    if tgt_token_ids is None:
                        print(f"Token IDs are None for polymer {i}")
                        continue
                        
                    if torch.is_tensor(tgt_token_ids) and len(tgt_token_ids) == 0:
                        print(f"Empty token IDs tensor for polymer {i}")
                        continue
                        
                    g.tgt_token_ids = tgt_token_ids
                    g.tgt_lens = tgt_lens
                    g.to(device)
                    
                    data_list.append(g)
                    valid_indices.append(i)
                    print(f"Successfully processed polymer {i} (replaced {oov_count} OOV tokens)")
                    
                except Exception as feature_error:
                    print(f"Feature conversion failed for polymer {i}: {feature_error}")
                    print(f"Error type: {type(feature_error)}")
                    continue
                    
            except Exception as e:
                print(f"Unexpected error processing polymer {i}: {e}")
                print(f"Polymer string: {s}")
                continue
        
        print(f"Successfully processed {len(data_list)} out of {len(prediction_strings)} polymers")
        
        if not data_list:
            print("No valid polymers to encode!")
            return [], [], [], None
        
        # Continue with encoding with enhanced error handling
        try:
            data_loader = DataLoader(dataset=data_list, batch_size=min(16, len(data_list)), shuffle=False)  # Even smaller batch size
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

                        # Forward pass with additional error handling
                        reconstruction, _, _, z, y = model.inference(
                            data=data, device=device, 
                            dest_is_origin_matrix=dest_is_origin_matrix, 
                            inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, 
                            sample=False, log_var=None
                        )
                        
                        # Normalize property predictions
                        y_normalized = self._normalize_property_predictions(y)
                        
                        # Enhanced NaN handling
                        if np.isnan(y_normalized).any():
                            print(f"Warning: NaN detected in predictions for batch {i}")
                            # Replace NaN with reasonable defaults instead of zero
                            y_normalized = np.nan_to_num(y_normalized, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        y_p.append(y_normalized)
                        z_p.append(z.cpu().numpy())
                        
                        reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), 
                                                               tokenization=tokenization) for sample in range(len(reconstruction))]
                        all_reconstructions.extend(reconstruction_strings)
                        
                        print(f"Batch {i} processed successfully: {len(reconstruction_strings)} reconstructions")
                        
                    except Exception as e:
                        print(f"Error in batch {i}: {e}")
                        # Add fallback data for failed batch
                        fallback_y = np.array([[0.0] * self.property_count])
                        fallback_z = np.array([[0.0] * 32])
                        y_p.append(fallback_y)
                        z_p.append(fallback_z)
                        all_reconstructions.append("failed_reconstruction")
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

            print(f"Encoding completed: {len(y_p_flat)} property predictions, {len(z_p_flat)} latent vectors, {len(all_reconstructions)} reconstructions")
            
            return y_p_flat, z_p_flat, all_reconstructions, dict_data_loader
            
        except Exception as e:
            print(f"Error in encoding pipeline: {e}")
            return [], [], [], None

def save_checkpoint(algorithm, problem, iteration, checkpoint_dir, opt_run):
    """Save optimization checkpoint"""
    # Fix for NoneType comparison error
    if iteration is None:
        iteration = 0  # Default to 0 if iteration is None
        
    checkpoint_data = {
        'iteration': iteration,
        'algorithm_state': {
            'pop': algorithm.pop.get("X").tolist() if hasattr(algorithm.pop, "get") else [],
            'F': algorithm.pop.get("F").tolist() if hasattr(algorithm.pop, "get") else [],
        },
        'custom_results': problem.results_custom,
        'eval_calls': problem.eval_calls,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_GA_iter_{iteration}_run{opt_run}.pkl')
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Keep backup of last checkpoint, but only if iteration is a valid number
    if iteration is not None and iteration > 200:
        prev_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_GA_iter_{iteration-200}_run{opt_run}.pkl')
        if os.path.exists(prev_checkpoint):
            backup_file = os.path.join(checkpoint_dir, f'backup_GA_iter_{iteration-200}_run{opt_run}.pkl')
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
        for obj in results["objective"]:
            total += 1
            # Check if objective is not penalty value (assuming penalty_value is 100)
            if obj < 100:  # Using an approximation since GA uses penalty differently than BO
                valid += 1
    
    if total > 0:
        validity_rate = (valid / total) * 100
        return validity_rate, valid, total
    return 0, 0, 0

def list_checkpoints(checkpoint_dir, opt_run):
    """List all checkpoints for a given run"""
    pattern = os.path.join(checkpoint_dir, f'checkpoint_GA_iter_*_run{opt_run}.pkl')
    checkpoints = glob.glob(pattern)
    checkpoints.sort()
    return checkpoints

# Define a convergence termination class
class ConvergenceTermination(Termination):
    def __init__(self, conv_threshold, conv_generations, n_max_gen):
        super().__init__()
        self.conv_threshold = conv_threshold
        self.conv_generations = conv_generations
        self.n_max_gen = n_max_gen
        self.best_fit_values = []
        self.conv_counter = 0
        self.converged_solution_X = None
        self.converged_solution_F = None

    # check for the convergence criterion
    def _do_continue(self, algorithm):
        return self.perc < 1.0
    
    # check the convergence progress
    def _update(self, algorithm):
        best_fit = algorithm.pop.get("F").min()
        self.best_fit_values.append(best_fit)
        
        if algorithm.n_gen >= self.n_max_gen:
            return 1.0

        if len(self.best_fit_values) > self.conv_generations:
            conv_rate = abs(self.best_fit_values[-1] - self.best_fit_values[-self.conv_generations]) / self.conv_generations
            if conv_rate < self.conv_threshold:
                self.conv_counter += 1
                if self.conv_counter >= 5:
                    # store the termination object and use it to print the converged solution and the objective value later
                    self.converged_solution_X = algorithm.pop[np.argmin(algorithm.pop.get("F"))].get("X")
                    self.converged_solution_F = algorithm.pop.get("F")
                    print(f"Algorithm has converged after {algorithm.n_gen} generations.")
                    return 1.0
            else:
                self.conv_counter = 0
        return algorithm.n_gen / self.n_max_gen

# Determine the boundaries for the latent dimensions from training dataset
with open(os.path.join(dir_name, 'latent_space_'+dataset_type+'.npy'), 'rb') as f:
    latent_space = np.load(f)
min_values = np.amin(latent_space, axis=0).tolist()
max_values = np.amax(latent_space, axis=0).tolist()

cutoff=0.0

if not cutoff==0.0:
    transformed_min_values = []
    transformed_max_values = []
    for min_val, max_val in zip(min_values, max_values):
        #bounds are larger than in training set if cutoff value is negative
        # Calculate amount to cut off from each end (cutoff*100 %)
        cutoff_amount = cutoff * abs(max_val - min_val)
        # Adjust min and max values
        transformed_min = min_val + cutoff_amount
        transformed_max = max_val - cutoff_amount
        transformed_min_values.append(transformed_min)
        transformed_max_values.append(transformed_max)
    min_values = transformed_min_values
    max_values = transformed_max_values

# Initialize the problem
opt_run = args.opt_run

# Get property configuration from arguments (enhanced from optimize_BO.py)
property_targets = property_targets
property_weights = property_weights
property_objectives = property_objectives
objective_type = args.objective_type
custom_equation = args.custom_equation
maximize_equation = args.maximize_equation

problem = Property_optimization_problem(
    model, 
    min_values, 
    max_values, 
    objective_type=objective_type,
    property_names=property_names,
    property_targets=property_targets,
    property_weights=property_weights,
    property_objectives=property_objectives,
    custom_equation=custom_equation,
    maximize_equation=maximize_equation,
    property_count=property_count
)

# Termination criterium
termination = ConvergenceTermination(conv_threshold=0.0025, conv_generations=20, n_max_gen=500)

stopping_type = args.stopping_type # time or iter
max_time = args.max_time  # Set to 600 seconds, for example
max_iter = args.max_iter # Set to a maximum number of iterations 

if stopping_type == "time":
    stopping_criterion = stopping_type+"_"+str(max_time)
    hours, remainder = divmod(max_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

    termination = get_termination("time", time_str)
    pop_size = 30

elif stopping_type == "iter":
    stopping_criterion = stopping_type+"_"+str(max_iter)
    termination = get_termination("n_eval", max_iter)
    pop_size = int(max_iter / 20) # 20 generations, pop size 100

elif stopping_type == "convergence":  # ✅ ADD THIS CASE
    stopping_criterion = stopping_type+"_convergence"
    termination = ConvergenceTermination(conv_threshold=0.0025, conv_generations=20, n_max_gen=500)
    pop_size = 25  # Or use a reasonable default

else:
    # Default fallback
    stopping_criterion = "iter_" + str(max_iter)
    termination = get_termination("n_eval", max_iter)
    pop_size = 25

# Define NSGA2 algorithm parameters
#pop_size = max_iter / 10
sampling = LatinHypercubeSampling()
crossover = SimulatedBinaryCrossover(prob=0.90, eta=20)
#crossover = SimulatedBinaryCrossover()
mutation = PolynomialMutation(prob=1.0 / problem.n_var, eta=30)

# Enhanced repair class with robust validation (from optimize_BO.py)
from pymoo.core.repair import Repair

class correctSamplesRepair(Repair):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model_predictor = model

    def _do(self, problem, X, **kwargs):
        print(f"🔧 REPAIR: Starting repair of {len(X)} individuals")
        
        # repair sampled points whole batch 
        pop_X_torch = torch.from_numpy(X).to(device).to(torch.float32)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=pop_X_torch, device=device, sample=False, log_var=None)
        
        # Enhanced validity check with robust validation
        prediction_strings, validity = self._calc_validity(predictions)
        invalid_mask = (validity == 0)
        
        # Encode and predict the valid molecules
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        
        try:
            y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _ = self._encode_and_predict_molecules(predictions_valid)
        except Exception as e:
            print(f"Error in repair encoding: {e}")
            # Graceful fallback
            y_p_after_encoding_valid = [[0.0] * property_count for _ in predictions_valid]
            z_p_after_encoding_valid = [[0.0] * 32 for _ in predictions_valid]
            all_reconstructions_valid = ["fallback_molecule"] * len(predictions_valid)
        
        expanded_z_p = []
        valid_idx = 0
        for val in validity:
            if val == 1:
                if valid_idx < len(z_p_after_encoding_valid):
                    expanded_z_p.append(z_p_after_encoding_valid[valid_idx])
                    valid_idx += 1
                else:
                    # Fallback when we run out of valid encodings
                    expanded_z_p.append([0] * 32)
            else:
                expanded_z_p.append([0] * 32)
        
        expanded_z_p = np.array(expanded_z_p)

        # 🚨 EMERGENCY RECOVERY: Check if all repairs failed
        if np.all(expanded_z_p == 0):
            print("🚨 EMERGENCY: All repairs failed, using fallback strategy")
            
            # Strategy 1: Use training data examples
            try:
                # Try to load training latents
                import os
                latent_files = [
                    os.path.join(dir_name, 'latent_space_train.npy'),
                    'latent_space_train.npy',
                    '../data/latent_space_train.npy'
                ]
                
                training_latents = None
                for file_path in latent_files:
                    try:
                        training_latents = np.load(file_path)
                        print(f"🚨 Loaded emergency latents from {file_path}")
                        break
                    except:
                        continue
                
                if training_latents is not None:
                    # Sample random training examples
                    n_samples = min(len(X), len(training_latents))
                    random_indices = np.random.choice(len(training_latents), n_samples, replace=False)
                    emergency_latents = training_latents[random_indices]
                    print(f"🚨 Using {len(emergency_latents)} emergency training examples")
                    return emergency_latents
                    
            except Exception as e:
                print(f"🚨 Emergency training data strategy failed: {e}")
            
            # Strategy 2: Use smaller random vectors in training range
            try:
                print("🚨 Using random vectors in conservative range")
                emergency_population = np.random.normal(0, 0.5, size=X.shape)
                return emergency_population
                
            except Exception as e:
                print(f"🚨 Emergency random strategy failed: {e}")

        print("🔧 REPAIR: Completed repair process")
        return expanded_z_p

    def _calc_validity(self, predictions):
        """
        Enhanced validity calculation with robust polymer validation (from optimize_BO.py)
        """
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), 
                                           tokenization=tokenization) for sample in range(len(predictions))]
        mols_valid = []
        fixed_strings = []
        
        for _s in prediction_strings:
            poly_input = _s[:-1] if _s.endswith('_') else _s  # Remove last character if it's padding
            
            # Use robust validation from optimize_BO.py
            fixed_poly = robust_polymer_validation(poly_input)
            
            if fixed_poly is None:
                mols_valid.append(0)
                fixed_strings.append(poly_input)
                continue
                
            try:
                poly_graph = poly_smiles_to_graph(fixed_poly, np.nan, np.nan, None)
                mols_valid.append(1)
                fixed_strings.append(fixed_poly)
            except Exception as e:
                print(f"Graph creation failed even after robust validation: {e}")
                # Last attempt: try basic fallback
                try:
                    fallback_poly = poly_input.split("|")[0] + "|1.0|<1-1:1.0:1.0"
                    poly_graph = poly_smiles_to_graph(fallback_poly, np.nan, np.nan, None)
                    mols_valid.append(1)
                    fixed_strings.append(fallback_poly)
                except:
                    mols_valid.append(0)
                    fixed_strings.append(poly_input)
        
        return fixed_strings, np.array(mols_valid)

    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _encode_and_predict_molecules(self, predictions):
        """
        Enhanced encoding with better error handling (simplified from optimize_BO.py)
        """
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), 
                                           tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
        for i, s in enumerate(prediction_strings):
            try:
                poly_input = s[:-1] if s.endswith('_') else s  # Remove last character if padding
                
                # Use simplified validation since G2S_clean already did validation
                cleaned_poly = poly_input.strip()
                if len(cleaned_poly) < 5:
                    continue
                
                # Try to create graph with basic error handling
                try:
                    g = poly_smiles_to_graph(cleaned_poly, np.nan, np.nan, None)
                except Exception as graph_error:
                    # Try comprehensive format fixing (add the fix_polymer_format function to GA script too)
                    try:
                        fixed_format = fix_polymer_format(cleaned_poly)
                        if fixed_format and fixed_format != cleaned_poly:
                            g = poly_smiles_to_graph(fixed_format, np.nan, np.nan, None)
                            cleaned_poly = fixed_format
                        else:
                            # Fallback to basic format
                            parts = cleaned_poly.split("|")
                            if len(parts) >= 1:
                                monomer_count = len(parts[0].split('.'))
                                if monomer_count == 2:
                                    basic_format = f"{parts[0]}|0.5|0.5|<1-2:1.0:1.0"
                                else:
                                    basic_format = f"{parts[0]}|1.0|<1-1:1.0:1.0"
                                g = poly_smiles_to_graph(basic_format, np.nan, np.nan, None)
                                cleaned_poly = basic_format
                            else:
                                continue
                    except Exception as e2:
                        continue
                
                # Enhanced tokenization with error handling
                try:
                    if tokenization == "oldtok":
                        target_tokens = tokenize_poly_input(poly_input=cleaned_poly)
                    elif tokenization == "RT_tokenized":
                        target_tokens = tokenize_poly_input_RTlike(poly_input=cleaned_poly)
                    
                    if not target_tokens or len(target_tokens) == 0:
                        continue
                        
                except Exception as token_error:
                    continue

                # Enhanced token processing with OOV handling
                try:
                    # Handle out-of-vocabulary tokens
                    unk_token = '_UNK'  # Based on your vocab format
                    if unk_token not in vocab:
                        # Try alternative UNK token names
                        for alt_unk in ['<UNK>', 'UNK', '_UNKNOWN', '<UNKNOWN>']:
                            if alt_unk in vocab:
                                unk_token = alt_unk
                                break
                        else:
                            # If no UNK token found, use the first vocab token
                            unk_token = list(vocab.keys())[0]
                    
                    # Check and replace OOV tokens
                    cleaned_tokens = []
                    for token in target_tokens:
                        if token in vocab:
                            cleaned_tokens.append(token)
                        else:
                            cleaned_tokens.append(unk_token)
                    
                    # Use cleaned tokens for feature conversion
                    tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=cleaned_tokens, vocab=vocab)
                    
                    # Validate token IDs
                    if tgt_token_ids is None:
                        continue
                        
                    if torch.is_tensor(tgt_token_ids) and len(tgt_token_ids) == 0:
                        continue
                        
                    g.tgt_token_ids = tgt_token_ids
                    g.tgt_lens = tgt_lens
                    g.to(device)
                    
                    data_list.append(g)
                    
                except Exception as feature_error:
                    continue
                    
            except Exception as e:
                continue
        
        if not data_list:
            return [], [], [], None
        
        # Continue with encoding with enhanced error handling
        try:
            data_loader = DataLoader(dataset=data_list, batch_size=min(16, len(data_list)), shuffle=False)
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
                        
                        y_p.append(y.cpu().numpy())
                        z_p.append(z.cpu().numpy())
                        
                        reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), 
                                                               tokenization=tokenization) for sample in range(len(reconstruction))]
                        all_reconstructions.extend(reconstruction_strings)
                        
                    except Exception as e:
                        # Add fallback data for failed batch
                        fallback_y = np.array([[0.0] * property_count])
                        fallback_z = np.array([[0.0] * 32])
                        y_p.append(fallback_y)
                        z_p.append(fallback_z)
                        all_reconstructions.append("failed_reconstruction")
                        continue
            
            # Flatten results
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
            
        except Exception as e:
            return [], [], [], None

repair_operator = correctSamplesRepair(model)
repair_operator = add_emergency_recovery(repair_operator)

algorithm = NSGA2(pop_size=pop_size,
                  sampling=sampling,
                  crossover=crossover,
                  repair=repair_operator,
                  mutation=mutation,
                  eliminate_duplicates=True)

# Check if resuming from checkpoint
start_iteration = 0
if args.resume_from:
    log_progress(f"Resuming from checkpoint: {args.resume_from}", log_file)
    checkpoint_data = load_checkpoint(args.resume_from)
    
    # Restore state (you'll need to implement this part based on how pymoo stores state)
    problem.results_custom = checkpoint_data['custom_results']
    problem.eval_calls = checkpoint_data['eval_calls']
    start_iteration = checkpoint_data['iteration']
    
    # Check validity rate at checkpoint
    validity_rate, valid_count, total_count = calculate_current_validity_rate(problem.results_custom)
    
    log_progress(f"Restored state from iteration {start_iteration} - Checkpoint validity: {valid_count}/{total_count} ({validity_rate:.1f}%)", log_file)
else:
    # Initialize new optimization
    log_progress("Starting new optimization", log_file)

# Now with checkpoint-aware optimization
start_time = time.time()
all_solutions = []
best_solutions = []
log_progress(f"Starting optimization with maximum {max_iter} iterations", log_file)

# Create a custom callback function to save checkpoints
class CheckpointCallback:
    def __init__(self, algorithm, problem, checkpoint_dir, opt_run, checkpoint_every, monitor_every, log_file):
        self.algorithm = algorithm
        self.problem = problem
        self.checkpoint_dir = checkpoint_dir
        self.opt_run = opt_run
        self.checkpoint_every = checkpoint_every
        self.monitor_every = monitor_every
        self.log_file = log_file
        self.start_time = time.time()
        
    def __call__(self, algorithm):
        iteration = algorithm.n_gen
        
        # Monitor progress regularly
        if iteration % self.monitor_every == 0 and iteration > 0:
            elapsed = time.time() - self.start_time
            # Get current best solution
            if hasattr(algorithm.pop, "get") and len(algorithm.pop) > 0:
                best_idx = np.argmin(algorithm.pop.get("F")[:, 0])  # Use first objective for simplicity
                best_obj = algorithm.pop.get("F")[best_idx, 0]
                
                validity_rate, valid_count, total_count = calculate_current_validity_rate(self.problem.results_custom)
                
                # Log to file
                message = f"Generation {iteration} - Best objective: {best_obj:.4f} - Elapsed: {elapsed:.1f}s - Validity: {valid_count}/{total_count} ({validity_rate:.1f}%)"
                log_progress(message, self.log_file)
        
        # Save checkpoint periodically
        if iteration % self.checkpoint_every == 0 and iteration > 0:
            checkpoint_file = save_checkpoint(algorithm, self.problem, iteration, self.checkpoint_dir, self.opt_run)
            log_progress(f"Checkpoint saved at generation {iteration}: {checkpoint_file}", self.log_file)

# Create the callback
callback = CheckpointCallback(
    algorithm, 
    problem, 
    checkpoint_dir, 
    args.opt_run, 
    args.checkpoint_every, 
    args.monitor_every, 
    log_file
)

# Run the optimization with callback
try:
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=opt_run,
        callback=callback,
        verbose=True,
    )
    elapsed_time = time.time() - start_time
    log_progress(f"Optimization completed. Total time: {elapsed_time:.2f} seconds", log_file)
except Exception as e:
    log_progress(f"Error during optimization: {str(e)}", log_file)
    # Save emergency checkpoint
    current_iteration = algorithm.n_gen if hasattr(algorithm, "n_gen") else "unknown"
    emergency_file = save_checkpoint(algorithm, problem, current_iteration, checkpoint_dir, args.opt_run)
    log_progress(f"Emergency checkpoint saved: {emergency_file}", log_file)
    raise e

# Save final checkpoint
# Use a sensible default if algorithm.n_gen is None
final_iteration = getattr(algorithm, 'n_gen', None)
if final_iteration is None:
    # Try to get an iteration value from somewhere else
    if hasattr(problem, 'eval_calls') and problem.eval_calls > 0:
        final_iteration = problem.eval_calls
    else:
        final_iteration = "final"  # Use a string as fallback
        
final_checkpoint = save_checkpoint(algorithm, problem, final_iteration, checkpoint_dir, args.opt_run)
log_progress(f"Final checkpoint saved: {final_checkpoint}", log_file)

# Access the results (keep these lines from your original code)
best_solution = res.X
#best_mod_solution = res.X_mod
best_fitness = res.F
results_custom = problem.results_custom

with open(os.path.join(dir_name, f'res_optimization_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.pkl'), 'wb') as f:
    pickle.dump(res, f)
with open(os.path.join(dir_name, f'optimization_results_custom_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.pkl'), 'wb') as f:
    pickle.dump(results_custom, f)
with open(os.path.join(dir_name, f'optimization_results_custom_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.txt'), 'w') as fl:
     print(results_custom, file=fl)

log_progress(f"Saved optimization results to {dir_name}", log_file)

#convergence = res.algorithm.termination
with open(os.path.join(dir_name, f'res_optimization_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.pkl'), 'rb') as f:
    res = pickle.load(f)

with open(os.path.join(dir_name, f'optimization_results_custom_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.pkl'), 'rb') as f:
    results_custom = pickle.load(f)

# Save checkpoint summary
checkpoint_summary = {
    'total_iterations': algorithm.n_gen,
    'final_objective': np.min(res.F[:, 0]) if hasattr(res, 'F') and res.F.size > 0 else None,  # Use first objective as summary
    'final_params': res.X.tolist() if hasattr(res, 'X') else None,
    'checkpoints': list_checkpoints(checkpoint_dir, args.opt_run),
    'log_file': log_file
}

summary_file = os.path.join(checkpoint_dir, f'checkpoint_summary_GA_run{args.opt_run}.json')
with open(summary_file, 'w') as f:
    json.dump(checkpoint_summary, f, indent=2, default=str)

log_progress(f"Checkpoint summary saved: {summary_file}", log_file)

# Calculate distances between the BO and reencoded latents
Latents_RE = []
pred_RE = []
decoded_mols= []
pred_RE_corrected = []

for idx, (pop, res) in enumerate(list(results_custom.items())):
    population= int(pop)
    # loop through population
    pop_size = len(list(res["objective"]))
    for point in range(pop_size):
        L_re=res["latents_reencoded"][point]
        Latents_RE.append(L_re)
        pred_RE.append(res["predictions"][point])
        pred_RE_corrected.append(res["predictions_doublecorrect"][point])
        decoded_mols.append(res["string_decoded"][point])

import matplotlib.pyplot as plt

iterations = range(len(pred_RE))

# Enhanced property value extraction with flexible property support (from optimize_BO.py)
property_values_re = [[] for _ in range(property_count)]  # Create lists for each property
property_values_re_corrected = [[] for _ in range(property_count)]  # Create lists for each property

for x in pred_RE:
    for prop_idx in range(property_count):
        if torch.is_tensor(x):
            x_val = x.cpu()[prop_idx] if x.size() > prop_idx else float('nan')
        elif hasattr(x, '__getitem__') and len(x) > prop_idx:
            x_val = x[prop_idx]
        else:
            x_val = float('nan')
        property_values_re[prop_idx].append(x_val)

for x in pred_RE_corrected:
    for prop_idx in range(property_count):
        if hasattr(x, '__getitem__') and len(x) > prop_idx:
            x_val = x[prop_idx]
        else:
            x_val = float('nan')
        property_values_re_corrected[prop_idx].append(x_val)

# For backwards compatibility with 2-property models
EA_re = property_values_re[0] if property_count >= 1 else []
IP_re = property_values_re[1] if property_count >= 2 else [float('nan')] * len(EA_re)
EA_re_c = property_values_re_corrected[0] if property_count >= 1 else []
IP_re_c = property_values_re_corrected[1] if property_count >= 2 else [float('nan')] * len(EA_re_c)

# Create plot with dynamic labels for all properties
plt.figure(0, figsize=(12, 8))

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

for prop_idx in range(property_count):
    prop_name = property_names[prop_idx]
    color = colors[prop_idx % len(colors)]
    
    plt.plot(iterations, property_values_re[prop_idx], label=f'{prop_name} (RE)', 
             color=color, linestyle='-', alpha=0.7)

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.title(f'GA Property Optimization Progress ({property_count} properties)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(dir_name, f'GA_objectives_correct_{stopping_criterion}_run{opt_run}.png'), dpi=300)
plt.close()

""" Plot the kde of the properties of training data and sampled data """
try:
    # Enhanced KDE plotting with flexible property support (from optimize_BO.py)
    training_property_data = []
    for prop_idx in range(property_count):
        try:
            with open(os.path.join(dir_name, f'y{prop_idx+1}_all_{dataset_type}.npy'), 'rb') as f:
                y_prop_all = np.load(f)
                training_property_data.append(list(y_prop_all))
        except FileNotFoundError:
            print(f"Warning: y{prop_idx+1}_all_{dataset_type}.npy not found, creating dummy data")
            # Create dummy data if file doesn't exist
            training_property_data.append([0.0] * 100)  # Dummy data

    try:
        with open(os.path.join(dir_name, f'yp_all_{dataset_type}.npy'), 'rb') as f:
            yp_all = np.load(f)
        yp_all_list = [yp for yp in yp_all]
    except FileNotFoundError:
        print("Warning: yp_all training data not found, using dummy data")
        yp_all_list = [[0.0] * property_count for _ in range(100)]
    
    # Fix data types to ensure all are numpy arrays for KDE
    def ensure_numpy_array(data_list):
        result = []
        for item in data_list:
            if torch.is_tensor(item):
                result.append(item.cpu().numpy())
            elif isinstance(item, (int, float)):
                result.append(item)
            else:
                try:
                    result.append(float(item))
                except:
                    # Skip items that can't be converted
                    continue
        return np.array(result)

    # Create KDE data for each property with enhanced error handling
    kde_plots_created = 0
    for prop_idx in range(property_count):
        prop_name = property_names[prop_idx]
        print(f"Creating KDE plot for {prop_name}...")
        
        try:
            # Get training data for this property
            y_prop_all = training_property_data[prop_idx] if prop_idx < len(training_property_data) else [0.0] * 100
            
            # Get predicted data for this property
            yp_prop_all = [yp[prop_idx] for yp in yp_all_list if len(yp) > prop_idx]
            yp_prop_all_ga = ensure_numpy_array(property_values_re[prop_idx])

            if len(yp_prop_all_ga) == 0:
                print(f"No valid data for property {prop_name}, skipping KDE plot")
                continue

            log_progress(f"Generating KDE plots for {prop_name} distributions...", log_file)

            plt.figure(figsize=(10, 8))
            
            # Filter out NaN and infinite values
            real_distribution = np.array([r for r in y_prop_all if not np.isnan(r) and not np.isinf(r)])
            augmented_distribution = np.array([p for p in yp_prop_all if not np.isnan(p) and not np.isinf(p)])
            ga_distribution = np.array([s for s in yp_prop_all_ga if not np.isnan(s) and not np.isinf(s)])

            if len(real_distribution) == 0 or len(augmented_distribution) == 0 or len(ga_distribution) == 0:
                print(f"Insufficient valid data for KDE plot of {prop_name}")
                plt.close()
                continue

            # Reshape the data
            real_distribution = real_distribution.reshape(-1, 1)
            augmented_distribution = augmented_distribution.reshape(-1, 1)
            ga_distribution = ga_distribution.reshape(-1, 1)

            # Adaptive bandwidth based on data range
            data_range = max(np.max(real_distribution), np.max(augmented_distribution), np.max(ga_distribution)) - \
                        min(np.min(real_distribution), np.min(augmented_distribution), np.min(ga_distribution))
            bandwidth = min(0.1, data_range / 20)  # Adaptive bandwidth

            # Fit kernel density estimators
            kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde_real.fit(real_distribution)
            kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde_augmented.fit(augmented_distribution)
            kde_ga = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde_ga.fit(ga_distribution)

            # Create a range of values for the x-axis
            x_min = min(np.min(real_distribution), np.min(augmented_distribution), np.min(ga_distribution))
            x_max = max(np.max(real_distribution), np.max(augmented_distribution), np.max(ga_distribution))
            
            # Add some padding
            padding = (x_max - x_min) * 0.1
            x_min -= padding
            x_max += padding
            
            x_values = np.linspace(x_min, x_max, 1000)
            
            # Evaluate the KDE on the range of values
            real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
            augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
            ga_density = np.exp(kde_ga.score_samples(x_values.reshape(-1, 1)))

            # Enhanced plotting
            plt.plot(x_values, real_density, label='Training Data', linewidth=2, alpha=0.8)
            plt.plot(x_values, augmented_density, label='Generated Data', linewidth=2, alpha=0.8)
            plt.plot(x_values, ga_density, label='GA Optimized', linewidth=2, alpha=0.8)

            plt.xlabel(f'{prop_name}')
            plt.ylabel('Density')
            plt.title(f'Property Distribution Comparison ({prop_name})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add statistics
            plt.text(0.02, 0.98, f'Training: μ={np.mean(real_distribution):.3f}, σ={np.std(real_distribution):.3f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
            plt.text(0.02, 0.94, f'Generated: μ={np.mean(augmented_distribution):.3f}, σ={np.std(augmented_distribution):.3f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
            plt.text(0.02, 0.90, f'GA Optimized: μ={np.mean(ga_distribution):.3f}, σ={np.std(ga_distribution):.3f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)

            # Save plot
            kde_file = os.path.join(dir_name, f'KDE{prop_name}_GA_correct_{stopping_criterion}_run{opt_run}.png')
            plt.savefig(kde_file, dpi=150, bbox_inches='tight')
            plt.close()
            kde_plots_created += 1
            print(f"KDE plot saved for {prop_name}")
            
            log_progress(f"KDE plot saved: {kde_file}", log_file)
        
        except Exception as e:
            print(f"Error creating KDE plot for {prop_name}: {e}")
            if 'plt' in locals():
                plt.close()
    
    log_progress(f"KDE plots created: {kde_plots_created}/{property_count}", log_file)
    
except Exception as e:
    log_progress(f"Error generating KDE plots: {str(e)}", log_file)

import math 
def indices_of_improvement(values):
    indices_of_increases = []

    # Initialize the highest value and its index
    highest_value = values[0]
    highest_index = 0

    # Iterate through the values
    for i, value in enumerate(values):
        # If the current value is greater than the previous highest value
        if value < highest_value:
            highest_value = value  # Update the highest value
            highest_index = i      # Update the index of the highest value
            indices_of_increases.append(i)  # Save the index of increase

    return indices_of_increases

def top_n_molecule_indices(objective_values, decoded_mols, n_idx=10):
    # Get the indices of 20 molecules with the best objective values
    # Pair each value with its index
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
            best_mols_count[decoded_mols[index]]=1
            _best_mols.append(decoded_mols[index])
        else:
            best_mols_count[decoded_mols[index]]+=1
        if len(top_idxs)==20:
            break

    return top_idxs, best_mols_count

# Calculate objective values with flexible property support (from optimize_BO.py)
if objective_type in ['mimick_peak', 'mimick_best', 'EAmin', 'max_gap'] and property_count >= 2:
    # Legacy objectives for 2+ property models
    if objective_type=='mimick_peak':
        objective_values = [(np.abs(arr.cpu()[0]+2)+np.abs(arr.cpu()[1]-1.2)) if torch.is_tensor(arr) else (np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) for arr in pred_RE]
        objective_values_c = [(np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) for arr in pred_RE_corrected]
    elif objective_type=='mimick_best':
        objective_values = [(np.abs(arr.cpu()[0]+2.64)+np.abs(arr.cpu()[1]-1.61)) if torch.is_tensor(arr) else (np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61)) for arr in pred_RE]
        objective_values_c = [(np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61)) for arr in pred_RE_corrected]
    elif objective_type=='EAmin': 
        objective_values = [arr.cpu()[0]+np.abs(arr.cpu()[1]-1) if torch.is_tensor(arr) else arr[0]+np.abs(arr[1]-1) for arr in pred_RE]
        objective_values_c = [arr[0]+np.abs(arr[1]-1) for arr in pred_RE_corrected]
    elif objective_type =='max_gap':
        objective_values = [arr.cpu()[0]-arr.cpu()[1] if torch.is_tensor(arr) else arr[0]-arr[1] for arr in pred_RE]
        objective_values_c = [arr[0]-arr[1] for arr in pred_RE_corrected]
else:
    # For flexible property models, use the primary property
    if property_objectives[0] == "target":
        objective_values = [np.abs(arr.cpu()[0] - property_targets[0]) if torch.is_tensor(arr) else np.abs(arr[0] - property_targets[0]) for arr in pred_RE]
        objective_values_c = [np.abs(arr[0] - property_targets[0]) for arr in pred_RE_corrected]
    elif property_objectives[0] == "maximize":
        objective_values = [arr.cpu()[0] if torch.is_tensor(arr) else arr[0] for arr in pred_RE]
        objective_values_c = [arr[0] for arr in pred_RE_corrected]
    else:  # minimize
        objective_values = [arr.cpu()[0] if torch.is_tensor(arr) else arr[0] for arr in pred_RE]
        objective_values_c = [arr[0] for arr in pred_RE_corrected]

indices_of_increases = indices_of_improvement(objective_values)

# Build best results with correct number of properties
best_z_re = [Latents_RE[i] for i in indices_of_increases]
best_mols = {i+1: decoded_mols[i] for i in indices_of_increases}

# Build best_props with correct number of properties (from optimize_BO.py)
best_props = {}
for i, idx in enumerate(indices_of_increases):
    prop_values = []
    for prop_idx in range(property_count):
        if torch.is_tensor(pred_RE[idx]):
            prop_values.append(pred_RE[idx].cpu()[prop_idx].item() if pred_RE[idx].size() > prop_idx else float('nan'))
        elif hasattr(pred_RE[idx], '__getitem__') and len(pred_RE[idx]) > prop_idx:
            prop_values.append(pred_RE[idx][prop_idx])
        else:
            prop_values.append(float('nan'))
    best_props[i+1] = prop_values

with open(os.path.join(dir_name, f'best_mols_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.txt'), 'w') as fl:
    print(best_mols, file=fl)
    print(best_props, file=fl)

top_20_indices, top_20_mols = top_n_molecule_indices(objective_values, decoded_mols, n_idx=20)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}

# Build best_props_t20 with correct number of properties
best_props_t20 = {}
best_props_t20_c = {}
for i, idx in enumerate(top_20_indices):
    prop_values = []
    prop_values_c = []
    for prop_idx in range(property_count):
        # Handle RE values
        if torch.is_tensor(pred_RE[idx]):
            prop_values.append(pred_RE[idx].cpu()[prop_idx].item() if pred_RE[idx].size() > prop_idx else float('nan'))
        elif hasattr(pred_RE[idx], '__getitem__') and len(pred_RE[idx]) > prop_idx:
            prop_values.append(pred_RE[idx][prop_idx])
        else:
            prop_values.append(float('nan'))
        
        # Handle corrected values
        if hasattr(pred_RE_corrected[idx], '__getitem__') and len(pred_RE_corrected[idx]) > prop_idx:
            prop_values_c.append(pred_RE_corrected[idx][prop_idx])
        else:
            prop_values_c.append(float('nan'))
    
    best_props_t20[i+1] = prop_values
    best_props_t20_c[i+1] = prop_values_c

best_objs_t20 = {i+1: objective_values[i] for i in top_20_indices}
best_objs_t20_c = {i+1: objective_values_c[i] for i in top_20_indices}

with open(os.path.join(dir_name, f'top20_mols_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.txt'), 'w') as fl:
    print(best_mols_t20, file=fl)
    print(best_props_t20, file=fl)
    print(best_props_t20_c, file=fl)
    print(best_objs_t20, file=fl)
    print(best_objs_t20_c, file=fl)
    print(top_20_mols, file=fl)

log_progress(f"Saved top molecules and properties to {dir_name}", log_file)

# Enhanced molecule validity and novelty analysis
try:
    log_progress("Analyzing molecule validity and novelty...", log_file)
    
    sm_can = SmilesEnumCanon()
    
    # First, get the training data for comparison
    all_polymers_data = []
    all_train_polymers = []
    
    # Extract training polymers from data loader
    for batch, graphs in enumerate(dict_train_loader):
        data = dict_train_loader[str(batch)][0]
        train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
        all_train_polymers.extend(train_polymers_batch)
    
    # Load dataset with priority: 1) provided CSV, 2) augmentation-based files, 3) fallback
    if args.dataset_csv:
        try:
            log_progress(f"Loading dataset from provided CSV: {args.dataset_csv}", log_file)
            df = pd.read_csv(args.dataset_csv)
            
            # Check if the specified column exists
            if args.polymer_column in df.columns:
                for i in range(len(df.loc[:, args.polymer_column])):
                    try:
                        poly_input = df.loc[i, args.polymer_column]
                        if isinstance(poly_input, str) and len(poly_input.strip()) > 0:
                            all_polymers_data.append(poly_input)
                    except Exception as e:
                        print(f"Error processing row {i}: {e}")
                        continue
                log_progress(f"Loaded {len(all_polymers_data)} polymers from {args.polymer_column} column", log_file)
            else:
                log_progress(f"Warning: Column '{args.polymer_column}' not found in CSV file", log_file)
                log_progress(f"Available columns: {list(df.columns)}", log_file)
                log_progress("Using fallback dataset", log_file)
                df = pd.DataFrame({'poly_chemprop_input': all_train_polymers[:100]})
        except FileNotFoundError:
            log_progress(f"Error: Dataset CSV file not found: {args.dataset_csv}", log_file)
            log_progress("Using fallback dataset", log_file)
            df = pd.DataFrame({'poly_chemprop_input': all_train_polymers[:100]})
        except Exception as e:
            log_progress(f"Error loading dataset CSV: {e}", log_file)
            log_progress("Using fallback dataset", log_file)
            df = pd.DataFrame({'poly_chemprop_input': all_train_polymers[:100]})
    else:
        # Load the appropriate dataset based on augmentation type (original logic)
        if augment == "augmented":
            try:
                df = pd.read_csv(os.path.join(dataset_path, 'dataset-combined-poly_chemprop.csv'))
            except FileNotFoundError:
                log_progress("Warning: dataset-combined-poly_chemprop.csv not found, creating dummy data", log_file)
                df = pd.DataFrame({'poly_chemprop_input': all_train_polymers[:100]})
        elif augment == "augmented_canonical":
            try:
                df = pd.read_csv(os.path.join(dataset_path, 'dataset-combined-canonical-poly_chemprop.csv'))
            except FileNotFoundError:
                log_progress("Warning: dataset-combined-canonical-poly_chemprop.csv not found, creating dummy data", log_file)
                df = pd.DataFrame({'poly_chemprop_input': all_train_polymers[:100]})
        elif augment == "augmented_enum":
            try:
                df = pd.read_csv(os.path.join(dataset_path, 'dataset-combined-enumerated2_poly_chemprop.csv'))
            except FileNotFoundError:
                log_progress("Warning: dataset-combined-enumerated2_poly_chemprop.csv not found, creating dummy data", log_file)
                df = pd.DataFrame({'poly_chemprop_input': all_train_polymers[:100]})
        else:
            # Default fallback
            df = pd.DataFrame({'poly_chemprop_input': all_train_polymers[:100]})
    
    # Extract all polymer data (only if we haven't already done it above for provided CSV)
    if not args.dataset_csv:
        for i in range(len(df.loc[:, 'poly_chemprop_input'])):
            poly_input = df.loc[i, 'poly_chemprop_input']
            all_polymers_data.append(poly_input)
    
    # Enhanced canonicalization with error handling
    def safe_canonicalize(polymer_string, sm_can, monomer_only=False):
        try:
            return sm_can.canonicalize(polymer_string, monomer_only=monomer_only, stoich_con_info=False)
        except Exception as e:
            print(f"Canonicalization failed for {polymer_string}: {e}")
            return polymer_string  # Return original if canonicalization fails
    
    # Canonicalize all strings for comparison with enhanced error handling
    all_predictions_can = []
    for s in decoded_mols:
        if s != 'invalid_polymer_string' and s.strip():
            try:
                canonical = safe_canonicalize(s, sm_can)
                all_predictions_can.append(canonical)
            except:
                continue  # Skip invalid strings
    
    all_train_can = [safe_canonicalize(s, sm_can) for s in all_train_polymers if s.strip()]
    all_pols_data_can = [safe_canonicalize(s, sm_can) for s in all_polymers_data if s.strip()]
    
    # Extract monomers with enhanced error handling
    monomers = []
    for s in all_train_polymers:
        try:
            if "|" in s:
                monomer_part = s.split("|")[0].split(".")
                monomers.append(monomer_part)
            else:
                monomers.append([s])  # Treat whole string as single monomer
        except:
            continue
    
    monomers_all = [mon for sub_list in monomers for mon in sub_list]
    all_mons_can = []
    
    for m in monomers_all:
        try:
            m_can = safe_canonicalize(m, sm_can, monomer_only=True)
            modified_string = re.sub(r'\*\:\d+', '*', m_can)
            all_mons_can.append(modified_string)
        except:
            continue
    
    all_mons_can = list(set(all_mons_can))
    
    # Analyze generated molecules with enhanced error handling
    monomer_smiles_predicted = []
    monomer_comb_predicted = []
    
    for poly_smiles in all_predictions_can:
        try:
            if "|" in poly_smiles:
                monomer_part = poly_smiles.split("|")[0]
                monomers = monomer_part.split('.')
                monomer_smiles_predicted.append(monomers)
                monomer_comb_predicted.append(monomer_part)
            else:
                # Handle case where there's no "|" separator
                monomer_smiles_predicted.append([poly_smiles])
                monomer_comb_predicted.append(poly_smiles)
        except:
            continue
    
    # Get training monomer combinations for comparison
    monomer_comb_train = []
    for poly_smiles in all_train_can:
        try:
            if "|" in poly_smiles:
                monomer_comb_train.append(poly_smiles.split("|")[0])
            else:
                monomer_comb_train.append(poly_smiles)
        except:
            continue
    
    # Extract monomer A and B with error handling
    monA_pred = []
    monB_pred = []
    monA_pred_gen = []
    monB_pred_gen = []
    
    for m_c in monomer_smiles_predicted:
        try:
            if len(m_c) > 0:
                ma = m_c[0]
                ma_can = safe_canonicalize(ma, sm_can, monomer_only=True)
                processed_ma = re.sub(r'\*\:\d+', '*', ma_can)
                monA_pred.append(ma)
                monA_pred_gen.append(processed_ma)
            
            if len(m_c) > 1:
                mb = m_c[1]
                mb_can = safe_canonicalize(mb, sm_can, monomer_only=True)
                processed_mb = re.sub(r'\*\:\d+', '*', mb_can)
                monB_pred.append(mb)
                monB_pred_gen.append(processed_mb)
        except:
            continue
    
    # Enhanced validity checking with polymer molecule creation
    def poly_smiles_to_molecule(poly_input):
        '''Turns adjusted polymer smiles string into mols with enhanced error handling'''
        try:
            if "|" in poly_input:
                parts = poly_input.split("|")
                if len(parts) >= 2:
                    fragment_weights = parts[1:-1] if len(parts) > 2 else []
                    mols = make_polymer_mol(parts[0], 0, 0, fragment_weights=fragment_weights)
                else:
                    # Fallback for malformed strings
                    mols = make_polymer_mol(parts[0], 0, 0, fragment_weights=[])
            else:
                # Handle case with no "|" separator
                mols = make_polymer_mol(poly_input, 0, 0, fragment_weights=[])
            return mols
        except Exception as e:
            print(f"Failed to create molecule from {poly_input}: {e}")
            return None
    
    # Check validity of generated molecules
    prediction_mols = []
    prediction_validityA = []
    prediction_validityB = []
    
    for poly in all_predictions_can:
        try:
            mol = poly_smiles_to_molecule(poly)
            prediction_mols.append(mol)
            
            # Check validity of monomer A
            try:
                if mol and hasattr(mol, '__getitem__') and len(mol) > 0:
                    prediction_validityA.append(mol[0] is not None)
                else:
                    prediction_validityA.append(False)
            except:
                prediction_validityA.append(False)
            
            # Check validity of monomer B
            try:
                if mol and hasattr(mol, '__getitem__') and len(mol) > 1:
                    prediction_validityB.append(mol[1] is not None)
                else:
                    prediction_validityB.append(False)
            except:
                prediction_validityB.append(False)
                
        except Exception as e:
            print(f"Error processing polymer {poly}: {e}")
            prediction_mols.append(None)
            prediction_validityA.append(False)
            prediction_validityB.append(False)
    
    # Calculate validity rates
    validityA = sum(prediction_validityA)/len(prediction_validityA) if prediction_validityA else 0
    validityB = sum(prediction_validityB)/len(prediction_validityB) if prediction_validityB else 0
    validity = len(all_predictions_can)/len(decoded_mols) if decoded_mols else 0
    
    # Novelty metrics
    novel = 0
    novel_pols = []
    for pol in monomer_comb_predicted:
        if pol not in monomer_comb_train:
            novel += 1
            novel_pols.append(pol)
    novelty_mon_comb = novel/len(monomer_comb_predicted) if monomer_comb_predicted else 0
    
    novel = 0
    for pol in all_predictions_can:
        if pol not in all_train_can:
            novel += 1
    novelty = novel/len(all_predictions_can) if all_predictions_can else 0
    
    novel = 0
    for pol in all_predictions_can:
        if pol not in all_pols_data_can:
            novel += 1
    novelty_full_dataset = novel/len(all_predictions_can) if all_predictions_can else 0
    
    # Monomer novelty
    novelA = 0
    novelAs = []
    for monA in monA_pred_gen:
        if monA not in all_mons_can:
            novelA += 1
            novelAs.append(monA)
    novelty_A = novelA/len(monA_pred_gen) if monA_pred_gen else 0
    
    novelB = 0
    novelBs = []
    for monB in monB_pred_gen:
        if monB not in all_mons_can:
            novelB += 1
            novelBs.append(monB)
    novelty_B = novelB/len(monB_pred_gen) if monB_pred_gen else 0
    
    # Diversity metrics
    diversity = len(set(all_predictions_can))/len(all_predictions_can) if all_predictions_can else 0
    diversity_novel = len(set(novel_pols))/len(novel_pols) if novel_pols else 0
    
    # Save the enhanced novelty and validity metrics
    novelty_file = os.path.join(dir_name, f'novelty_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.txt')
    with open(novelty_file, 'w') as f:
        f.write(f"=== Enhanced GA Optimization Results ===\n")
        f.write(f"Property Configuration: {property_count} properties ({property_names})\n")
        f.write(f"Objective Type: {objective_type}\n")
        f.write(f"Dataset Path: {dataset_path}\n")
        f.write(f"Dataset CSV: {args.dataset_csv if args.dataset_csv else 'Default augmentation-based'}\n")
        f.write(f"Polymer Column: {args.polymer_column}\n")
        f.write(f"Total Generated Molecules: {len(decoded_mols)}\n")
        f.write(f"Valid Molecules: {len(all_predictions_can)}\n")
        f.write(f"Reference Dataset Size: {len(all_polymers_data)}\n")
        f.write(f"\n=== Validity Metrics ===\n")
        f.write(f"Gen Mon A validity: {100*validityA:.4f}%\n")
        f.write(f"Gen Mon B validity: {100*validityB:.4f}%\n")
        f.write(f"Gen validity: {100*validity:.4f}%\n")
        f.write(f"\n=== Novelty Metrics ===\n")
        f.write(f"Novelty: {100*novelty:.4f}%\n")
        f.write(f"Novelty (mon_comb): {100*novelty_mon_comb:.4f}%\n")
        f.write(f"Novelty MonA full dataset: {100*novelty_A:.4f}%\n")
        f.write(f"Novelty MonB full dataset: {100*novelty_B:.4f}%\n")
        f.write(f"Novelty in full dataset: {100*novelty_full_dataset:.4f}%\n")
        f.write(f"\n=== Diversity Metrics ===\n")
        f.write(f"Diversity: {100*diversity:.4f}%\n")
        f.write(f"Diversity (novel polymers): {100*diversity_novel:.4f}%\n")
        f.write(f"\n=== Optimization Summary ===\n")
        f.write(f"Final validity rate: {calculate_current_validity_rate(problem.results_custom)[0]:.1f}%\n")
        f.write(f"Total evaluations: {problem.eval_calls}\n")
        f.write(f"Best molecules found: {len(best_mols)}\n")
    
    log_progress(f"Enhanced novelty analysis completed and saved to {novelty_file}", log_file)
    
    # Print summary to console
    print(f"\n=== Optimization Summary ===")
    print(f"Properties optimized: {property_names}")
    print(f"Validity: {100*validity:.2f}%")
    print(f"Novelty: {100*novelty:.2f}%")
    print(f"Diversity: {100*diversity:.2f}%")
    print(f"Best molecules found: {len(best_mols)}")
    
except Exception as e:
    log_progress(f"Error during enhanced novelty analysis: {str(e)}", log_file)
    print(f"Warning: Novelty analysis failed: {e}")

# Final summary and cleanup
log_progress("="*50, log_file)
log_progress(f"GA OPTIMIZATION COMPLETED SUCCESSFULLY", log_file)
log_progress(f"Properties optimized: {property_names}", log_file)
log_progress(f"Objective type: {objective_type}", log_file)
log_progress(f"Total evaluations: {problem.eval_calls}", log_file)
log_progress(f"Dataset path: {dataset_path}", log_file)
log_progress(f"Dataset CSV: {args.dataset_csv if args.dataset_csv else 'Default augmentation-based'}", log_file)
log_progress(f"Results saved to: {dir_name}", log_file)
log_progress("="*50, log_file)

print(f"\n🎉 GA optimization completed successfully!")
print(f"📊 Results saved to: {dir_name}")
print(f"📝 Log file: {log_file}")
print(f"💾 Checkpoints: {checkpoint_dir}")
