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

# setting device on GPU if available, else CPU
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

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
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--max_time", type=int, default=3600)
parser.add_argument("--stopping_type", type=str, default="iter", choices=["iter","time"])
parser.add_argument("--opt_run", type=int, default=1)
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load model checkpoints from and save results to")
parser.add_argument("--checkpoint_every", type=int, default=200, help="Save checkpoint every N iterations")
parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint file to resume from")
parser.add_argument("--monitor_every", type=int, default=50, help="Print progress every N iterations")
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties to optimize")
parser.add_argument("--property_targets", type=float, nargs='+', default=None,
                    help="Target values for each property (one per property)")
parser.add_argument("--property_weights", type=float, nargs='+', default=None,
                    help="Weights for each property in the objective function (one per property)")
parser.add_argument("--property_objectives", type=str, nargs='+', default=None, 
                    choices=["minimize", "maximize", "target"],
                    help="Objective for each property: minimize, maximize, or target a specific value")
parser.add_argument("--objective_type", type=str, default="EAmin",
                    choices=["EAmin", "mimick_peak", "mimick_best", "max_gap", "custom"],
                    help="Type of objective function to use (legacy option)")
parser.add_argument("--custom_equation", type=str, default=None,
                    help="Custom equation for objective function. Use 'p[i]' to reference property i. Example: '(1 + p[0] - p[1])*2'")
parser.add_argument("--maximize_equation", action="store_true", 
                    help="Maximize the custom equation instead of minimizing it")


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

# Load model
# Create an instance of the G2S model from checkpoint
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
filepath = os.path.join(args.save_dir, model_name, "model_best_loss.pt")

if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_VAE_PPguided
    else: 
        model_type = G2S_VAE_PPguideddisabled
        
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    batch_size = model_config['batch_size']
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

# PropertyPrediction class - modified initialization
class PropertyPrediction():
    def __init__(self, model, nr_vars, objective_type="EAmin", property_names=None, 
                 property_targets=None, property_weights=None, property_objectives=None,
                 custom_equation=None, maximize_equation=False):
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
        
        # New flexible property configuration
        self.property_names = property_names if property_names else ["EA", "IP"]
        self.property_weights = property_weights if property_weights else [1.0] * len(self.property_names)
        self.property_targets = property_targets
        self.property_objectives = property_objectives if property_objectives else ["minimize"] * len(self.property_names)
        
        # Validate inputs
        if self.property_weights and len(self.property_weights) != len(self.property_names):
            raise ValueError("Number of property weights must match number of properties")
        if self.property_targets and len(self.property_targets) != len(self.property_names):
            raise ValueError("Number of property targets must match number of properties")
        if self.property_objectives and len(self.property_objectives) != len(self.property_names):
            raise ValueError("Number of property objectives must match number of properties")
            
        # Check that targets are provided for 'target' objectives
        if self.property_objectives:
            for i, obj in enumerate(self.property_objectives):
                if obj == "target" and (not self.property_targets or self.property_targets[i] is None):
                    raise ValueError(f"Target value must be provided for property {self.property_names[i]}")
                    
        # If using a custom equation, we need at least one property defined
        if self.custom_equation and not property_names:
            raise ValueError("Property names must be provided when using a custom equation")

    
    def evaluate(self, **params):
        # Assuming x is a 1D array containing the 32 numerical parameters

        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        #x = torch.from_numpy(np.array(list(params.values()))).to(device).to(torch.float32)
        self.eval_calls += 1
        print(params)
        _vector = [params[f'x{i}'] for i in range(self.nr_vars)]
        print(_vector)
        
        x = torch.from_numpy(np.array(_vector)).to(device).to(torch.float32)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=x, device=device, sample=False, log_var=None)
        # Validity check of the decoded molecule + penalize invalid molecules
        prediction_strings, validity = self._calc_validity(predictions)
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        prediction_strings_valid = [j for j, valid in zip(prediction_strings, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _ = self._encode_and_predict_decode_molecules(predictions_valid)
        invalid_mask = (validity == 0)
        # Encode and predict the valid molecules
        expanded_y_p = np.array([y_p_after_encoding_valid.pop(0) if val == 1 else [np.nan,np.nan] for val in list(validity)])
        expanded_z_p = np.array([z_p_after_encoding_valid.pop(0) if val == 1 else [0] * 32 for val in list(validity)])
        #print(x, expanded_z_p)

        
        # If using a custom equation
        if self.objective_type == "custom" and self.custom_equation:
            if validity[0]:
                # Extract property values into a list
                property_values = [expanded_y_p[~invalid_mask, i][0] for i in range(len(self.property_names))]
                
                # Create a safe evaluation environment
                eval_locals = {'p': property_values, 'abs': abs, 'np': np, 'math': math}
                
                try:
                    # Evaluate the custom equation with the property values
                    equation_result = eval(self.custom_equation, {"__builtins__": {}}, eval_locals)
                    
                    # Apply maximization if requested
                    if self.maximize_equation:
                        equation_result = -equation_result
                        
                    aggr_obj = -equation_result  # The negative sign is because BO maximizes by default
                except Exception as e:
                    print(f"Error evaluating custom equation: {e}")
                    aggr_obj = self.penalty_value
            else:
                aggr_obj = self.penalty_value
        # If using legacy objective types, handle those
        elif self.objective_type != "custom":
            if self.objective_type=='EAmin':
                obj1 = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]
                obj2 = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1)
            elif self.objective_type=='mimick_peak':
                obj1 = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2)
                obj2 = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.2)
            elif self.objective_type=='mimick_best':
                obj1 = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2.64)
                obj2 = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.61)
            elif self.objective_type=='max_gap':
                obj1 = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]
                obj2 = - self.weight_ionization_potential * expanded_y_p[~invalid_mask, 1]
            
            if validity[0]:
                obj3 = 0
                aggr_obj = -(obj1[0] + obj2[0] + obj3)
            else:
                obj3 = self.penalty_value
                aggr_obj = obj3
        else:
            # New flexible property handling without custom equation
            if validity[0]:
                obj_values = []
                
                for i, prop_name in enumerate(self.property_names):
                    prop_idx = i  # Index of the property in the predicted values
                    
                    if self.property_objectives[i] == "minimize":
                        # For minimization, use the raw value
                        obj_value = expanded_y_p[~invalid_mask, prop_idx][0]
                    elif self.property_objectives[i] == "maximize":
                        # For maximization, negate the value
                        obj_value = -expanded_y_p[~invalid_mask, prop_idx][0]
                    elif self.property_objectives[i] == "target":
                        # For targeting a specific value, use absolute difference
                        obj_value = np.abs(expanded_y_p[~invalid_mask, prop_idx][0] - self.property_targets[i])
                    
                    # Apply the weight
                    obj_value *= self.property_weights[i]
                    obj_values.append(obj_value)
                
                # Sum up all objective components
                aggr_obj = -sum(obj_values)
            else:
                aggr_obj = self.penalty_value
        
        # results

        results_dict = {
        "objective":aggr_obj,
        "latents_BO": x,
        "latents_reencoded": expanded_z_p, 
        "predictions_BO": y,
        "predictions_reencoded": expanded_y_p,
        "string_decoded": prediction_strings, 
        "string_reconstructed": all_reconstructions_valid,
        }
        self.results_custom[str(self.eval_calls)] = results_dict
        
        # Remove the verbose printing that interferes with default output
        
        return aggr_obj

    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _calc_validity(self, predictions):
        # Molecule validity check     
        # Return a boolean array indicating whether each solution is valid or not
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        mols_valid= []
        for _s in prediction_strings:
            poly_input = _s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            try: 
                poly_graph=poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
                mols_valid.append(1)
            except:
                poly_graph = None
                mols_valid.append(0)
        mols_valid = np.array(mols_valid) # List of lists
        return prediction_strings, mols_valid
    
    def _encode_and_predict_decode_molecules(self, predictions):
        # create data that can be encoded again
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
        print(prediction_strings)
        for i, s in enumerate(prediction_strings):
            poly_input = s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            g = poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
            if tokenization=="oldtok":
                target_tokens = tokenize_poly_input(poly_input=poly_input)
            elif tokenization=="RT_tokenized":
                target_tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
            tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=target_tokens, vocab=vocab)
            g.tgt_token_ids = tgt_token_ids
            g.tgt_token_lens = tgt_lens
            g.to(device)
            data_list.append(g)
        data_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
        dict_data_loader = MP_Matrix_Creator(data_loader, device)

        #Encode and predict
        batches = list(range(len(dict_data_loader)))
        y_p = []
        z_p = []
        all_reconstructions = []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                data = dict_data_loader[str(batch)][0]
                data.to(device)
                dest_is_origin_matrix = dict_data_loader[str(batch)][1]
                dest_is_origin_matrix.to(device)
                inc_edges_to_atom_matrix = dict_data_loader[str(batch)][2]
                inc_edges_to_atom_matrix.to(device)

                # Perform a single forward pass.
                reconstruction, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
                y_p.append(y.cpu().numpy())
                z_p.append(z.cpu().numpy())
                reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(reconstruction))]
                all_reconstructions.extend(reconstruction_strings)
        #Return the predictions from the encoded latents
        y_p_flat = [sublist.tolist() for array_ in y_p for sublist in array_]
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

cutoff=-0.05

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

# Instantiating the PropertyPrediction class - updated
nr_vars = 32

# Get property configuration from arguments
property_names = args.property_names
property_targets = args.property_targets
property_weights = args.property_weights
property_objectives = args.property_objectives
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
    maximize_equation=maximize_equation
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
#utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.01)
utility = UtilityFunction(kind="ucb")

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
    init_points = 20
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

#optimizer.maximize(init_points=20, n_iter=500, acquisition_function=utility)
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
#print(results_custom)
# Calculate distances between the BO and reencoded latents
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

# Extract data for the curves
iterations = range(len(pred_BO))

# Handle different possible formats safely with proper error checking
EA_bo = []
IP_bo = []
for x in pred_BO:
    # First ensure x is a numpy array that we can work with
    if torch.is_tensor(x):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x) if not isinstance(x, np.ndarray) else x
    
    # Access elements safely - handle (1,2) shaped arrays
    if x_np.ndim > 1 and x_np.shape[1] >= 2:  # Handle (1,2) shaped arrays
        EA_bo.append(x_np[0, 0])
        IP_bo.append(x_np[0, 1])
    elif x_np.size >= 2:  # Check if flat array has at least 2 elements
        EA_bo.append(x_np[0])
        IP_bo.append(x_np[1])
    elif x_np.size == 1:  # Handle arrays with only 1 element
        EA_bo.append(x_np[0])
        IP_bo.append(float('nan'))  # Use NaN as a placeholder
    else:  # Handle empty arrays
        EA_bo.append(float('nan'))
        IP_bo.append(float('nan'))

# Similarly handle pred_RE with proper error checking
EA_re = []
IP_re = []
for x in pred_RE:
    if isinstance(x, np.ndarray):
        if x.ndim > 1 and x.shape[1] >= 2:  # Handle (1,2) shaped arrays
            EA_re.append(x[0, 0])
            IP_re.append(x[0, 1])
        elif x.size >= 2:  # Check if flat array has at least 2 elements
            EA_re.append(x[0])
            IP_re.append(x[1])
        elif x.size == 1:
            EA_re.append(x[0])
            IP_re.append(float('nan'))
        else:
            EA_re.append(float('nan'))
            IP_re.append(float('nan'))
    else:
        # Try to handle as a list or other indexable object
        try:
            if len(x) >= 2:
                EA_re.append(x[0])
                IP_re.append(x[1])
            elif len(x) == 1:
                EA_re.append(x[0])
                IP_re.append(float('nan'))
            else:
                EA_re.append(float('nan'))
                IP_re.append(float('nan'))
        except:
            EA_re.append(float('nan'))
            IP_re.append(float('nan'))

EA_re = [x[0] if isinstance(x, np.ndarray) and x.size > 0 else 
         
         
         
         
         
         ('nan') for x in pred_RE]
IP_re = [x[1] if isinstance(x, np.ndarray) and x.size > 0 else float('nan') for x in pred_RE]

# Create plot
plt.figure(0)

plt.plot(iterations, EA_bo, label='EA (BO)')
plt.plot(iterations, IP_bo, label='IP (BO)')
plt.plot(iterations, EA_re, label='EA (RE)')
plt.plot(iterations, IP_re, label='IP (RE)')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.savefig(dir_name+'BO_objectives_'+str(cutoff)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png',  dpi=300)
plt.close()


# Plot the training data pca and optimization points
    
# Dimensionality reduction
#PCA
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


# PCA projection colored by EA
plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=2, c='black')
plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=2, c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(dir_name+'BO_projected_to_pca_'+str(cutoff)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png',  dpi=300)
plt.close()
#pca = PCA(n_components=2)



# PCA projection colored by IP
plt.figure(1)
plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y2_all_train, cmap='plasma')
clb = plt.colorbar()
clb.ax.set_title('Ionization potential')
plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=2, c='black')
plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=2, c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection Colored by Ionization Potential')
plt.savefig(dir_name + 'BO_projected_to_pca_IP_' + str(cutoff) + '_' + str(stopping_criterion) + '_run' + str(opt_run) + '.png', dpi=300)
plt.close()




### Do the same but only for improved points
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
    # Get the indices of 20 molecules with the highest objective values
    # Pair each value with its index
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
        if len(top_idxs)==20:
            break

    return top_idxs, best_mols_count

# Extract data for the curves
if objective_type=='mimick_peak':
    objective_values = [-(np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) if not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]
elif objective_type=='mimick_best':
    objective_values = [-((np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61))) if not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]
elif objective_type=='EAmin': 
    objective_values = [-(arr[0]+np.abs(arr[1]-1)) if not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]
elif objective_type =='max_gap':
    objective_values = [-(arr[0]-arr[1]) if not np.isnan(arr[0]) and not np.isnan(arr[1]) else float('-inf') for arr in pred_RE]

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

# Now proceed with using the indices
EA_bo_imp = [EA_bo[i] for i in indices_of_increases]
IP_bo_imp = [IP_bo[i] for i in indices_of_increases]
EA_re_imp = [EA_re[i] for i in indices_of_increases]
IP_re_imp = [IP_re[i] for i in indices_of_increases]
best_z_re = [Latents_RE[i] for i in indices_of_increases]
best_mols = {i+1: decoded_mols[i] for i in indices_of_increases}
best_props = {i+1: [EA_re[i], EA_bo[i], IP_re[i], IP_bo[i]] for i in indices_of_increases}
best_mols_rec = {i+1: rec_mols[i] for i in indices_of_increases}


with open(dir_name+'best_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols, file=fl)
    print(best_props, file=fl)
with open(dir_name+'best_recon_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols_rec, file=fl)

top_20_indices, top_20_mols = top_n_molecule_indices(objective_values, n_idx=20)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}
best_props_t20 = {i+1: [EA_re[i], EA_bo[i], IP_re[i], IP_bo[i]] for i in top_20_indices}
with open(dir_name+'top20_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols_t20, file=fl)
    print(best_props_t20, file=fl)


print(objective_values)
print(indices_of_improvement)
latents_BO_np_imp = np.stack([Latents_BO[i] for i in indices_of_increases])
z_embedded_BO_imp = reducer.transform(latents_BO_np_imp)

latents_RE_np_imp = np.stack([Latents_RE[i] for i in indices_of_increases])
z_embedded_RE_imp = reducer.transform(latents_RE_np_imp)

plt.figure(2)


plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis', alpha=0.2)
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
#plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=1, c='black', marker="1")
#plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=1, c='red',marker="2")
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

plt.figure(3)


plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis', alpha=0.2)
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
#plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=1, c='black', marker="1")
#plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=1, c='red',marker="2")
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



""" Sample around seed molecule - seed being the optimal solution found by optimizer """
# Sample around the optimal molecule and predict the property values

all_prediction_strings=[]
all_reconstruction_strings=[]


# Add these debug prints BEFORE the above line:
print(f"\nCHECKING BEST MOLECULE:")
print(f"best_z_re length: {len(best_z_re) if 'best_z_re' in locals() else 'not found'}")
print(f"best_z_re type: {type(best_z_re) if 'best_z_re' in locals() else 'not found'}")
if 'best_z_re' in locals() and len(best_z_re) > 0:
    print(f"Last best_z_re: {best_z_re[-1]}")
else:
    print("No valid best molecules found!")

seed_z = best_z_re[-1] # last reencoded best molecule

# extract the predictions of best molecule together with the latents of best molecule(by encoding? or BO one)
seed_z = torch.from_numpy(np.array(seed_z)).unsqueeze(0).repeat(64,1).to(device).to(torch.float32)
print(seed_z)
with torch.no_grad():
    predictions, _, _, _, y_seed = model.inference(data=seed_z, device=device, sample=False, log_var=None)
    seed_string = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
sampled_z = []
all_y_p = []
#model.eval()

with torch.no_grad():
    # Define the mean and standard deviation of the Gaussian noise
    for r in range(8):
        mean = 0
        std = args.epsilon
        
        # Create a tensor of the same size as the original tensor with random noise
        noise = torch.tensor(np.random.normal(mean, std, size=seed_z.size()), dtype=torch.float, device=device)

        # Add the noise to the original tensor
        seed_z_noise = seed_z + noise
        sampled_z.append(seed_z_noise.cpu().numpy())
        predictions, _, _, _, y = model.inference(data=seed_z_noise, device=device, sample=False, log_var=None)
        prediction_strings, validity = prop_predictor._calc_validity(predictions)
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        prediction_strings_valid = [j for j, valid in zip(prediction_strings, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, reconstructions_valid, _ = prop_predictor._encode_and_predict_decode_molecules(predictions_valid)
        all_prediction_strings.extend(prediction_strings_valid)
        all_reconstruction_strings.extend(reconstructions_valid)
        all_y_p.extend(y_p_after_encoding_valid)


# Add these debug prints BEFORE the above lines:
print(f"\nSAMPLING RESULTS CHECK:")
print(f"all_prediction_strings length: {len(all_prediction_strings)}")
print(f"all_reconstruction_strings length: {len(all_reconstruction_strings)}")
print(f"all_y_p length: {len(all_y_p)}")
if all_y_p:
    print(f"First 3 y_p values: {all_y_p[:3]}")
    print(f"Types in all_y_p: {[type(x) for x in all_y_p[:3]]}")
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

all_predictions=all_prediction_strings.copy()
sm_can = SmilesEnumCanon()
all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
prediction_validityA= []
prediction_validityB =[]
data_dir = os.path.join(main_dir_path,'data/')


if augment=="augmented":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop.csv')
elif augment=="augmented_canonical":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-canonical-poly_chemprop.csv')
elif augment=="augmented_enum":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-enumerated2_poly_chemprop.csv')
all_polymers_data= []
all_train_polymers = []

for batch, graphs in enumerate(dict_train_loader):
    data = dict_train_loader[str(batch)][0]
    train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
    all_train_polymers.extend(train_polymers_batch)
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    all_polymers_data.append(poly_input)

 
all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
all_train_can = list(map(sm_can.canonicalize, all_train_polymers))
all_pols_data_can = list(map(sm_can.canonicalize, all_polymers_data))
monomers= [s.split("|")[0].split(".") for s in all_train_polymers]
monomers_all=[mon for sub_list in monomers for mon in sub_list]
all_mons_can = []
for m in monomers_all:
    m_can = sm_can.canonicalize(m, monomer_only=True, stoich_con_info=False)
    modified_string = re.sub(r'\*\:\d+', '*', m_can)
    all_mons_can.append(modified_string)
all_mons_can = list(set(all_mons_can))
print(len(all_mons_can), all_mons_can[1:3])

with open(data_dir+'all_train_pols_can'+'.pkl', 'wb') as f:
    pickle.dump(all_train_can, f)
with open(data_dir+'all_pols_data_can'+'.pkl', 'wb') as f:
    pickle.dump(all_pols_data_can, f)
with open(data_dir+'all_mons_train_can'+'.pkl', 'wb') as f:
    pickle.dump(all_mons_can, f)

with open(data_dir+'all_train_pols_can'+'.pkl', 'rb') as f:
    all_train_can = pickle.load(f)
with open(data_dir+'all_pols_data_can'+'.pkl', 'rb') as f:
    all_pols_data_can= pickle.load(f)
with open(data_dir+'all_mons_train_can'+'.pkl', 'rb') as f:
    all_mons_can = pickle.load(f)


# C
prediction_mols = list(map(poly_smiles_to_molecule, all_predictions))
for mon in prediction_mols: 
    try: prediction_validityA.append(mon[0] is not None)
    except: prediction_validityA.append(False)
    try: prediction_validityB.append(mon[1] is not None)
    except: prediction_validityB.append(False)


#predBvalid = []
#for mon in prediction_mols:
#    try: 
#        predBvalid.append(mon[1] is not None)
#    except: 
#        predBvalid.append(False)

#prediction_validityB.append(predBvalid)
#reconstructed_SmilesA = list(map(Chem.MolToSmiles, [mon[0] for mon in prediction_mols]))
#reconstructed_SmilesB = list(map(Chem.MolToSmiles, [mon[1] for mon in prediction_validity]))


# Evaluation of validation set reconstruction accuracy (inference)
monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_comb_predicted = [poly_smiles.split("|")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_comb_train = [poly_smiles.split("|")[0] for poly_smiles in all_train_can if poly_smiles]

monA_pred = [mon[0] for mon in monomer_smiles_predicted]
monB_pred = [mon[1] for mon in monomer_smiles_predicted]
monA_pred_gen = []
monB_pred_gen = []
for m_c in monomer_smiles_predicted:
    ma = m_c[0]
    mb = m_c[1]
    ma_can = sm_can.canonicalize(ma, monomer_only=True, stoich_con_info=False)

    monA_pred_gen.append(re.sub(r'\*\:\d+', '*', ma_can))
    mb_can = sm_can.canonicalize(mb, monomer_only=True, stoich_con_info=False)
    monB_pred_gen.append(re.sub(r'\*\:\d+', '*', mb_can))

monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_con_predicted = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']


#prediction_validityA= [num for elem in prediction_validityA for num in elem]
#prediction_validityB = [num for elem in prediction_validityB for num in elem]
validityA = sum(prediction_validityA)/len(prediction_validityA)
validityB = sum(prediction_validityB)/len(prediction_validityB)

# Novelty metrics
novel = 0
novel_pols=[]
for pol in monomer_comb_predicted:
    if not pol in monomer_comb_train:
        novel+=1
        novel_pols.append(pol)
novelty_mon_comb = novel/len(monomer_comb_predicted)
novel = 0
novel_pols=[]
for pol in all_predictions_can:
    if not pol in all_train_can:
        novel+=1
        novel_pols.append(pol)
novelty = novel/len(all_predictions_can)
novel = 0
for pol in all_predictions_can:
    if not pol in all_pols_data_can:
        novel+=1
novelty_full_dataset = novel/len(all_predictions_can)
novelA = 0
novelAs = []
for monA in monA_pred_gen:
    if not monA in all_mons_can:
        novelA+=1
        novelAs.append(monA)
print(novelAs, len(list(set(novelAs))))
novelty_A = novelA/len(monA_pred_gen)
novelB = 0
novelBs = []
for monB in monB_pred_gen:
    if not monB in all_mons_can:
        novelB+=1
        novelBs.append(monB)
print(novelBs, len(list(set(novelBs))))

novelty_B = novelB/len(monB_pred_gen)

diversity = len(list(set(all_predictions_can)))/len(all_predictions_can)
diversity_novel = len(list(set(novel_pols)))/len(novel_pols)

classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
#if data_augment=='new':
#    classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
#else:
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
whole_valid = len(monomer_smiles_predicted)
validity = whole_valid/len(all_predictions)
with open(dir_name+'novelty_BO_seed_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as f:
    f.write("Gen Mon A validity: %.4f %% Gen Mon B validity: %.4f %% "% (100*validityA, 100*validityB,))
    f.write("Gen validity: %.4f %% "% (100*validity,))
    f.write("Novelty: %.4f %% "% (100*novelty,))
    f.write("Novelty (mon_comb): %.4f %% "% (100*novelty_mon_comb,))
    f.write("Novelty MonA full dataset: %.4f %% "% (100*novelty_A,))
    f.write("Novelty MonB full dataset: %.4f %% "% (100*novelty_B,))
    f.write("Novelty in full dataset: %.4f %% "% (100*novelty_full_dataset,))
    f.write("Diversity: %.4f %% "% (100*diversity,))
    f.write("Diversity (novel polymers): %.4f %% "% (100*diversity_novel,))


""" Plot the kde of the properties of training data and sampled data """
from sklearn.neighbors import KernelDensity

with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all = np.load(f)
with open(dir_name+'y2_all_'+dataset_type+'.npy', 'rb') as f:
    y2_all = np.load(f)
with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)

y1_all=list(y1_all)
y2_all=list(y2_all)
yp1_all = [yp[0] for yp in yp_all]
yp2_all = [yp[1] for yp in yp_all]
yp1_all_seed = [yp[0] for yp in all_y_p]
yp2_all_seed = [yp[1] for yp in all_y_p]

# debug prints:
print(f"\nBEFORE KDE PLOTTING:")
print(f"Length of all_y_p: {len(all_y_p)}")
print(f"Data for KDE y1: {yp1_all_seed}")
print(f"Data for KDE y2: {yp2_all_seed}")
print(f"Length of y1 data: {len(yp1_all_seed)}")
print(f"Length of y2 data: {len(yp2_all_seed)}")

# Replace the entire KDE plotting section with this fixed version:

""" y1 """
plt.figure(figsize=(10, 8))  # Specify figure size
real_distribution = np.array([r for r in y1_all if not np.isnan(r)])
augmented_distribution = np.array([p for p in yp1_all])
seed_distribution = np.array([s for s in yp1_all_seed])

# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)
seed_distribution = seed_distribution.reshape(-1, 1)

# Define bandwidth (bandwidth controls the smoothness of the kernel density estimate)
bandwidth = 0.1

# Fit kernel density estimator for real data
kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_real.fit(real_distribution)
# Fit kernel density estimator for augmented data
kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_augmented.fit(augmented_distribution)
# Fit kernel density estimator for sampled data
kde_sampled_seed = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_sampled_seed.fit(seed_distribution)

# Create a range of values for the x-axis
x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(seed_distribution)), 
                      max(np.max(real_distribution), np.max(augmented_distribution), np.max(seed_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
seed_density = np.exp(kde_sampled_seed.score_samples(x_values.reshape(-1, 1)))

# Plotting
plt.plot(x_values, real_density, label='Real Data', linewidth=2)
plt.plot(x_values, augmented_density, label='Augmented Data', linewidth=2)
plt.plot(x_values, seed_density, label='Sampled around optimal molecule', linewidth=2)

plt.xlabel('EA (eV)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (Electron affinity)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save BEFORE show()
plt.savefig(dir_name+'KDEy1_BO_seed'+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png', dpi=150, bbox_inches='tight')
plt.close()  # Close the figure to free memory

""" y2 """
plt.figure(figsize=(10, 8))  # Specify figure size
real_distribution = np.array([r for r in y2_all if not np.isnan(r)])
augmented_distribution = np.array([p for p in yp2_all])
seed_distribution = np.array([s for s in yp2_all_seed])

# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)
seed_distribution = seed_distribution.reshape(-1, 1)

# Define bandwidth (bandwidth controls the smoothness of the kernel density estimate)
bandwidth = 0.1

# Fit kernel density estimator for real data
kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_real.fit(real_distribution)
# Fit kernel density estimator for augmented data
kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_augmented.fit(augmented_distribution)
# Fit kernel density estimator for sampled data
kde_sampled_seed = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_sampled_seed.fit(seed_distribution)

# Create a range of values for the x-axis
x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(seed_distribution)), 
                      max(np.max(real_distribution), np.max(augmented_distribution), np.max(seed_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
seed_density = np.exp(kde_sampled_seed.score_samples(x_values.reshape(-1, 1)))

# Plotting
plt.plot(x_values, real_density, label='Real Data', linewidth=2)
plt.plot(x_values, augmented_density, label='Augmented Data', linewidth=2)
plt.plot(x_values, seed_density, label='Sampled around optimal molecule', linewidth=2)

plt.xlabel('IP (eV)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (Ionization potential)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save BEFORE show()
plt.savefig(dir_name+'KDEy2_BO_seed'+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png', dpi=150, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# Add a final verification
print(f"\nKDE plot files created:")
import os
kde_file1 = dir_name+'KDEy1_BO_seed'+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png'
kde_file2 = dir_name+'KDEy2_BO_seed'+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png'

if os.path.exists(kde_file1):
    size1 = os.path.getsize(kde_file1)
    print(f"KDEy1 file: {size1} bytes")
    
if os.path.exists(kde_file2):
    size2 = os.path.getsize(kde_file2)
    print(f"KDEy2 file: {size2} bytes")
