
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

import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.rdkit_poly import make_polymer_mol


# setting device on GPU if available, else CPU
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
parser.add_argument("--objective_type", type=str, default="EAmin", choices=["EAmin", "mimick_peak", "mimick_best", "max_gap"], 
                    help="Type of objective function to use")

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

if args.save_dir:
    filepath = os.path.join(args.save_dir, model_name, "model_best_loss.pt")
else:
    filepath = os.path.join(main_dir_path,'Checkpoints/', model_name, "model_best_loss.pt")


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

if args.save_dir:
    dir_name = os.path.join(args.save_dir, model_name)
else:
    dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)

# Create checkpoint directory
checkpoint_dir = os.path.join(dir_name, 'checkpoints_GA')
os.makedirs(checkpoint_dir, exist_ok=True)

# Setup log file for monitoring
log_file = os.path.join(dir_name, f'optimization_log_GA_{args.opt_run}.txt')

class Property_optimization_problem(Problem):
    def __init__(self, model, x_min, x_max, objective_type):
        super().__init__(n_var=len(x_min), n_obj=2, n_constr=0, xl=x_min, xu=x_max)
        self.model_predictor = model
        self.weight_electron_affinity = 1  # Adjust the weight for electron affinity
        self.weight_ionization_potential = 1  # Adjust the weight for ionization potential
        self.weight_z_distances = 5  # Adjust the weight for distance between GA chosen z and reencoded z
        self.penalty_value = 100  # Adjust the weight for penalty of validity
        self.modified_solution = None # Initialize the class variable that later stores the recalculated latents
        self.modified_solution_history = []  # Initialize list to store modified solutions
        self.results_custom = {}
        self.eval_calls = 0
        self.objective_type = objective_type


    def _evaluate(self, x, out, *args, **kwargs):
        # Assuming x is a 1D array containing the 32 numerical parameters

        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        self.eval_calls += 1
        x_torch = torch.from_numpy(x).to(device).to(torch.float32) 
        print("Evaluation should be repaired")
        print(x)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=x_torch, device=device, sample=False, log_var=None)
        # Validity check of the decoded molecule + penalize invalid molecules
        prediction_strings, validity = self._calc_validity(predictions)
        
        invalid_mask = (validity == 0)
        zero_vector = np.zeros(self.n_var)
        validity_mask = np.all(x != zero_vector, axis=1)
        print(len(invalid_mask))
        print(validity_mask.shape[0])
        print(x.shape[0])
        print(np.array(y.cpu()).shape[0])
        print(out["F"])
        out["F"] = np.zeros((x.shape[0], 2))
        print(out["F"].shape[0])
        if self.objective_type=='mimick_peak':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.abs(np.array(y.cpu())[validity_mask,0]+2)
            out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y.cpu())[validity_mask,1] - 1.2)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type=='mimick_best':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.abs(np.array(y.cpu())[validity_mask,0]+2.64)
            out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y.cpu())[validity_mask,1] - 1.61)
        elif self.objective_type=='EAmin':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.array(y.cpu())[validity_mask,0]  # Minimize the first property (electron affinity)
            out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y.cpu())[validity_mask,1] - 1.0)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type =='max_gap':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.array(y.cpu())[validity_mask,0]  # Minimize the first property (electron affinity)
            out["F"][validity_mask, 1] = -self.weight_ionization_potential * np.array(y.cpu())[validity_mask,1]  # Maximize IP (is in general positive)
        out["F"][~validity_mask] += self.penalty_value


        
        # Encode and predict the valid molecules
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _=self._encode_and_predict_molecules(predictions_valid)
        expanded_y_p = np.array([y_p_after_encoding_valid.pop(0) if val == 1 else [np.nan,np.nan] for val in list(validity)])
        expanded_z_p = np.array([z_p_after_encoding_valid.pop(0) if val == 1 else [0] * 32 for val in list(validity)])
        all_reconstructions = [all_reconstructions_valid.pop(0) if val == 1 else "" for val in list(validity)]
        print("evaluation should not change")
        print(expanded_z_p)


        out["F_corrected"] = np.zeros((x.shape[0], 2))
        if self.objective_type=='mimick_peak':
            #out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]  # Minimize the first property (electron affinity)
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2) # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.2)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type=='mimick_best':
            #out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]  # Minimize the first property (electron affinity)
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2.64) # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.61)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type=='EAmin':
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0] # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.0) 
        elif self.objective_type =='max_gap':
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0] # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = -self.weight_ionization_potential * expanded_y_p[~invalid_mask, 1] 


        # results
        #print(out["F"])
        aggr_obj = np.sum(out["F"], axis=1)
        aggr_obj_corrected = np.sum(out["F_corrected"], axis=1)
        results_dict = {
            "objective":aggr_obj,
            "objective_corrected": aggr_obj_corrected,
            "latents_reencoded": x, 
            "predictions": y,
            "predictions_doublecorrect": expanded_y_p,
            "string_decoded": prediction_strings, 
        }
        self.results_custom[str(self.eval_calls)] = results_dict
    
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
    
    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _encode_and_predict_molecules(self, predictions):
        # create data that can be encoded again
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
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

def save_checkpoint(algorithm, problem, iteration, checkpoint_dir, opt_run):
    """Save optimization checkpoint"""
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
    
    # Keep backup of last checkpoint
    if iteration > 200:
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
with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
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
objective_type = args.objective_type
problem = Property_optimization_problem(model, min_values, max_values, objective_type)

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
    pop_size = 100


elif stopping_type == "iter":
    stopping_criterion = stopping_type+"_"+str(max_iter)
    termination = get_termination("n_eval", max_iter)
    pop_size = int(max_iter / 20) # 20 generations, pop size 100


# Define NSGA2 algorithm parameters
#pop_size = max_iter / 10
sampling = LatinHypercubeSampling()
crossover = SimulatedBinaryCrossover(prob=0.90, eta=20)
#crossover = SimulatedBinaryCrossover()
mutation = PolynomialMutation(prob=1.0 / problem.n_var, eta=30)

# Initialize the NSGA2 algorithm
# algorithm = MyCustomNSGA2(pop_size=pop_size,
#                   sampling=sampling,
#                   crossover=crossover,
#                   mutation=mutation,
#                   eliminate_duplicates=True)


from pymoo.core.repair import Repair
class correctSamplesRepair(Repair):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model_predictor = model

    def _do(self, problem, Z, **kwargs):
        # repair sampled points whole batch 
        #pop_Z = []
        #for i in range(len(Z)): 
        #    pop_Z.append(Z[i])
        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        # Repair the sampled population

        #pop_Z_np=np.array(pop_Z)
        #pop_Z_torch = torch.from_numpy(pop_Z_np).to(device).to(torch.float32)
        pop_Z_torch = torch.from_numpy(Z).to(device).to(torch.float32)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=pop_Z_torch, device=device, sample=False, log_var=None)
        # Validity check of the decoded molecule + penalize invalid molecules
        prediction_strings, validity = self._calc_validity(predictions)
        invalid_mask = (validity == 0)
        # Encode and predict the valid molecules
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _=self._encode_and_predict_molecules(predictions_valid)
        expanded_z_p = np.array([z_p_after_encoding_valid.pop(0) if val == 1 else [0] * 32 for val in list(validity)])

        print("repaired population")
        print(expanded_z_p)
        Z = Population().create(*expanded_z_p)
        return Z


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

    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _encode_and_predict_molecules(self, predictions):
        # create data that can be encoded again
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
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


algorithm = NSGA2(pop_size=pop_size,
                  sampling=sampling,
                  crossover=crossover,
                  repair=correctSamplesRepair(model),
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
final_checkpoint = save_checkpoint(algorithm, problem, algorithm.n_gen, checkpoint_dir, args.opt_run)
log_progress(f"Final checkpoint saved: {final_checkpoint}", log_file)

# Access the results (keep these lines from your original code)
best_solution = res.X
#best_mod_solution = res.X_mod
best_fitness = res.F
results_custom = problem.results_custom


with open(dir_name+'res_optimization_GA_correct_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'wb') as f:
    pickle.dump(res, f)
with open(dir_name+'optimization_results_custom_GA_correct_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'wb') as f:
    pickle.dump(results_custom, f)
with open(dir_name+'optimization_results_custom_GA_correct_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
     print(results_custom, file=fl)

log_progress(f"Saved optimization results to {dir_name}", log_file)

#convergence = res.algorithm.termination
with open(dir_name+'res_optimization_GA_correct_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'rb') as f:
    res = pickle.load(f)

with open(dir_name+'optimization_results_custom_GA_correct_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.pkl', 'rb') as f:
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
EA_re= [arr[0].cpu() for arr in pred_RE]
IP_re = [arr[1].cpu() for arr in pred_RE]
EA_re_c= [arr[0] for arr in pred_RE_corrected]
IP_re_c = [arr[1] for arr in pred_RE_corrected]

# Create plot
plt.figure(0)

plt.plot(iterations, EA_re, label='EA (RE)')
plt.plot(iterations, IP_re, label='IP (RE)')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.savefig(dir_name+'GA_objectives_correct_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png',  dpi=300)
plt.close()

""" Plot the kde of the properties of training data and sampled data """
try:
    with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
        y1_all = np.load(f)
    with open(dir_name+'y2_all_'+dataset_type+'.npy', 'rb') as f:
        y2_all = np.load(f)
    with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
        yp_all = np.load(f)

    y1_all = list(y1_all)
    y2_all = list(y2_all)
    yp1_all = [yp[0] for yp in yp_all]
    yp2_all = [yp[1] for yp in yp_all]
    
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

    yp1_all_ga = ensure_numpy_array([x.cpu().numpy() if torch.is_tensor(x) else x for x in EA_re])
    yp2_all_ga = ensure_numpy_array([x.cpu().numpy() if torch.is_tensor(x) else x for x in IP_re])

    log_progress("Generating KDE plots for property distributions...", log_file)

    """ y1 """
    plt.figure(figsize=(10, 8))
    real_distribution = np.array([r for r in y1_all if not np.isnan(r)])
    augmented_distribution = np.array([p for p in yp1_all])
    ga_distribution = yp1_all_ga

    # Reshape the data
    real_distribution = real_distribution.reshape(-1, 1)
    augmented_distribution = augmented_distribution.reshape(-1, 1)
    ga_distribution = ga_distribution.reshape(-1, 1)

    # Define bandwidth
    bandwidth = 0.1

    # Fit kernel density estimator for real data
    kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_real.fit(real_distribution)
    # Fit kernel density estimator for augmented data
    kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_augmented.fit(augmented_distribution)
    # Fit kernel density estimator for GA data
    kde_ga = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_ga.fit(ga_distribution)

    # Create a range of values for the x-axis
    x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(ga_distribution)), 
                          max(np.max(real_distribution), np.max(augmented_distribution), np.max(ga_distribution)), 1000)
    # Evaluate the KDE
    real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
    augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
    ga_density = np.exp(kde_ga.score_samples(x_values.reshape(-1, 1)))

    # Plotting
    plt.plot(x_values, real_density, label='Real Data', linewidth=2)
    plt.plot(x_values, augmented_density, label='Augmented Data', linewidth=2)
    plt.plot(x_values, ga_density, label='GA Optimized', linewidth=2)

    plt.xlabel('EA (eV)')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation (Electron affinity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    kde_file1 = dir_name+'KDEy1_GA_correct_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png'
    plt.savefig(kde_file1, dpi=150, bbox_inches='tight')
    plt.close()

    """ y2 """
    plt.figure(figsize=(10, 8))
    real_distribution = np.array([r for r in y2_all if not np.isnan(r)])
    augmented_distribution = np.array([p for p in yp2_all])
    ga_distribution = yp2_all_ga

    # Reshape the data
    real_distribution = real_distribution.reshape(-1, 1)
    augmented_distribution = augmented_distribution.reshape(-1, 1)
    ga_distribution = ga_distribution.reshape(-1, 1)

    # Fit KDEs
    kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_real.fit(real_distribution)
    kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_augmented.fit(augmented_distribution)
    kde_ga = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_ga.fit(ga_distribution)

    # Create x values and evaluate
    x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(ga_distribution)), 
                          max(np.max(real_distribution), np.max(augmented_distribution), np.max(ga_distribution)), 1000)
    real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
    augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
    ga_density = np.exp(kde_ga.score_samples(x_values.reshape(-1, 1)))

    # Plot
    plt.plot(x_values, real_density, label='Real Data', linewidth=2)
    plt.plot(x_values, augmented_density, label='Augmented Data', linewidth=2)
    plt.plot(x_values, ga_density, label='GA Optimized', linewidth=2)

    plt.xlabel('IP (eV)')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation (Ionization potential)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    kde_file2 = dir_name+'KDEy2_GA_correct_'+str(stopping_criterion)+'_run'+str(opt_run)+'.png'
    plt.savefig(kde_file2, dpi=150, bbox_inches='tight')
    plt.close()
    
    log_progress(f"KDE plots saved: {kde_file1} and {kde_file2}", log_file)
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

# Extract data for the curves
if objective_type=='mimick_peak':
    objective_values = [(np.abs(arr.cpu()[0]+2)+np.abs(arr.cpu()[1]-1.2)) for arr in pred_RE]
    objective_values_c = [(np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) for arr in pred_RE_corrected]
elif objective_type=='mimick_best':
    objective_values = [(np.abs(arr.cpu()[0]+2.64)+np.abs(arr.cpu()[1]-1.61)) for arr in pred_RE]
    objective_values_c = [(np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61)) for arr in pred_RE_corrected]
elif objective_type=='EAmin': 
    objective_values = [arr.cpu()[0]+np.abs(arr.cpu()[1]-1) for arr in pred_RE]
    objective_values_c = [arr[0]+np.abs(arr[1]-1) for arr in pred_RE_corrected]
elif objective_type =='max_gap':
    objective_values = [arr.cpu()[0]-arr.cpu()[1] for arr in pred_RE]
    objective_values_c = [arr[0]-arr[1] for arr in pred_RE_corrected]

indices_of_increases = indices_of_improvement(objective_values)


EA_re_imp = [EA_re[i] for i in indices_of_increases]
IP_re_imp = [IP_re[i] for i in indices_of_increases]
best_z_re = [Latents_RE[i] for i in indices_of_increases]
best_mols = {i+1: decoded_mols[i] for i in indices_of_increases}
best_props = {i+1: [EA_re[i],IP_re[i]] for i in indices_of_increases}
with open(dir_name+'best_mols_GA_correct_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols, file=fl)
    print(best_props, file=fl)

top_20_indices, top_20_mols = top_n_molecule_indices(objective_values, decoded_mols, n_idx=20)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}
best_props_t20 = {i+1: [EA_re[i], IP_re[i]] for i in top_20_indices}
best_props_t20_c = {i+1: [EA_re_c[i], IP_re_c[i]] for i in top_20_indices}
best_objs_t20 = {i+1: objective_values[i] for i in top_20_indices}
best_objs_t20_c = {i+1: objective_values_c[i] for i in top_20_indices}
with open(dir_name+'top20_mols_GA_correct_'+str(objective_type)+'_'+str(stopping_criterion)+'_run'+str(opt_run)+'.txt', 'w') as fl:
    print(best_mols_t20, file=fl)
    print(best_props_t20, file=fl)
    print(best_props_t20_c, file=fl)
    print(best_objs_t20, file=fl)
    print(best_objs_t20_c, file=fl)
    print(top_20_mols, file=fl)

# AFTER (Add this at the end of your script)
""" Check the molecules for validity and novelty """
try:
    log_progress("Analyzing molecule validity and novelty...", log_file)
    
    sm_can = SmilesEnumCanon()
    # First, get the training data for comparison
    all_polymers_data = []
    all_train_polymers = []
    for batch, graphs in enumerate(dict_train_loader):
        data = dict_train_loader[str(batch)][0]
        train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
        all_train_polymers.extend(train_polymers_batch)
    if augment=="augmented":
        df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop.csv')
    elif augment=="augmented_canonical":
        df = pd.read_csv(main_dir_path+'/data/dataset-combined-canonical-poly_chemprop.csv')
    elif augment=="augmented_enum":
        df = pd.read_csv(main_dir_path+'/data/dataset-combined-enumerated2_poly_chemprop.csv')
    for i in range(len(df.loc[:, 'poly_chemprop_input'])):
        poly_input = df.loc[i, 'poly_chemprop_input']
        all_polymers_data.append(poly_input)
    
    # Canonicalize all strings for comparison
    all_predictions_can = [sm_can.canonicalize(s) for s in decoded_mols if s != 'invalid_polymer_string']
    all_train_can = [sm_can.canonicalize(s) for s in all_train_polymers]
    all_pols_data_can = [sm_can.canonicalize(s) for s in all_polymers_data]
    
    # Extract monomers
    monomers = [s.split("|")[0].split(".") for s in all_train_polymers]
    monomers_all = [mon for sub_list in monomers for mon in sub_list]
    all_mons_can = []
    for m in monomers_all:
        m_can = sm_can.canonicalize(m, monomer_only=True, stoich_con_info=False)
        modified_string = re.sub(r'\*\:\d+', '*', m_can)
        all_mons_can.append(modified_string)
    all_mons_can = list(set(all_mons_can))
    
    # Analyze generated molecules
    monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions_can]
    monomer_comb_predicted = [poly_smiles.split("|")[0] for poly_smiles in all_predictions_can]
    monomer_comb_train = [poly_smiles.split("|")[0] for poly_smiles in all_train_can]
    
    # Extract monomer A and B
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
    
    # Validity metrics
    prediction_validityA = []
    prediction_validityB = []
    
    def poly_smiles_to_molecule(poly_input):
        '''Turns adjusted polymer smiles string into mols'''
        try:
            mols = make_monomer_mols(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1])
            return mols
        except:
            return None
    
    # Check validity of generated molecules
    prediction_mols = []
    for poly in all_predictions_can:
        try:
            mol = poly_smiles_to_molecule(poly)
            prediction_mols.append(mol)
        except:
            prediction_mols.append(None)
    
    for mon in prediction_mols:
        try: 
            prediction_validityA.append(mon[0] is not None if mon else False)
        except: 
            prediction_validityA.append(False)
        
        try: 
            prediction_validityB.append(mon[1] is not None if mon else False)
        except: 
            prediction_validityB.append(False)
    
    # Calculate validity rates
    validityA = sum(prediction_validityA)/len(prediction_validityA) if prediction_validityA else 0
    validityB = sum(prediction_validityB)/len(prediction_validityB) if prediction_validityB else 0
    validity = len(monomer_smiles_predicted)/len(decoded_mols) if decoded_mols else 0
    
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
    
    # Save the novelty and validity metrics
    with open(dir_name+f'novelty_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.txt', 'w') as f:
        f.write(f"Gen Mon A validity: {100*validityA:.4f}% Gen Mon B validity: {100*validityB:.4f}%\n")
        f.write(f"Gen validity: {100*validity:.4f}%\n")
        f.write(f"Novelty: {100*novelty:.4f}%\n")
        f.write(f"Novelty (mon_comb): {100*novelty_mon_comb:.4f}%\n")
        f.write(f"Novelty MonA full dataset: {100*novelty_A:.4f}%\n")
        f.write(f"Novelty MonB full dataset: {100*novelty_B:.4f}%\n")
        f.write(f"Novelty in full dataset: {100*novelty_full_dataset:.4f}%\n")
        f.write(f"Diversity: {100*diversity:.4f}%\n")
        f.write(f"Diversity (novel polymers): {100*diversity_novel:.4f}%\n")
    
    log_progress(f"Novelty analysis completed and saved to {dir_name}novelty_GA_correct_{objective_type}_{stopping_criterion}_run{opt_run}.txt", log_file)
except Exception as e:
    log_progress(f"Error during novelty analysis: {str(e)}", log_file)
