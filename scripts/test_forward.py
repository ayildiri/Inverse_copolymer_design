# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *

import time
from datetime import datetime
import sys
import random
# deep learning packages
import torch
from statistics import mean
import os
import numpy as np
import argparse
import pickle

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
parser.add_argument("--embedding_dim", type=int, help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--alpha", default="fixed", choices=["fixed","schedule"])  # Added alpha parameter
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
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing")
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load model checkpoints from and save results to")

# Add flexible property arguments
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties used in the model")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")

# ADD THE DATASET_PATH ARGUMENT
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Path to custom dataset files (will use default naming pattern if not specified)")

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

print(f"Testing model for {property_count} properties: {property_names}")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "train"
data_augment = "old" # new or old

# MODIFIED DATA LOADING LOGIC TO SUPPORT CUSTOM DATASET PATH
if args.dataset_path:
    # Use custom dataset path
    data_path_prefix = os.path.join(args.dataset_path, f'dict_{{}}_loader_{augment}_{tokenization}.pt')
    vocab_file = os.path.join(args.dataset_path, f'poly_smiles_vocab_{augment}_{tokenization}.txt')
    print(f"Using custom dataset path: {args.dataset_path}")
else:
    # Use default paths
    data_path_prefix = main_dir_path+'/data/dict_{}_loader_'+augment+'_'+tokenization+'.pt'
    vocab_file = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    print(f"Using default dataset path: {main_dir_path}/data/")

# Load training data
dict_train_loader = torch.load(data_path_prefix.format('train'))

num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

# Include property info in model name
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+property_str+'/'

# Updated path handling to use save_dir if provided
if args.save_dir is not None:
    filepath = os.path.join(args.save_dir, model_name, "model_best_loss.pt")
else:
    filepath = os.path.join(main_dir_path, 'Checkpoints/', model_name, "model_best_loss.pt")

if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_VAE_PPguided
    else: 
        model_type = G2S_VAE_PPguideddisabled

    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    
    # Get property information from model config if available
    model_property_count = model_config.get('property_count', 2)
    model_property_names = model_config.get('property_names', ["EA", "IP"])
    
    # Validate that the specified properties match the model
    if property_count != model_property_count:
        print(f"Warning: Specified property count ({property_count}) doesn't match model property count ({model_property_count})")
        print(f"Using model property count: {model_property_count}")
        property_count = model_property_count
    
    if property_names != model_property_names:
        print(f"Warning: Specified property names ({property_names}) don't match model property names ({model_property_names})")
        print(f"Using model property names: {model_property_names}")
        property_names = model_property_names
    
    print(f"Model trained for {property_count} properties: {property_names}")
    
    batch_size = model_config['batch_size']
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    
    # Load vocabulary using the configured path
    vocab = load_vocab(vocab_file=vocab_file)
    
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: 
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Directory to save results - updated to use save_dir if provided
    if args.save_dir is not None:
        dir_name = os.path.join(args.save_dir, model_name)
    else:
        dir_name = os.path.join(main_dir_path, 'Checkpoints/', model_name)
        
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    def extract_properties_from_data(data):
        """Extract property values from data object dynamically."""
        properties = []
        for i in range(property_count):
            prop_attr = f'y{i+1}'
            if hasattr(data, prop_attr):
                properties.append(getattr(data, prop_attr).cpu().numpy())
            else:
                print(f"Warning: Property {prop_attr} not found in data")
                properties.append(np.full(data.num_graphs, np.nan))
        return properties

    def run_forward_pass(data_loader, dataset_name):
        """Run forward pass on dataset and extract results."""
        print(f'\nRunning forward pass on {dataset_name} set')
        
        batches = list(range(len(data_loader)))
        test_ce_losses = []
        test_total_losses = []
        test_kld_losses = []
        test_accs = []
        test_mses = []

        model.eval()
        model.beta = model_config['max_beta']
        model.alpha = model_config['max_alpha']
        
        # Initialize lists for properties dynamically
        latents = []
        properties_real = [[] for _ in range(property_count)]  # List of lists for each property
        y_p = []
        monomers = []
        stoichiometry = []
        connectivity_pattern = []
        
        with torch.no_grad():
            for i, batch in enumerate(batches):
                # Limit to 500 batches for augmented train set to save time
                if augment=='augmented' and dataset_name=='train': 
                    if i>=500: 
                        break
                        
                data = data_loader[str(batch)][0]
                data.to(device)
                dest_is_origin_matrix = data_loader[str(batch)][1]
                dest_is_origin_matrix.to(device)
                inc_edges_to_atom_matrix = data_loader[str(batch)][2]
                inc_edges_to_atom_matrix.to(device)

                # Check for NaNs in property labels
                has_nan = False
                for j in range(property_count):
                    prop_attr = f'y{j+1}'
                    if hasattr(data, prop_attr):
                        if torch.isnan(getattr(data, prop_attr)).any():
                            has_nan = True
                            break
                
                if has_nan and dataset_name == 'test':
                    print(f"⚠️ Skipping {dataset_name} batch {i} due to NaNs in labels.")
                    continue

                # Perform a single forward pass
                loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y_pred = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
                
                # Check for NaN in outputs
                if torch.isnan(recon_loss) or torch.isnan(loss) or torch.isnan(kl_loss) or torch.isnan(acc):
                    print(f"⚠️ Skipping {dataset_name} batch {i} due to NaN in outputs")
                    continue
                
                # Store results
                latents.append(z.cpu().numpy())
                y_p.append(y_pred.cpu().numpy())
                
                # Extract properties dynamically
                prop_values = extract_properties_from_data(data)
                for j, prop_val in enumerate(prop_values):
                    properties_real[j].append(prop_val)
                
                # Extract monomers
                if augment=="augmented_canonical":
                    monomers.append(data.monomer_smiles_nocan)
                else: 
                    monomers.append(data.monomer_smiles)
                
                # Extract stoichiometry and connectivity
                targ_list = target.tolist()
                stoichiometry.extend(["|".join(combine_tokens(tokenids_to_vocab(targ_list_sub, vocab),tokenization=tokenization).split("|")[1:3]) for targ_list_sub in targ_list])
                connectivity_pattern.extend([combine_tokens(tokenids_to_vocab(targ_list_sub, vocab), tokenization=tokenization).split("|")[-1].split(':')[1] for targ_list_sub in targ_list])
                
                # Store losses
                test_ce_losses.append(recon_loss.item())
                test_total_losses.append(loss.item())
                test_kld_losses.append(kl_loss.item())
                test_accs.append(acc.item())
                test_mses.append(mse.item())

        # Calculate average metrics
        test_total = mean(test_total_losses) if test_total_losses else 0
        test_kld = mean(test_kld_losses) if test_kld_losses else 0
        test_acc = mean(test_accs) if test_accs else 0

        # Concatenate results
        latent_space = np.concatenate(latents, axis=0) if latents else np.array([])
        y_p_all = np.concatenate(y_p, axis=0) if y_p else np.array([])
        
        # Concatenate properties dynamically
        properties_all = []
        for j in range(property_count):
            if properties_real[j]:
                prop_all = np.concatenate(properties_real[j], axis=0)
                properties_all.append(prop_all)
            else:
                properties_all.append(np.array([]))

        # Save results
        print(f"Saving results for {dataset_name} set...")
        
        # Save common results
        with open(os.path.join(dir_name, f'stoichiometry_{dataset_name}'), 'wb') as f:
            pickle.dump(stoichiometry, f)
        with open(os.path.join(dir_name, f'connectivity_{dataset_name}'), 'wb') as f:
            pickle.dump(connectivity_pattern, f)
        with open(os.path.join(dir_name, f'monomers_{dataset_name}'), 'wb') as f:
            pickle.dump(monomers, f)
        with open(os.path.join(dir_name, f'latent_space_{dataset_name}.npy'), 'wb') as f:
            np.save(f, latent_space)
        with open(os.path.join(dir_name, f'yp_all_{dataset_name}.npy'), 'wb') as f:
            np.save(f, y_p_all)
        
        # Save properties dynamically (y1_all, y2_all, etc.)
        for j, prop_all in enumerate(properties_all):
            with open(os.path.join(dir_name, f'y{j+1}_all_{dataset_name}.npy'), 'wb') as f:
                np.save(f, prop_all)
        
        print(f"{dataset_name.capitalize()}set: Total Loss: {test_total:.5f} | KLD: {test_kld:.5f} | ACC: {test_acc:.5f}")
        
        # Print data shapes for verification
        print(f"Data shapes saved for {dataset_name}:")
        print(f"  Latent space: {latent_space.shape}")
        print(f"  Predictions: {y_p_all.shape}")
        for j, prop_all in enumerate(properties_all):
            print(f"  {property_names[j]} (y{j+1}): {prop_all.shape}")

    # Run forward pass on training set
    run_forward_pass(dict_train_loader, 'train')

    # Run forward pass on test set
    print('\n' + '='*60)
    print('STARTING TEST')
    print('='*60)
    
    dataset_type = "test"
    
    # MODIFIED TEST LOADER TO USE CONFIGURED PATH
    dict_test_loader = torch.load(data_path_prefix.format('test'))
    run_forward_pass(dict_test_loader, 'test')

    print('\n' + '='*60)
    print('FORWARD PASS COMPLETED SUCCESSFULLY')
    print('='*60)
    print(f"Results saved to: {dir_name}")
    print(f"Properties extracted: {property_names}")

else: 
    print("The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
