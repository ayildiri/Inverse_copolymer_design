import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *

# deep learning packages
import torch
import pickle
import argparse
import numpy as np
import time


# Necessary lists
all_predictions = []
all_real = []
prediction_validityA = []
prediction_validityB = []
monA_pred = []
monB_pred = []
monA_true = []
monB_true = []
monomer_weights_predicted = []
monomer_weights_real = []
monomer_con_predicted = []
monomer_con_real = []


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
parser.add_argument("--alpha", default="fixed", choices=["fixed","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initialization", default="random", choices=["random", "xavier", "kaiming"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.01)
parser.add_argument("--max_alpha", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load model checkpoints from and save results to")

# Add flexible property arguments (same as other scripts)
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties used in the model")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Path to custom dataset files (will use default naming pattern if not specified)")
parser.add_argument("--save_properties", action="store_true",
                    help="Save property predictions alongside reconstruction results")
# FIXED: Add argument to control homopolymer enforcement with default=False
parser.add_argument("--enforce_homopolymer", action="store_true", default=False,
                    help="Enforce homopolymer format in predicted structures")

# 🔥 NEW: Add debugging and performance arguments
parser.add_argument("--test_batches", type=int, default=None,
                    help="Limit number of batches for quick testing (e.g., 10 for fast test)")
parser.add_argument("--verbose", action="store_true", default=False,
                    help="Enable verbose progress reporting")

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

print(f"Running inference for model with {property_count} properties: {property_names}")
if args.save_properties:
    print("Property predictions will be saved alongside reconstruction results")
if args.enforce_homopolymer:
    print("Homopolymer format will be enforced on all predictions")
if args.test_batches:
    print(f"🔍 TESTING MODE: Only processing first {args.test_batches} batches")
if args.verbose:
    print("🔍 VERBOSE MODE: Detailed progress reporting enabled")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "test"
data_augment = "old" # new or old

# Handle dataset path flexibility
if args.dataset_path:
    # Use custom dataset path
    data_path = os.path.join(args.dataset_path, f'dict_test_loader_{augment}_{tokenization}.pt')
    vocab_file_path = os.path.join(args.dataset_path, f'poly_smiles_vocab_{augment}_{tokenization}.txt')
else:
    # Use default paths
    data_path = main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt'
    vocab_file_path = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'

print(f"Loading test data from: {data_path}")
print(f"Loading vocabulary from: {vocab_file_path}")

dict_test_loader = torch.load(data_path)

num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

def extract_monomer_info(polymer_string):
    """
    Extract monomer information from polymer string, handling both copolymers and homopolymers.
    This function matches the robust logic used in reconstruction_validity.py
    """
    try:
        monA = monB = stoich = connectivity = ""
        
        if '|' in polymer_string:
            parts = polymer_string.split('|')
            
            # Extract from explicit monomer fields if available
            if len(parts) >= 3:
                # Try to get monomers from the expected positions
                monA = parts[1] if len(parts) > 1 else ""
                monB = parts[2] if len(parts) > 2 else ""
                
                # For backwards compatibility, check if monomers might be in field 0
                if (not monA or not monB) and '.' in parts[0]:
                    monomers = parts[0].split('.')
                    if len(monomers) >= 2:
                        monA = monA or monomers[0]
                        monB = monB or monomers[1]
                
                # Get stoichiometry and connectivity if available
                stoich = parts[3] if len(parts) > 3 else ""
                connectivity = parts[4] if len(parts) > 4 else ""
            
            # Try alternative format where monomers are in first field with a dot separator
            elif '.' in parts[0]:
                monomers = parts[0].split('.')
                if len(monomers) >= 2:
                    monA = monomers[0]
                    monB = monomers[1]
        
        # Handle simple format without pipes
        elif '.' in polymer_string:
            monomers = polymer_string.split('.')
            if len(monomers) >= 2:
                monA = monomers[0]
                monB = monomers[1]
        else:
            # Assume single monomer (homopolymer)
            monA = monB = polymer_string
            
        return monA, monB, stoich, connectivity
    except:
        return "", "", "", ""

def convert_to_homopolymer_format(polymer_string):
    """
    Convert any polymer string to homopolymer format by making monA = monB.
    Uses the robust monomer extraction function for consistency.
    """
    if not polymer_string:
        return polymer_string
    
    try:
        # Extract components using the robust extraction function
        monA, monB, stoich, connectivity = extract_monomer_info(polymer_string)
        
        # If we couldn't extract proper components, return original
        if not monA:
            return polymer_string
        
        # Build homopolymer format: START|monA|monA|1:1|connectivity
        if '|' in polymer_string:
            # Extract the START part
            parts = polymer_string.split('|')
            start_part = parts[0] if parts else "START"
            
            # Build new homopolymer string
            new_parts = [start_part, monA, monA, "1:1"]
            
            # Add connectivity if it exists
            if connectivity:
                new_parts.append(connectivity)
            
            return "|".join(new_parts)
        else:
            # For simple formats, just return the monomer
            return monA
    
    except Exception as e:
        print(f"Warning: Error converting to homopolymer format for '{polymer_string}': {e}")
        return polymer_string

def extract_monomers_from_strings(polymer_strings):
    """
    Extract monomer information from polymer strings using the robust extraction function.
    Returns lists of monA, monB, stoichiometry, and connectivity.
    """
    monA_list = []
    monB_list = []
    weights_list = []
    connectivity_list = []
    
    for polymer_str in polymer_strings:
        monA, monB, stoich, connectivity = extract_monomer_info(polymer_str)
        
        monA_list.append(monA)
        monB_list.append(monB)
        weights_list.append(stoich)
        connectivity_list.append(connectivity)
    
    return monA_list, monB_list, weights_list, connectivity_list

# Load model
# Create an instance of the G2S model from checkpoint
# Include property info in model name for consistency
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+str(property_str)+'/'

# Updated path handling to use save_dir if provided
if args.save_dir is not None:
    filepath = os.path.join(args.save_dir, model_name, "model_best_loss.pt")
else:
    filepath = os.path.join(main_dir_path, 'Checkpoints/', model_name, "model_best_loss.pt")

print(f"Looking for model checkpoint at: {filepath}")

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
    
    vocab = load_vocab(vocab_file=vocab_file_path)
    
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file_path)
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

    print(f'Running inference on test set')
    print(f'Results will be saved to: {dir_name}')

    # Run over all batches
    batches = list(range(len(dict_test_loader)))
    
    # 🔥 NEW: Limit batches for testing if specified
    if args.test_batches:
        original_batch_count = len(batches)
        batches = batches[:args.test_batches]
        print(f"🔍 LIMITED TESTING: Processing {len(batches)}/{original_batch_count} batches for quick testing")
    
    print(f"📊 Total batches to process: {len(batches)}")
    
    vocab = load_vocab(vocab_file=vocab_file_path)

    # Initialize property storage if saving properties
    if args.save_properties:
        all_property_predictions = [[] for _ in range(property_count)]
        all_real_properties = [[] for _ in range(property_count)]

    # 🔥 NEW: Progress tracking variables
    start_time = time.time()
    batch_times = []
    
    ### INFERENCE ###
    print("\n🚀 Starting inference...")
    print("=" * 80)
    
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            
            # 🔥 NEW: Progress reporting
            if batch_idx % 10 == 0 or args.verbose:
                elapsed_time = time.time() - start_time
                if batch_idx > 0:
                    avg_batch_time = elapsed_time / batch_idx
                    estimated_total_time = avg_batch_time * len(batches)
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"📍 Batch [{batch_idx:4d}/{len(batches):4d}] ({100*batch_idx/len(batches):5.1f}%) | "
                          f"Elapsed: {elapsed_time/60:.1f}min | ETA: {remaining_time/60:.1f}min")
                else:
                    print(f"📍 Batch [{batch_idx:4d}/{len(batches):4d}] - Starting...")
            
            if args.verbose:
                print(f"  🔍 Loading batch data...")
            
            data = dict_test_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_test_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)
            model.beta = 1.0
            
            if args.verbose:
                print(f"  🔍 Running model inference...")
            
            predictions, _, _, z, y_pred = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
            
            if args.verbose:
                print(f"  🔍 Processing predictions...")
            
            # Convert predictions to strings
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
            all_predictions.extend(prediction_strings)
            
            real_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization) for sample in range(len(data))]
            all_real.extend(real_strings)
            
            # Store property predictions and real values if requested
            if args.save_properties:
                # Property predictions from model
                if y_pred is not None:
                    y_pred_np = y_pred.cpu().numpy()
                    for i in range(property_count):
                        if i < y_pred_np.shape[1]:
                            all_property_predictions[i].append(y_pred_np[:, i])
                        else:
                            print(f"Warning: Property {i+1} not available in predictions")
                            all_property_predictions[i].append(np.full(y_pred_np.shape[0], np.nan))
                
                # Real property values from data
                real_props = extract_properties_from_data(data)
                for i, prop_vals in enumerate(real_props):
                    all_real_properties[i].append(prop_vals)
            
            # 🔥 NEW: Track batch timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if args.verbose:
                print(f"  ✅ Batch {batch_idx} completed in {batch_time:.2f}s")
            
            # 🔥 NEW: Memory management for large datasets
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()  # Clear GPU cache periodically
                if args.verbose:
                    print(f"  🧹 GPU cache cleared")

    # 🔥 NEW: Final timing report
    total_time = time.time() - start_time
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    
    print("=" * 80)
    print(f"✅ INFERENCE COMPLETED!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Average batch time: {avg_batch_time:.2f} seconds")
    print(f"🎯 Processed {len(batches)} batches, {len(all_predictions)} total samples")
    
    if args.test_batches:
        full_time_estimate = (total_time / len(batches)) * len(dict_test_loader)
        print(f"📈 Estimated full dataset time: {full_time_estimate/60:.1f} minutes")
    
    print("=" * 80)

    print(f'Saving inference results')
    print(f'Total predictions: {len(all_predictions)}')
    print(f'Total real samples: {len(all_real)}')
    
    # Save original predictions before homopolymer enforcement
    with open(os.path.join(dir_name, 'all_val_prediction_strings_original.pkl'), 'wb') as f:
        pickle.dump(all_predictions, f)

    # Apply homopolymer enforcement if specified
    if args.enforce_homopolymer:
        print(f'Converting all predictions to homopolymer format')
        homopolymer_predictions = [convert_to_homopolymer_format(pred) for pred in all_predictions]
        
        # Save homopolymer-enforced predictions as the main result
        with open(os.path.join(dir_name, 'all_val_prediction_strings.pkl'), 'wb') as f:
            pickle.dump(homopolymer_predictions, f)
        
        # Use the homopolymer-enforced predictions for monomer extraction
        final_predictions = homopolymer_predictions
    else:
        # Save original predictions if homopolymer enforcement not specified
        with open(os.path.join(dir_name, 'all_val_prediction_strings.pkl'), 'wb') as f:
            pickle.dump(all_predictions, f)
        
        # Use original predictions for monomer extraction
        final_predictions = all_predictions
    
    # Save real samples
    with open(os.path.join(dir_name, 'all_val_real_strings.pkl'), 'wb') as f:
        pickle.dump(all_real, f)
    
    # Extract monomer information using the robust extraction function
    print(f'Extracting monomer information from predictions and real data')
    monA_pred, monB_pred, monomer_weights_predicted, monomer_con_predicted = extract_monomers_from_strings(final_predictions)
    monA_true, monB_true, monomer_weights_real, monomer_con_real = extract_monomers_from_strings(all_real)
    
    # Save monomer analysis results
    with open(os.path.join(dir_name, 'monA_predicted.pkl'), 'wb') as f:
        pickle.dump(monA_pred, f)
    with open(os.path.join(dir_name, 'monB_predicted.pkl'), 'wb') as f:
        pickle.dump(monB_pred, f)
    with open(os.path.join(dir_name, 'monA_real.pkl'), 'wb') as f:
        pickle.dump(monA_true, f)
    with open(os.path.join(dir_name, 'monB_real.pkl'), 'wb') as f:
        pickle.dump(monB_true, f)
    with open(os.path.join(dir_name, 'weights_predicted.pkl'), 'wb') as f:
        pickle.dump(monomer_weights_predicted, f)
    with open(os.path.join(dir_name, 'weights_real.pkl'), 'wb') as f:
        pickle.dump(monomer_weights_real, f)
    with open(os.path.join(dir_name, 'connectivity_predicted.pkl'), 'wb') as f:
        pickle.dump(monomer_con_predicted, f)
    with open(os.path.join(dir_name, 'connectivity_real.pkl'), 'wb') as f:
        pickle.dump(monomer_con_real, f)
    
    # Save property predictions if requested
    if args.save_properties:
        print(f'Saving property predictions for {property_count} properties: {property_names}')
        
        for i in range(property_count):
            # Save predicted properties
            if all_property_predictions[i]:
                pred_props_concat = np.concatenate(all_property_predictions[i], axis=0)
                np.save(os.path.join(dir_name, f'{property_names[i]}_predicted.npy'), pred_props_concat)
                print(f'  Saved {property_names[i]} predictions: {pred_props_concat.shape}')
            
            # Save real properties
            if all_real_properties[i]:
                real_props_concat = np.concatenate(all_real_properties[i], axis=0)
                np.save(os.path.join(dir_name, f'{property_names[i]}_real.npy'), real_props_concat)
                print(f'  Saved {property_names[i]} real values: {real_props_concat.shape}')
    
    # Summary statistics
    homopolymer_count_pred = sum(1 for a, b in zip(monA_pred, monB_pred) if a == b and a != "")
    homopolymer_count_real = sum(1 for a, b in zip(monA_true, monB_true) if a == b and a != "")
    
    print(f'\n=== INFERENCE SUMMARY ===')
    print(f'Total samples processed: {len(all_predictions)}')
    print(f'Homopolymers in predictions: {homopolymer_count_pred}/{len(final_predictions)} ({100*homopolymer_count_pred/len(final_predictions):.1f}%)')
    print(f'Homopolymers in real data: {homopolymer_count_real}/{len(all_real)} ({100*homopolymer_count_real/len(all_real):.1f}%)')
    if args.enforce_homopolymer:
        print(f'Homopolymer format enforced on all predictions')
        print(f'Original predictions saved as: all_val_prediction_strings_original.pkl')
    print(f'Results saved to: {dir_name}')
    print(f'Model properties: {property_names}')
    if args.save_properties:
        print(f'Property predictions saved for: {property_names}')
    if args.test_batches:
        print(f'🔍 THIS WAS A LIMITED TEST - only {args.test_batches} batches processed')
        print(f'To run full inference, remove --test_batches argument')
    print(f'⏱️  Total processing time: {total_time/60:.1f} minutes')
    print(f'=========================')

else: 
    print("The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
