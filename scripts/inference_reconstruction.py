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
# Add argument to control homopolymer enforcement
parser.add_argument("--enforce_homopolymer", action="store_true", default=True,
                    help="Enforce homopolymer format in predicted structures")

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

def convert_to_homopolymer_format(polymer_string):
    """
    Convert any polymer string to homopolymer format by making monA = monB.
    Assumes the format: START|monA|monB|stoich|connectivity
    """
    if not polymer_string or '|' not in polymer_string:
        return polymer_string
    
    parts = polymer_string.split('|')
    if len(parts) < 3:
        return polymer_string
    
    # Extract parts
    start_part = parts[0]
    monA = parts[1]
    
    # Use monA for both monomers (homopolymer)
    new_parts = [start_part, monA, monA]
    
    # Add stoichiometry (1:1 for homopolymers)
    if len(parts) > 3:
        new_parts.append("1:1")
    else:
        new_parts.append("1:1")
    
    # Add connectivity if present
    if len(parts) > 4:
        new_parts.append(parts[4])
    
    # Reconstruct the polymer string
    return "|".join(new_parts)

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

def extract_monomers_from_strings(polymer_strings, is_homopolymer_check=True):
    """Extract monomer information from polymer strings, handling both copolymers and homopolymers."""
    monA_list = []
    monB_list = []
    weights_list = []
    connectivity_list = []
    
    for polymer_str in polymer_strings:
        try:
            # Split the polymer string to extract components
            parts = polymer_str.split('|')
            if len(parts) >= 4:
                # Extract monomers (assuming format: START|monA|monB|stoich|connectivity:pattern)
                monA = parts[1] if len(parts) > 1 else ""
                monB = parts[2] if len(parts) > 2 else ""
                stoich_part = parts[3] if len(parts) > 3 else ""
                
                # Handle homopolymers - if monA and monB are the same or monB is empty
                if monA == monB or monB == "" or (is_homopolymer_check and monA != "" and monB == ""):
                    monB = monA  # Ensure homopolymer consistency
                    stoich_part = "1:1" if stoich_part == "" else stoich_part
                
                monA_list.append(monA)
                monB_list.append(monB)
                weights_list.append(stoich_part)
                
                # Extract connectivity pattern
                if len(parts) > 4 and ':' in parts[4]:
                    connectivity = parts[4].split(':')[1] if ':' in parts[4] else ""
                else:
                    connectivity = ""
                connectivity_list.append(connectivity)
            else:
                # Handle incomplete strings
                monA_list.append("")
                monB_list.append("")
                weights_list.append("")
                connectivity_list.append("")
        except Exception as e:
            print(f"Warning: Error parsing polymer string '{polymer_str}': {e}")
            monA_list.append("")
            monB_list.append("")
            weights_list.append("")
            connectivity_list.append("")
    
    return monA_list, monB_list, weights_list, connectivity_list

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
    vocab = load_vocab(vocab_file=vocab_file_path)

    # Initialize property storage if saving properties
    if args.save_properties:
        all_property_predictions = [[] for _ in range(property_count)]
        all_real_properties = [[] for _ in range(property_count)]

    ### INFERENCE ###
    with torch.no_grad():
        model.eval()
        for batch in batches:
            data = dict_test_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_test_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)
            model.beta = 1.0
            
            predictions, _, _, z, y_pred = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
            
            # Convert predictions to strings
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
            all_predictions.extend(prediction_strings)
            
            real_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization) for sample in range(len(data))]
            all_real.extend(real_strings)
            
            # Extract and store monomer information for both predicted and real strings
            pred_monA, pred_monB, pred_weights, pred_connectivity = extract_monomers_from_strings(prediction_strings)
            real_monA, real_monB, real_weights, real_connectivity = extract_monomers_from_strings(real_strings)
            
            monA_pred.extend(pred_monA)
            monB_pred.extend(pred_monB)
            monA_true.extend(real_monA)
            monB_true.extend(real_monB)
            monomer_weights_predicted.extend(pred_weights)
            monomer_weights_real.extend(real_weights)
            monomer_con_predicted.extend(pred_connectivity)
            monomer_con_real.extend(real_connectivity)
            
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
        
        # Re-extract monomer information after homopolymer enforcement
        monA_pred, monB_pred, monomer_weights_predicted, monomer_con_predicted = extract_monomers_from_strings(homopolymer_predictions)
    else:
        # Save original predictions if homopolymer enforcement not specified
        with open(os.path.join(dir_name, 'all_val_prediction_strings.pkl'), 'wb') as f:
            pickle.dump(all_predictions, f)
    
    # Save real samples
    with open(os.path.join(dir_name, 'all_val_real_strings.pkl'), 'wb') as f:
        pickle.dump(all_real, f)
    
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
    print(f'Homopolymers in predictions: {homopolymer_count_pred}/{len(all_predictions)} ({100*homopolymer_count_pred/len(all_predictions):.1f}%)')
    print(f'Homopolymers in real data: {homopolymer_count_real}/{len(all_real)} ({100*homopolymer_count_real/len(all_real):.1f}%)')
    if args.enforce_homopolymer:
        print(f'Homopolymer format enforced on all predictions')
        print(f'Original predictions saved as: all_val_prediction_strings_original.pkl')
    print(f'Results saved to: {dir_name}')
    print(f'Model properties: {property_names}')
    if args.save_properties:
        print(f'Property predictions saved for: {property_names}')
    print(f'=========================')

else: 
    print("The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
