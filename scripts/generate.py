# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *

# deep learning packages
import torch
import pickle
import argparse
import random
import numpy as np

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
parser.add_argument("--alpha", default="fixed", choices=["fixed","schedule"])
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
parser.add_argument("--save_dir", type=str, required=True, help="Path to save model and generated results")

# Add flexible property arguments
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties used in the model")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
parser.add_argument("--save_properties", action="store_true",
                    help="Save predicted properties of generated molecules")
# Add option to enforce homopolymer format
parser.add_argument("--enforce_homopolymer", action="store_true",
                    help="Enforce homopolymer format in generated structures")

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

print(f"Loading model trained for {property_count} properties: {property_names}")
if args.enforce_homopolymer:
    print("Homopolymer format will be enforced in all generated outputs")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "test"
data_augment = "old" # new or old
dict_test_loader = torch.load(main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')

num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

# Functions for cleaning and processing outputs
def clean_output(polymer_string):
    """Remove all padding underscores from generated polymer strings"""
    return polymer_string.rstrip('_')

def convert_to_homopolymer_format(polymer_string):
    """
    Convert any polymer string to homopolymer format by making monA = monB.
    Assumes the format: START|monA|monB|stoich|connectivity
    """
    polymer_string = clean_output(polymer_string)  # First remove padding
    
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

def process_generated_string(polymer_string, enforce_homopolymer=False):
    """Process a generated polymer string by removing padding and optionally enforcing homopolymer format"""
    # First remove trailing underscores
    clean_string = clean_output(polymer_string)
    
    # Optionally enforce homopolymer format
    if enforce_homopolymer:
        return convert_to_homopolymer_format(clean_string)
    else:
        return clean_string

# Include property info in model name
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+property_str+'/'

filepath = os.path.join(args.save_dir, model_name, "model_best_loss.pt")

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
    
    batch_size = model_config.get('batch_size', 64)
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    model_config["max_alpha"] = args.max_alpha
    
    vocab_file = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    vocab = load_vocab(vocab_file=vocab_file)
    
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: 
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Directory to save results
    dir_name = os.path.join(args.save_dir, model_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f'📝 Results will be saved to: {dir_name}')
    print(f'🎲 Generate random samples')

    ### RANDOM GENERATION ###
    torch.manual_seed(args.seed)
    all_predictions = []
    all_properties = [] if args.save_properties else None
    
    with torch.no_grad():
        model.eval()
        for i in range(250):
            z_rand = torch.randn((64, embedding_dimension), device=device) * args.epsilon
            predictions_rand, _, _, z, y = model.inference(data=z_rand, device=device, sample=False, log_var=None)
            print(f'Generated batch {i+1}/250')
            
            # Convert predictions to strings and clean up
            prediction_strings = [
                process_generated_string(
                    combine_tokens(tokenids_to_vocab(predictions_rand[sample][0].tolist(), vocab), tokenization=tokenization),
                    enforce_homopolymer=args.enforce_homopolymer
                ) for sample in range(len(predictions_rand))
            ]
            all_predictions.extend(prediction_strings)
            
            # Save property predictions if requested
            if args.save_properties:
                if torch.is_tensor(y):
                    properties_batch = y.cpu().numpy()
                    all_properties.extend(properties_batch)
   
    # Save random generation results
    with open(os.path.join(dir_name, 'generated_polymers.pkl'), 'wb') as f:
        pickle.dump(all_predictions, f)
    print(f"✅ Saved {len(all_predictions)} random generations to generated_polymers.pkl")
    
    if args.save_properties and all_properties:
        all_properties = np.array(all_properties)
        with open(os.path.join(dir_name, 'generated_polymers_properties.npy'), 'wb') as f:
            np.save(f, all_properties)
        print(f"✅ Saved properties of generated polymers: {all_properties.shape}")
        
        # Save property summary
        with open(os.path.join(dir_name, 'generated_polymers_property_summary.txt'), 'w') as f:
            f.write(f"Property Summary for {len(all_predictions)} Generated Polymers\n")
            f.write("="*60 + "\n\n")
            f.write(f"Properties: {property_names}\n\n")
            for i, prop_name in enumerate(property_names):
                prop_values = all_properties[:, i]
                f.write(f"{prop_name}:\n")
                f.write(f"  Mean: {np.mean(prop_values):.4f}\n")
                f.write(f"  Std:  {np.std(prop_values):.4f}\n")
                f.write(f"  Min:  {np.min(prop_values):.4f}\n")
                f.write(f"  Max:  {np.max(prop_values):.4f}\n\n")

    ### SEED-BASED GENERATION ###
    print(f'🌱 Generate samples around seed molecule')
    
    all_predictions_seed = []
    all_properties_seed = [] if args.save_properties else None
    
    batches = list(range(len(dict_test_loader)))
    random.seed(args.seed)
    batch = random.choice(batches)
    
    with torch.no_grad():
        model.eval()

        data = dict_test_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_test_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)
        
        _, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        
        # Randomly select a seed molecule
        ind = random.choice(list(range(64)))
        seed_z = z[ind]
        seed_z = seed_z.unsqueeze(0).repeat(64, 1)
        seed_string_raw = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind], vocab), tokenization=tokenization)
        seed_string = clean_output(seed_string_raw)  # Clean seed string
        
        print(f"🌱 Seed molecule: {seed_string}")
        
        sampled_z = []
        for r in range(8):
            # Define the mean and standard deviation of the Gaussian noise
            mean = 0
            std = args.epsilon / 2  # half of epsilon
            
            # Create a tensor of the same size as the original tensor with random noise
            noise = torch.tensor(np.random.normal(mean, std, size=seed_z.size()), dtype=torch.float, device=device)

            # Add the noise to the original tensor
            seed_z_noise = seed_z + noise
            sampled_z.append(seed_z_noise.cpu().numpy())
            
            predictions_seed, _, _, z_new, y_new = model.inference(data=seed_z_noise, device=device, sample=False, log_var=None)
            
            # Convert predictions to strings and clean up
            prediction_strings = [
                process_generated_string(
                    combine_tokens(tokenids_to_vocab(predictions_seed[sample][0].tolist(), vocab), tokenization=tokenization),
                    enforce_homopolymer=args.enforce_homopolymer
                ) for sample in range(len(predictions_seed))
            ]
            all_predictions_seed.extend(prediction_strings)
            
            # Save property predictions if requested
            if args.save_properties and torch.is_tensor(y_new):
                properties_batch = y_new.cpu().numpy()
                all_properties_seed.extend(properties_batch)

    # Save seed-based generation results
    print(f'💾 Saving generated strings around seed molecule')
    
    with open(os.path.join(dir_name, 'seed_polymer.txt'), 'w') as f:
        f.write(f'Seed molecule: {seed_string}\n')
        f.write(f'Properties: {property_names}\n')

    std = args.epsilon / 2
    with open(os.path.join(dir_name, f'seed_polymers_noise{std:.4f}.txt'), 'w') as f:
        f.write(f"Seed molecule: {seed_string}\n")
        f.write(f"Properties: {property_names}\n")
        f.write("The following are the generations from seed (mean) with noise\n")
        for i, s in enumerate(all_predictions_seed):
            f.write(f"{i+1}: {s}\n")
            
    with open(os.path.join(dir_name, f'seed_polymers_latents_noise{std:.4f}.npy'), 'wb') as f:
        sampled_z = np.stack(sampled_z)
        np.save(f, sampled_z)
        
    with open(os.path.join(dir_name, 'seed_polymer_z.npy'), 'wb') as f:
        seed_z_original = seed_z.cpu().numpy()
        np.save(f, seed_z_original)
        
    with open(os.path.join(dir_name, f'generated_polymers_from_seed_noise{std:.4f}.pkl'), 'wb') as f:
        pickle.dump(all_predictions_seed, f)
        
    print(f"✅ Saved {len(all_predictions_seed)} seed-based generations")
    
    if args.save_properties and all_properties_seed:
        all_properties_seed = np.array(all_properties_seed)
        with open(os.path.join(dir_name, f'seed_polymers_properties_noise{std:.4f}.npy'), 'wb') as f:
            np.save(f, all_properties_seed)
        print(f"✅ Saved properties of seed-based generations: {all_properties_seed.shape}")

    ### INTERPOLATION ###
    print(f'🔄 Generate interpolated samples between molecules')
    
    all_predictions_interp_all = []
    all_properties_interp_all = [] if args.save_properties else None
    
    random.seed(args.seed)
    batch = random.choice(batches)
    
    with torch.no_grad():
        model.eval()

        data = dict_test_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_test_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)
        
        _, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        
        examples = 10
        for e in range(examples):
            all_predictions_interp = []
            all_properties_interp = [] if args.save_properties else None
            
            # Randomly select two different molecules
            ind1 = random.choice(list(range(64)))
            ind2 = random.choice(list(range(64)))
            while ind1 == ind2:  # Ensure they're different
                ind2 = random.choice(list(range(64)))
                
            start_mol_raw = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind1], vocab), tokenization=tokenization)
            end_mol_raw = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind2], vocab), tokenization=tokenization)
            
            # Clean up the strings
            start_mol = clean_output(start_mol_raw)
            end_mol = clean_output(end_mol_raw)
            
            seed_z1 = z[ind1]
            seed_z2 = z[ind2]
            
            print(f"🔄 Interpolation {e+1}/10: {start_mol} ↔ {end_mol}")

            # Number of steps for interpolation
            num_steps = 10

            # Calculate the step size for each dimension
            step_sizes = (seed_z2 - seed_z1) / (num_steps + 1)

            # Generate interpolated vectors
            interpolated_vectors = [seed_z1 + i * step_sizes for i in range(1, num_steps + 1)]

            # Include the endpoints
            interpolated_vectors = torch.stack([seed_z1] + interpolated_vectors + [seed_z2])

            # Generate molecules for each interpolated vector
            for s in range(interpolated_vectors.shape[0]):
                prediction_interp, _, _, _, y_interp = model.inference(data=interpolated_vectors[s].unsqueeze(0), device=device, sample=False, log_var=None)
                
                raw_string = combine_tokens(tokenids_to_vocab(prediction_interp[0][0].tolist(), vocab), tokenization=tokenization)
                processed_string = process_generated_string(raw_string, enforce_homopolymer=args.enforce_homopolymer)
                all_predictions_interp.append(processed_string)
                
                # Save property predictions if requested
                if args.save_properties and torch.is_tensor(y_interp):
                    property_values = y_interp.cpu().numpy()
                    all_properties_interp.append(property_values)

            # Save interpolation results for this example
            with open(os.path.join(dir_name, f'interpolated_polymers_example{e}.txt'), 'w') as f:
                f.write(f"Properties: {property_names}\n")
                f.write(f"Molecule1: {start_mol}\n")
                f.write(f"Molecule2: {end_mol}\n")
                f.write("The following are the stepwise interpolated molecules:\n")
                for s, mol in enumerate(all_predictions_interp):
                    f.write(f"Step {s}: {mol}\n")
                    
            all_predictions_interp_all.extend(all_predictions_interp)
            if args.save_properties and all_properties_interp:
                all_properties_interp_all.extend(all_properties_interp)

    print(f"✅ Saved {examples} interpolation examples with {len(all_predictions_interp_all)} total interpolated molecules")
    
    if args.save_properties and all_properties_interp_all:
        all_properties_interp_all = np.array(all_properties_interp_all)
        with open(os.path.join(dir_name, 'interpolated_polymers_properties.npy'), 'wb') as f:
            np.save(f, all_properties_interp_all)
        print(f"✅ Saved properties of interpolated molecules: {all_properties_interp_all.shape}")

    # Final summary
    print('\n' + '='*60)
    print('🎉 GENERATION COMPLETED SUCCESSFULLY')
    print('='*60)
    print(f"📊 Generation Summary:")
    print(f"  Random generations: {len(all_predictions)}")
    print(f"  Seed-based generations: {len(all_predictions_seed)}")
    print(f"  Interpolated molecules: {len(all_predictions_interp_all)}")
    print(f"  Total molecules generated: {len(all_predictions) + len(all_predictions_seed) + len(all_predictions_interp_all)}")
    print(f"📁 Results saved to: {dir_name}")
    if args.enforce_homopolymer:
        print(f"🧪 Homopolymer format enforced on all generated structures")
    if args.save_properties:
        print(f"🔬 Property predictions saved for all generated molecules")
        print(f"📋 Properties: {property_names}")

else: 
    print("❌ The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
