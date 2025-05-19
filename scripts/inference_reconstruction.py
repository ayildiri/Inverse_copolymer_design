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
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load model checkpoints from and save results to")
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
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

# Add flexible property arguments
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties used in the model")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
parser.add_argument("--save_properties", action="store_true",
                    help="Save property predictions alongside reconstruction results")

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

print(f"Running inference reconstruction for model with {property_count} properties: {property_names}")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "test"
data_augment = "old" # new or old

# Load test data
dict_test_loader = torch.load(main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')
num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

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
    
    batch_size = model_config['batch_size']
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    
    # Load vocabulary
    vocab_file = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    vocab = load_vocab(vocab_file=vocab_file)
    
    # Initialize model
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
    print(f'🔄 Running inference reconstruction on test set')

    # Initialize storage lists
    all_predictions = []
    all_real = []
    all_latents = []
    all_properties_pred = [] if args.save_properties else None
    all_properties_real = [] if args.save_properties else None

    # Run inference
    batches = list(range(len(dict_test_loader)))
    
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}")
            
            data = dict_test_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_test_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)
            
            model.beta = 1.0
            predictions, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
            
            # Convert predictions to strings
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
            all_predictions.extend(prediction_strings)
            
            # Convert real molecules to strings
            real_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization) for sample in range(len(data))]
            all_real.extend(real_strings)
            
            # Save latent vectors
            all_latents.extend(z.cpu().numpy())
            
            # Save property predictions if requested
            if args.save_properties:
                if torch.is_tensor(y):
                    properties_pred_batch = y.cpu().numpy()
                    all_properties_pred.extend(properties_pred_batch)
                
                # Extract real property values dynamically
                properties_real_batch = []
                for j in range(property_count):
                    prop_attr = f'y{j+1}'
                    if hasattr(data, prop_attr):
                        prop_values = getattr(data, prop_attr).cpu().numpy()
                        if j == 0:  # Initialize for first property
                            properties_real_batch = prop_values.reshape(-1, 1)
                        else:  # Stack additional properties
                            properties_real_batch = np.column_stack([properties_real_batch, prop_values])
                    else:
                        print(f"Warning: Property {prop_attr} not found in data")
                        # Add NaN values for missing property
                        nan_values = np.full((len(data), 1), np.nan)
                        if j == 0:
                            properties_real_batch = nan_values
                        else:
                            properties_real_batch = np.column_stack([properties_real_batch, nan_values])
                
                if properties_real_batch.size > 0:
                    all_properties_real.extend(properties_real_batch)

    print(f'💾 Saving inference reconstruction results')

    # Save reconstruction results
    with open(os.path.join(dir_name, 'all_test_prediction_strings.pkl'), 'wb') as f:
        pickle.dump(all_predictions, f)
    with open(os.path.join(dir_name, 'all_test_real_strings.pkl'), 'wb') as f:
        pickle.dump(all_real, f)
    with open(os.path.join(dir_name, 'all_test_latents.npy'), 'wb') as f:
        np.save(f, np.array(all_latents))

    print(f"✅ Saved {len(all_predictions)} reconstructed molecules")
    print(f"✅ Saved {len(all_real)} real molecules")
    print(f"✅ Saved latent vectors: {np.array(all_latents).shape}")

    # Save property results if requested
    if args.save_properties:
        if all_properties_pred:
            all_properties_pred = np.array(all_properties_pred)
            with open(os.path.join(dir_name, 'all_test_properties_predicted.npy'), 'wb') as f:
                np.save(f, all_properties_pred)
            print(f"✅ Saved predicted properties: {all_properties_pred.shape}")
        
        if all_properties_real:
            all_properties_real = np.array(all_properties_real)
            with open(os.path.join(dir_name, 'all_test_properties_real.npy'), 'wb') as f:
                np.save(f, all_properties_real)
            print(f"✅ Saved real properties: {all_properties_real.shape}")

    # Calculate and save reconstruction accuracy
    exact_matches = sum(1 for pred, real in zip(all_predictions, all_real) if pred == real)
    total_molecules = len(all_predictions)
    reconstruction_accuracy = exact_matches / total_molecules

    # Save reconstruction summary
    with open(os.path.join(dir_name, 'reconstruction_summary.txt'), 'w') as f:
        f.write("Inference Reconstruction Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Properties: {property_names}\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Total molecules: {total_molecules}\n")
        f.write(f"Exact reconstructions: {exact_matches}\n")
        f.write(f"Reconstruction accuracy: {reconstruction_accuracy:.4f} ({reconstruction_accuracy*100:.2f}%)\n\n")
        
        if args.save_properties and all_properties_pred is not None:
            f.write("Property Prediction Summary:\n")
            f.write("-"*30 + "\n")
            for i, prop_name in enumerate(property_names):
                pred_values = all_properties_pred[:, i]
                f.write(f"{prop_name} (Predicted):\n")
                f.write(f"  Mean: {np.mean(pred_values):.4f}\n")
                f.write(f"  Std:  {np.std(pred_values):.4f}\n")
                f.write(f"  Min:  {np.min(pred_values):.4f}\n")
                f.write(f"  Max:  {np.max(pred_values):.4f}\n\n")
        
        if args.save_properties and all_properties_real is not None:
            f.write("Real Property Summary:\n")
            f.write("-"*30 + "\n")
            for i, prop_name in enumerate(property_names):
                real_values = all_properties_real[:, i]
                valid_values = real_values[~np.isnan(real_values)]
                if len(valid_values) > 0:
                    f.write(f"{prop_name} (Real):\n")
                    f.write(f"  Mean: {np.mean(valid_values):.4f}\n")
                    f.write(f"  Std:  {np.std(valid_values):.4f}\n")
                    f.write(f"  Min:  {np.min(valid_values):.4f}\n")
                    f.write(f"  Max:  {np.max(valid_values):.4f}\n\n")

    # Save detailed comparison for first 100 molecules
    with open(os.path.join(dir_name, 'reconstruction_examples.txt'), 'w') as f:
        f.write("Reconstruction Examples (First 100 molecules)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Properties: {property_names}\n\n")
        
        for i in range(min(100, len(all_predictions))):
            f.write(f"Molecule {i+1}:\n")
            f.write(f"  Real:        {all_real[i]}\n")
            f.write(f"  Predicted:   {all_predictions[i]}\n")
            f.write(f"  Match:       {'✓' if all_real[i] == all_predictions[i] else '✗'}\n")
            
            if args.save_properties and all_properties_pred is not None and all_properties_real is not None:
                f.write("  Properties:\n")
                for j, prop_name in enumerate(property_names):
                    real_val = all_properties_real[i, j] if not np.isnan(all_properties_real[i, j]) else "N/A"
                    pred_val = all_properties_pred[i, j]
                    f.write(f"    {prop_name}: Real={real_val}, Pred={pred_val:.4f}\n")
            f.write("\n")

    print('\n' + '='*60)
    print('🎉 INFERENCE RECONSTRUCTION COMPLETED')
    print('='*60)
    print(f"📊 Summary:")
    print(f"  Total molecules: {total_molecules}")
    print(f"  Exact reconstructions: {exact_matches}")
    print(f"  Reconstruction accuracy: {reconstruction_accuracy:.4f} ({reconstruction_accuracy*100:.2f}%)")
    print(f"📁 Results saved to: {dir_name}")
    if args.save_properties:
        print(f"🔬 Property predictions and comparisons saved")
        print(f"📋 Properties: {property_names}")

else: 
    print("❌ The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
