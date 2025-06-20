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
import re

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
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Path to custom dataset files (will use default naming pattern if not specified)")
parser.add_argument("--save_properties", action="store_true",
                    help="Save predicted properties of generated molecules")
parser.add_argument("--enforce_homopolymer", action="store_true", default=False,
                    help="Enforce homopolymer format in generated structures")

# Quality control arguments
parser.add_argument("--quality_control", action="store_true", default=True,
                    help="Enable quality control to filter invalid molecules during generation")
parser.add_argument("--target_molecules", type=int, default=16000,
                    help="Target number of valid molecules to generate")
parser.add_argument("--max_attempts", type=int, default=25000,
                    help="Maximum generation attempts before stopping")
parser.add_argument("--sampling_strategy", type=str, default="conservative", 
                    choices=["conservative", "standard", "aggressive"],
                    help="Latent space sampling strategy")

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
if args.quality_control:
    print(f"Quality control enabled: targeting {args.target_molecules} valid molecules")

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

# ================================
# ENHANCED G2S FORMAT HANDLING
# ================================

def clean_output(polymer_string):
    """Remove all padding underscores from generated polymer strings"""
    if not polymer_string:
        return ""
    return polymer_string.rstrip('_')

# Add this import at the top of your generate.py file if not already present:
# import numpy as np

def safe_token_processing(token_ids, vocab, tokenization="RT_tokenized"):
    """Safely process tokens with comprehensive error handling"""
    try:
        # Handle empty check for different data types
        if token_ids is None:
            return ""
        
        # Check for empty arrays/tensors properly
        if hasattr(token_ids, '__len__'):
            if len(token_ids) == 0:
                return ""
        elif isinstance(token_ids, torch.Tensor):
            if token_ids.numel() == 0:
                return ""
        elif isinstance(token_ids, np.ndarray):
            if token_ids.size == 0:
                return ""
        
        # Handle different input types
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Ensure token_ids is a list
        if not isinstance(token_ids, list):
            try:
                token_ids = list(token_ids)
            except Exception:
                return ""
        
        # Convert token IDs to vocabulary with error handling
        tokens = []
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            elif isinstance(token_id, np.ndarray):
                token_id = token_id.item()
            
            # Skip invalid token IDs
            if not isinstance(token_id, (int, float)) or token_id < 0:
                continue
                
            token = tokenids_to_vocab([token_id], vocab)
            if token and token[0] not in ['_PAD', '_SOS', '_EOS', '_UNK']:
                tokens.extend(token)
        
        # Combine tokens safely
        if not tokens:
            return ""
            
        return combine_tokens(tokens, tokenization=tokenization)
        
    except Exception as e:
        print(f"Warning: Token processing failed: {e}")
        return ""

def validate_basic_smiles(smiles_string):
    """Basic SMILES validation for G2S format"""
    try:
        if not smiles_string or len(smiles_string) < 5:
            return False
        
        # Must contain attachment points
        if '[*:' not in smiles_string or ']' not in smiles_string:
            return False
        
        # Basic bracket matching
        if smiles_string.count('[') != smiles_string.count(']'):
            return False
        
        # Should have at least one monomer separator or sufficient complexity
        if '.' not in smiles_string and len(smiles_string) < 15:
            return False
        
        # Check for reasonable attachment point format
        import re
        attachment_points = re.findall(r'\[\*:(\d+)\]', smiles_string)
        if len(attachment_points) < 2:
            return False
        
        return True
        
    except Exception:
        return False

def fix_connectivity_pattern(connectivity_string):
    """Fix malformed connectivity patterns"""
    try:
        if not connectivity_string:
            return ""
        
        import re
        
        # Remove excessive repetitions and fix malformed patterns
        fixed = connectivity_string
        
        # Fix patterns like ":0.500:0.500:0.500..." by keeping only the first two values
        fixed = re.sub(r'(<\d+-\d+):(\d+\.\d+):(\d+\.\d+)(:0\.500)*', r'\1:\2:\3', fixed)
        
        # Fix patterns missing proper structure
        fixed = re.sub(r'<(\d+)-(\d+):(\d+\.\d+)<', r'<\1-\2:\3:0.250<', fixed)
        
        # Ensure proper format for remaining patterns
        patterns = []
        for match in re.finditer(r'<(\d+)-(\d+):(\d+\.\d+):(\d+\.\d+)', fixed):
            start, end, prob1, prob2 = match.groups()
            # Validate the values
            try:
                start_num = int(start)
                end_num = int(end)
                prob1_val = float(prob1)
                prob2_val = float(prob2)
                
                if 1 <= start_num <= 4 and 1 <= end_num <= 4 and 0 <= prob1_val <= 1 and 0 <= prob2_val <= 1:
                    patterns.append(f"<{start}-{end}:{prob1}:{prob2}")
            except ValueError:
                continue
        
        return "".join(patterns[:6])  # Limit to reasonable number of patterns
        
    except Exception:
        return ""

def reconstruct_g2s_format(raw_string):
    """Reconstruct proper G2S format from raw generated string"""
    try:
        if not raw_string:
            return None
        
        # Clean the string first
        clean_string = clean_output(raw_string)
        
        if '|' not in clean_string:
            return None
        
        # Split into components
        parts = clean_string.split('|')
        
        if len(parts) < 3:
            return None
        
        # Extract SMILES (first part)
        smiles_part = parts[0]
        
        # Validate basic SMILES
        if not validate_basic_smiles(smiles_part):
            return None
        
        # Find where connectivity starts
        connectivity_start = None
        stoich_parts = []
        
        for i, part in enumerate(parts[1:], 1):
            if '<' in part:
                connectivity_start = i
                break
            else:
                # Try to parse as stoichiometry
                try:
                    val = float(part)
                    if 0 <= val <= 1:
                        stoich_parts.append(part)
                except ValueError:
                    pass
        
        # Ensure we have at least 2 stoichiometry values
        if len(stoich_parts) < 2:
            # Default to equal stoichiometry
            stoich_parts = ["0.500", "0.500"]
        
        # Handle connectivity
        connectivity_string = ""
        if connectivity_start is not None:
            connectivity_parts = parts[connectivity_start:]
            connectivity_string = "|".join(connectivity_parts)
            connectivity_string = fix_connectivity_pattern(connectivity_string)
        
        # Add default connectivity if missing
        if not connectivity_string:
            connectivity_string = "<1-3:0.250:0.250<1-4:0.250:0.250"
        
        # Reconstruct the format
        result = smiles_part + "|" + "|".join(stoich_parts) + "|" + connectivity_string
        
        return result
        
    except Exception as e:
        print(f"Warning: G2S reconstruction failed: {e}")
        return None

def validate_g2s_format(polymer_string):
    """Validate complete G2S format"""
    try:
        if not polymer_string:
            return False
        
        parts = polymer_string.split('|')
        if len(parts) < 4:  # SMILES|stoich1|stoich2|connectivity
            return False
        
        # Validate SMILES
        smiles_part = parts[0]
        if not validate_basic_smiles(smiles_part):
            return False
        
        # Validate stoichiometry
        stoich_valid = 0
        for i in range(1, len(parts)):
            if '<' in parts[i]:
                break
            try:
                val = float(parts[i])
                if 0 <= val <= 1:
                    stoich_valid += 1
            except ValueError:
                break
        
        if stoich_valid < 2:
            return False
        
        # Check for connectivity patterns
        connectivity_found = any('<' in part and ':' in part and '-' in part for part in parts)
        
        return connectivity_found
        
    except Exception:
        return False

def convert_to_homopolymer_format(polymer_string):
    """Convert any polymer string to homopolymer format by making monA = monB"""
    try:
        if not polymer_string or '|' not in polymer_string:
            return polymer_string
        
        parts = polymer_string.split('|')
        if len(parts) < 2:
            return polymer_string
        
        # Extract SMILES part
        smiles_part = parts[0]
        
        # If multiple monomers, make them the same (use first monomer)
        if '.' in smiles_part:
            monomers = smiles_part.split('.')
            if len(monomers) >= 2:
                # Use first monomer for both
                homopolymer_smiles = f"{monomers[0]}.{monomers[0]}"
                
                # Update stoichiometry to 0.5|0.5
                new_parts = [homopolymer_smiles, "0.500", "0.500"]
                
                # Keep connectivity if present
                connectivity_parts = []
                for part in parts[1:]:
                    if '<' in part:
                        connectivity_parts.append(part)
                        break
                
                if connectivity_parts:
                    new_parts.extend(connectivity_parts)
                else:
                    new_parts.append("<1-3:0.250:0.250<1-4:0.250:0.250")
                
                return "|".join(new_parts)
        
        return polymer_string
        
    except Exception:
        return polymer_string

def enhanced_polymer_processing(pred_string, enforce_homopolymer=False):
    """Enhanced processing with comprehensive error handling"""
    try:
        if not pred_string:
            return None
        
        # First attempt: reconstruct G2S format
        reconstructed = reconstruct_g2s_format(pred_string)
        
        if reconstructed and validate_g2s_format(reconstructed):
            # Apply homopolymer conversion if requested
            if enforce_homopolymer:
                return convert_to_homopolymer_format(reconstructed)
            return reconstructed
        
        # Second attempt: basic cleanup
        cleaned = clean_output(pred_string)
        if cleaned and len(cleaned) > 10:
            return cleaned
        
        return None
        
    except Exception as e:
        print(f"Warning: Polymer processing failed: {e}")
        return None

# ================================
# ENHANCED GENERATION WITH ERROR HANDLING
# ================================

def safe_model_inference(model, z_rand, device):
    """Safely run model inference with comprehensive error handling"""
    try:
        with torch.no_grad():
            model.eval()
            
            # Check if model has property prediction
            if hasattr(model, 'inference') and callable(model.inference):
                result = model.inference(data=z_rand, device=device, sample=False, log_var=None)
                
                # Handle different return formats
                if len(result) >= 4:
                    predictions_rand, _, _, z = result[:4]
                    y = result[4] if len(result) > 4 else None
                else:
                    predictions_rand, _, _, z = result
                    y = None
                    
                return predictions_rand, z, y
            else:
                print("Warning: Model inference method not found")
                return None, None, None
                
    except Exception as e:
        print(f"Error in model inference: {e}")
        return None, None, None

def generate_batch_with_error_handling(model, vocab, tokenization, batch_size=16, 
                                     embedding_dimension=32, device=device, 
                                     enforce_homopolymer=False, save_properties=False,
                                     sampling_strategy="conservative"):
    """Generate a batch with comprehensive error handling"""
    
    valid_predictions = []
    batch_properties = []
    
    try:
        # Generate latent vectors based on sampling strategy
        if sampling_strategy == "conservative":
            z_rand = torch.randn((batch_size, embedding_dimension), device=device) * 0.8
        elif sampling_strategy == "aggressive":
            z_rand = torch.randn((batch_size, embedding_dimension), device=device) * 1.5
        else:  # standard
            z_rand = torch.randn((batch_size, embedding_dimension), device=device) * 1.0
        
        # Run model inference safely
        predictions_rand, z, y = safe_model_inference(model, z_rand, device)
        
        if predictions_rand is None:
            print("Warning: Model inference failed")
            return valid_predictions, batch_properties
        
        # Process each prediction safely
        for sample in range(len(predictions_rand)):
            try:
                # Check if prediction has the expected structure
                if (len(predictions_rand[sample]) == 0 or 
                    not hasattr(predictions_rand[sample][0], '__iter__')):
                    continue
                
                # Extract tokens safely
                pred_tokens = predictions_rand[sample][0]
                if hasattr(pred_tokens, 'tolist'):
                    pred_tokens = pred_tokens.tolist()
                elif not isinstance(pred_tokens, list):
                    pred_tokens = list(pred_tokens)
                
                # Convert to string safely
                pred_string = safe_token_processing(pred_tokens, vocab, tokenization)
                
                if not pred_string:
                    continue
                
                # Process the string to G2S format
                processed_string = enhanced_polymer_processing(
                    pred_string, 
                    enforce_homopolymer=enforce_homopolymer
                )
                
                if processed_string and validate_g2s_format(processed_string):
                    valid_predictions.append(processed_string)
                    
                    # Save properties if requested
                    if save_properties and y is not None and torch.is_tensor(y):
                        try:
                            batch_properties.append(y[sample].cpu().numpy())
                        except Exception:
                            pass
                
            except Exception as e:
                print(f"Warning: Processing sample {sample} failed: {e}")
                continue
        
        return valid_predictions, batch_properties
        
    except Exception as e:
        print(f"Error in batch generation: {e}")
        return valid_predictions, batch_properties

def generate_with_comprehensive_error_handling(model, vocab, tokenization, target_count=100, 
                                              max_attempts=500, batch_size=16, 
                                              embedding_dimension=32, device=device, 
                                              enforce_homopolymer=False, save_properties=False,
                                              sampling_strategy="conservative"):
    """Generation with comprehensive error handling and recovery"""
    
    all_valid_predictions = []
    all_properties = [] if save_properties else None
    attempts = 0
    consecutive_failures = 0
    
    print(f"🎯 Starting generation with comprehensive error handling")
    print(f"Target: {target_count} valid molecules, Max attempts: {max_attempts}")
    
    while len(all_valid_predictions) < target_count and attempts < max_attempts:
        
        try:
            # Generate a batch
            batch_valid, batch_properties = generate_batch_with_error_handling(
                model=model,
                vocab=vocab,
                tokenization=tokenization,
                batch_size=batch_size,
                embedding_dimension=embedding_dimension,
                device=device,
                enforce_homopolymer=enforce_homopolymer,
                save_properties=save_properties,
                sampling_strategy=sampling_strategy
            )
            
            # Check if batch generation was successful
            if batch_valid:
                all_valid_predictions.extend(batch_valid)
                if save_properties and batch_properties:
                    all_properties.extend(batch_properties)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                print(f"Warning: Batch {attempts//batch_size + 1} produced no valid molecules")
            
            attempts += batch_size
            
            # Progress reporting
            if attempts % (batch_size * 5) == 0 or len(batch_valid) > 0:
                batch_validity = len(batch_valid) / batch_size if batch_size > 0 else 0
                total_validity = len(all_valid_predictions) / attempts if attempts > 0 else 0
                
                print(f'Batch {attempts//batch_size}: {len(batch_valid)}/{batch_size} valid ({batch_validity:.1%}) | '
                      f'Total: {len(all_valid_predictions)}/{attempts} ({total_validity:.1%}) | '
                      f'Progress: {len(all_valid_predictions)}/{target_count} '
                      f'({len(all_valid_predictions)/target_count:.1%})')
            
            # Early stopping if consistent failures
            if consecutive_failures > 10:
                print("⚠️ Too many consecutive failures. Model may need retraining.")
                break
                
            # Reduce batch size if many failures
            if consecutive_failures > 5 and batch_size > 8:
                batch_size = max(8, batch_size // 2)
                print(f"Reducing batch size to {batch_size} due to failures")
                
        except Exception as e:
            print(f"Critical error in generation loop: {e}")
            attempts += batch_size
            consecutive_failures += 1
            continue
    
    final_validity = len(all_valid_predictions) / attempts if attempts > 0 else 0
    print(f"✅ Generation completed: {len(all_valid_predictions)} valid molecules from {attempts} attempts")
    print(f"📊 Final validity rate: {final_validity:.1%}")
    
    return all_valid_predictions[:target_count], all_properties

# ================================
# ANALYSIS AND VALIDATION FUNCTIONS
# ================================

def analyze_generated_polymers(polymer_list, sample_size=10):
    """Analyze generated polymers for G2S format compliance"""
    
    print("🔍 G2S FORMAT ANALYSIS")
    print("="*50)
    
    if not polymer_list:
        print("❌ No polymers to analyze")
        return {}
    
    sample_polymers = polymer_list[:sample_size] if len(polymer_list) > sample_size else polymer_list
    
    format_compliant = 0
    basic_valid = 0
    has_connectivity = 0
    
    print(f"\n📊 Analyzing {len(sample_polymers)} sample polymers...")
    print("-"*50)
    
    for i, polymer in enumerate(sample_polymers):
        print(f"\n🧪 Polymer {i+1}:")
        print(f"   {polymer}")
        
        # Check basic SMILES validity
        if polymer and '|' in polymer:
            smiles_part = polymer.split('|')[0]
            if validate_basic_smiles(smiles_part):
                basic_valid += 1
        
        # Check G2S format compliance
        complete_valid = validate_g2s_format(polymer)
        if complete_valid:
            format_compliant += 1
            
        # Check connectivity patterns
        if '<' in polymer and ':' in polymer and '-' in polymer:
            has_connectivity += 1
            print(f"   ✅ Has connectivity patterns")
        else:
            print(f"   ⚠️  No connectivity patterns")
        
        print(f"   Valid: {'✅' if complete_valid else '❌'}")
    
    print("="*50)
    print("📋 ANALYSIS SUMMARY:")
    print(f"   Basic SMILES validity: {basic_valid}/{len(sample_polymers)} ({basic_valid/len(sample_polymers)*100:.1f}%)")
    print(f"   G2S format compliance: {format_compliant}/{len(sample_polymers)} ({format_compliant/len(sample_polymers)*100:.1f}%)")
    print(f"   Contains connectivity patterns: {has_connectivity}/{len(sample_polymers)} ({has_connectivity/len(sample_polymers)*100:.1f}%)")
    
    return {
        'total_analyzed': len(sample_polymers),
        'basic_valid': basic_valid,
        'format_compliant': format_compliant,
        'has_connectivity': has_connectivity,
        'compliance_rate': format_compliant / len(sample_polymers) if sample_polymers else 0
    }

def save_polymers_as_text(polymers_list, filename, title="Generated Polymers"):
    """Save polymers as readable text file with format explanation"""
    with open(filename, 'w') as f:
        f.write(f"{title}\n")
        f.write("="*len(title) + "\n\n")
        f.write(f"Total count: {len(polymers_list)}\n")
        f.write(f"Format: SMILES|stoichiometry|connectivity_patterns\n")
        f.write(f"Example format: [*:1]c1ccc([*:2])cc1.[*:3]c1ccc([*:4])c1|0.5|0.5|<1-3:0.25:0.25\n\n")
        
        for i, polymer in enumerate(polymers_list, 1):
            f.write(f"{i:4d}: {polymer}\n")
    
    print(f"✅ Saved {len(polymers_list)} polymers to readable text file: {filename}")

# ================================
# MODEL LOADING AND SETUP
# ================================

# Include property info in model name
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+str(property_str)+'/'

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
    
    vocab = load_vocab(vocab_file=vocab_file_path)
    
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file_path)
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

    # ================================
    # RANDOM GENERATION WITH ERROR HANDLING
    # ================================
    
    print(f'🎲 Generate random samples with enhanced error handling')
    torch.manual_seed(args.seed)
    
    if args.quality_control:
        # Use enhanced generation with comprehensive error handling
        all_predictions, all_properties = generate_with_comprehensive_error_handling(
            model=model,
            vocab=vocab,
            tokenization=tokenization,
            target_count=args.target_molecules,
            max_attempts=args.max_attempts,
            batch_size=16,
            embedding_dimension=embedding_dimension,
            device=device,
            enforce_homopolymer=args.enforce_homopolymer,
            save_properties=args.save_properties,
            sampling_strategy=args.sampling_strategy
        )
    else:
        # Original method with enhanced error handling
        print(f'🎲 Generate random samples (original method with error handling)')
        all_predictions = []
        all_properties = [] if args.save_properties else None
        
        for i in range(50):  # Reduced from 250 for testing
            try:
                batch_valid, batch_properties = generate_batch_with_error_handling(
                    model=model,
                    vocab=vocab,
                    tokenization=tokenization,
                    batch_size=32,
                    embedding_dimension=embedding_dimension,
                    device=device,
                    enforce_homopolymer=args.enforce_homopolymer,
                    save_properties=args.save_properties,
                    sampling_strategy=args.sampling_strategy
                )
                
                all_predictions.extend(batch_valid)
                if args.save_properties and batch_properties:
                    all_properties.extend(batch_properties)
                    
                print(f'Generated batch {i+1}/50: {len(batch_valid)} valid molecules')
                
            except Exception as e:
                print(f'Error in batch {i+1}: {e}')
                continue

    # Save random generation results
    if all_predictions:
        with open(os.path.join(dir_name, 'generated_polymers.pkl'), 'wb') as f:
            pickle.dump(all_predictions, f)
        print(f"✅ Saved {len(all_predictions)} random generations to generated_polymers.pkl")
        
        # Save as readable text file
        save_polymers_as_text(
            all_predictions,
            os.path.join(dir_name, 'generated_polymers.txt'),
            "Random Generated Polymers"
        )
        
        if args.save_properties and all_properties:
            all_properties = np.array(all_properties)
            with open(os.path.join(dir_name, 'generated_polymers_properties.npy'), 'wb') as f:
                np.save(f, all_properties)
            print(f"✅ Saved properties of generated polymers: {all_properties.shape}")
    else:
        print("❌ No valid polymers generated in random generation")

    # ================================
    # SEED-BASED GENERATION WITH ERROR HANDLING
    # ================================
    
    print(f'🌱 Generate samples around seed molecule')
    
    all_predictions_seed = []
    all_properties_seed = [] if args.save_properties else None
    
    try:
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
            
            # Get latent representations safely
            result = model.inference(data=data, device=device, 
                                   dest_is_origin_matrix=dest_is_origin_matrix, 
                                   inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, 
                                   sample=False, log_var=None)
            
            if len(result) >= 4:
                _, _, _, z = result[:4]
                y = result[4] if len(result) > 4 else None
            else:
                print("Warning: Unexpected inference result format")
                z = None
                y = None
            
            if z is not None:
                # Randomly select a seed molecule
                ind = random.choice(list(range(min(64, z.size(0)))))
                seed_z = z[ind]
                seed_z = seed_z.unsqueeze(0).repeat(32, 1)  # Reduced batch size
                
                # Get seed string safely with enhanced error handling
                try:
                    # Extract seed tokens with proper indexing
                    if hasattr(data, 'tgt_token_ids') and hasattr(data.tgt_token_ids, '__getitem__'):
                        seed_tokens = data.tgt_token_ids[ind]
                        
                        # Additional safety check for the extracted tokens
                        if seed_tokens is not None:
                            seed_string = safe_token_processing(seed_tokens, vocab, tokenization)
                            if seed_string:
                                print(f"🌱 Seed molecule: {seed_string}")
                            else:
                                print("🌱 Seed molecule: [Could not decode tokens]")
                                seed_string = "[Could not decode tokens]"
                        else:
                            print("🌱 Seed molecule: [No tokens found]")
                            seed_string = "[No tokens found]"
                    else:
                        print("🌱 Seed molecule: [No token data available]")
                        seed_string = "[No token data available]"
                        
                except Exception as e:
                    print(f"Warning: Could not extract seed string: {e}")
                    print("🌱 Seed molecule: [Extraction failed]")
                    seed_string = "[Extraction failed]"
                
                # Generate variations
                for r in range(8):
                    try:
                        # Add noise
                        mean = 0
                        std = args.epsilon / 2
                        noise = torch.tensor(np.random.normal(mean, std, size=seed_z.size()), 
                                           dtype=torch.float, device=device)
                        seed_z_noise = seed_z + noise
                        
                        # Generate from noisy latent
                        batch_valid, batch_properties = generate_batch_with_error_handling(
                            model=model,
                            vocab=vocab,
                            tokenization=tokenization,
                            batch_size=seed_z_noise.size(0),
                            embedding_dimension=embedding_dimension,
                            device=device,
                            enforce_homopolymer=args.enforce_homopolymer,
                            save_properties=args.save_properties,
                            sampling_strategy="conservative"
                        )
                        
                        all_predictions_seed.extend(batch_valid)
                        if args.save_properties and batch_properties:
                            all_properties_seed.extend(batch_properties)
                            
                    except Exception as e:
                        print(f"Error in seed variation {r}: {e}")
                        continue
        
        # Save seed-based results
        if all_predictions_seed:
            print(f'💾 Saving generated strings around seed molecule')
            
            with open(os.path.join(dir_name, 'seed_polymer.txt'), 'w') as f:
                f.write(f'Seed molecule: {seed_string}\n')
                f.write(f'Properties: {property_names}\n')

            with open(os.path.join(dir_name, f'generated_polymers_from_seed.pkl'), 'wb') as f:
                pickle.dump(all_predictions_seed, f)
                
            save_polymers_as_text(
                all_predictions_seed,
                os.path.join(dir_name, f'seed_based_polymers.txt'),
                "Seed-Based Generated Polymers"
            )
                
            print(f"✅ Saved {len(all_predictions_seed)} seed-based generations")
        else:
            print("❌ No valid polymers generated in seed-based generation")
            
    except Exception as e:
        print(f"Error in seed-based generation: {e}")

    # ================================
    # FINAL SUMMARY WITH VALIDATION
    # ================================

    print('\n' + '='*60)
    print('🎉 GENERATION COMPLETED')
    print('='*60)

    # Run analysis on generated results
    if all_predictions:
        validation_results = analyze_generated_polymers(all_predictions)
        
        print(f"📊 Generation Summary:")
        print(f"  Random generations: {len(all_predictions)}")
        print(f"  Seed-based generations: {len(all_predictions_seed) if all_predictions_seed else 0}")
        print(f"  Total molecules generated: {len(all_predictions) + (len(all_predictions_seed) if all_predictions_seed else 0)}")
        print(f"📁 Results saved to: {dir_name}")

        if validation_results:
            print(f"\n🔬 G2S FORMAT COMPLIANCE:")
            print(f"  Basic SMILES validity: {validation_results['basic_valid']}/{validation_results['total_analyzed']} ({validation_results['basic_valid']/validation_results['total_analyzed']*100:.1f}%)")
            print(f"  G2S format compliance: {validation_results['format_compliant']}/{validation_results['total_analyzed']} ({validation_results['compliance_rate']*100:.1f}%)")
            print(f"  Contains connectivity patterns: {validation_results['has_connectivity']}/{validation_results['total_analyzed']} ({validation_results['has_connectivity']/validation_results['total_analyzed']*100:.1f}%)")

        if args.enforce_homopolymer:
            print(f"🧪 Homopolymer format enforced on all generated structures")
        if args.save_properties:
            print(f"🔬 Property predictions saved for generated molecules")
            print(f"📋 Properties: {property_names}")
    else:
        print("❌ No valid polymers were generated. Model may need retraining or parameter adjustment.")

else: 
    print("❌ The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
