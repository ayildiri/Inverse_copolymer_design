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
# PROPER POLYMER FORMAT HANDLING
# ================================

def clean_output(polymer_string):
    """Remove all padding underscores from generated polymer strings"""
    return polymer_string.rstrip('_')

def convert_to_homopolymer_format(polymer_string):
    """
    Convert any polymer string to homopolymer format by making monA = monB.
    Assumes the format: SMILES|stoichiometry|connectivity
    """
    polymer_string = clean_output(polymer_string)  # First remove padding
    
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
            new_parts = [homopolymer_smiles, "0.5", "0.5"]
            
            # Keep connectivity if present
            if len(parts) > 3:
                new_parts.extend(parts[3:])
            
            return "|".join(new_parts)
    
    return polymer_string

def validate_smiles_component(smiles_string):
    """Validate SMILES component with proper RDKit checking"""
    try:
        from rdkit import Chem
        
        if not smiles_string or smiles_string.strip() == '':
            return False
        
        # Handle multiple monomers separated by '.'
        if '.' in smiles_string:
            monomers = smiles_string.split('.')
            valid_count = 0
            for monomer in monomers:
                monomer = monomer.strip()
                if monomer:
                    # Replace attachment points for testing
                    test_monomer = re.sub(r'\[\*:\d+\]', '*', monomer)
                    mol = Chem.MolFromSmiles(test_monomer)
                    if mol is not None and mol.GetNumAtoms() > 0:
                        valid_count += 1
            return valid_count > 0
        else:
            # Single monomer
            test_smiles = re.sub(r'\[\*:\d+\]', '*', smiles_string)
            mol = Chem.MolFromSmiles(test_smiles)
            return mol is not None and mol.GetNumAtoms() > 0
    
    except Exception:
        return False

def extract_polymer_components(polymer_string):
    """
    Extract components from full polymer string format:
    SMILES|stoich1|stoich2|...|<connectivity patterns>
    """
    if not polymer_string:
        return None, [], []
    
    # Clean padding first
    polymer_string = clean_output(polymer_string)
    
    if '|' not in polymer_string:
        # Just SMILES
        return polymer_string, [], []
    
    parts = polymer_string.split('|')
    
    # First part is always SMILES
    smiles_part = parts[0]
    
    # Extract stoichiometry and connectivity
    stoich_parts = []
    connectivity_parts = []
    
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
            
        if part.startswith('<'):
            # Connectivity pattern
            connectivity_parts.append(part)
        else:
            # Try to parse as number (stoichiometry)
            try:
                val = float(part)
                if 0 <= val <= 1:
                    stoich_parts.append(val)
            except ValueError:
                # Could be part of connectivity that got split
                if ':' in part or '-' in part:
                    connectivity_parts.append(part)
    
    return smiles_part, stoich_parts, connectivity_parts

def fix_connectivity_patterns(connectivity_parts):
    """Fix and validate connectivity patterns"""
    if not connectivity_parts:
        return []
    
    fixed_patterns = []
    current_pattern = ""
    
    for part in connectivity_parts:
        if part.startswith('<'):
            # New pattern starts
            if current_pattern:
                fixed_patterns.append(current_pattern)
            current_pattern = part
        else:
            # Continuation of pattern
            current_pattern += part
    
    # Add the last pattern
    if current_pattern:
        fixed_patterns.append(current_pattern)
    
    # Validate and fix each pattern
    validated_patterns = []
    for pattern in fixed_patterns:
        # Pattern should be like <1-3:0.25:0.25
        if re.match(r'<\d+-\d+:[\d.]+:[\d.]+', pattern):
            validated_patterns.append(pattern)
        else:
            # Try to fix common issues
            # Remove extra < symbols
            pattern = re.sub(r'<+', '<', pattern)
            if re.match(r'<\d+-\d+:[\d.]+:[\d.]+', pattern):
                validated_patterns.append(pattern)
    
    return validated_patterns

def reconstruct_polymer_string(smiles_part, stoich_parts, connectivity_parts):
    """Reconstruct proper polymer string from components"""
    if not smiles_part:
        return None
    
    # Start with SMILES
    result = smiles_part
    
    # Add stoichiometry
    if stoich_parts:
        for stoich in stoich_parts:
            result += f"|{stoich:.3f}"
    else:
        # Default stoichiometry
        if '.' in smiles_part:
            num_monomers = len(smiles_part.split('.'))
            equal_stoich = 1.0 / num_monomers
            for _ in range(num_monomers):
                result += f"|{equal_stoich:.3f}"
        else:
            result += "|1.000"
    
    # Add connectivity patterns
    for pattern in connectivity_parts:
        result += f"|{pattern}"
    
    return result

def fix_polymer_format_complete(polymer_string):
    """Fix polymer format while preserving complete structure"""
    if not polymer_string:
        return None
    
    # Extract components
    smiles_part, stoich_parts, connectivity_parts = extract_polymer_components(polymer_string)
    
    if not smiles_part:
        return None
    
    # Validate and fix SMILES
    if not validate_smiles_component(smiles_part):
        return None
    
    # Fix connectivity patterns
    fixed_connectivity = fix_connectivity_patterns(connectivity_parts)
    
    # Validate stoichiometry
    if stoich_parts:
        # Normalize stoichiometry to sum to 1.0
        total = sum(stoich_parts)
        if total > 0:
            stoich_parts = [s/total for s in stoich_parts]
    
    # Reconstruct
    return reconstruct_polymer_string(smiles_part, stoich_parts, fixed_connectivity)

def validate_complete_polymer_format(polymer_string, verbose=False):
    """
    Validate complete polymer format including connectivity patterns
    """
    if not polymer_string:
        if verbose: print("❌ Empty string")
        return False
    
    try:
        # Extract components
        smiles_part, stoich_parts, connectivity_parts = extract_polymer_components(polymer_string)
        
        if not smiles_part:
            if verbose: print("❌ No SMILES component found")
            return False
        
        # Validate SMILES
        if not validate_smiles_component(smiles_part):
            if verbose: print("❌ Invalid SMILES component")
            return False
        
        # Check stoichiometry
        if stoich_parts:
            total = sum(stoich_parts)
            if abs(total - 1.0) > 0.1:
                if verbose: print(f"❌ Stoichiometry doesn't sum to ~1.0: {total}")
                return False
        
        # Validate connectivity patterns
        for pattern in connectivity_parts:
            if not re.match(r'<\d+-\d+:[\d.]+:[\d.]+', pattern):
                if verbose: print(f"❌ Invalid connectivity pattern: {pattern}")
                return False
        
        if verbose:
            print("✅ Valid complete polymer format!")
            print(f"  - SMILES: {smiles_part}")
            print(f"  - Stoichiometry: {stoich_parts}")
            print(f"  - Connectivity: {len(connectivity_parts)} patterns")
        
        return True
        
    except Exception as e:
        if verbose: print(f"❌ Validation error: {str(e)}")
        return False

def process_generated_string(polymer_string, enforce_homopolymer=False):
    """Process generated string with proper format preservation"""
    if not polymer_string:
        return None
    
    # Clean padding
    clean_string = clean_output(polymer_string)
    
    # Try to fix format while preserving structure
    fixed_string = fix_polymer_format_complete(clean_string)
    
    if not fixed_string:
        return None
    
    # Optionally enforce homopolymer
    if enforce_homopolymer:
        return convert_to_homopolymer_format(fixed_string)
    
    return fixed_string

def enhanced_polymer_processing_complete(pred_string, enforce_homopolymer=False):
    """Enhanced processing that preserves the complete polymer format"""
    
    # First check if already valid
    if validate_complete_polymer_format(pred_string):
        if enforce_homopolymer:
            return convert_to_homopolymer_format(pred_string)
        return pred_string
    
    # Try to fix
    fixed = process_generated_string(pred_string, enforce_homopolymer)
    
    if fixed and validate_complete_polymer_format(fixed):
        return fixed
    
    return None

# ================================
# ANALYSIS AND VALIDATION FUNCTIONS
# ================================

def analyze_generated_polymers(polymer_list, sample_size=10):
    """Analyze generated polymers for complete format compliance"""
    
    print("🔍 COMPLETE POLYMER FORMAT ANALYSIS")
    print("="*60)
    
    if not polymer_list:
        print("❌ No polymers to analyze")
        return
    
    sample_polymers = polymer_list[:sample_size] if len(polymer_list) > sample_size else polymer_list
    
    format_compliant = 0
    basic_valid = 0
    has_connectivity = 0
    
    print(f"\n📊 Analyzing {len(sample_polymers)} sample polymers...")
    print("-"*60)
    
    for i, polymer in enumerate(sample_polymers):
        print(f"\n🧪 Polymer {i+1}:")
        print(f"   {polymer}")
        
        # Check basic SMILES validity
        smiles_part, _, _ = extract_polymer_components(polymer)
        if smiles_part and validate_smiles_component(smiles_part):
            basic_valid += 1
            
        # Check complete format compliance
        complete_valid = validate_complete_polymer_format(polymer, verbose=False)
        if complete_valid:
            format_compliant += 1
            
        # Check if has connectivity patterns
        _, _, connectivity = extract_polymer_components(polymer)
        if connectivity:
            has_connectivity += 1
            print(f"   ✅ Has {len(connectivity)} connectivity patterns")
        else:
            print(f"   ⚠️  No connectivity patterns")
        
        print(f"   Valid: {'✅' if complete_valid else '❌'}")
    
    print("="*60)
    print("📋 ANALYSIS SUMMARY:")
    print(f"   Basic SMILES validity: {basic_valid}/{len(sample_polymers)} ({basic_valid/len(sample_polymers)*100:.1f}%)")
    print(f"   Complete format compliance: {format_compliant}/{len(sample_polymers)} ({format_compliant/len(sample_polymers)*100:.1f}%)")
    print(f"   Contains connectivity patterns: {has_connectivity}/{len(sample_polymers)} ({has_connectivity/len(sample_polymers)*100:.1f}%)")
    
    if has_connectivity == 0:
        print("\n⚠️  CRITICAL ISSUE:")
        print("   No polymers contain connectivity patterns!")
        print("   The model may not be generating the complete format.")
        print("   Check tokenization and model architecture.")
    elif has_connectivity < len(sample_polymers):
        print(f"\n⚠️  PARTIAL ISSUE:")
        print(f"   Only {has_connectivity}/{len(sample_polymers)} polymers have connectivity patterns")
        print("   Some polymers are missing connectivity information")
    else:
        print("\n✅ EXCELLENT: All polymers contain connectivity patterns!")
    
    return {
        'total_analyzed': len(sample_polymers),
        'basic_valid': basic_valid,
        'format_compliant': format_compliant,
        'has_connectivity': has_connectivity,
        'compliance_rate': format_compliant / len(sample_polymers) if sample_polymers else 0
    }

def enhanced_validation_of_results(all_predictions):
    """Run comprehensive validation on generated results"""
    
    print("\n" + "="*60)
    print("🔬 COMPREHENSIVE VALIDATION OF GENERATED POLYMERS")
    print("="*60)
    
    # Analyze sample of generated polymers
    analysis_result = analyze_generated_polymers(all_predictions, sample_size=min(20, len(all_predictions)))
    
    # Save analysis report
    if analysis_result:
        with open(os.path.join(dir_name, 'polymer_format_analysis.txt'), 'w') as f:
            f.write("Complete Polymer Format Analysis Report\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total polymers analyzed: {analysis_result['total_analyzed']}\n")
            f.write(f"Basic SMILES validity: {analysis_result['basic_valid']}\n")
            f.write(f"Complete format compliance: {analysis_result['format_compliant']}\n")
            f.write(f"Contains connectivity patterns: {analysis_result['has_connectivity']}\n")
            f.write(f"Compliance rate: {analysis_result['compliance_rate']:.1%}\n\n")
            
            f.write("Sample polymers:\n")
            for i, polymer in enumerate(all_predictions[:10]):
                f.write(f"{i+1}: {polymer}\n")
    
    return analysis_result

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
# ENHANCED GENERATION FUNCTION
# ================================

def generate_with_enhanced_decoding(model, vocab, tokenization, target_count=100, max_attempts=500,
                                   batch_size=16, embedding_dimension=32, device=device, 
                                   enforce_homopolymer=False, save_properties=False):
    """Generation with enhanced decoding that preserves complete format"""
    
    all_valid_predictions = []
    all_properties = [] if save_properties else None
    attempts = 0
    
    print(f"🎯 Enhanced decoding generation: targeting {target_count} valid polymers with complete format")
    
    with torch.no_grad():
        model.eval()
        
        while len(all_valid_predictions) < target_count and attempts < max_attempts:
            
            # Conservative sampling for better quality
            z_rand = torch.randn((batch_size, embedding_dimension), device=device) * 0.5
            
            try:
                predictions_rand, _, _, z, y = model.inference(
                    data=z_rand, device=device, sample=False, log_var=None
                )
                
                batch_valid = []
                batch_properties = []
                
                for sample in range(len(predictions_rand)):
                    pred_tokens = predictions_rand[sample][0].tolist()
                    pred_string = combine_tokens(
                        tokenids_to_vocab(pred_tokens, vocab), 
                        tokenization=tokenization
                    )
                    
                    # Process with complete format preservation
                    processed_string = enhanced_polymer_processing_complete(
                        pred_string, 
                        enforce_homopolymer=enforce_homopolymer
                    )
                    
                    if processed_string and validate_complete_polymer_format(processed_string):
                        batch_valid.append(processed_string)
                        
                        if save_properties and torch.is_tensor(y):
                            batch_properties.append(y[sample].cpu().numpy())
                
                all_valid_predictions.extend(batch_valid)
                if save_properties and batch_properties:
                    all_properties.extend(batch_properties)
                
                attempts += batch_size
                
                # Quick format analysis on first batch
                if len(all_valid_predictions) >= 5 and attempts == batch_size:
                    print("\n🔍 Quick format analysis on first 5 polymers:")
                    for i, polymer in enumerate(all_valid_predictions[:5]):
                        _, _, connectivity = extract_polymer_components(polymer)
                        compliant = validate_complete_polymer_format(polymer, verbose=False)
                        print(f"  Polymer {i+1}: {'✅' if compliant else '❌'} | Connectivity patterns: {len(connectivity)}")
                    print()
                
                # Progress reporting
                if attempts % (batch_size * 5) == 0 or len(batch_valid) > 0:
                    batch_validity = len(batch_valid) / batch_size if batch_size > 0 else 0
                    total_validity = len(all_valid_predictions) / attempts if attempts > 0 else 0
                    
                    print(f'Batch {attempts//batch_size}: {len(batch_valid)}/{batch_size} valid ({batch_validity:.1%}) | '
                          f'Total: {len(all_valid_predictions)}/{attempts} ({total_validity:.1%}) | '
                          f'Progress: {len(all_valid_predictions)}/{target_count} '
                          f'({len(all_valid_predictions)/target_count:.1%})')
                
                # Early stopping if very poor performance
                if attempts > 200 and len(all_valid_predictions) < attempts * 0.01:
                    print("⚠️  Very low validity rate. Model may need retraining for complete format.")
                    break
                    
            except Exception as e:
                print(f"Error in generation batch: {e}")
                attempts += batch_size
                continue
    
    final_validity = len(all_valid_predictions) / attempts if attempts > 0 else 0
    print(f"✅ Enhanced generation completed: {len(all_valid_predictions)} valid polymers from {attempts} attempts")
    print(f"📊 Final validity rate: {final_validity:.1%}")
    
    return all_valid_predictions[:target_count], all_properties

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
    # TESTING COMPLETE FORMAT PROCESSING
    # ================================
    
    print("🧪 Testing Complete Format Processing:")
    print("-" * 50)
    
    # Test cases based on the expected G2S VAE format
    test_cases = [
        "[*:1]c1ccc([*:2])cc1.[*:3]c1ccc([*:4])c1|0.5|0.5|<1-3:0.25:0.25<1-4:0.25:0.25",
        "[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2.[*:3]c1ccc([*:4])c(N)c1|0.25|0.75|<1-3:0.25:0.25<1-4:0.25:0.25",
        "O=C([*:1])c1ccc([*:2])cc1|1.0|<1-2:0.5:0.5",  # Simplified case
        "O=C([*:1])[*:2]|1.0|",  # Missing connectivity
    ]
    
    for i, test in enumerate(test_cases):
        print(f"Test {i+1}:")
        print(f"  Input:     {test}")
        
        # Extract components
        smiles, stoich, connectivity = extract_polymer_components(test)
        print(f"  SMILES:    {smiles}")
        print(f"  Stoich:    {stoich}")
        print(f"  Connect:   {connectivity}")
        
        # Validate
        valid = validate_complete_polymer_format(test, verbose=False)
        print(f"  Valid:     {valid}")
        print()

    ### RANDOM GENERATION ###
    if args.quality_control:
        print(f'🎲 Generate random samples with complete format preservation')
        torch.manual_seed(args.seed)
        
        # Use enhanced generation that preserves complete format
        all_predictions, all_properties = generate_with_enhanced_decoding(
            model=model,
            vocab=vocab,
            tokenization=tokenization,
            target_count=args.target_molecules,
            max_attempts=args.max_attempts,
            batch_size=16,
            embedding_dimension=embedding_dimension,
            device=device,
            enforce_homopolymer=args.enforce_homopolymer,
            save_properties=args.save_properties
        )
        
    else:
        # Original generation method (for backward compatibility)
        print(f'🎲 Generate random samples (original method)')
        torch.manual_seed(args.seed)
        all_predictions = []
        all_properties = [] if args.save_properties else None
        
        with torch.no_grad():
            model.eval()
            for i in range(250):
                z_rand = torch.randn((64, embedding_dimension), device=device) * args.epsilon
                predictions_rand, _, _, z, y = model.inference(data=z_rand, device=device, sample=False, log_var=None)
                print(f'Generated batch {i+1}/250')
                
                # Convert predictions to strings with complete format processing
                prediction_strings = []
                for sample in range(len(predictions_rand)):
                    pred_string = combine_tokens(tokenids_to_vocab(predictions_rand[sample][0].tolist(), vocab), tokenization=tokenization)
                    processed_string = enhanced_polymer_processing_complete(
                        pred_string,
                        enforce_homopolymer=args.enforce_homopolymer
                    )
                    if processed_string:
                        prediction_strings.append(processed_string)
                    else:
                        # Keep original if processing fails
                        prediction_strings.append(clean_output(pred_string))
                        
                all_predictions.extend(prediction_strings)
                
                # Save property predictions if requested
                if args.save_properties:
                    if torch.is_tensor(y):
                        properties_batch = y.cpu().numpy()
                        all_properties.extend(properties_batch)
   
    # Save random generation results (BOTH pickle AND text)
    with open(os.path.join(dir_name, 'generated_polymers.pkl'), 'wb') as f:
        pickle.dump(all_predictions, f)
    print(f"✅ Saved {len(all_predictions)} random generations to generated_polymers.pkl")
    
    # Save as readable text file with format explanation
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
        seed_string = clean_output(seed_string_raw)
        
        print(f"🌱 Seed molecule: {seed_string}")
        
        sampled_z = []
        for r in range(8):
            # Define the mean and standard deviation of the Gaussian noise
            mean = 0
            std = args.epsilon / 2
            
            # Create a tensor of the same size as the original tensor with random noise
            noise = torch.tensor(np.random.normal(mean, std, size=seed_z.size()), dtype=torch.float, device=device)

            # Add the noise to the original tensor
            seed_z_noise = seed_z + noise
            sampled_z.append(seed_z_noise.cpu().numpy())
            
            predictions_seed, _, _, z_new, y_new = model.inference(data=seed_z_noise, device=device, sample=False, log_var=None)
            
            # Convert predictions to strings with complete format processing
            prediction_strings = []
            for sample in range(len(predictions_seed)):
                pred_string = combine_tokens(tokenids_to_vocab(predictions_seed[sample][0].tolist(), vocab), tokenization=tokenization)
                processed_string = enhanced_polymer_processing_complete(
                    pred_string,
                    enforce_homopolymer=args.enforce_homopolymer
                )
                if processed_string:
                    prediction_strings.append(processed_string)
                else:
                    prediction_strings.append(clean_output(pred_string))
            
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
        
    # Save seed-based as text too
    save_polymers_as_text(
        all_predictions_seed,
        os.path.join(dir_name, f'seed_based_polymers.txt'),
        "Seed-Based Generated Polymers"
    )
        
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
            while ind1 == ind2:
                ind2 = random.choice(list(range(64)))
                
            start_mol_raw = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind1], vocab), tokenization=tokenization)
            end_mol_raw = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind2], vocab), tokenization=tokenization)
            
            start_mol = clean_output(start_mol_raw)
            end_mol = clean_output(end_mol_raw)
            
            seed_z1 = z[ind1]
            seed_z2 = z[ind2]
            
            print(f"🔄 Interpolation {e+1}/10: {start_mol[:50]}... ↔ {end_mol[:50]}...")

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
                processed_string = enhanced_polymer_processing_complete(raw_string, enforce_homopolymer=args.enforce_homopolymer)
                
                if processed_string:
                    all_predictions_interp.append(processed_string)
                else:
                    all_predictions_interp.append(clean_output(raw_string))
                
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
    
    # Save interpolated as text too
    save_polymers_as_text(
        all_predictions_interp_all,
        os.path.join(dir_name, 'interpolated_polymers.txt'),
        "Interpolated Polymers"
    )
    
    if args.save_properties and all_properties_interp_all:
        all_properties_interp_all = np.array(all_properties_interp_all)
        with open(os.path.join(dir_name, 'interpolated_polymers_properties.npy'), 'wb') as f:
            np.save(f, all_properties_interp_all)
        print(f"✅ Saved properties of interpolated molecules: {all_properties_interp_all.shape}")

    # ================================
    # FINAL SUMMARY WITH COMPREHENSIVE VALIDATION
    # ================================

    # Final summary with comprehensive validation
    print('\n' + '='*60)
    print('🎉 GENERATION COMPLETED SUCCESSFULLY')
    print('='*60)

    # Run comprehensive validation
    validation_results = enhanced_validation_of_results(all_predictions)

    print(f"📊 Generation Summary:")
    print(f"  Random generations: {len(all_predictions)}")
    print(f"  Seed-based generations: {len(all_predictions_seed)}")
    print(f"  Interpolated molecules: {len(all_predictions_interp_all)}")
    print(f"  Total molecules generated: {len(all_predictions) + len(all_predictions_seed) + len(all_predictions_interp_all)}")
    print(f"📁 Results saved to: {dir_name}")

    if validation_results:
        print(f"\n🔬 COMPLETE FORMAT COMPLIANCE:")
        print(f"  Basic SMILES validity: {validation_results['basic_valid']}/{validation_results['total_analyzed']} ({validation_results['basic_valid']/validation_results['total_analyzed']*100:.1f}%)")
        print(f"  Complete format compliance: {validation_results['format_compliant']}/{validation_results['total_analyzed']} ({validation_results['compliance_rate']*100:.1f}%)")
        print(f"  Contains connectivity patterns: {validation_results['has_connectivity']}/{validation_results['total_analyzed']} ({validation_results['has_connectivity']/validation_results['total_analyzed']*100:.1f}%)")

    if args.enforce_homopolymer:
        print(f"🧪 Homopolymer format enforced on all generated structures")
    if args.save_properties:
        print(f"🔬 Property predictions saved for all generated molecules")
        print(f"📋 Properties: {property_names}")
    
    print(f"\n📄 SAVED FILES:")
    print(f"  - generated_polymers.pkl (binary)")
    print(f"  - generated_polymers.txt (readable with format explanation)")
    print(f"  - seed_based_polymers.txt (readable)")
    print(f"  - interpolated_polymers.txt (readable)")
    print(f"  - polymer_format_analysis.txt (validation report)")

else: 
    print("❌ The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
