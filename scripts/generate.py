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
# ORIGINAL UTILITY FUNCTIONS (PRESERVED)
# ================================

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

# ================================
# ENHANCED SMILES FIXING FUNCTIONS
# ================================

def clean_obvious_smiles_junk(smiles_string):
    """Remove obvious junk that corrupts SMILES"""
    if not smiles_string:
        return smiles_string
    
    # Remove trailing junk patterns that got mixed in
    patterns_to_remove = [
        r'\)\|\d+$',       # )|0, )|1 etc at end
        r'\|[\d:]*$',      # |0, |:0, |1:2 etc at end
        r'\d+\)$',         # trailing numbers before )
    ]
    
    for pattern in patterns_to_remove:
        smiles_string = re.sub(pattern, '', smiles_string)
    
    return smiles_string

def fix_complex_ring_notation(smiles_string):
    """Fix complex ring notation issues"""
    if not smiles_string:
        return smiles_string
    
    # Problem: "O=C(C[*:3])3)[*:4])" 
    # The "3)" after [*:3] is invalid ring notation
    
    # Fix pattern: [*:n])digit) -> [*:n])
    smiles_string = re.sub(r'(\[\*:\d+\])\)\d+\)', r'\1)', smiles_string)
    
    # Fix pattern: )digit) -> )
    smiles_string = re.sub(r'\)\d+\)', ')', smiles_string)
    
    # Fix orphaned digits that look like ring numbers but aren't
    # Pattern: atom-digit-) where digit is not a valid ring closer
    smiles_string = re.sub(r'([CNOSPFBrClI])\d+\)', r'\1)', smiles_string)
    
    # Remove standalone digits that aren't ring numbers
    # Keep only single digits that are properly part of ring notation
    smiles_string = re.sub(r'(?<![CNOSPFBrClI])\d+(?!\])', '', smiles_string)
    
    return smiles_string

def fix_parentheses_and_brackets_aggressive(smiles_string):
    """Aggressive parentheses fixing for heavily corrupted SMILES"""
    if not smiles_string:
        return smiles_string
    
    # Count parentheses
    open_paren = smiles_string.count('(')
    close_paren = smiles_string.count(')')
    
    if open_paren > close_paren:
        # Add missing closing parentheses
        smiles_string += ')' * (open_paren - close_paren)
    elif close_paren > open_paren:
        # Remove excess closing parentheses - be more aggressive
        excess = close_paren - open_paren
        
        # Strategy: Remove from the end first, then from problematic areas
        # Remove from end
        while excess > 0 and smiles_string.endswith(')'):
            smiles_string = smiles_string[:-1]
            excess -= 1
        
        # If still excess, remove from obvious problem areas
        if excess > 0:
            # Remove )()) -> ()
            smiles_string = re.sub(r'\)\(\)', '()', smiles_string)
            
            # Recount
            new_close = smiles_string.count(')')
            new_open = smiles_string.count('(')
            if new_close > new_open:
                # Still excess, remove more aggressively
                temp = smiles_string[::-1]  # Reverse
                for _ in range(min(excess, new_close - new_open)):
                    temp = temp.replace(')', '', 1)
                smiles_string = temp[::-1]  # Reverse back
    
    # Same for brackets
    open_bracket = smiles_string.count('[')
    close_bracket = smiles_string.count(']')
    
    if open_bracket > close_bracket:
        smiles_string += ']' * (open_bracket - close_bracket)
    elif close_bracket > open_bracket:
        excess = close_bracket - open_bracket
        temp = smiles_string[::-1]
        for _ in range(excess):
            temp = temp.replace(']', '', 1)
        smiles_string = temp[::-1]
    
    return smiles_string

def final_smiles_cleanup(smiles_string):
    """Final cleanup of SMILES syntax"""
    if not smiles_string:
        return smiles_string
    
    # Remove double punctuation
    smiles_string = re.sub(r'\)\)', ')', smiles_string)  # )) -> )
    smiles_string = re.sub(r'\(\(', '(', smiles_string)   # (( -> (
    smiles_string = re.sub(r'\]\]', ']', smiles_string)  # ]] -> ]
    smiles_string = re.sub(r'\[\[', '[', smiles_string)   # [[ -> [
    
    # Remove empty parentheses
    smiles_string = re.sub(r'\(\)', '', smiles_string)
    
    # Fix common atom notation issues
    smiles_string = re.sub(r'\bcl\b', 'Cl', smiles_string, flags=re.IGNORECASE)
    smiles_string = re.sub(r'\bbr\b', 'Br', smiles_string, flags=re.IGNORECASE)
    
    return smiles_string.strip()

def validate_fixed_smiles(smiles_string):
    """Test if the fixed SMILES is actually valid"""
    if not smiles_string:
        return False
    
    try:
        from rdkit import Chem
        
        # Replace attachment points for testing
        test_smiles = re.sub(r'\[\*:\d+\]', '*', smiles_string)
        
        # Try to parse
        mol = Chem.MolFromSmiles(test_smiles)
        if mol is not None and mol.GetNumAtoms() > 0:
            return True
            
        # If failed, try without attachment points entirely
        test_smiles_clean = re.sub(r'\[\*:\d+\]', '', smiles_string)
        if test_smiles_clean:
            mol = Chem.MolFromSmiles(test_smiles_clean)
            return mol is not None and mol.GetNumAtoms() > 0
        
        return False
        
    except Exception:
        return False

def aggressive_smiles_salvage(original_smiles):
    """Last resort: try to salvage something useful from heavily corrupted SMILES"""
    if not original_smiles:
        return None
    
    # Strategy: Look for recognizable chemical patterns and rebuild
    
    # Find atoms and basic connectivity
    atoms = re.findall(r'[CNOSPFBrClI]', original_smiles, re.IGNORECASE)
    if len(atoms) < 2:
        return None
    
    # Find attachment points
    attachment_points = re.findall(r'\[\*:\d+\]', original_smiles)
    
    # Try to build a minimal valid structure
    if 'C' in atoms and 'O' in atoms:
        # Common pattern: carbonyl
        if len(attachment_points) >= 2:
            return f"O=C({attachment_points[0]}){attachment_points[1] if len(attachment_points) > 1 else '[*:2]'}"
        else:
            return "O=C([*:1])[*:2]"
    elif 'C' in atoms:
        # Just carbon chain
        if len(attachment_points) >= 2:
            return f"C({attachment_points[0]}){attachment_points[1] if len(attachment_points) > 1 else '[*:2]'}"
        else:
            return "C([*:1])[*:2]"
    
    return None

def fix_smiles_component(smiles_part):
    """Enhanced SMILES component fixing for complex corruption cases"""
    if not smiles_part or smiles_part.strip() == '':
        return None
    
    smiles_part = smiles_part.strip()
    
    # If it's just numbers, try to see if it's a corrupted valid SMILES
    if re.match(r'^\d+$', smiles_part):
        # Pure numbers like "500" - likely too corrupted to fix
        return None
    
    # Check if it contains actual chemical elements
    has_atoms = bool(re.search(r'[CNOSPFBrClI]', smiles_part, re.IGNORECASE))
    if not has_atoms:
        return None
    
    # Multi-stage fixing approach
    fixed = smiles_part
    
    # Stage 1: Clean obvious junk first
    fixed = clean_obvious_smiles_junk(fixed)
    
    # Stage 2: Fix ring notation issues BEFORE parentheses (important!)
    fixed = fix_complex_ring_notation(fixed)
    
    # Stage 3: Fix parentheses and brackets
    fixed = fix_parentheses_and_brackets_aggressive(fixed)
    
    # Stage 4: Final cleanup
    fixed = final_smiles_cleanup(fixed)
    
    # Stage 5: Validate the fix worked
    if validate_fixed_smiles(fixed):
        return fixed
    
    # Stage 6: If still broken, try aggressive salvage
    salvaged = aggressive_smiles_salvage(smiles_part)
    if salvaged and validate_fixed_smiles(salvaged):
        return salvaged
    
    return None

# ================================
# POLYMER FORMAT HANDLING FUNCTIONS
# ================================

def is_stoichiometry_like(part):
    """Check if part looks like stoichiometry values"""
    # Should be numbers, decimals, maybe some basic punctuation
    return bool(re.match(r'^[0-9.:]+$', part))

def is_connectivity_like(part):
    """Check if part looks like connectivity patterns"""
    # Should contain < symbols and pattern like <1-2:0.5:0.5
    return '<' in part and ':' in part

def fix_stoichiometry_values(stoich_string):
    """Fix stoichiometry values like 0.25, 0.75"""
    # Extract numbers that look like stoichiometry
    numbers = re.findall(r'\d*\.?\d+', stoich_string)
    
    if numbers:
        # Convert to floats and normalize if needed
        try:
            float_values = [float(x) for x in numbers if float(x) > 0]
            if float_values:
                # Normalize so they sum to 1.0
                total = sum(float_values)
                if total > 0:
                    normalized = [x/total for x in float_values]
                    return "|".join(f"{x:.3f}" for x in normalized)
        except ValueError:
            pass
    
    # Fallback
    return "1.0"

def fix_connectivity_values(conn_string):
    """Fix connectivity patterns like <1-2:0.5:0.5"""
    # Try to extract connectivity patterns
    patterns = re.findall(r'<?\d+-\d+:[\d.]+:[\d.]+', conn_string)
    
    if patterns:
        # Clean up the patterns
        fixed_patterns = []
        for pattern in patterns:
            if not pattern.startswith('<'):
                pattern = '<' + pattern
            fixed_patterns.append(pattern)
        
        return "".join(fixed_patterns)
    
    # Fallback: create simple connectivity for attachment points
    return "<1-2:0.5:0.5"

def fix_stoichiometry_section(stoich_parts):
    """Fix the stoichiometry and connectivity sections"""
    if not stoich_parts:
        return "1.0|"
    
    fixed_parts = []
    
    for part in stoich_parts:
        part = part.strip()
        if not part:
            continue
            
        # Check if this looks like stoichiometry (numbers, decimals, colons)
        if is_stoichiometry_like(part):
            fixed_stoich = fix_stoichiometry_values(part)
            if fixed_stoich:
                fixed_parts.append(fixed_stoich)
        
        # Check if this looks like connectivity (< symbols, colons, numbers)
        elif is_connectivity_like(part):
            fixed_conn = fix_connectivity_values(part)
            if fixed_conn:
                fixed_parts.append(fixed_conn)
    
    # If we couldn't fix anything, provide defaults
    if not fixed_parts:
        return "1.0|"
    
    return "|".join(fixed_parts) + "|"

def fix_polymer_representation_correct(polymer_string):
    """Fix polymer representation understanding the correct format"""
    if not polymer_string or polymer_string.strip() == '':
        return None
    
    # Clean basic padding
    clean_string = polymer_string.rstrip('_').strip()
    
    # Handle no | case
    if '|' not in clean_string:
        fixed_smiles = fix_smiles_component(clean_string)
        if fixed_smiles:
            return f"{fixed_smiles}|1.0|"
        return None
    
    # Split by | to get components
    parts = clean_string.split('|')
    
    if len(parts) < 2:
        return None
    
    # Fix SMILES component with enhanced method
    fixed_smiles = fix_smiles_component(parts[0])
    if not fixed_smiles:
        return None
    
    # Fix stoichiometry section
    fixed_stoich = fix_stoichiometry_section(parts[1:])
    
    # Combine
    if fixed_stoich:
        return f"{fixed_smiles}|{fixed_stoich}"
    else:
        return f"{fixed_smiles}|1.0|"

# ================================
# VALIDATION FUNCTIONS
# ================================

def validate_smiles_component(smiles_string):
    """Validate just the SMILES part"""
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
                    # Try parsing with attachment points replaced
                    test_monomer = monomer
                    test_monomer = re.sub(r'\[\*:\d+\]', '*', test_monomer)
                    
                    mol = Chem.MolFromSmiles(test_monomer)
                    if mol is not None:
                        valid_count += 1
            
            return valid_count > 0
        else:
            # Single monomer
            test_smiles = smiles_string
            test_smiles = re.sub(r'\[\*:\d+\]', '*', test_smiles)
            
            mol = Chem.MolFromSmiles(test_smiles)
            return mol is not None
    
    except Exception:
        return False

def validate_polymer_format_correct(polymer_string):
    """Validate polymer using correct format understanding"""
    if not polymer_string:
        return False
    
    try:
        # Split into components
        if '|' not in polymer_string:
            # Just SMILES - validate that
            return validate_smiles_component(polymer_string)
        
        parts = polymer_string.split('|')
        if len(parts) < 2:
            return False
        
        # Validate SMILES component
        smiles_part = parts[0]
        if not validate_smiles_component(smiles_part):
            return False
        
        # Validate stoichiometry (numbers between 0 and 1)
        for i in range(1, len(parts)):
            part = parts[i].strip()
            if part and not part.startswith('<'):
                try:
                    val = float(part)
                    if val < 0 or val > 1:
                        return False
                except ValueError:
                    # Not a number, might be connectivity - skip
                    pass
        
        return True
        
    except Exception:
        return False

def enhanced_polymer_processing_correct(pred_string, enforce_homopolymer=False):
    """Enhanced processing with improved SMILES fixing"""
    
    # Try multiple fixing strategies in order of preference
    candidates = []
    
    # Strategy 1: Original string
    candidates.append(pred_string)
    
    # Strategy 2: Main fix with enhanced SMILES handling
    main_fix = fix_polymer_representation_correct(pred_string)
    if main_fix:
        candidates.append(main_fix)
    
    # Strategy 3: Extract and fix just the SMILES part if polymer format is broken
    if '|' in pred_string:
        smiles_only = pred_string.split('|')[0]
        smiles_fixed = fix_smiles_component(smiles_only)
        if smiles_fixed:
            candidates.append(f"{smiles_fixed}|1.0|")
    
    # Strategy 4: Try basic cleaning
    basic_clean = clean_output(pred_string)
    if basic_clean != pred_string:
        candidates.append(basic_clean)
    
    # Remove None and duplicate candidates
    candidates = list(set([c for c in candidates if c is not None]))
    
    # Test each candidate
    for candidate in candidates:
        if validate_polymer_format_correct(candidate):
            if enforce_homopolymer:
                return convert_to_homopolymer_format(candidate)
            return candidate
    
    return None

# ================================
# ENHANCED GENERATION FUNCTION
# ================================

def generate_with_enhanced_decoding(model, vocab, tokenization, target_count=100, max_attempts=500,
                                   batch_size=16, embedding_dimension=32, device=device, 
                                   enforce_homopolymer=False, save_properties=False):
    """Generation with enhanced decoding strategies"""
    
    all_valid_predictions = []
    all_properties = [] if save_properties else None
    attempts = 0
    
    print(f"🎯 Enhanced decoding generation: targeting {target_count} valid molecules")
    
    with torch.no_grad():
        model.eval()
        
        while len(all_valid_predictions) < target_count and attempts < max_attempts:
            
            # Use very conservative sampling
            z_rand = torch.randn((batch_size, embedding_dimension), device=device) * 0.1
            
            try:
                # Standard inference
                predictions_rand, _, _, z, y = model.inference(
                    data=z_rand, device=device, sample=False, log_var=None
                )
                
                # Enhanced processing for each prediction
                batch_valid = []
                batch_properties = []
                
                for sample in range(len(predictions_rand)):
                    pred_tokens = predictions_rand[sample][0].tolist()
                    pred_string = combine_tokens(
                        tokenids_to_vocab(pred_tokens, vocab), 
                        tokenization=tokenization
                    )
                    
                    # Enhanced processing with correct format understanding
                    fixed_string = enhanced_polymer_processing_correct(
                        pred_string, 
                        enforce_homopolymer=enforce_homopolymer
                    )
                    
                    if fixed_string and validate_polymer_format_correct(fixed_string):
                        batch_valid.append(fixed_string)
                        
                        if save_properties and torch.is_tensor(y):
                            batch_properties.append(y[sample].cpu().numpy())
                
                all_valid_predictions.extend(batch_valid)
                if save_properties and batch_properties:
                    all_properties.extend(batch_properties)
                
                attempts += batch_size
                
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
                    print("⚠️  Very low validity rate. Decoder may need retraining.")
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
    # TESTING ENHANCED PROCESSING
    # ================================
    
    print("🧪 Testing Enhanced Processing on Sample Data:")
    print("-" * 50)
    
    # Test cases based on your error log
    test_cases = [
        "500|0",  # Should be rejected (pure numbers)
        "O=C(C[*:3])3)[*:4])|0",  # Should fix complex SMILES corruption
        "O=C([*:1])c1ccc([*:2])cc1|0.5|0.5|",  # Should validate correctly
        "O=C(O[*:4]",  # Should fix missing parentheses
    ]
    
    for i, test in enumerate(test_cases):
        fixed = enhanced_polymer_processing_correct(test)
        valid = validate_polymer_format_correct(fixed) if fixed else False
        print(f"Test {i+1}:")
        print(f"  Original: {test}")
        print(f"  Fixed:    {fixed}")
        print(f"  Valid:    {valid}")
        print()

    ### RANDOM GENERATION ###
    if args.quality_control:
        print(f'🎲 Generate random samples with enhanced decoding')
        torch.manual_seed(args.seed)
        
        # Use enhanced decoding generation
        all_predictions, all_properties = generate_with_enhanced_decoding(
            model=model,
            vocab=vocab,
            tokenization=tokenization,
            target_count=args.target_molecules,
            max_attempts=args.max_attempts,
            batch_size=16,  # Smaller batch size for better control
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
            
            # Convert predictions to strings with enhanced processing
            prediction_strings = []
            for sample in range(len(predictions_seed)):
                pred_string = combine_tokens(tokenids_to_vocab(predictions_seed[sample][0].tolist(), vocab), tokenization=tokenization)
                processed_string = enhanced_polymer_processing_correct(
                    pred_string,
                    enforce_homopolymer=args.enforce_homopolymer
                )
                if processed_string:
                    prediction_strings.append(processed_string)
                else:
                    # Fallback to basic processing if enhanced fails
                    prediction_strings.append(process_generated_string(pred_string, enforce_homopolymer=args.enforce_homopolymer))
            
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
                processed_string = enhanced_polymer_processing_correct(raw_string, enforce_homopolymer=args.enforce_homopolymer)
                
                if processed_string:
                    all_predictions_interp.append(processed_string)
                else:
                    # Fallback to basic processing
                    all_predictions_interp.append(process_generated_string(raw_string, enforce_homopolymer=args.enforce_homopolymer))
                
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
    if args.quality_control:
        final_validity = len(all_predictions) / max(args.max_attempts, len(all_predictions)) if len(all_predictions) > 0 else 0
        print(f"🎯 Quality control used - generated {len(all_predictions)} valid molecules with enhanced processing")

else: 
    print("❌ The model training diverged and there is no trained model file!")
    print(f"Expected model file: {filepath}")
