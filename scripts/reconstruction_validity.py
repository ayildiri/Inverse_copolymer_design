import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from data_processing.Smiles_enum_canon import SmilesEnumCanon

import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pickle
from statistics import mean
import argparse
from functools import partial


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
parser.add_argument("--initialization", default="random", choices=["random"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.1)
parser.add_argument("--max_alpha", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load results from and save validation metrics to")

# Add flexible property arguments (same as other scripts)
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties used in the model")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
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

print(f"Validating reconstruction for model with {property_count} properties: {property_names}")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "test"
data_augment = "old"

# Handle dataset path flexibility
if args.dataset_path:
    vocab_file_path = os.path.join(args.dataset_path, f'poly_smiles_vocab_{augment}_{tokenization}.txt')
else:
    vocab_file_path = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'

print(f"Loading vocabulary from: {vocab_file_path}")
vocab = load_vocab(vocab_file=vocab_file_path)

# Directory to load results from
# Include property info in model name for consistency
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+str(property_str)+'/'

# Updated path handling to use save_dir if provided
if args.save_dir is not None:
    dir_name = os.path.join(args.save_dir, model_name)
else:
    dir_name = os.path.join(main_dir_path, 'Checkpoints/', model_name)

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

print(f'Validity check of validation set using inference decoding')
print(f'Loading results from: {dir_name}')

def safe_canonicalize(smile_string, sm_can, monomer_only=False):
    """Safely canonicalize a SMILES string, returning None if invalid.
    Special handling for stoichiometry values like '0.5'."""
    if not smile_string:
        return None
        
    # Skip canonicalization for stoichiometry values (numbers)
    if smile_string.replace('.', '').isdigit():
        return smile_string
        
    try:
        result = sm_can.canonicalize(smile_string, monomer_only=monomer_only)
        return result if result != 'invalid_monomer_string' else None
    except Exception as e:
        # Optionally print the error for debugging
        # print(f"Error canonicalizing {smile_string}: {e}")
        return None

def is_homopolymer_string(polymer_string):
    """Check if a polymer string represents a homopolymer (monA = monB)."""
    try:
        parts = polymer_string.split('|')
        if len(parts) >= 3:
            monA = parts[1] if len(parts) > 1 else ""
            monB = parts[2] if len(parts) > 2 else ""
            return monA == monB and monA != ""
        return False
    except:
        return False

def extract_monomer_info(polymer_string):
    """Extract monomer information from polymer string, handling the format:
    monomerA.monomerB|stoichA|stoichB|connectivity"""
    try:
        # Default empty values
        monA = monB = stoichA = stoichB = connectivity = ""
        
        # Split by pipes to get the main parts
        if '|' in polymer_string:
            parts = polymer_string.split('|')
            
            # First part contains the monomers separated by '.'
            if len(parts) > 0 and '.' in parts[0]:
                monomers = parts[0].split('.')
                if len(monomers) >= 2:
                    monA = monomers[0]
                    monB = monomers[1]
                elif len(monomers) == 1:
                    # Homopolymer case
                    monA = monB = monomers[0]
            else:
                # Handle case where there's no dot in the first part
                monA = monB = parts[0] if parts else ""
            
            # Second part is stoichA
            if len(parts) > 1:
                stoichA = parts[1]
                
            # Third part is stoichB
            if len(parts) > 2:
                stoichB = parts[2]
            
            # Fourth part is connectivity
            if len(parts) > 3:
                connectivity = parts[3]
        
        # Handle simple format without pipes
        elif '.' in polymer_string:
            monomers = polymer_string.split('.')
            if len(monomers) >= 2:
                monA = monomers[0]
                monB = monomers[1]
            else:
                monA = monB = monomers[0] if monomers else ""
        else:
            # Assume single monomer (homopolymer)
            monA = monB = polymer_string
            
        # For homopolymers with no explicit stoichiometry, use 1.0
        if monA == monB and not stoichA and not stoichB:
            stoichA = stoichB = "1.0"
            
        # Use combined stoichiometry value for comparison (stoichA + stoichB)
        stoich = stoichA + "|" + stoichB if stoichA or stoichB else ""
            
        return monA, monB, stoich, connectivity
    except Exception as e:
        print(f"Error in extract_monomer_info: {e} for string: {polymer_string[:30]}...")
        return "", "", "", ""

# Load prediction and real strings
pred_file = os.path.join(dir_name, 'all_val_prediction_strings.pkl')
real_file = os.path.join(dir_name, 'all_val_real_strings.pkl')

if not os.path.exists(pred_file) or not os.path.exists(real_file):
    print(f"Error: Required files not found!")
    print(f"Expected files:")
    print(f"  {pred_file}")
    print(f"  {real_file}")
    print("Please run inference_reconstruction.py first to generate these files.")
    exit(1)

with open(pred_file, 'rb') as f:
    all_predictions = pickle.load(f)
with open(real_file, 'rb') as f:
    all_real = pickle.load(f)

print(f"Loaded {len(all_predictions)} predictions and {len(all_real)} real samples")

# Remove all '_' from the strings (EOS token)
all_predictions = [s.split('_', 1)[0] for s in all_predictions]
all_real = [s.split('_', 1)[0] for s in all_real]

# Save cleaned strings to text files
with open(os.path.join(dir_name, 'all_val_prediction_strings.txt'), 'w') as f:
    for s in all_predictions:
        f.write(s+'\n')
with open(os.path.join(dir_name, 'all_val_real_strings.txt'), 'w') as f:
    for s in all_real:
        f.write(s+'\n')

# Canonicalize both the prediction and real string and check if they are the same
sm_can = SmilesEnumCanon()

# Use safe_canonicalize instead of direct mapping to handle errors gracefully
print("Safely canonicalizing SMILES strings...")
all_predictions_can = []
all_real_can = []
for s in all_predictions:
    can_s = safe_canonicalize(s, sm_can)
    all_predictions_can.append(can_s if can_s else "invalid_smiles")
    
for s in all_real:
    can_s = safe_canonicalize(s, sm_can)
    all_real_can.append(can_s if can_s else "invalid_smiles")

    # Add debug printing for first few samples to understand structure
    print("\nDebug: Examining polymer string formats")
    for i, (s_r, s_p) in enumerate(zip(all_real[:5], all_predictions[:5])):
        print(f"\nSample {i+1} Real: {s_r}")
        monA_r, monB_r, stoich_r, con_r = extract_monomer_info(s_r)
        print(f"  → Extracted: monA='{monA_r}', monB='{monB_r}', stoich='{stoich_r}', con='{con_r}'")
        
        print(f"Sample {i+1} Pred: {s_p}")
        monA_p, monB_p, stoich_p, con_p = extract_monomer_info(s_p)
        print(f"  → Extracted: monA='{monA_p}', monB='{monB_p}', stoich='{stoich_p}', con='{con_p}'")
        
        # Show pipe-delimited structure
        if '|' in s_r:
            print(f"  Real pipe parts: {s_r.split('|')}")
        if '|' in s_p:
            print(f"  Pred pipe parts: {s_p.split('|')}")
        
        # Show stoichiometry comparison result
        try:
            # Check stoichiometry reconstruction - now using the combined stoichA|stoichB format
            stoich_match = False
            if stoich_p == stoich_r:
                stoich_match = True
            else:
                # Handle special cases for homopolymers
                is_homo_real = monA_r == monB_r and monA_r
                is_homo_pred = monA_p == monB_p and monA_p
                
                if is_homo_real and is_homo_pred:
                    # For homopolymers, stoichiometry might be "1.0|1.0" or empty in both
                    if (not stoich_r or stoich_r == "1.0|1.0") and (not stoich_p or stoich_p == "1.0|1.0"):
                        stoich_match = True
                else:
                    # Try comparing individual stoichiometry values
                    try:
                        # Split combined stoich values
                        stoich_r_parts = stoich_r.split('|') if stoich_r else []
                        stoich_p_parts = stoich_p.split('|') if stoich_p else []
                        
                        # Need at least 2 parts to compare
                        if len(stoich_r_parts) >= 2 and len(stoich_p_parts) >= 2:
                            # Try numeric comparison
                            stoich_r_a = float(stoich_r_parts[0]) if stoich_r_parts[0] else 0
                            stoich_r_b = float(stoich_r_parts[1]) if stoich_r_parts[1] else 0
                            stoich_p_a = float(stoich_p_parts[0]) if stoich_p_parts[0] else 0
                            stoich_p_b = float(stoich_p_parts[1]) if stoich_p_parts[1] else 0
                            
                            # Check if numeric values are close enough (allow small floating point differences)
                            if abs(stoich_r_a - stoich_p_a) < 0.001 and abs(stoich_r_b - stoich_p_b) < 0.001:
                                stoich_match = True
                    except ValueError:
                        pass
            
            print(f"  Stoichiometry match: {stoich_match}")
        except Exception as e:
            print(f"  Error checking stoich: {e}")
        
    print("\nContinuing with validation...")

# Initialize validation lists
prediction_validityA = []
prediction_validityB = []
rec_A = []
rec_B = []
rec = []
rec_stoich = []
rec_con = []

# Counters for statistics
total_samples = len(all_real)
homopolymer_count_real = 0
homopolymer_count_pred = 0
copolymer_count_real = 0
copolymer_count_pred = 0
invalid_pred_count = 0
invalid_real_count = 0

for i, (s_r, s_p) in enumerate(zip(all_real, all_predictions)):
    # Check if homopolymer or copolymer
    is_homo_real = is_homopolymer_string(s_r)
    is_homo_pred = is_homopolymer_string(s_p)
    
    if is_homo_real:
        homopolymer_count_real += 1
    else:
        copolymer_count_real += 1
        
    if is_homo_pred:
        homopolymer_count_pred += 1
    else:
        copolymer_count_pred += 1
    
    # Both canonicalized strings are the same - use safe canonicalization
    # Only attempt to canonicalize the full strings if they don't match exactly already
    if s_r == s_p:
        rec.append(True)
        prediction_validityA.append(True)
        prediction_validityB.append(True)
        rec_A.append(True)
        rec_B.append(True)
        rec_stoich.append(True)
        rec_con.append(True)
        continue
        
    s_r_can = safe_canonicalize(s_r, sm_can)
    s_p_can = safe_canonicalize(s_p, sm_can)
    
    # Track invalid SMILES counts
    if not s_p_can:
        invalid_pred_count += 1
    if not s_r_can:
        invalid_real_count += 1
    
    if s_r_can and s_p_can and s_r_can == s_p_can:
        rec.append(True)
        prediction_validityA.append(True)
        prediction_validityB.append(True)
        rec_A.append(True)
        rec_B.append(True)
        rec_stoich.append(True)
        rec_con.append(True)
    else:
        rec.append(False)
        
        # Extract monomer information for detailed analysis
        try:
            monA_r, monB_r, stoich_r, con_r = extract_monomer_info(s_r)
            monA_p, monB_p, stoich_p, con_p = extract_monomer_info(s_p)
            
            # Safely canonicalize monomers
            monA_r_can = safe_canonicalize(monA_r, sm_can, monomer_only=True)
            monB_r_can = safe_canonicalize(monB_r, sm_can, monomer_only=True)
            monA_p_can = safe_canonicalize(monA_p, sm_can, monomer_only=True)
            monB_p_can = safe_canonicalize(monB_p, sm_can, monomer_only=True)
            
            # Check monomer A validity and reconstruction
            if monA_p_can:
                prediction_validityA.append(True)
                if monA_r_can and monA_p_can == monA_r_can:
                    rec_A.append(True)
                else:
                    rec_A.append(False)
            else:
                prediction_validityA.append(False)
                rec_A.append(False)
            
            # Check monomer B validity and reconstruction
            # For homopolymers, monB should equal monA
            if is_homo_real and is_homo_pred:
                # Both are homopolymers
                if monA_p_can:
                    prediction_validityB.append(True)
                    if monA_r_can and monA_p_can == monA_r_can:  # For homopolymers, monB = monA
                        rec_B.append(True)
                    else:
                        rec_B.append(False)
                else:
                    prediction_validityB.append(False)
                    rec_B.append(False)
            else:
                # Handle copolymers or mixed cases
                if monB_p_can:
                    prediction_validityB.append(True)
                    if monB_r_can and monB_p_can == monB_r_can:
                        rec_B.append(True)
                    else:
                        rec_B.append(False)
                else:
                    prediction_validityB.append(False)
                    rec_B.append(False)
            
            # Check stoichiometry reconstruction - now using the combined stoichA|stoichB format
            if stoich_p == stoich_r:
                rec_stoich.append(True)
            else:
                # Handle special cases for homopolymers
                if is_homo_real and is_homo_pred:
                    # For homopolymers, stoichiometry might be "1.0|1.0" or empty in both
                    if (not stoich_r or stoich_r == "1.0|1.0") and (not stoich_p or stoich_p == "1.0|1.0"):
                        rec_stoich.append(True)
                    else:
                        rec_stoich.append(False)
                else:
                    # Try comparing individual stoichiometry values
                    try:
                        # Split combined stoich values
                        stoich_r_parts = stoich_r.split('|') if stoich_r else []
                        stoich_p_parts = stoich_p.split('|') if stoich_p else []
                        
                        # Need at least 2 parts to compare
                        if len(stoich_r_parts) >= 2 and len(stoich_p_parts) >= 2:
                            # Try numeric comparison
                            stoich_r_a = float(stoich_r_parts[0]) if stoich_r_parts[0] else 0
                            stoich_r_b = float(stoich_r_parts[1]) if stoich_r_parts[1] else 0
                            stoich_p_a = float(stoich_p_parts[0]) if stoich_p_parts[0] else 0
                            stoich_p_b = float(stoich_p_parts[1]) if stoich_p_parts[1] else 0
                            
                            # Check if numeric values are close enough (allow small floating point differences)
                            if abs(stoich_r_a - stoich_p_a) < 0.001 and abs(stoich_r_b - stoich_p_b) < 0.001:
                                rec_stoich.append(True)
                            else:
                                rec_stoich.append(False)
                        else:
                            rec_stoich.append(False)
                    except ValueError:
                        rec_stoich.append(False)
            
            # Check connectivity reconstruction - direct string comparison, no canonicalization
            if con_p == con_r:
                rec_con.append(True)
            else:
                rec_con.append(False)
                
        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            # Default to False for all metrics if parsing fails
            prediction_validityA.append(False)
            prediction_validityB.append(False)
            rec_A.append(False)
            rec_B.append(False)
            rec_stoich.append(False)
            rec_con.append(False)

    # Print stoichiometry distribution info for insights
    stoich_values_real = {}
    stoich_values_pred = {}
    
    print("\nStoichiometry distribution analysis:")
    for i, (s_r, s_p) in enumerate(zip(all_real, all_predictions)):
        _, _, stoich_r, _ = extract_monomer_info(s_r)
        _, _, stoich_p, _ = extract_monomer_info(s_p)
        
        # Count occurrences
        stoich_values_real[stoich_r] = stoich_values_real.get(stoich_r, 0) + 1
        stoich_values_pred[stoich_p] = stoich_values_pred.get(stoich_p, 0) + 1
    
    # Show most common stoichiometry values
    print("Real data stoichiometry values:")
    for stoich, count in sorted(stoich_values_real.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  '{stoich}': {count} occurrences ({100*count/total_samples:.1f}%)")
        
    print("Prediction stoichiometry values:")
    for stoich, count in sorted(stoich_values_pred.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  '{stoich}': {count} occurrences ({100*count/total_samples:.1f}%)")
        
# Calculate reconstruction accuracies
if len(rec) > 0:
    rec_accuracy = sum(1 for entry in rec if entry) / len(rec)
    rec_accuracyA = sum(1 for entry in rec_A if entry) / len(rec_A)
    rec_accuracyB = sum(1 for entry in rec_B if entry) / len(rec_B)
    rec_accuracy_stoich = sum(1 for entry in rec_stoich if entry) / len(rec_stoich)
    rec_accuracy_con = sum(1 for entry in rec_con if entry) / len(rec_con)
else:
    rec_accuracy = 0
    rec_accuracyA = 0
    rec_accuracyB = 0
    rec_accuracy_stoich = 0
    rec_accuracy_con = 0

validityA = sum(1 for entry in prediction_validityA if entry) / len(prediction_validityA)
validityB = sum(1 for entry in prediction_validityB if entry) / len(prediction_validityB)
valid_pred_percentage = 100 * (1 - invalid_pred_count / total_samples)
valid_real_percentage = 100 * (1 - invalid_real_count / total_samples)

# Print detailed results
print(f'\n=== RECONSTRUCTION VALIDATION RESULTS ===')
print(f'Total samples: {total_samples}')
print(f'Valid SMILES in predictions: {total_samples - invalid_pred_count} ({valid_pred_percentage:.1f}%)')
print(f'Valid SMILES in real data: {total_samples - invalid_real_count} ({valid_real_percentage:.1f}%)')
print(f'Homopolymers in real data: {homopolymer_count_real} ({100*homopolymer_count_real/total_samples:.1f}%)')
print(f'Homopolymers in predictions: {homopolymer_count_pred} ({100*homopolymer_count_pred/total_samples:.1f}%)')
print(f'Copolymers in real data: {copolymer_count_real} ({100*copolymer_count_real/total_samples:.1f}%)')
print(f'Copolymers in predictions: {copolymer_count_pred} ({100*copolymer_count_pred/total_samples:.1f}%)')
print(f'')
print(f'Full reconstruction accuracy: {100*rec_accuracy:.2f}%')
print(f'Monomer A reconstruction: {100*rec_accuracyA:.2f}%')
print(f'Monomer B reconstruction: {100*rec_accuracyB:.2f}%')
print(f'Stoichiometry reconstruction: {100*rec_accuracy_stoich:.2f}%')
print(f'Connectivity reconstruction: {100*rec_accuracy_con:.2f}%')
print(f'')
print(f'Monomer A validity: {100*validityA:.2f}%')
print(f'Monomer B validity: {100*validityB:.2f}%')
print(f'')
print(f'Model properties: {property_names}')
print(f'==========================================')

# Save detailed results
metrics_file = os.path.join(dir_name, 'reconstruction_metrics.txt')
print(f'Saving detailed metrics to: {metrics_file}')

with open(metrics_file, 'w') as f:
    f.write("=== RECONSTRUCTION VALIDATION RESULTS ===\n")
    f.write(f"Total samples: {total_samples}\n")
    f.write(f"Valid SMILES in predictions: {total_samples - invalid_pred_count} ({valid_pred_percentage:.1f}%)\n")
    f.write(f"Valid SMILES in real data: {total_samples - invalid_real_count} ({valid_real_percentage:.1f}%)\n")
    f.write(f"Homopolymers in real data: {homopolymer_count_real} ({100*homopolymer_count_real/total_samples:.1f}%)\n")
    f.write(f"Homopolymers in predictions: {homopolymer_count_pred} ({100*homopolymer_count_pred/total_samples:.1f}%)\n")
    f.write(f"Copolymers in real data: {copolymer_count_real} ({100*copolymer_count_real/total_samples:.1f}%)\n")
    f.write(f"Copolymers in predictions: {copolymer_count_pred} ({100*copolymer_count_pred/total_samples:.1f}%)\n")
    f.write(f"\n")
    f.write(f"Full reconstruction accuracy: {100*rec_accuracy:.4f}%\n")
    f.write(f"Monomer A reconstruction: {100*rec_accuracyA:.4f}%\n")
    f.write(f"Monomer B reconstruction: {100*rec_accuracyB:.4f}%\n")
    f.write(f"Stoichiometry reconstruction: {100*rec_accuracy_stoich:.4f}%\n")
    f.write(f"Connectivity reconstruction: {100*rec_accuracy_con:.4f}%\n")
    f.write(f"\n")
    f.write(f"Monomer A validity: {100*validityA:.4f}%\n")
    f.write(f"Monomer B validity: {100*validityB:.4f}%\n")
    f.write(f"\n")
    f.write(f"Model properties: {', '.join(property_names)}\n")
    f.write(f"Property count: {property_count}\n")
    f.write(f"Arguments used: {vars(args)}\n")

# Save detailed breakdown for further analysis
detailed_results = {
    'total_samples': total_samples,
    'invalid_pred_count': invalid_pred_count,
    'invalid_real_count': invalid_real_count,
    'homopolymer_count_real': homopolymer_count_real,
    'homopolymer_count_pred': homopolymer_count_pred,
    'copolymer_count_real': copolymer_count_real,
    'copolymer_count_pred': copolymer_count_pred,
    'rec_accuracy': rec_accuracy,
    'rec_accuracyA': rec_accuracyA,
    'rec_accuracyB': rec_accuracyB,
    'rec_accuracy_stoich': rec_accuracy_stoich,
    'rec_accuracy_con': rec_accuracy_con,
    'validityA': validityA,
    'validityB': validityB,
    'property_names': property_names,
    'property_count': property_count,
    'full_reconstruction_list': rec,
    'monA_reconstruction_list': rec_A,
    'monB_reconstruction_list': rec_B,
    'stoich_reconstruction_list': rec_stoich,
    'connectivity_reconstruction_list': rec_con,
    'monA_validity_list': prediction_validityA,
    'monB_validity_list': prediction_validityB
}

with open(os.path.join(dir_name, 'detailed_reconstruction_results.pkl'), 'wb') as f:
    pickle.dump(detailed_results, f)

print(f"Detailed results saved to: {os.path.join(dir_name, 'detailed_reconstruction_results.pkl')}")
print("Reconstruction validation completed successfully!")
