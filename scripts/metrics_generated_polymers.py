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
import pandas as pd
import argparse


def clean_output(polymer_string):
    """Remove all padding underscores from generated polymer strings"""
    return polymer_string.rstrip('_')

def poly_smiles_to_molecule(poly_input):
    '''
    Turns adjusted polymer smiles string into PyG data objects
    '''
    try:
        # Clean up the polymer string first to remove padding
        poly_input = clean_output(poly_input)
        
        # Turn into RDKIT mol object
        mols = make_monomer_mols(poly_input.split("|")[0], 0, 0,  # smiles
                                fragment_weights=poly_input.split("|")[1:-1])
        return mols
    except Exception as e:
        print(f"Error processing polymer: {poly_input}")
        print(f"Error details: {e}")
        return [None, None]  # Return None for both monomers to indicate invalid

def extract_monomers_safely(polymer_string):
    """Safely extract monomer information from a polymer string."""
    try:
        parts = polymer_string.split('|')
        monomers = []
        
        if len(parts) > 0:
            if '.' in parts[0]:
                monomers = parts[0].split('.')
            else:
                # Single monomer case
                monomers = [parts[0], parts[0]]
                
        # Ensure we have at least two monomers
        if len(monomers) < 2:
            monomers = [monomers[0], monomers[0]] if monomers else ["", ""]
            
        # Get weights if available
        weights = parts[1:-1] if len(parts) > 2 else []
        
        # Get connectivity if available
        connectivity = parts[-1].split("_")[0] if len(parts) > 1 else ""
        
        return monomers, weights, connectivity
    except Exception as e:
        # Return empty values on error
        return ["", ""], [], ""

def valid_scores(smiles):
    return np.array(list(map(make_polymer_mol, smiles)), dtype=np.float32)


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
parser.add_argument("--alpha", default="fixed", choices=["fixed","schedule"])  # Add after beta parameter
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to load model checkpoints from and save results to")
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

# Add flexible property arguments for consistency with other scripts
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties (used for model identification)")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
parser.add_argument("--sample_analysis", type=int, default=5,
                    help="Number of sample polymers to analyze in detail")

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

print(f"Evaluating generated polymer metrics for model trained on {property_count} properties: {property_names}")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
dataset_type = "test"
data_augment ="old"
vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
vocab = load_vocab(vocab_file=vocab_file)

# Include property info in model name for consistency with other scripts
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+property_str+'/'

# Directory to save results
dir_name = os.path.join(args.save_dir, model_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

print(f'Validity check and metrics for newly generated samples')
print(f'Model properties: {property_names}')
print(f'Results directory: {dir_name}')

# Load the generated polymers
with open(os.path.join(dir_name, 'generated_polymers.pkl'), 'rb') as f:
    all_predictions = pickle.load(f)

print(f"Loaded {len(all_predictions)} generated polymers")

# Clean up predictions by removing trailing underscores
cleaned_predictions = [clean_output(p) for p in all_predictions]
print(f"Cleaned trailing underscores from predictions")

# Save the cleaned predictions
with open(os.path.join(dir_name, 'generated_polymers_cleaned.pkl'), 'wb') as f:
    pickle.dump(cleaned_predictions, f)
print(f"Saved cleaned predictions to 'generated_polymers_cleaned.pkl'")

# Analyze first few samples in detail
print("\nDetailed analysis of sample polymers:")
print("-" * 80)
for i in range(min(args.sample_analysis, len(cleaned_predictions))):
    print(f"Sample {i+1}:")
    print(f"  Original: {all_predictions[i]}")
    print(f"  Cleaned:  {cleaned_predictions[i]}")
    
    # Try to parse with RDKit
    try:
        if '|' in cleaned_predictions[i]:
            monomer_part = cleaned_predictions[i].split('|')[0]
            print(f"  Monomer part: {monomer_part}")
            
            if '.' in monomer_part:
                monomers = monomer_part.split('.')
                print(f"  Monomer A: {monomers[0]}")
                
                # Check monomer A with RDKit
                mol_a = Chem.MolFromSmiles(monomers[0])
                print(f"  Monomer A valid: {mol_a is not None}")
                
                if len(monomers) > 1:
                    print(f"  Monomer B: {monomers[1]}")
                    # Check monomer B with RDKit
                    mol_b = Chem.MolFromSmiles(monomers[1])
                    print(f"  Monomer B valid: {mol_b is not None}")
            else:
                print(f"  Single monomer: {monomer_part}")
                mol = Chem.MolFromSmiles(monomer_part)
                print(f"  Monomer valid: {mol is not None}")
        else:
            print(f"  No pipe delimiter found, treating as single molecule")
            mol = Chem.MolFromSmiles(cleaned_predictions[i])
            print(f"  Molecule valid: {mol is not None}")
    except Exception as e:
        print(f"  Error analyzing structure: {e}")
    
    print("-" * 80)

# Load appropriate dataset based on augmentation type
if augment == "original":
    df = pd.read_csv(main_dir_path+'/data/dataset-poly_chemprop.csv')
elif augment == "augmented":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop.csv')
else:
    raise ValueError(f"Unknown augment type: {augment}")

all_polymers_data= []
all_train_polymers = []

# Extract training polymers
for batch, graphs in enumerate(dict_train_loader):
    data = dict_train_loader[str(batch)][0]
    train_polymers_batch = [clean_output(combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization)) for sample in range(len(data))]
    all_train_polymers.extend(train_polymers_batch)

# Extract all polymers from dataset
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    all_polymers_data.append(poly_input)

print(f"Loaded {len(all_train_polymers)} training polymers and {len(all_polymers_data)} total dataset polymers")

# Canonicalize all polymer strings
print("Canonicalizing polymer strings...")
sm_can = SmilesEnumCanon()

# Use safe canonicalization
def safe_canonicalize(smile_string):
    """Safely canonicalize a SMILES string, returning 'invalid_polymer_string' if invalid."""
    try:
        result = sm_can.canonicalize(smile_string)
        return result 
    except Exception as e:
        # Optionally print the error for debugging
        # print(f"Error canonicalizing {smile_string}: {e}")
        return 'invalid_polymer_string'

# Canonicalize with progress tracking
all_predictions_can = []
invalid_count = 0
for i, poly in enumerate(cleaned_predictions):
    can_poly = safe_canonicalize(poly)
    all_predictions_can.append(can_poly)
    if can_poly == 'invalid_polymer_string':
        invalid_count += 1
    if (i+1) % 1000 == 0 or i+1 == len(cleaned_predictions):
        print(f"Canonicalized {i+1}/{len(cleaned_predictions)} polymers")

print(f"Found {invalid_count} invalid polymers ({100*invalid_count/len(cleaned_predictions):.1f}%)")

# Categorize common errors
print("\nAnalyzing common error patterns...")
error_categories = {
    "unbalanced_parentheses": 0,
    "unclosed_rings": 0,
    "invalid_atoms": 0,
    "invalid_format": 0,
    "other_errors": 0
}

for poly in cleaned_predictions:
    if '(' in poly or ')' in poly:
        # Check for parentheses balance
        if poly.count('(') != poly.count(')'):
            error_categories["unbalanced_parentheses"] += 1
            continue
            
    # Check for ring closures (numbers used only once)
    digit_counts = {}
    for char in poly:
        if char.isdigit():
            digit_counts[char] = digit_counts.get(char, 0) + 1
    
    if any(count % 2 != 0 for count in digit_counts.values()):
        error_categories["unclosed_rings"] += 1
        continue
    
    # Check for invalid atoms
    valid_atoms = set('CHONFPSIBrcnofpsibl()[]:=#-+*.')
    if any(c not in valid_atoms and not c.isdigit() for c in poly):
        error_categories["invalid_atoms"] += 1
        continue
        
    # Check for polymer format (should have at least one pipe)
    if '|' not in poly:
        error_categories["invalid_format"] += 1
        continue
        
    # Other errors
    if safe_canonicalize(poly) == 'invalid_polymer_string':
        error_categories["other_errors"] += 1

print("Common error categories:")
for category, count in error_categories.items():
    print(f"  {category}: {count} ({100*count/len(cleaned_predictions):.1f}%)")

# Canonicalize training and dataset polymers
all_train_can = [safe_canonicalize(poly) for poly in all_train_polymers]
all_pols_data_can = [safe_canonicalize(poly) for poly in all_polymers_data]

# Check validity of generated polymers
print("Checking chemical validity...")
prediction_validityA = []
prediction_validityB = []

for i, poly in enumerate(cleaned_predictions):
    mols = poly_smiles_to_molecule(poly)
    
    # Check validity of monomers
    try:
        validA = mols[0] is not None
        validB = mols[1] is not None
    except:
        validA = False
        validB = False
    
    prediction_validityA.append(validA)
    prediction_validityB.append(validB)
    
    if (i+1) % 1000 == 0 or i+1 == len(cleaned_predictions):
        print(f"Processed {i+1}/{len(cleaned_predictions)} polymers for validity")

# Calculate validity percentages
validityA = sum(prediction_validityA)/len(prediction_validityA) if prediction_validityA else 0
validityB = sum(prediction_validityB)/len(prediction_validityB) if prediction_validityB else 0

# Extract monomer information safely
monomer_data = [extract_monomers_safely(poly) for poly in cleaned_predictions]
monomer_smiles_predicted = [monomers for monomers, _, _ in monomer_data if monomers[0] and monomers[1]]

print(f"Extracted monomers from {len(monomer_smiles_predicted)} polymers")

# Handle case where all polymers might be invalid
if len(monomer_smiles_predicted) == 0:
    print("WARNING: No valid polymer structures found in the predictions!")
    monA_pred = []
    monB_pred = []
    
    # Set default metrics
    novelty = 0
    novelty_full_dataset = 0
    novelty_A = 0
    novelty_B = 0
    novelty_oneMon = 0
    novelty_both = 0
    diversity = 0
    diversity_novel = 0
    validity = 0
    
    # Skip further processing
else:
    # Extract monomer info when we have valid structures
    monA_pred = [mon[0] for mon in monomer_smiles_predicted]
    monB_pred = [mon[1] for mon in monomer_smiles_predicted]

    # Extract monomer information from training set and full dataset
    monomer_smiles_train = []
    for poly_smiles in all_train_can:
        if poly_smiles != 'invalid_polymer_string':
            monomers, _, _ = extract_monomers_safely(poly_smiles)
            if monomers[0] and monomers[1]:
                monomer_smiles_train.append(monomers)
                
    monomer_smiles_d = []
    for poly_smiles in all_pols_data_can:
        if poly_smiles != 'invalid_polymer_string':
            monomers, _, _ = extract_monomers_safely(poly_smiles)
            if monomers[0] and monomers[1]:
                monomer_smiles_d.append(monomers)

    monA_t = [mon[0] for mon in monomer_smiles_train]
    monB_t = [mon[1] for mon in monomer_smiles_train]

    monA_d = [mon[0] for mon in monomer_smiles_d]
    monB_d = [mon[1] for mon in monomer_smiles_d]
    unique_mons = list(set(monA_d) | set(monB_d))

    # Extract weights and connectivity
    weights_and_con = [(w, c) for _, w, c in monomer_data if w or c]
    monomer_weights_predicted = [w for w, _ in weights_and_con]
    monomer_con_predicted = [c for _, c in weights_and_con]

    # Calculate novelty metrics with error handling
    novel_pols = [p for p in all_predictions_can if p != 'invalid_polymer_string' and p not in all_train_can]
    novel = len(novel_pols)
    novelty = novel/len(all_predictions) if len(all_predictions) > 0 else 0

    novel_full = len([p for p in all_predictions_can if p != 'invalid_polymer_string' and p not in all_pols_data_can])
    novelty_full_dataset = novel_full/len(all_predictions) if len(all_predictions) > 0 else 0

    # Safe calculation of novelty metrics for monomers
    if len(monA_pred) > 0:
        novelA = sum(1 for monA in monA_pred if monA not in unique_mons)
        novelty_A = novelA/len(monA_pred)
    else:
        novelty_A = 0
        
    if len(monB_pred) > 0:
        novelB = sum(1 for monB in monB_pred if monB not in unique_mons)
        novelty_B = novelB/len(monB_pred)
    else:
        novelty_B = 0

    # Calculate combined novelty metrics
    if len(monA_pred) == len(monB_pred) and len(monA_pred) > 0:
        novelboth = sum(1 for monA, monB in zip(monA_pred, monB_pred) 
                        if monA not in unique_mons and monB not in unique_mons)
        novelone = sum(1 for monA, monB in zip(monA_pred, monB_pred) 
                       if monA not in unique_mons or monB not in unique_mons)

        novelty_oneMon = novelone/len(monB_pred)
        novelty_both = novelboth/len(monB_pred)
    else:
        novelty_oneMon = 0
        novelty_both = 0

    # Calculate diversity metrics
    unique_valid_preds = set(p for p in all_predictions_can if p != 'invalid_polymer_string')
    diversity = len(unique_valid_preds)/len(all_predictions) if len(all_predictions) > 0 else 0
    diversity_novel = len(set(novel_pols))/len(novel_pols) if novel_pols else 0

    # Calculate overall validity
    whole_valid = len([p for p in all_predictions_can if p != 'invalid_polymer_string'])
    validity = whole_valid/len(all_predictions) if len(all_predictions) > 0 else 0

# Create property-aware output filename
properties_suffix = "_".join(property_names[:2]) if len(property_names) <= 2 else f"{len(property_names)}props"
metrics_filename = f'generated_polymers_metrics_{properties_suffix}.txt'
examples_filename = f'generated_polymers_examples_{properties_suffix}.txt'
errors_filename = f'generated_polymers_errors_{properties_suffix}.txt'

# Save metrics with property context
with open(os.path.join(dir_name, metrics_filename), 'w') as f:
    f.write(f"Generated Polymer Metrics\n")
    f.write(f"Model Properties: {', '.join(property_names)}\n")
    f.write(f"Property Count: {property_count}\n")
    f.write(f"Total Generated Polymers: {len(all_predictions)}\n\n")
    f.write("="*50 + "\n")
    f.write("VALIDITY METRICS:\n")
    f.write("-"*20 + "\n")
    f.write(f"Valid polymer structures: {whole_valid}/{len(all_predictions)} ({100*validity:.2f}%)\n")
    f.write(f"Gen Mon A validity: {100*validityA:.4f}%\n")
    f.write(f"Gen Mon B validity: {100*validityB:.4f}%\n")
    f.write(f"Gen validity: {100*validity:.4f}%\n")
    
    f.write("\nError categories:\n")
    for category, count in error_categories.items():
        f.write(f"  {category}: {count} ({100*count/len(cleaned_predictions):.1f}%)\n")
        
    f.write("\n\nNOVELTY METRICS:\n")
    f.write("-"*20 + "\n")
    f.write(f"Novelty: {100*novelty:.4f}%\n")
    f.write(f"Novelty one Mon (at least): {100*novelty_oneMon:.4f}%\n")
    f.write(f"Novelty both Mons {100*novelty_both:.4f}%\n")
    f.write(f"Novelty MonA full dataset: {100*novelty_A:.4f}%\n")
    f.write(f"Novelty MonB full dataset: {100*novelty_B:.4f}%\n")
    f.write(f"Novelty in full dataset: {100*novelty_full_dataset:.4f}%\n")
    f.write("\n\nDIVERSITY METRICS:\n")
    f.write("-"*20 + "\n")
    f.write(f"Diversity: {100*diversity:.4f}%\n")
    f.write(f"Diversity (novel polymers): {100*diversity_novel:.4f}%\n")

# Save examples
with open(os.path.join(dir_name, examples_filename), 'w') as f:
    f.write(f"Generated Polymer Examples\n")
    f.write(f"Model Properties: {', '.join(property_names)}\n")
    f.write(f"Total Count: {len(all_predictions)}\n")
    f.write("="*50 + "\n\n")
    for i, (original, cleaned) in enumerate(zip(all_predictions, cleaned_predictions)):
        valid = "✅" if i < len(prediction_validityA) and prediction_validityA[i] and prediction_validityB[i] else "❌"
        f.write(f"{i+1:4d}: {cleaned} {valid}\n")
        if original != cleaned:
            f.write(f"     Original with padding: {original}\n")
        f.write("\n")

# Save error examples for detailed analysis
with open(os.path.join(dir_name, errors_filename), 'w') as f:
    f.write(f"Generated Polymer Error Analysis\n")
    f.write(f"Model Properties: {', '.join(property_names)}\n")
    f.write("="*50 + "\n\n")
    
    # Unbalanced parentheses examples
    f.write("UNBALANCED PARENTHESES EXAMPLES:\n")
    f.write("-"*40 + "\n")
    count = 0
    for i, poly in enumerate(cleaned_predictions):
        if poly.count('(') != poly.count(')') and count < 5:
            f.write(f"Example {count+1}: {poly}\n")
            f.write(f"  Open parentheses: {poly.count('(')}\n")
            f.write(f"  Close parentheses: {poly.count(')')}\n\n")
            count += 1
    f.write("\n")
    
    # Unclosed rings examples
    f.write("UNCLOSED RINGS EXAMPLES:\n")
    f.write("-"*40 + "\n")
    count = 0
    for i, poly in enumerate(cleaned_predictions):
        digit_counts = {}
        for char in poly:
            if char.isdigit():
                digit_counts[char] = digit_counts.get(char, 0) + 1
                
        if any(count % 2 != 0 for count in digit_counts.values()) and count < 5:
            f.write(f"Example {count+1}: {poly}\n")
            f.write(f"  Ring numbers: {digit_counts}\n\n")
            count += 1
    f.write("\n")
    
    # Invalid format examples
    f.write("INVALID FORMAT EXAMPLES:\n")
    f.write("-"*40 + "\n")
    count = 0
    for i, poly in enumerate(cleaned_predictions):
        if '|' not in poly and count < 5:
            f.write(f"Example {count+1}: {poly}\n\n")
            count += 1
    f.write("\n")
    
    # Other errors
    f.write("OTHER ERROR EXAMPLES:\n")
    f.write("-"*40 + "\n")
    count = 0
    for i, poly in enumerate(cleaned_predictions):
        if ('|' in poly and 
            poly.count('(') == poly.count(')') and
            safe_canonicalize(poly) == 'invalid_polymer_string' and
            count < 5):
            f.write(f"Example {count+1}: {poly}\n\n")
            count += 1

# Print summary
print("\nGenerated Polymer Metrics Summary:")
print(f"Properties: {property_names}")
print(f"Total generated: {len(all_predictions)}")
print(f"Overall validity: {100*validity:.2f}%")
print(f"Monomer A validity: {100*validityA:.2f}%")
print(f"Monomer B validity: {100*validityB:.2f}%")

print("\nCommon error categories:")
for category, count in error_categories.items():
    print(f"  {category}: {count} ({100*count/len(cleaned_predictions):.1f}%)")

print(f"\nFiles saved successfully to:")
print(f"1. {os.path.join(dir_name, metrics_filename)} - Contains all metrics")
print(f"2. {os.path.join(dir_name, examples_filename)} - Contains examples with validity markers")
print(f"3. {os.path.join(dir_name, errors_filename)} - Contains detailed error analysis")
print(f"4. {os.path.join(dir_name, 'generated_polymers_cleaned.pkl')} - Polymers with padding removed")

print("\nRecommendation: Focus on fixing the model's generation capability rather than repairing outputs.")
