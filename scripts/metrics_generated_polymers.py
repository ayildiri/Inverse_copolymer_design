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


def poly_smiles_to_molecule(poly_input):
    '''
    Turns adjusted polymer smiles string into PyG data objects
    '''
    try:
        # Turn into RDKIT mol object
        mols = make_monomer_mols(poly_input.split("|")[0], 0, 0,  # smiles
                                fragment_weights=poly_input.split("|")[1:-1])
        return mols
    except Exception as e:
        print(f"Error processing polymer: {poly_input}")
        print(f"Error details: {e}")
        return [None, None]  # Return None for both monomers to indicate invalid

def valid_scores(smiles):
    return np.array(list(map(make_polymer_mol, smiles)), dtype=np.float32)


prediction_validityA = []
prediction_validityB = []


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

with open(dir_name+'generated_polymers.pkl', 'rb') as f:
    all_predictions=pickle.load(f)

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
    train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
    all_train_polymers.extend(train_polymers_batch)

# Extract all polymers from dataset
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    all_polymers_data.append(poly_input)

# Canonicalize all polymer strings
sm_can = SmilesEnumCanon()
all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
all_train_can = list(map(sm_can.canonicalize, all_train_polymers))
all_pols_data_can = list(map(sm_can.canonicalize, all_polymers_data))

# Check validity of generated polymers
prediction_mols = list(map(poly_smiles_to_molecule, all_predictions))
for mon in prediction_mols: 
    try: prediction_validityA.append(mon[0] is not None)
    except: prediction_validityA.append(False)
    try: prediction_validityB.append(mon[1] is not None)
    except: prediction_validityB.append(False)

# Calculate validity percentages
validityA = sum(prediction_validityA)/len(prediction_validityA)
validityB = sum(prediction_validityB)/len(prediction_validityB)

# Extract monomer information
monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monA_pred = [mon[0] for mon in monomer_smiles_predicted]
monB_pred = [mon[1] for mon in monomer_smiles_predicted]

monomer_smiles_train = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_train_can]
monA_t = [mon[0] for mon in monomer_smiles_train]
monB_t = [mon[1] for mon in monomer_smiles_train]

monomer_smiles_d = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_pols_data_can]
monA_d = [mon[0] for mon in monomer_smiles_d]
monB_d = [mon[1] for mon in monomer_smiles_d]
unique_mons = list(set(monA_d) | set(monB_d))

monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_con_predicted = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']

# Calculate novelty metrics
novel = 0
novel_pols=[]
for pol in all_predictions_can:
    if not pol in all_train_can:
        novel+=1
        novel_pols.append(pol)
novelty = novel/len(all_predictions)

novel = 0
for pol in all_predictions_can:
    if not pol in all_pols_data_can:
        novel+=1
novelty_full_dataset = novel/len(all_predictions)

novelA = 0
for monA in monA_pred:
    if not monA in unique_mons:
        novelA+=1
novelty_A = novelA/len(monA_pred)

novelB = 0
for monB in monB_pred:
    if not monB in unique_mons:
        novelB+=1
novelty_B = novelB/len(monB_pred)

novelboth = 0
novelone = 0
for monA, monB in zip(monA_pred,monB_pred):
    if (not monB in unique_mons) and (not monA in unique_mons):
        novelboth+=1
    if (not monB in unique_mons) or (not monA in unique_mons):
        novelone+=1

novelty_oneMon = novelone/len(monB_pred)
novelty_both= novelboth/len(monB_pred)

# Calculate diversity metrics
diversity = len(list(set(all_predictions)))/len(all_predictions)
diversity_novel = len(list(set(novel_pols)))/len(novel_pols) if novel_pols else 0

# Calculate overall validity
whole_valid = len(monomer_smiles_predicted)
validity = whole_valid/len(all_predictions)

# Create property-aware output filename
properties_suffix = "_".join(property_names[:2]) if len(property_names) <= 2 else f"{len(property_names)}props"
metrics_filename = f'generated_polymers_metrics_{properties_suffix}.txt'
examples_filename = f'generated_polymers_examples_{properties_suffix}.txt'

# Save metrics with property context
with open(dir_name + metrics_filename, 'w') as f:
    f.write(f"Generated Polymer Metrics\n")
    f.write(f"Model Properties: {', '.join(property_names)}\n")
    f.write(f"Property Count: {property_count}\n")
    f.write(f"Total Generated Polymers: {len(all_predictions)}\n\n")
    f.write("="*50 + "\n")
    f.write("VALIDITY METRICS:\n")
    f.write("-"*20 + "\n")
    f.write("Gen Mon A validity: %.4f %% " % (100*validityA,))
    f.write("Gen Mon B validity: %.4f %% " % (100*validityB,))
    f.write("Gen validity: %.4f %% " % (100*validity,))
    f.write("\n\nNOVELTY METRICS:\n")
    f.write("-"*20 + "\n")
    f.write("Novelty: %.4f %% " % (100*novelty,))
    f.write("Novelty one Mon (at least): %.4f %% " % (100*novelty_oneMon,))
    f.write("Novelty both Mons %.4f %% " % (100*novelty_both,))
    f.write("Novelty MonA full dataset: %.4f %% " % (100*novelty_A,))
    f.write("Novelty MonB full dataset: %.4f %% " % (100*novelty_B,))
    f.write("Novelty in full dataset: %.4f %% " % (100*novelty_full_dataset,))
    f.write("\n\nDIVERSITY METRICS:\n")
    f.write("-"*20 + "\n")
    f.write("Diversity: %.4f %% " % (100*diversity,))
    f.write("Diversity (novel polymers): %.4f %% " % (100*diversity_novel,))

# Save examples with property context
with open(dir_name + examples_filename, 'w') as f:
    f.write(f"Generated Polymer Examples\n")
    f.write(f"Model Properties: {', '.join(property_names)}\n")
    f.write(f"Total Count: {len(all_predictions)}\n")
    f.write("="*50 + "\n\n")
    for i, polymer in enumerate(all_predictions):
        f.write(f"{i+1:4d}: {polymer}\n")

# Print summary
print("\nGenerated Polymer Metrics Summary:")
print(f"Properties: {property_names}")
print(f"Total generated: {len(all_predictions)}")
print(f"Overall validity: {100*validity:.2f}%")
print(f"Novelty (vs training): {100*novelty:.2f}%")
print(f"Novelty (vs full dataset): {100*novelty_full_dataset:.2f}%")
print(f"Diversity: {100*diversity:.2f}%")

print(f"\nFiles saved successfully to:")
print(f"1. {dir_name}{metrics_filename} - Contains all metrics")
print(f"2. {dir_name}{examples_filename} - Contains examples of generated polymers")
