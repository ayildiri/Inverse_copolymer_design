import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import torch
from rdkit import Chem
from data_processing.data_utils import *
import argparse

def validate_training_data(data_path, vocab_path, tokenization="RT_tokenized"):
    """Validate that training data contains valid SMILES"""
    
    print("🔍 Validating training data...")
    
    # Load data
    dict_train_loader = torch.load(data_path)
    vocab = load_vocab(vocab_path)
    
    total_polymers = 0
    valid_polymers = 0
    invalid_examples = []
    
    for batch_key in dict_train_loader:
        if batch_key in ['dest_is_origin_matrix', 'inc_edges_to_atom_matrix']:
            continue
            
        data = dict_train_loader[batch_key][0]
        
        for sample in range(len(data.tgt_token_ids)):
            total_polymers += 1
            
            try:
                # Convert tokens back to string
                tokens = tokenids_to_vocab(data.tgt_token_ids[sample], vocab)
                polymer_string = combine_tokens(tokens, tokenization=tokenization)
                
                # Extract SMILES part (before first |)
                if '|' in polymer_string:
                    smiles_part = polymer_string.split('|')[0]
                else:
                    smiles_part = polymer_string
                
                # Validate with RDKit
                if '.' in smiles_part:
                    # Handle copolymer (multiple monomers)
                    monomers = smiles_part.split('.')
                    all_valid = True
                    for monomer in monomers:
                        if not monomer.strip():
                            continue
                        mol = Chem.MolFromSmiles(monomer)
                        if mol is None:
                            all_valid = False
                            break
                    
                    if all_valid:
                        valid_polymers += 1
                    else:
                        invalid_examples.append(polymer_string)
                else:
                    # Single monomer
                    mol = Chem.MolFromSmiles(smiles_part)
                    if mol is not None:
                        valid_polymers += 1
                    else:
                        invalid_examples.append(polymer_string)
                        
            except Exception as e:
                invalid_examples.append(f"Error processing: {e}")
                
            # Show progress
            if total_polymers % 100 == 0:
                print(f"Processed {total_polymers} polymers...")
    
    validity_rate = valid_polymers / total_polymers * 100
    
    print(f"\n📊 TRAINING DATA VALIDATION RESULTS:")
    print(f"Total polymers: {total_polymers}")
    print(f"Valid polymers: {valid_polymers}")
    print(f"Validity rate: {validity_rate:.1f}%")
    
    if invalid_examples:
        print(f"\n❌ First 5 invalid examples:")
        for i, example in enumerate(invalid_examples[:5]):
            print(f"  {i+1}: {example}")
    
    return validity_rate > 90  # Return True if training data is mostly valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--vocab_path", required=True) 
    parser.add_argument("--tokenization", default="RT_tokenized")
    args = parser.parse_args()
    
    is_valid = validate_training_data(args.data_path, args.vocab_path, args.tokenization)
    
    if not is_valid:
        print("\n⚠️ WARNING: Training data has validity issues!")
        print("Consider cleaning the training data or checking tokenization.")
