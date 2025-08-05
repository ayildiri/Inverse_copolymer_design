"""
Precompute FM4M embeddings for all molecules in the dataset
"""
import torch
import argparse
import os
from tqdm import tqdm
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_utils import tokenids_to_vocab, combine_tokens, load_vocab

def extract_unique_smiles(data_loader, vocab, tokenization):
    """Extract all unique SMILES from data loader"""
    unique_smiles = set()
    
    for batch_key in tqdm(data_loader, desc="Extracting SMILES"):
        if batch_key in ['dest_is_origin_matrix', 'inc_edges_to_atom_matrix']:
            continue
            
        data = data_loader[batch_key][0]
        if hasattr(data, 'tgt_token_ids'):
            for seq in data.tgt_token_ids:
                try:
                    # Handle tensor conversion
                    if hasattr(seq, 'cpu'):
                        seq = seq.cpu().tolist()
                    tokens = tokenids_to_vocab(seq, vocab)
                    smiles = combine_tokens(tokens, tokenization=tokenization)
                    
                    # Extract just the SMILES part (before polymer notation)
                    if '|' in smiles:
                        smiles = smiles.split('|')[0]
                    
                    # Skip empty or invalid SMILES
                    if smiles and len(smiles) > 1:
                        unique_smiles.add(smiles)
                except Exception as e:
                    print(f"Error processing sequence: {e}")
                    continue
    
    return list(unique_smiles)

def compute_fm4m_embeddings(smiles_list, model_name, batch_size=32):
    """Compute FM4M embeddings for a list of SMILES"""
    try:
        from fm4m import FM4M_Kit
        fm4m_kit = FM4M_Kit()
    except ImportError:
        raise ImportError("FM4M not installed. Please install fm4m-kit first.")
    
    embeddings_dict = {}
    
    # Process in batches
    for i in tqdm(range(0, len(smiles_list), batch_size), desc=f"Computing {model_name}"):
        batch_smiles = smiles_list[i:i+batch_size]
        
        try:
            # Get embeddings from FM4M
            batch_embeddings = fm4m_kit.get_representation(
                model=model_name,
                data=batch_smiles
            )
            
            # Convert to tensor if needed
            if not isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = torch.tensor(batch_embeddings, dtype=torch.float32)
            
            # Store each embedding
            for smiles, embedding in zip(batch_smiles, batch_embeddings):
                embeddings_dict[smiles] = embedding.cpu()
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            # Store None for failed embeddings
            for smiles in batch_smiles:
                embeddings_dict[smiles] = None
    
    return embeddings_dict

def main():
    parser = argparse.ArgumentParser(description="Precompute FM4M embeddings for polymer dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory")
    parser.add_argument("--tokenization", default="RT_tokenized", choices=["RT_tokenized", "oldtok"])
    parser.add_argument("--fm4m_models", nargs='+', default=['SMI-TED', 'MHG-GED'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage", type=int, default=1, help="Training stage (affects which data to load)")
    parser.add_argument("--output_dir", default=None, help="Output directory (defaults to dataset_path)")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.dataset_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocabulary
    vocab_file = os.path.join(args.dataset_path, f'poly_smiles_vocab_augmented_{args.tokenization}.txt')
    if not os.path.exists(vocab_file):
        # Try without augmented
        vocab_file = os.path.join(args.dataset_path, f'poly_smiles_vocab_{args.tokenization}.txt')
    
    print(f"Loading vocabulary from {vocab_file}")
    vocab = load_vocab(vocab_file)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Determine which data files to load based on stage
    data_files = []
    if args.stage in [0, 1]:
        # Stage 0/1: Use source property data
        data_files = [
            f'dict_train_loader_augmented_{args.tokenization}_EA_IP.pt',
            f'dict_val_loader_augmented_{args.tokenization}_EA_IP.pt',
            f'dict_test_loader_augmented_{args.tokenization}_EA_IP.pt'
        ]
    else:
        # Stage 2: Use target property data
        data_files = [
            f'dict_train_loader_augmented_{args.tokenization}_bandgap.pt',
            f'dict_val_loader_augmented_{args.tokenization}_bandgap.pt',
            f'dict_test_loader_augmented_{args.tokenization}_bandgap.pt'
        ]
    
    # Fallback to generic names if specific files don't exist
    if not os.path.exists(os.path.join(args.dataset_path, data_files[0])):
        data_files = [
            f'dict_train_loader_augmented_{args.tokenization}.pt',
            f'dict_val_loader_augmented_{args.tokenization}.pt',
            f'dict_test_loader_augmented_{args.tokenization}.pt'
        ]
    
    # Extract unique SMILES from all data files
    all_smiles = set()
    
    for data_file in data_files:
        file_path = os.path.join(args.dataset_path, data_file)
        if os.path.exists(file_path):
            print(f"\nLoading {data_file}...")
            data = torch.load(file_path)
            smiles_list = extract_unique_smiles(data, vocab, args.tokenization)
            print(f"Found {len(smiles_list)} SMILES")
            all_smiles.update(smiles_list)
        else:
            print(f"Warning: {file_path} not found, skipping...")
    
    all_smiles = list(all_smiles)
    print(f"\nTotal unique SMILES: {len(all_smiles)}")
    
    # Compute embeddings for each model
    for model_name in args.fm4m_models:
        print(f"\n{'='*50}")
        print(f"Computing {model_name} embeddings...")
        print(f"{'='*50}")
        
        # Compute embeddings
        embeddings_dict = compute_fm4m_embeddings(all_smiles, model_name, args.batch_size)
        
        # Count successful embeddings
        valid_embeddings = sum(1 for v in embeddings_dict.values() if v is not None)
        print(f"Successfully computed {valid_embeddings}/{len(embeddings_dict)} embeddings")
        
        # Save embeddings
        cache_file = os.path.join(output_dir, f'fm4m_cache_{model_name}_{args.tokenization}_stage{args.stage}.pt')
        torch.save(embeddings_dict, cache_file)
        print(f"Saved to {cache_file}")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'tokenization': args.tokenization,
            'stage': args.stage,
            'num_smiles': len(all_smiles),
            'num_valid_embeddings': valid_embeddings,
            'embedding_shape': next(iter(embeddings_dict.values())).shape if valid_embeddings > 0 else None
        }
        metadata_file = cache_file.replace('.pt', '_metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata to {metadata_file}")

if __name__ == "__main__":
    main()
