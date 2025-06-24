import pandas as pd
import torch
from torch_geometric.data import Data, Batch
import numpy as np
from .data_utils import get_graph_from_polymer_string, get_seq_features_from_line

def load_transfer_data(csv_path, stage, source_properties, target_properties, 
                      batch_size, tokenization, vocab, sample_weight=1.0, device='cuda'):
    """
    Load data for transfer learning based on stage and property availability
    
    Stage 1: Use all data with source properties + unlabeled data
    Stage 2: Use only data with target properties
    """
    
    # Load the combined dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded combined dataset with {len(df)} entries")
    
    # Filter based on stage
    if stage == 1:
        # Stage 1: Use data with source properties OR unlabeled data
        mask = pd.Series([False] * len(df))
        
        # Include data with any source property
        for prop in source_properties:
            if prop in df.columns:
                mask |= df[prop].notna()
        
        # Include unlabeled data (no properties)
        all_props = source_properties + target_properties
        unlabeled_mask = pd.Series([True] * len(df))
        for prop in all_props:
            if prop in df.columns:
                unlabeled_mask &= df[prop].isna()
        
        mask |= unlabeled_mask
        
        # Weighted sampling if specified
        if sample_weight != 1.0:
            # Oversample data with source properties
            labeled_indices = df[mask & df[source_properties[0]].notna()].index
            unlabeled_indices = df[mask & df[source_properties[0]].isna()].index
            
            # Sample with weights
            n_labeled = int(len(labeled_indices) * sample_weight)
            n_unlabeled = len(unlabeled_indices)
            
            sampled_indices = np.concatenate([
                np.random.choice(labeled_indices, n_labeled, replace=True),
                unlabeled_indices
            ])
            
            df_filtered = df.loc[sampled_indices]
        else:
            df_filtered = df[mask]
            
        print(f"Stage 1: Using {len(df_filtered)} samples")
        print(f"  - With source properties: {df_filtered[source_properties[0]].notna().sum() if source_properties[0] in df_filtered.columns else 0}")
        print(f"  - Unlabeled: {(df_filtered[all_props].isna().all(axis=1)).sum() if all([p in df_filtered.columns for p in all_props]) else 'N/A'}")
        
    else:  # Stage 2
        # Stage 2: Use only data with target properties
        mask = pd.Series([False] * len(df))
        
        for prop in target_properties:
            if prop in df.columns:
                mask |= df[prop].notna()
        
        df_filtered = df[mask]
        print(f"Stage 2: Using {len(df_filtered)} samples with {target_properties}")
    
    # Convert to graph data
    return create_data_loaders(df_filtered, source_properties, target_properties, 
                             batch_size, tokenization, vocab, stage, device)

def create_data_loaders(df, source_properties, target_properties, batch_size, 
                       tokenization, vocab, stage, device):
    """Convert dataframe to graph data loaders"""
    
    # Split data
    n_samples = len(df)
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create data loaders
    train_loader = create_dict_loader(df.iloc[train_indices], source_properties, 
                                     target_properties, batch_size, tokenization, 
                                     vocab, stage, device)
    val_loader = create_dict_loader(df.iloc[val_indices], source_properties, 
                                   target_properties, batch_size, tokenization, 
                                   vocab, stage, device)
    test_loader = create_dict_loader(df.iloc[test_indices], source_properties, 
                                    target_properties, batch_size, tokenization, 
                                    vocab, stage, device)
    
    return train_loader, val_loader, test_loader

def create_dict_loader(df, source_properties, target_properties, batch_size, 
                      tokenization, vocab, stage, device):
    """Create dictionary-style data loader"""
    
    dict_loader = {}
    n_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        # Create batch data
        batch_graphs = []
        
        for _, row in batch_df.iterrows():
            # Get polymer string (adjust column name as needed)
            polymer_string = row['poly_chemprop_input']
            
            # Create graph
            graph = get_graph_from_polymer_string(polymer_string)
            
            # Add properties based on stage
            if stage == 1:
                # Stage 1: Use source properties
                for j, prop in enumerate(source_properties):
                    if prop in row and pd.notna(row[prop]):
                        setattr(graph, f'y{j+1}', torch.tensor([row[prop]], dtype=torch.float))
                    else:
                        setattr(graph, f'y{j+1}', torch.tensor([float('nan')], dtype=torch.float))
            else:
                # Stage 2: Use target properties
                for j, prop in enumerate(target_properties):
                    if prop in row and pd.notna(row[prop]):
                        setattr(graph, f'y{j+1}', torch.tensor([row[prop]], dtype=torch.float))
                    else:
                        setattr(graph, f'y{j+1}', torch.tensor([float('nan')], dtype=torch.float))
            
            # Add sequence features
            seq_features = get_seq_features_from_line(polymer_string, vocab, max_tgt_len=512)
            graph.tgt_token_ids = seq_features[0]
            
            batch_graphs.append(graph)
        
        # Create batch
        batch = Batch.from_data_list(batch_graphs)
        
        # Create matrices (you'll need to implement these based on your setup)
        dest_is_origin_matrix = create_dest_is_origin_matrix(batch)
        inc_edges_to_atom_matrix = create_inc_edges_to_atom_matrix(batch)
        
        dict_loader[str(i)] = [batch, dest_is_origin_matrix, inc_edges_to_atom_matrix]
    
    return dict_loader
