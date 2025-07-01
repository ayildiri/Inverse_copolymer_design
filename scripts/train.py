import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import time
from datetime import datetime
import random
#from G2S import *
from model.G2S_clean import *
from data_processing.data_utils import *
# deep learning packages
import torch
import torch.nn as nn
from statistics import mean
import pickle
import math
import argparse
import numpy as np
import csv

def debug_vocab_and_embeddings(vocab_file_path, dataset_path=None):
    """Debug vocabulary and embedding setup"""
    
    print("=== VOCABULARY DEBUG ===")
    
    # Load and inspect vocabulary
    try:
        vocab = load_vocab(vocab_file_path)
        print(f"Vocabulary size: {len(vocab)}")
        print(f"First 10 tokens: {list(vocab.items())[:10]}")
        print(f"Last 10 tokens: {list(vocab.items())[-10:]}")
        
        # Check special tokens
        special_tokens = ['_PAD', '_SOS', '_EOS', '_UNK']
        for token in special_tokens:
            if token in vocab:
                print(f"{token}: {vocab[token]}")
            else:
                print(f"WARNING: {token} not found in vocabulary!")
        
        return vocab
    except Exception as e:
        print(f"ERROR loading vocabulary: {e}")
        raise

def validate_model_configuration(model, vocab, dict_train_loader):
    """Comprehensive validation of model configuration - FIXED"""
    
    print("🔧 VALIDATING MODEL CONFIGURATION...")
    
    # 1. Vocabulary validation
    decoder_vocab_size = model.Decoder.output_layer.out_features
    actual_vocab_size = len(vocab)
    
    assert decoder_vocab_size == actual_vocab_size, \
        f"Output layer size mismatch! Model has {decoder_vocab_size} but vocab has {actual_vocab_size}"
    print(f"✅ Vocabulary size validated: {actual_vocab_size}")
    
    # 2. Embedding configuration validation - FIXED
    embeddings = model.Decoder.decoder_embeddings
    print(f"✅ Embedding configuration:")
    print(f"   Word vec size: {embeddings.word_vec_size}")
    
    # FIXED: Safely access embedding information
    try:
        if hasattr(embeddings, 'make_embedding'):
            if hasattr(embeddings.make_embedding, '__len__') and len(embeddings.make_embedding) > 0:
                # Try to get the first embedding layer
                first_emb = embeddings.make_embedding[0]
                if hasattr(first_emb, 'num_embeddings'):
                    print(f"   Word vocab size: {first_emb.num_embeddings}")
                elif hasattr(first_emb, '__len__'):
                    print(f"   Embedding modules: {len(first_emb)} modules")
                else:
                    print(f"   Embedding type: {type(first_emb).__name__}")
            else:
                print(f"   Make embedding type: {type(embeddings.make_embedding).__name__}")
        
        print(f"   Position encoding: {embeddings.position_encoding}")
        
        # Check for feature embeddings safely
        if hasattr(embeddings, 'make_embedding'):
            try:
                emb_len = len(embeddings.make_embedding) if hasattr(embeddings.make_embedding, '__len__') else 1
                if emb_len > 1:
                    print(f"   Feature embeddings detected: {emb_len - 1} feature types")
                else:
                    print(f"   Feature embeddings: None (word-only)")
            except:
                print(f"   Feature embeddings: Could not determine structure")
    except Exception as e:
        print(f"   ⚠️ Could not fully analyze embedding structure: {e}")
        print(f"   Embedding type: {type(embeddings).__name__}")
    
    # 3. Test forward pass with sample data - ENHANCED
    print("🧪 Testing forward pass...")
    
    try:
        # Get a small batch for testing
        first_batch_key = list(dict_train_loader.keys())[0]
        test_data = dict_train_loader[first_batch_key][0]
        test_dest = dict_train_loader[first_batch_key][1]
        test_inc = dict_train_loader[first_batch_key][2]
        
        # Move to device
        test_data.to(device)
        test_dest.to(device)
        test_inc.to(device)
        
        # ENHANCED: Check data structure first
        print(f"   Test batch info:")
        print(f"     Num graphs: {test_data.num_graphs}")
        print(f"     Node features: {test_data.num_node_features}")
        print(f"     Edge features: {test_data.num_edge_features}")
        
        if hasattr(test_data, 'tgt_token_ids'):
            print(f"     Target sequences: {len(test_data.tgt_token_ids)}")
            if len(test_data.tgt_token_ids) > 0:
                sample_seq = test_data.tgt_token_ids[0]
                if isinstance(sample_seq, torch.Tensor):
                    sample_tokens = sample_seq.tolist()
                else:
                    sample_tokens = list(sample_seq)
                print(f"     Sample sequence length: {len(sample_tokens)}")
                print(f"     Sample token range: [{min(sample_tokens)}, {max(sample_tokens)}]")
        
        # Test encoding
        with torch.no_grad():
            if hasattr(model, 'Encoder'):
                mu, logvar = model.Encoder(test_data, test_dest, test_inc, device)
                print(f"✅ Encoder output shapes: mu={mu.shape}, logvar={logvar.shape}")
                
                # Test sample from latent space
                z = model.sample(mu, logvar, eps_scale=model.eps)
                print(f"✅ Latent sample shape: {z.shape}")
                
                # Test decoder embeddings specifically - ENHANCED
                if hasattr(test_data, 'tgt_token_ids') and len(test_data.tgt_token_ids) > 0:
                    # Use first 5 tokens from first sequence for testing
                    sample_tokens = test_data.tgt_token_ids[0][:5]  
                    # CRITICAL FIX: Don't unsqueeze - keep shape as [1, seq_len]
                    target = torch.tensor([sample_tokens], device=device).unsqueeze(-1)  # Shape: [1, 5, 1]
                    
                    print(f"   Testing embedding with shape: {target.shape}")
                    print(f"   Token range in test: [{target.min().item()}, {target.max().item()}]")
                    
                    try:
                        emb_output = model.Decoder.decoder_embeddings(target)
                        print(f"✅ Embedding output shape: {emb_output.shape}")
                        print(f"✅ Embedding dimension: {emb_output.size(-1)}")
                    except Exception as emb_error:
                        print(f"❌ EMBEDDING ERROR: {emb_error}")
                        print(f"   Target shape: {target.shape}")
                        print(f"   Target dtype: {target.dtype}")
                        print(f"   Target range: [{target.min().item()}, {target.max().item()}]")
                        print(f"   Vocab size: {len(vocab)}")
                        
                        # CRITICAL DEBUG: Check if this is the Elementwise error
                        if "assert len(self) == len(emb_)" in str(emb_error):
                            print(f"\n🔥 FOUND THE ROOT CAUSE!")
                            print(f"   This is the Elementwise dimension mismatch error!")
                            print(f"   The embeddings module expects different input dimensions")
                            print(f"   Target last dimension: {target.size(-1)}")
                            print(f"   Expected by Elementwise: different size")
                            
                            # Check embedding configuration
                            print(f"   Debugging embedding configuration...")
                            if hasattr(embeddings, 'make_embedding'):
                                print(f"   make_embedding type: {type(embeddings.make_embedding)}")
                                if hasattr(embeddings.make_embedding, '__len__'):
                                    print(f"   make_embedding length: {len(embeddings.make_embedding)}")
                        
                        raise
                else:
                    print(f"   ⚠️ No target token IDs found for embedding test")
                        
        print("✅ Forward pass validation completed successfully!")
        
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}")
        print("🔧 This indicates a configuration mismatch.")
        
        # Enhanced error diagnosis
        if "Elementwise" in str(e):
            print(f"\n🔥 ELEMENTWISE ERROR DETECTED!")
            print(f"This means the embeddings are configured with feature dimensions")
            print(f"but the input data doesn't match those dimensions.")
            print(f"\n🔧 POSSIBLE SOLUTIONS:")
            print(f"1. Regenerate embeddings with feat_vocab_sizes=[]")
            print(f"2. Ensure input data has correct feature dimensions")
            print(f"3. Check if data preprocessing matches model expectations")
        
        raise

# In train.py, fix the debug_elementwise_error function (around line 260-310)

def debug_elementwise_error(model, vocab, dict_train_loader):
    """
    Specific debugging for the Elementwise error
    """
    print(f"\n🔍 DEBUGGING ELEMENTWISE ERROR...")
    
    embeddings = model.Decoder.decoder_embeddings
    
    print(f"📊 Embeddings configuration:")
    print(f"   Type: {type(embeddings)}")
    print(f"   Word vec size: {embeddings.word_vec_size}")
    
    if hasattr(embeddings, 'make_embedding'):
        make_emb = embeddings.make_embedding
        print(f"   make_embedding type: {type(make_emb)}")
        
        if hasattr(make_emb, '__len__'):
            print(f"   make_embedding length: {len(make_emb)}")
            
            for i, layer in enumerate(make_emb):
                print(f"   Layer {i}: {type(layer)} - {layer}")
                if hasattr(layer, 'num_embeddings'):
                    print(f"     Vocab size: {layer.num_embeddings}")
                if hasattr(layer, 'embedding_dim'):
                    print(f"     Embedding dim: {layer.embedding_dim}")
        
        # Check if this is an Elementwise container
        if "Elementwise" in str(type(make_emb)):
            print(f"   🔍 Found Elementwise container!")
            print(f"   This means multiple embedding tables are expected")
            
            # Get expected input dimensions
            if hasattr(make_emb, '__len__'):
                expected_dims = len(make_emb)
                print(f"   Expected input dimensions: {expected_dims}")
                
                # Check sample data
                first_batch = dict_train_loader[list(dict_train_loader.keys())[0]][0]
                if hasattr(first_batch, 'tgt_token_ids'):
                    sample = first_batch.tgt_token_ids[0][:5]
                    # CRITICAL FIX: Don't unsqueeze - check actual input shape
                    test_input = torch.tensor([sample], device=device).unsqueeze(-1)  # Shape: [1, 5, 1]
                    print(f"   Actual input dimensions: {test_input.ndim}")
                    print(f"   Input shape: {test_input.shape}")
                    
                    # The issue is that Elementwise expects the last dimension to match its length
                    # But we're providing 2D input when it expects 3D with last dim matching expected_dims
                    print(f"   \n🔥 ROOT CAUSE:")
                    print(f"   Elementwise expects input with last dimension = {expected_dims}")
                    print(f"   But the input is 2D with shape {test_input.shape}")
                    print(f"   \n🔧 SOLUTION:")
                    print(f"   Remove .unsqueeze(-1) from the forward method")
                    print(f"   Keep input as 2D tensor [batch, sequence]")
    
    return embeddings

def safe_model_creation(model_class, *args, **kwargs):
    """Safely create model with better error reporting"""
    
    try:
        model = model_class(*args, **kwargs)
        print("✅ Model created successfully")
        return model
    except Exception as e:
        print(f"❌ MODEL CREATION FAILED: {e}")
        print(f"Model class: {model_class.__name__}")
        print(f"Arguments: {args}")
        print(f"Keyword arguments: {kwargs}")
        
        # Provide specific guidance for common errors
        if "AssertionError" in str(e):
            print("\n🔧 ASSERTION ERROR DETECTED:")
            print("This usually means dimension mismatch in embeddings.")
            print("Check:")
            print("1. Vocabulary size matches model configuration")
            print("2. Feature embedding configuration")
            print("3. Data preprocessing consistency")
        
        raise

def load_transfer_data_safely(csv_path, stage, source_properties, target_properties, 
                              batch_size, tokenization, vocab, device, **kwargs):
    """Safely load transfer learning data with vocabulary validation"""
    
    print(f"🔄 Loading transfer learning data for stage {stage}")
    print(f"Source properties: {source_properties}")
    print(f"Target properties: {target_properties}")
    
    try:
        # Import your transfer data loading function
        from data_processing.transfer_data_utils import load_transfer_data
        
        dict_train_loader, dict_val_loader, dict_test_loader = load_transfer_data(
            csv_path=csv_path,
            stage=stage,
            source_properties=source_properties,
            target_properties=target_properties,
            batch_size=batch_size,
            tokenization=tokenization,
            vocab=vocab,
            device=device,
            **kwargs
        )
        
        # Validate data consistency
        print("🔍 Validating loaded data...")
        
        # Check first batch
        first_key = list(dict_train_loader.keys())[0]
        first_batch = dict_train_loader[first_key][0]
        
        if hasattr(first_batch, 'tgt_token_ids'):
            max_token = max(max(seq) for seq in first_batch.tgt_token_ids)
            if max_token >= len(vocab):
                raise ValueError(f"Token ID {max_token} exceeds vocabulary size {len(vocab)}")
        
        print(f"✅ Transfer data loaded successfully")
        return dict_train_loader, dict_val_loader, dict_test_loader
        
    except ImportError:
        print("❌ transfer_data_utils not found. Using standard data loading...")
        # Fallback to standard data loading
        data_path_prefix = os.path.join(os.path.dirname(csv_path), f'dict_{{}}_loader_{tokenization}.pt')
        try:
            dict_train_loader = torch.load(data_path_prefix.format('train'))
            dict_val_loader = torch.load(data_path_prefix.format('val'))
            dict_test_loader = torch.load(data_path_prefix.format('test'))
            return dict_train_loader, dict_val_loader, dict_test_loader
        except Exception as e:
            print(f"❌ Standard data loading also failed: {e}")
            raise
        
    except Exception as e:
        print(f"❌ Transfer data loading failed: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if transfer_data_utils.py exists and is correct")
        print("2. Verify CSV file format and column names")
        print("3. Ensure vocabulary file matches the tokenization used")
        print("4. Check if property names match CSV columns")
        raise

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
        
class EarlyStopping:
    def __init__(self, dir, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_dir = dir

    def __call__(self, val_loss, model_dict):
        val_loss = round(val_loss,4)
        if self.best_score is None:
            self.best_score = val_loss
            torch.save(model_dict, os.path.join(self.save_dir,"model_best_loss.pt"))
            #torch.save(model.state_dict(), self.save_dir + "/model_best_loss.pth")
            return True  # Indicate that a new best model was saved
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement
        else:
            self.best_score = val_loss
            torch.save(model_dict, os.path.join(self.save_dir,"model_best_loss.pt"))
            #torch.save(model.state_dict(), self.save_dir + "/model_best_loss.pth")
            self.counter = 0
            return True  # Indicate that a new best model was saved

class EarlyStoppingWithValidity:
    def __init__(self, dir, patience, validity_weight=0.3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_validity = 0
        self.early_stop = False
        self.save_dir = dir
        self.validity_weight = validity_weight

    def __call__(self, val_loss, model_dict, generation_validity=0):
        # Combine loss and validity into a single score
        # Lower loss is better, higher validity is better
        combined_score = val_loss * (1 - self.validity_weight) - generation_validity * self.validity_weight
        
        if self.best_score is None or combined_score < self.best_score:
            self.best_score = combined_score
            self.best_validity = generation_validity
            torch.save(model_dict, os.path.join(self.save_dir, "model_best_combined.pt"))
            print(f"💾 New best model: loss={val_loss:.4f}, validity={generation_validity:.1%}")
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def train(dict_train_loader, global_step, monotonic_step, gradient_clip_threshold, epoch, total_epochs):
    # shuffle batches every epoch
    order_batches = list(range(len(dict_train_loader)))
    random.shuffle(order_batches)

    ce_losses = []
    total_losses = []
    kld_losses = []
    accs = []
    mses = []

    model.train()
    
    # Calculate teacher forcing ratio based on epoch
    # Start at 1.0 and gradually decrease to 0.5 over training
    teacher_forcing_ratio = max(0.5, 1.0 - (epoch / total_epochs) * 0.5)
    
    # Log teacher forcing ratio periodically
    if epoch % 10 == 0:
        print(f"📚 Teacher forcing ratio: {teacher_forcing_ratio:.3f}")
    
    # Iterate in batches over the training dataset.
    for i, batch in enumerate(order_batches):
        # CRITICAL FIX: Clear decoder state completely before each batch
        if hasattr(model, 'Decoder'):
            model.Decoder.clear_decoder_state_completely()
        
        # Clear any accumulated gradients
        if i % args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        
        if model_config['beta']=="schedule":
            # determine beta at time step t
            if global_step >= len(beta_schedule):
                beta_t = model.beta #stays the same
            else:
                beta_t = beta_schedule[global_step]
        
            model.beta = beta_t
        if model_config['alpha']=="schedule":
            # determine alpha at time step t
            if global_step >= len(alpha_schedule):
                alpha_t = model.alpha #stays the same
            else:
                alpha_t = alpha_schedule[global_step]
            model.alpha = alpha_t
        
        # get graphs & matrices for MP from dictionary
        data = dict_train_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_train_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_train_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)

        try:
            # FIXED: Handle both basic VAE and PP-guided VAE with teacher forcing
            # Pass teacher forcing ratio to the model
            result = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device, teacher_forcing_ratio=teacher_forcing_ratio)

            if len(result) == 7:  # Basic G2S_VAE
                loss, recon_loss, kl_loss, acc, predictions, target, z = result
                mse = torch.tensor(0.0, device=device)  # Dummy MSE
                y = torch.tensor(0.0, device=device)    # Dummy property prediction
            elif len(result) == 9:  # PP-guided VAE
                loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y = result
            else:
                raise ValueError(f"Unexpected number of return values from model: {len(result)}")
            
            # Check for unstable loss values before backpropagation
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"WARNING: NaN or Inf detected in loss at batch {i}")
                print(f"Loss: {loss.item()}, Recon: {recon_loss.item()}, KLD: {kl_loss.item()}")
                continue  # Skip this batch

            # Monitor and skip bad batches with extreme KLD
            if kl_loss.item() > 1000 or torch.isnan(loss).any():
                print(f"WARNING: Skipping batch {i} due to instability (KLD: {kl_loss.item():.2f})")
                optimizer.zero_grad()  # Clear any accumulated gradients
                continue

            # Check if KLD spike indicates instability
            if i > 0 and kl_loss.item() > args.kld_spike_threshold * np.mean(kld_losses[-min(10, len(kld_losses)):]):
                print(f"WARNING: KLD spike detected at batch {i}")
                print(f"Current KLD: {kl_loss.item()}, Recent mean: {np.mean(kld_losses[-min(10, len(kld_losses)):]):.2f}")

            # 🔥 CRITICAL FIX: Add gradient penalty to force gradients through context attention
            loss_with_penalty = add_gradient_penalty_to_loss(model, loss)
            # Add this debug print:
            if i == 0 and epoch % 10 == 0:  # First batch every 10 epochs
                print(f"🔍 DEBUG: Original loss: {loss.item():.6f}")
                print(f"🔍 DEBUG: Loss with penalty: {loss_with_penalty.item():.6f}")
                print(f"🔍 DEBUG: Penalty amount: {(loss_with_penalty - loss).item():.9f}")
                print(f"📚 DEBUG: Teacher forcing ratio: {teacher_forcing_ratio:.3f}")
            
            # Add validity reward after warmup
            if epoch > 20 and i % 20 == 0:  # Check periodically
                with torch.no_grad():
                    # Generate a few samples
                    z_sample = z[:min(5, z.size(0))]
                    try:
                        sample_preds = model.Decoder.inference(z_sample, temperature=0.8)
                        
                        # Calculate validity reward
                        validity_reward = 0.0
                        for pred in sample_preds:
                            tokens = tokenids_to_vocab(pred[0], vocab)
                            smiles = combine_tokens(tokens, tokenization="RT_tokenized")
                            
                            # Format validity rewards
                            if len(smiles) > 20 and '|' in smiles:
                                validity_reward += 0.02
                            if smiles.count('|') >= 3:
                                validity_reward += 0.02
                            if '[*:' in smiles and '<' in smiles:
                                validity_reward += 0.02
                        
                        # Apply as bonus (negative loss)
                        validity_reward = validity_reward / len(sample_preds)
                        if validity_reward > 0:
                            validity_bonus = torch.tensor(validity_reward * 0.1, device=device, requires_grad=False)
                            loss_with_penalty = loss_with_penalty - validity_bonus
                            if i == 0:  # Log first batch
                                print(f"   💎 Validity reward: {validity_reward:.4f}")
                    except:
                        pass  # Ignore errors during validity check
            
            loss_with_penalty.backward()
            
            # Monitor gradient norms before clipping
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            if total_grad_norm > 10.0:  # Warning threshold
                print(f"WARNING: Large gradient norm detected: {total_grad_norm:.2f}")
            
            # Use configurable gradient clipping threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_threshold)

            # Step optimizer only after accumulating gradients
            if (i + 1) % args.accumulate_grad_batches == 0:
                optimizer.step()
            
            # Store metrics
            ce_losses.append(recon_loss.item())
            total_losses.append(loss.item())
            kld_losses.append(kl_loss.item())
            accs.append(acc.item())
            mses.append(mse.item())
            
            # Log semi-supervised info for first batch of first epoch
            if args.training_stage == 1 and epoch == epoch_cp and i == 0:
                print("\n📊 Semi-supervised Stage 1 - First batch analysis:")
                print(f"   Reconstruction loss applied to: ALL {data.num_graphs} molecules")
                print(f"   KLD loss applied to: ALL {data.num_graphs} molecules")
                print(f"   Teacher forcing ratio: {teacher_forcing_ratio:.3f}")
                # Count molecules with valid properties
                valid_props = 0
                if hasattr(data, 'y1') and hasattr(data, 'y2'):
                    for idx in range(data.num_graphs):
                        if idx < len(data.y1) and idx < len(data.y2):
                            if not torch.isnan(data.y1[idx]) and not torch.isnan(data.y2[idx]):
                                valid_props += 1
                print(f"   Property loss (MSE) applied to: {valid_props} molecules with EA/IP labels")
                print(f"   Unlabeled molecules in batch: {data.num_graphs - valid_props}")
            
        except RuntimeError as e:
            if "backward through the graph a second time" in str(e):
                print(f"WARNING: Graph retention error at batch {i}, skipping batch")
                # Add dummy metrics to maintain consistency
                if len(ce_losses) > 0:
                    ce_losses.append(ce_losses[-1])
                    total_losses.append(total_losses[-1])
                    kld_losses.append(kld_losses[-1])
                    accs.append(accs[-1])
                    mses.append(mses[-1])
                else:
                    ce_losses.append(0.0)
                    total_losses.append(0.0)
                    kld_losses.append(0.0)
                    accs.append(0.0)
                    mses.append(0.0)
                continue
            else:
                raise e
        
        # NEW: Add gradient monitoring here
        if i % 50 == 0:  # Only print every 50 batches to avoid spam
            print("\n📊 Decoder gradient norms:")
            for name, param in model.named_parameters():
                if param.grad is not None and 'decoder' in name.lower():
                    print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
        
        if i % 10 == 0:
            print(f"\nBatch [{i:4d} / {len(order_batches):4d}]")
            print("-" * 70)
            # NEW (FIXED):
            alpha_str = f"{model.alpha:.6f}" if hasattr(model, 'alpha') else "N/A"
            print(f"Recon: {ce_losses[-1]:.6f} | Total: {total_losses[-1]:.6f} | KLD: {kld_losses[-1]:.6f} | Acc: {accs[-1]:.6f} | MSE: {mses[-1]:.6f} | Beta: {model.beta:.6f} | Alpha: {alpha_str}")
            print("-" * 70)
            
        global_step += 1
        
    return model, ce_losses, total_losses, kld_losses, accs, mses, global_step, monotonic_step


def test(dict_loader):
    batches = list(range(len(dict_loader)))
    ce_losses = []
    total_losses = []
    kld_losses = []
    accs = []
    mses = []

    model.eval()
    test_loss = 0
    # Iterate in batches over the training/test dataset.
    with torch.no_grad():
        for batch in batches:
            data = dict_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)

            # FIXED: Handle both basic VAE and PP-guided VAE
            result = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

            if len(result) == 7:  # Basic G2S_VAE
                loss, recon_loss, kl_loss, acc, predictions, target, z = result
                mse = torch.tensor(0.0, device=device)  # Dummy MSE
                y = torch.tensor(0.0, device=device)    # Dummy property prediction
            elif len(result) == 9:  # PP-guided VAE
                loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y = result
            else:
                raise ValueError(f"Unexpected number of return values from model: {len(result)}")

            ce_losses.append(recon_loss.item())
            total_losses.append(loss.item())
            kld_losses.append(kl_loss.item())
            accs.append(acc.item())
            mses.append(mse.item())
        
    return ce_losses, total_losses, kld_losses, accs, mses

def save_epoch_metrics_to_csv(epoch, train_metrics, val_metrics, directory_path, resume_from_checkpoint=False, generation_validity=None, rdkit_validity=None):
    csv_file = os.path.join(directory_path, 'training_log.csv')
    flag_file = os.path.join(directory_path, '.csv_initialized')
    
    # For fresh training (not resuming), reset the CSV file once at the beginning
    if not resume_from_checkpoint and not os.path.exists(flag_file):
        mode = 'w'  # Write mode (overwrite)
        print(f"[INFO] Fresh training — resetting log: {csv_file}")
        # Create flag file to mark that we've initialized the CSV for this training run
        with open(flag_file, 'w') as f:
            f.write(str(time.time()))
    else:
        # Either resuming or not the first time writing to the CSV in this run
        mode = 'a'  # Append mode
    
    # Write to the CSV file
    with open(csv_file, mode, newline='') as f:
        writer = csv.writer(f)
        # Write header only if we're in write mode or the file doesn't exist yet
        if mode == 'w' or not os.path.exists(csv_file):
            writer.writerow([
                'epoch', 'train_loss_mean', 'train_kld_mean', 'train_acc_mean', 'train_mse_mean',
                'val_loss_mean', 'val_kld_mean', 'val_acc_mean', 'val_mse_mean', 
                'generation_validity', 'rdkit_validity'  # Added rdkit_validity
            ])
        # Always write the data row
        writer.writerow([
            epoch,
            train_metrics['loss'], train_metrics['kld'], train_metrics['acc'], train_metrics['mse'],
            val_metrics['loss'], val_metrics['kld'], val_metrics['acc'], val_metrics['mse'],
            generation_validity if generation_validity is not None else 0.0,
            rdkit_validity if rdkit_validity is not None else 0.0  # Added
        ])

def load_existing_loss_dicts(directory_path):
    """Load existing loss dictionaries if they exist"""
    train_loss_file = os.path.join(directory_path, 'train_loss.pkl')
    val_loss_file = os.path.join(directory_path, 'val_loss.pkl')
    
    train_loss_dict = {}
    val_loss_dict = {}
    
    if os.path.exists(train_loss_file):
        try:
            with open(train_loss_file, 'rb') as f:
                train_loss_dict = pickle.load(f)
            print(f"[INFO] Loaded existing train_loss.pkl with {len(train_loss_dict)} epochs")
        except Exception as e:
            print(f"[WARNING] Could not load train_loss.pkl: {e}")
    
    if os.path.exists(val_loss_file):
        try:
            with open(val_loss_file, 'rb') as f:
                val_loss_dict = pickle.load(f)
            print(f"[INFO] Loaded existing val_loss.pkl with {len(val_loss_dict)} epochs")
        except Exception as e:
            print(f"[WARNING] Could not load val_loss.pkl: {e}")
    
    return train_loss_dict, val_loss_dict

def save_loss_dicts(train_loss_dict, val_loss_dict, directory_path):
    """Save loss dictionaries with error handling"""
    try:
        with open(os.path.join(directory_path, 'train_loss.pkl'), 'wb') as file:
            pickle.dump(train_loss_dict, file)
        print(f"[INFO] Saved train_loss.pkl with {len(train_loss_dict)} epochs")
    except Exception as e:
        print(f"[ERROR] Could not save train_loss.pkl: {e}")
    
    try:
        with open(os.path.join(directory_path, 'val_loss.pkl'), 'wb') as file:
            pickle.dump(val_loss_dict, file)
        print(f"[INFO] Saved val_loss.pkl with {len(val_loss_dict)} epochs")
    except Exception as e:
        print(f"[ERROR] Could not save val_loss.pkl: {e}")

def validate_generation_quality(model, vocab, device, num_samples=100):
    """Test generation quality with both format and chemical validity"""
    model.eval()
    format_valid_count = 0
    rdkit_valid_count = 0
    
    # Import RDKit if available
    try:
        from rdkit import Chem
        import re
        rdkit_available = True
    except ImportError:
        rdkit_available = False
        print("Warning: RDKit not available for chemical validation")
    
    with torch.no_grad():
        try:
            z_random = torch.randn(num_samples, model.embedding_dim, device=device)
            result = model.inference(data=z_random, device=device, sample=False, log_var=None)
            
            if len(result) >= 4:
                predictions, _, _, _ = result[:4]
            else:
                predictions = result[0] if result else None
            
            if predictions is None:
                return 0.0, 0.0
            
            for i in range(min(len(predictions), num_samples)):
                try:
                    if len(predictions[i]) > 0 and hasattr(predictions[i][0], '__iter__'):
                        pred_tokens = predictions[i][0]
                        smiles_string = safe_token_processing(pred_tokens, vocab, "RT_tokenized")
                        
                        # Format validity check
                        if (len(smiles_string) > 10 and 
                            '|' in smiles_string and
                            '[*:' in smiles_string and
                            smiles_string.count('(') == smiles_string.count(')') and
                            smiles_string.count('[') == smiles_string.count(']')):
                            format_valid_count += 1
                            
                            # Chemical validity check with RDKit
                            if rdkit_available and '|' in smiles_string:
                                smiles_part = smiles_string.split('|')[0]
                                try:
                                    if '.' in smiles_part:
                                        # Copolymer
                                        monomers = smiles_part.split('.')
                                        all_valid = True
                                        for monomer in monomers:
                                            if monomer.strip():
                                                # CRITICAL FIX: Replace [*:n] with * for RDKit
                                                monomer_clean = re.sub(r'\[\*:\d+\]', '*', monomer)
                                                mol = Chem.MolFromSmiles(monomer_clean)
                                                if mol is None:
                                                    all_valid = False
                                                    break
                                        if all_valid:
                                            rdkit_valid_count += 1
                                    else:
                                        # Single monomer
                                        # CRITICAL FIX: Replace [*:n] with * for RDKit
                                        smiles_clean = re.sub(r'\[\*:\d+\]', '*', smiles_part)
                                        mol = Chem.MolFromSmiles(smiles_clean)
                                        if mol is not None:
                                            rdkit_valid_count += 1
                                except:
                                    pass
                except:
                    continue
                    
        except Exception as e:
            print(f"Generation validation failed: {e}")
            return 0.0, 0.0
    
    format_validity = format_valid_count / num_samples if num_samples > 0 else 0.0
    rdkit_validity = rdkit_valid_count / num_samples if num_samples > 0 else 0.0
    
    model.train()
    return format_validity, rdkit_validity

def enhanced_validate_generation(model, vocab, device, num_samples=200):
    """Multi-level validation with detailed metrics"""
    model.eval()
    results = {
        'format_valid': 0,
        'chemical_valid': 0,
        'novel': 0,
        'unique_count': 0,
        'complete_g2s': 0,
        'avg_length': 0,
        'examples': []
    }
    
    generated_smiles = []
    unique_smiles = set()
    
    # Import RDKit if available
    try:
        from rdkit import Chem
        import re
        rdkit_available = True
    except ImportError:
        rdkit_available = False
    
    with torch.no_grad():
        try:
            # Try different generation strategies
            for i in range(num_samples):
                # Vary temperature for diversity
                temperature = 0.8 if i % 2 == 0 else 1.0
                
                # Generate
                z = torch.randn(1, model.embedding_dim, device=device)
                
                # Try constrained beam search for some samples
                if hasattr(model.Decoder, 'constrained_beam_search') and i % 3 == 0:
                    predictions = model.Decoder.constrained_beam_search(z, beam_size=3, temperature=temperature)
                    pred_tokens = predictions[0]
                else:
                    result = model.inference(data=z, device=device, sample=(i % 2 == 0))
                    predictions = result[0]
                    if predictions and len(predictions) > 0:
                        pred_tokens = predictions[0][0] if hasattr(predictions[0], '__iter__') else predictions[0]
                    else:
                        continue
                
                # Process tokens
                smiles_string = safe_token_processing(pred_tokens, vocab, "RT_tokenized")
                
                # Apply post-generation fixes
                if hasattr(model.Decoder, 'fix_polymer_format'):
                    smiles_string = model.Decoder.fix_polymer_format(smiles_string)
                
                generated_smiles.append(smiles_string)
                unique_smiles.add(smiles_string)
                
                # Track length
                results['avg_length'] += len(smiles_string)
                
                # Save examples
                if i < 5:
                    results['examples'].append(smiles_string)
                
                # Check format validity
                format_checks = [
                    len(smiles_string) > 20,
                    '|' in smiles_string,
                    smiles_string.count('|') >= 3,
                    '[*:' in smiles_string,
                    smiles_string.count('(') == smiles_string.count(')'),
                    smiles_string.count('[') == smiles_string.count(']'),
                    '<' in smiles_string,  # Has connectivity
                    ':' in smiles_string.split('|')[-1] if '|' in smiles_string else False
                ]
                
                if all(format_checks):
                    results['format_valid'] += 1
                    
                    # Check G2S completeness
                    if smiles_string.count('|') == 3 and '-' in smiles_string:
                        results['complete_g2s'] += 1
                    
                    # Check chemical validity
                    if rdkit_available and '|' in smiles_string:
                        smiles_part = smiles_string.split('|')[0]
                        try:
                            if '.' in smiles_part:
                                # Copolymer
                                monomers = smiles_part.split('.')
                                all_valid = True
                                for monomer in monomers:
                                    if monomer.strip():
                                        monomer_clean = re.sub(r'\[\*:\d+\]', '*', monomer)
                                        mol = Chem.MolFromSmiles(monomer_clean)
                                        if mol is None:
                                            all_valid = False
                                            break
                                if all_valid:
                                    results['chemical_valid'] += 1
                            else:
                                # Single monomer
                                smiles_clean = re.sub(r'\[\*:\d+\]', '*', smiles_part)
                                mol = Chem.MolFromSmiles(smiles_clean)
                                if mol is not None:
                                    results['chemical_valid'] += 1
                        except:
                            pass
            
            # Calculate final metrics
            results['unique_count'] = len(unique_smiles)
            results['avg_length'] = results['avg_length'] / num_samples if num_samples > 0 else 0
            
            # Convert counts to rates
            results['format_valid_rate'] = results['format_valid'] / num_samples
            results['chemical_valid_rate'] = results['chemical_valid'] / num_samples
            results['complete_g2s_rate'] = results['complete_g2s'] / num_samples
            results['uniqueness'] = results['unique_count'] / num_samples
            
        except Exception as e:
            print(f"Enhanced validation failed: {e}")
            results['format_valid_rate'] = 0
            results['chemical_valid_rate'] = 0
    
    model.train()
    return results

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0, n_cycle=5, ratio_increase=0.5, ratio_zero=0.3):
    """Beta scheduling function from Optimus paper"""
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L

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
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument("--scheduler_patience", type=int, default=10, help="Patience for learning rate scheduler")
parser.add_argument("--es_patience", type=int, default=5, help="Patience for early stopping")
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.1)
parser.add_argument("--max_alpha", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--validation_samples", type=int, default=100, 
                    help="Number of samples to generate for validity checking")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume training from")
parser.add_argument("--save_dir", type=str, default=None, help="Custom directory to save model checkpoints")

# Add flexible property arguments (same as BO and GA scripts)
parser.add_argument("--property_names", type=str, nargs='+', default=["bandgap"],
                    help="Names of the properties to train the model to predict")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Path to custom dataset files (will use default naming pattern if not specified)")

# NEW: Anti-overfitting and training stability arguments
parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for regularization")
parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization weight decay")
parser.add_argument("--gradient_clip_threshold", type=float, default=0.5, help="Gradient clipping threshold")
parser.add_argument("--lr_decay_factor", type=float, default=0.5, help="Learning rate decay factor for scheduler")
parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for scheduler")
parser.add_argument("--validation_freq", type=int, default=1, help="Run validation every N epochs")
parser.add_argument("--checkpoint_freq", type=int, default=5, help="Save checkpoint every N epochs")
parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs for AE warmup when AE_Warmup is True")
parser.add_argument("--max_grad_norm_warning", type=float, default=10.0, help="Threshold for gradient norm warning")
parser.add_argument("--kld_spike_threshold", type=float, default=5.0, help="Threshold multiplier for KLD spike detection")
parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Number of batches to accumulate gradients")

# Transfer learning arguments
parser.add_argument("--training_stage", type=int, default=1, choices=[1, 2],
                    help="Stage 1: Pretrain on all data, Stage 2: Fine-tune on target property")
parser.add_argument("--pretrained_model_path", type=str, default=None,
                    help="Path to pretrained model for stage 2")
parser.add_argument("--source_properties", type=str, nargs='+', default=["EA", "IP"],
                    help="Properties used in stage 1 pretraining")
parser.add_argument("--target_properties", type=str, nargs='+', default=["bandgap"],
                    help="Properties to predict in stage 2")
parser.add_argument("--freeze_encoder", action="store_true", default=False,
                    help="Freeze encoder during stage 2")
parser.add_argument("--freeze_decoder", action="store_true", default=False,
                    help="Freeze decoder during stage 2")
parser.add_argument("--stage1_sample_weight", type=float, default=1.0,
                    help="Weight for sampling data with source properties in stage 1")
parser.add_argument("--combined_dataset_path", type=str, 
                    default="/content/drive/MyDrive/X_Materials_Organized_Files_V1/combined_polymer_database_2.csv",
                    help="Path to combined dataset CSV")

args = parser.parse_args()

# Handle property configuration for basic VAE
if not args.ppguided:
    # Force no properties for basic VAE
    property_names = []
    property_count = 0
    print(f"Using basic VAE (no property prediction)")
else:
    # Use provided properties for PP-guided VAE
    property_names = args.property_names
    if args.property_count is not None:
        property_count = args.property_count
    else:
        property_count = len(property_names)
        
    # Validate that property count matches property names
    if len(property_names) != property_count:
        raise ValueError(f"Number of property names ({len(property_names)}) must match property count ({property_count})")

print(f"Training model to predict {property_count} properties: {property_names}")

# Define resume_from_checkpoint as a boolean for logic control
resume_from_checkpoint = args.resume_from_checkpoint is not None and os.path.exists(args.resume_from_checkpoint)

# First set the seed for reproducible results
seed = args.seed
torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#torch.cuda.manual_seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
#random.seed(seed)
#np.random.seed(seed)

augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

# Model config and vocab
if args.dataset_path:
    # Use custom dataset path
    vocab_file_path = os.path.join(args.dataset_path, f'poly_smiles_vocab_{augment}_{tokenization}.txt')
    data_path_prefix = os.path.join(args.dataset_path, f'dict_{{}}_loader_{augment}_{tokenization}.pt')
else:
    # Use default paths
    vocab_file_path = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    data_path_prefix = main_dir_path+'/data/dict_{}_loader_'+augment+'_'+tokenization+'.pt'

# QUICK FIX: Add debugging and validation before model creation
print("🔧 APPLYING EMBEDDING FIX...")
vocab = debug_vocab_and_embeddings(vocab_file_path, args.dataset_path)

# Ensure vocabulary consistency
print(f"Vocabulary size check: {len(vocab)}")
assert len(vocab) > 0, "Vocabulary is empty!"

# Check if special tokens exist
required_tokens = ['_PAD', '_SOS', '_EOS', '_UNK']
for token in required_tokens:
    if token not in vocab:
        print(f"WARNING: {token} missing from vocabulary")

print("✅ Pre-creation checks passed")

print(f"DEBUG: Constructed vocab file path: {vocab_file_path}")
print(f"DEBUG: File exists: {os.path.exists(vocab_file_path)}")
print(f"DEBUG: Current working directory: {os.getcwd()}")
print(f"DEBUG: Absolute path: {os.path.abspath(vocab_file_path)}")
print(f"\n🔍 DEBUG: Vocabulary Analysis")
print(f"Vocab size: {len(vocab)}")
print(f"First 10 tokens: {list(vocab.items())[:10]}")
print(f"Last 10 tokens: {list(vocab.items())[-10:]}")
print(f"Special tokens: _PAD={vocab.get('_PAD', 'MISSING')}, _SOS={vocab.get('_SOS', 'MISSING')}, _EOS={vocab.get('_EOS', 'MISSING')}, _UNK={vocab.get('_UNK', 'MISSING')}")
print()


model_config = {
    "embedding_dim": args.embedding_dim, # latent dimension needs to be embedding dimension of word vectors
    "beta": args.beta,
    "max_beta":args.max_beta,
    "epsilon":args.epsilon,
    "decoder_num_layers": args.dec_layers,
    "num_attention_heads":4,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'hidden_dimension': 300, #hidden dimension of nodes
    'n_nodes_pool': 10, #how many representative nodes are used for attention based pooling
    'pooling': 'mean', #mean or custom
    'learning_rate': args.learning_rate,
    'es_patience': args.es_patience,
    'loss': args.loss, # focal or ce
    'max_alpha': args.max_alpha,
    'alpha': args.alpha,
    # Add property configuration to model config
    'property_count': property_count,
    'property_names': property_names,
    # Add new regularization parameters
    'dropout_rate': args.dropout_rate,
    'weight_decay': args.weight_decay
}
batch_size = model_config['batch_size']
epochs = model_config['epochs']
hidden_dimension = model_config['hidden_dimension']
embedding_dim = model_config['embedding_dim']
loss = model_config['loss']

# %% Call data
if args.training_stage == 1:
    # Stage 1: Load all available data
    print(f"Stage 1: Loading combined dataset for pretraining on {args.source_properties}")
    
    # Use safe data loading
    dict_train_loader, dict_val_loader, dict_test_loader = load_transfer_data_safely(
        csv_path=args.combined_dataset_path,
        stage=1,
        source_properties=args.source_properties,
        target_properties=args.target_properties,
        batch_size=batch_size,
        tokenization=tokenization,
        vocab=vocab,
        sample_weight=args.stage1_sample_weight,
        device=device
    )
    
    # Update property names for stage 1
    property_names = args.source_properties
    property_count = len(property_names)
    
else:  # Stage 2
    # Stage 2: Load only target property data
    print(f"Stage 2: Loading data for fine-tuning on {args.target_properties}")
    
    dict_train_loader, dict_val_loader, dict_test_loader = load_transfer_data_safely(
        csv_path=args.combined_dataset_path,
        stage=2,
        source_properties=args.source_properties,
        target_properties=args.target_properties,
        batch_size=batch_size,
        tokenization=tokenization,
        vocab=vocab,
        device=device
    )
    
    # Update property names for stage 2
    property_names = args.target_properties
    property_count = len(property_names)

# CRITICAL FIX: Verify data/vocab consistency AFTER loading data
try:
    first_batch = dict_train_loader['0'][0]
    if hasattr(first_batch, 'tgt_token_ids'):
        max_token_id = max(max(seq) for seq in first_batch.tgt_token_ids)
        if max_token_id >= len(vocab):
            raise ValueError(f"Data contains token ID {max_token_id} but vocab only has {len(vocab)} tokens")
        print(f"✅ Data/vocab consistency verified. Max token ID: {max_token_id}, Vocab size: {len(vocab)}")
except Exception as e:
    print(f"WARNING: Could not validate data consistency: {e}")

num_train_graphs = len(list(dict_train_loader.keys())[
    :-2])*batch_size + dict_train_loader[list(dict_train_loader.keys())[-1]][0].num_graphs
num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

assert dict_train_loader['0'][0].num_graphs == batch_size, 'Batch_sizes of data and model do not match'

# %% Create an instance of the G2S model
# only for wce loss we calculate the token weights from vocabulary
if model_config['loss']=="wce":
    class_weights = token_weights(vocab_file_path)
    class_weights = torch.FloatTensor(class_weights)
if model_config['loss']=="ce":
    class_weights=None

def reset_context_attention(model):
    """Reset only the problematic context attention components"""
    import torch.nn as nn
    
    reset_count = 0
    for layer in model.Decoder.Decoder.transformer_layers:
        if hasattr(layer, 'context_attn'):
            # Reset the dead components
            nn.init.xavier_uniform_(layer.context_attn.linear_keys.weight)
            nn.init.xavier_uniform_(layer.context_attn.linear_query.weight)
            if hasattr(layer, 'layer_norm_2'):
                nn.init.ones_(layer.layer_norm_2.weight)
                nn.init.zeros_(layer.layer_norm_2.bias)
            reset_count += 1
    
    print(f"🔧 Reset context attention in {reset_count} transformer layers")
    print("🎯 This should wake up the dead context attention components!")
    return model

def fix_dead_attention(model):
    """More aggressive fix for dead attention"""
    import torch.nn as nn
    
    for i, layer in enumerate(model.Decoder.Decoder.transformer_layers):
        if hasattr(layer, 'context_attn'):
            # Use different initialization for problematic layers
            if i >= 2:  # Layers 2 and 3 seem most affected
                nn.init.xavier_normal_(layer.context_attn.linear_keys.weight, gain=2.0)
                nn.init.xavier_normal_(layer.context_attn.linear_query.weight, gain=2.0)
                nn.init.xavier_normal_(layer.context_attn.linear_values.weight, gain=1.5)
            
            # Add small random bias to prevent symmetry
            if hasattr(layer.context_attn, 'linear_keys') and hasattr(layer.context_attn.linear_keys, 'bias'):
                if layer.context_attn.linear_keys.bias is not None:
                    nn.init.normal_(layer.context_attn.linear_keys.bias, 0, 0.01)
    
    print(f"🔧 Applied aggressive fix for dead attention layers")
    return model

def add_gradient_penalty_to_loss(model, loss):
    """Add penalty to force gradients through context attention keys/queries"""
    penalty = 0.0
    
    # Add small penalty based on context attention weights
    for layer in model.Decoder.Decoder.transformer_layers:
        if hasattr(layer, 'context_attn'):
            # Force gradient flow by adding norm of weights to loss
            if hasattr(layer.context_attn, 'linear_keys'):
                penalty += layer.context_attn.linear_keys.weight.norm() * 1e-7
            if hasattr(layer.context_attn, 'linear_query'):
                penalty += layer.context_attn.linear_query.weight.norm() * 1e-7
    
    return loss + penalty

def add_attention_regularization(model):
    """Add regularization to encourage non-uniform attention"""
    print("\n🎯 ADDING ATTENTION REGULARIZATION...")
    
    # Hook to add regularization loss
    def attention_hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights = output[1]  # Usually the attention weights are the second output
            
            # Encourage non-uniform attention by penalizing uniform distributions
            if attention_weights is not None and attention_weights.numel() > 0:
                # Calculate entropy of attention weights
                entropy = -(attention_weights * (attention_weights + 1e-10).log()).sum(dim=-1).mean()
                
                # Add a small penalty to the loss to discourage high entropy (uniform attention)
                if hasattr(module, '_attention_penalty'):
                    module._attention_penalty = entropy * 0.01
                else:
                    module._attention_penalty = entropy * 0.01
    
    # Register hooks
    for layer in model.Decoder.Decoder.transformer_layers:
        if hasattr(layer, 'context_attn'):
            layer.context_attn.register_forward_hook(attention_hook)
    
    print("✅ Attention regularization added!")
    return model

# %% Create an instance of the G2S model with safe creation
if args.training_stage == 1:
    # Stage 1: Regular training
    if args.ppguided:
        model_type = G2S_VAE_PPguided
    else:
        model_type = G2S_VAE
    
    model = safe_model_creation(
        model_type,
        num_node_features, num_edge_features, hidden_dimension, 
        embedding_dim, device, model_config, vocab, seed, 
        loss_weights=class_weights, add_latent=add_latent
    )
else:
    # Stage 2: Transfer learning
    print(f"Loading pretrained model from: {args.pretrained_model_path}")
    
    # Load pretrained model
    pretrained_checkpoint = torch.load(args.pretrained_model_path)
    pretrained_config = pretrained_checkpoint['model_config']
    
    # Create pretrained model
    if args.ppguided:
        pretrained_model_type = G2S_VAE_PPguided
    else:
        pretrained_model_type = G2S_VAE
    
    pretrained_model = safe_model_creation(
        pretrained_model_type,
        num_node_features, num_edge_features, hidden_dimension,
        embedding_dim, device, pretrained_config, vocab, seed,
        loss_weights=class_weights, add_latent=add_latent
    )
    pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])
    
    # Update model config for transfer learning
    model_config['source_properties'] = args.source_properties
    model_config['target_properties'] = args.target_properties
    
    # Create transfer model
    model = safe_model_creation(
        G2S_VAE_Transfer,
        num_node_features, num_edge_features, hidden_dimension,
        embedding_dim, device, model_config, vocab, seed,
        loss_weights=class_weights, add_latent=add_latent,
        pretrained_model=pretrained_model,
        freeze_components={'encoder': args.freeze_encoder, 'decoder': args.freeze_decoder}
    )

model.to(device)

# CRITICAL: Add comprehensive validation after model creation
try:
    validate_model_configuration(model, vocab, dict_train_loader)
except Exception as e:
    print(f"❌ Model validation failed: {e}")
    
    # Run specific debugging for Elementwise errors
    print(f"\n🔧 Running specialized Elementwise debugging...")
    debug_elementwise_error(model, vocab, dict_train_loader)
    
    # Stop here so we can fix the issue
    raise

# Validate vocabulary size matches model
print(f"\n🔍 Model Validation:")
print(f"Output layer size: {model.Decoder.output_layer.out_features}")
print(f"Vocabulary size: {len(vocab)}")
assert model.Decoder.output_layer.out_features == len(vocab), \
    f"Output layer size mismatch! Model has {model.Decoder.output_layer.out_features} but vocab has {len(vocab)}"
print("✅ Model vocabulary size validated!\n")

# 🔧 CRITICAL FIX: Reset dead context attention components
model = reset_context_attention(model)

# 🔧 Apply more aggressive fix for dead attention
model = fix_dead_attention(model)

# 🎯 NEW: Add attention regularization
model = add_attention_regularization(model)

print(model)

# Use configurable warmup epochs
n_iter = int(20 * num_train_graphs/batch_size) # 20 epochs
# Beta scheduling function from Optimus paper 
# Beta scheduling function from Optimus paper 
def two_stage_beta_schedule(n_iter, warmup_ratio=0.4, max_beta=None):
    """Two-stage beta: low for reconstruction, then increase for generation"""
    if max_beta is None:
        max_beta = model_config['max_beta']
    
    warmup_steps = int(n_iter * warmup_ratio)
    
    # Stage 1: Gradual increase from very low (not constant!)
    stage1 = np.linspace(0.0, max_beta * 0.2, warmup_steps)
    
    # Stage 2: Cyclical increase to full beta
    remaining_steps = n_iter - warmup_steps
    stage2 = frange_cycle_zero_linear(
        remaining_steps, 
        start=max_beta * 0.2, 
        stop=max_beta,  # Use full max_beta, not 10x
        n_cycle=4,
        ratio_increase=0.6,
        ratio_zero=0.1
    )
    
    return np.concatenate([stage1, stage2])

def smoother_beta_schedule(n_iter, warmup_epochs, max_beta=0.5):
    """Smoother transition after warmup"""
    warmup_steps = warmup_epochs * (num_train_graphs / batch_size)
    
    # Very gradual increase after warmup
    warmup = np.zeros(int(warmup_steps))
    
    # Smooth sigmoid transition instead of sharp increase
    transition_steps = int(n_iter * 0.2)  # 20% for transition
    transition = max_beta / (1 + np.exp(-10 * (np.arange(transition_steps) - transition_steps/2) / transition_steps))
    
    # Rest of training
    remaining = int(n_iter - warmup_steps - transition_steps)
    if remaining > 0:
        rest = frange_cycle_zero_linear(remaining, start=max_beta*0.8, stop=max_beta, n_cycle=3)
    else:
        rest = np.array([])
    
    return np.concatenate([warmup, transition, rest])


# Beta schedule initialization
if model_config['beta'] == "schedule":
    if args.AE_Warmup:
        # Use smoother schedule when AE warmup is enabled
        beta_schedule = smoother_beta_schedule(n_iter=n_iter, warmup_epochs=args.warmup_epochs, max_beta=model_config['max_beta'])
    else:
        beta_schedule = two_stage_beta_schedule(n_iter=n_iter, max_beta=model_config['max_beta'])
elif model_config['beta'] == "normalVAE":
    beta_schedule = np.ones(1)

if model_config['alpha'] == "schedule":
    alpha_schedule = frange_cycle_zero_linear(n_iter=n_iter, start=0.0, stop=model_config['max_alpha'], n_cycle=5, ratio_increase=0.5, ratio_zero=0.3)
elif model_config['alpha'] == "fixed":
    alpha_schedule = np.ones(1)

# %%# %% Train

# Enhanced optimizer with weight decay
if args.training_stage == 1:
    # Stage 1: Normal optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    # Stage 2: Different learning rates for different components
    param_groups = [
        {'params': model.PP_lin1.parameters(), 'lr': args.learning_rate},
        {'params': model.PP_lin2.parameters(), 'lr': args.learning_rate}
    ]
    
    if not args.freeze_encoder:
        param_groups.append({'params': model.Encoder.parameters(), 'lr': args.learning_rate * 0.1})
    
    if not args.freeze_decoder:
        param_groups.append({'params': model.Decoder.parameters(), 'lr': args.learning_rate * 0.01})
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    print(f"Stage 2 optimizer with differential learning rates created")

# Enhanced learning rate scheduler with configurable parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_factor, 
                              patience=args.scheduler_patience, verbose=True, min_lr=args.min_lr)

# Add learning rate warmup
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(epoch):
    if epoch < args.warmup_epochs:
        return (epoch + 1) / args.warmup_epochs
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda)

# Early stopping callback
# Log directory creation

data_augment="old"
# Include property info in model name for better organization
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+str(property_str)+'/'

# Always use model_name in the path structure, regardless of save_dir
if args.save_dir is not None:
    # Create the expected directory structure inside save_dir
    directory_path = os.path.join(args.save_dir, model_name)
else:
    directory_path = os.path.join(main_dir_path,'Checkpoints/', model_name)

# Create directory with all parent directories
os.makedirs(directory_path, exist_ok=True)
print(f"✅ Checkpoint directory created/verified: {directory_path}")

es_patience = model_config['es_patience']
# Keep BOTH early stopping mechanisms
# Traditional early stopping for best loss
earlystopping_loss = EarlyStopping(dir=directory_path, patience=es_patience)
# Combined early stopping for best combined metric
earlystopping = EarlyStoppingWithValidity(dir=directory_path, patience=es_patience, validity_weight=0.3)

print(f'STARTING TRAINING')
print(f'Model will predict {property_count} properties: {property_names}')
print(f'Enhanced features: dropout_rate={args.dropout_rate}, weight_decay={args.weight_decay}, gradient_clip={args.gradient_clip_threshold}')

# Prepare dictionaries for training or load checkpoint
checkpoint_file = None

# ------------------ Resume checkpoint logic ------------------
checkpoint_file = None

if args.resume_from_checkpoint:
    if os.path.exists(args.resume_from_checkpoint):
        checkpoint_file = args.resume_from_checkpoint
        print(f"[INFO] Resuming training from checkpoint: {checkpoint_file}")
        resume_from_checkpoint = True
    else:
        print(f"[WARNING] Checkpoint path not found: {args.resume_from_checkpoint}. Starting from scratch.")
        resume_from_checkpoint = False
else:
    print("[INFO] No checkpoint specified. Starting from scratch.")
    resume_from_checkpoint = False

# ENHANCED: Proper loss dictionary handling for resuming
if resume_from_checkpoint:
    # Load existing loss dictionaries first (most up-to-date)
    train_loss_dict, val_loss_dict = load_existing_loss_dicts(directory_path)
    
    # Load checkpoint
    print(f"Loading model from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_cp = checkpoint['epoch']
    
    # Merge checkpoint loss dicts with existing ones (checkpoint might be outdated)
    if 'loss_dict' in checkpoint and 'val_loss_dict' in checkpoint:
        checkpoint_train_dict = checkpoint['loss_dict']
        checkpoint_val_dict = checkpoint['val_loss_dict']
        
        # Update with checkpoint data (but don't overwrite newer data)
        for epoch_key in checkpoint_train_dict:
            if epoch_key not in train_loss_dict:
                train_loss_dict[epoch_key] = checkpoint_train_dict[epoch_key]
        
        for epoch_key in checkpoint_val_dict:
            if epoch_key not in val_loss_dict:
                val_loss_dict[epoch_key] = checkpoint_val_dict[epoch_key]
        
        print(f"[INFO] Merged checkpoint loss data. Total epochs in train_loss_dict: {len(train_loss_dict)}")
    
    if model_config['beta'] == "schedule":
        global_step = checkpoint.get('global_step', 0)
        monotonic_step = checkpoint.get('monotonic_step', 0)
        model.beta = model_config['max_beta']
else:
    # Fresh training - reset everything
    train_loss_dict = {}
    val_loss_dict = {}
    epoch_cp = 0
    global_step = 0
    monotonic_step = 0
    
    # Reset CSV flag if starting from scratch
    flag_file = os.path.join(directory_path, '.csv_initialized')
    if os.path.exists(flag_file):
        print("[INFO] Removing old .csv_initialized to allow clean training log overwrite.")
        os.remove(flag_file)

for epoch in range(epoch_cp, epochs):
    print(f"Epoch {epoch + 1}\n" + "-" * 30)

    t1 = time.time()
    model, train_ce_losses, train_total_losses, train_kld_losses, train_accs, train_mses, global_step, monotonic_step = train(dict_train_loader, global_step, monotonic_step, args.gradient_clip_threshold, epoch, epochs)
    t2 = time.time()

    epoch_time = t2 - t1
    hours = int(epoch_time // 3600)
    minutes = int((epoch_time % 3600) // 60)
    seconds = epoch_time % 60
    time_str = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"

    # Add beta/alpha cycling monitor
    if epoch % 5 == 0 and model_config['beta'] == "schedule":
        print(f"📊 Schedule status:")
        print(f"   Beta: {model.beta:.6f} (max: {model_config['max_beta']})")
        print(f"   Alpha: {model.alpha:.6f} (max: {model_config['max_alpha']})")
        print(f"   Global step: {global_step} / {len(beta_schedule)}")

    # Run validation based on frequency
    if (epoch + 1) % args.validation_freq == 0:
        val_ce_losses, val_total_losses, val_kld_losses, val_accs, val_mses = test(dict_val_loader)
        
        train_loss = mean(train_total_losses)
        val_loss = mean(val_total_losses)
        train_kld_loss = mean(train_kld_losses)
        val_kld_loss = mean(val_kld_losses)
        train_acc = mean(train_accs)
        val_acc = mean(val_accs)
        train_mse = mean(train_mses)
        val_mse = mean(val_mses)

        # Update learning rate
        scheduler.step(val_loss)
        # 📊 Property Prediction Diagnostic (only for PP-guided VAE)
        if args.ppguided and epoch % 10 == 0 and val_mse > 0:
            model.eval()
            with torch.no_grad():
                # Get a small batch for inspection
                sample_batch_key = '0'  # First batch
                if sample_batch_key in dict_val_loader:
                    sample_data = dict_val_loader[sample_batch_key][0]
                    sample_data.to(device)
                    sample_dest = dict_val_loader[sample_batch_key][1]
                    sample_dest.to(device)
                    sample_inc = dict_val_loader[sample_batch_key][2]
                    sample_inc.to(device)
                    
                    # Get predictions
                    result = model(sample_data, sample_dest, sample_inc, device)
                    
                    if len(result) == 9:  # PP-guided VAE
                        _, _, _, _, _, _, _, _, y_pred = result
                        
                        # Get true values
                        y_true = sample_data.y1.float() if hasattr(sample_data, 'y1') else None
                        
                        if y_pred is not None and y_true is not None:
                            print(f"\n📊 Property Prediction Check (Epoch {epoch + 1}):")
                            print(f"Predicted values (first 10): {y_pred[:10].squeeze().tolist()}")
                            print(f"True values (first 10): {y_true[:10].tolist()}")
                            print(f"Prediction std: {y_pred.std().item():.4f}")
                            print(f"Prediction mean: {y_pred.mean().item():.4f}")
                            print(f"True value std: {y_true.std().item():.4f}")
                            print(f"True value mean: {y_true.mean().item():.4f}")
            model.train()
            
    else:
        # Skip validation but still compute training metrics
        train_loss = mean(train_total_losses)
        train_kld_loss = mean(train_kld_losses)
        train_acc = mean(train_accs)
        train_mse = mean(train_mses)
        
        # Use previous validation metrics or set to None
        val_loss = train_loss  # Fallback for scheduler
        val_total_losses = train_total_losses
        val_kld_losses = train_kld_losses
        val_accs = train_accs
        val_mses = train_mses
        val_kld_loss = train_kld_loss
        val_acc = train_acc
        val_mse = train_mse

    # Apply learning rate warmup
    if epoch < args.warmup_epochs:
        warmup_scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch time: {time_str}")
    print(f"Current learning rate: {current_lr:.6f}")

    # Save checkpoint before critical transitions
    if args.AE_Warmup and epoch == args.warmup_epochs - 1:
        transition_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_dict': train_loss_dict,
            'val_loss_dict': val_loss_dict,
            'model_config': model_config,
            'global_step': global_step,
            'monotonic_step': monotonic_step,
        }
        torch.save(transition_dict, os.path.join(directory_path, f"model_before_transition_epoch_{epoch}.pt"))
        print(f"💾 Saved checkpoint before AE warmup ends")

    # Save checkpoint based on frequency
    if (epoch + 1) % args.checkpoint_freq == 0 or epoch == epochs - 1:
        model_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_dict': train_loss_dict,
            'val_loss_dict': val_loss_dict,
            'model_config': model_config,
            'global_step': global_step,
            'monotonic_step': monotonic_step,
        }
        torch.save(model_dict, os.path.join(directory_path, "model_latest.pt"))
        print(f"Saved latest checkpoint *after* epoch {epoch + 1}")

    # Test generation validity every epoch (or at least when validation runs)
    generation_validity = 0.0
    rdkit_validity = 0.0
    if (epoch + 1) % args.validation_freq == 0:  # Test when validation runs
        print(f"🧪 Testing generation quality...")
        generation_validity, rdkit_validity = validate_generation_quality(model, vocab, device, num_samples=args.validation_samples)
        print(f"📊 Format validity: {generation_validity:.1%}, Chemical validity: {rdkit_validity:.1%}")
        
        # Add checkpoint saving based on validity
        if generation_validity > 0.15:  # Save when validity is decent
            validity_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_dict': train_loss_dict,
                'val_loss_dict': val_loss_dict,
                'model_config': model_config,
                'generation_validity': generation_validity,
                'val_loss': val_loss
            }
            torch.save(validity_checkpoint, os.path.join(directory_path, f"model_validity_{generation_validity:.3f}_epoch_{epoch+1}.pt"))
            print(f"💾 Saved validity checkpoint: {generation_validity:.1%} at epoch {epoch+1}")
        
        # Only show detailed analysis every 10 epochs to reduce clutter
        if (epoch + 1) % 10 == 0:
            # FIXED: Improved format-specific validation with better error handling
            print("🧪 Detailed format analysis:")
            try:
                # Generate sample predictions with safe error handling
                sample_predictions = model.inference(torch.randn(10, model.embedding_dim, device=device), device)[0]
                for i, pred in enumerate(sample_predictions[:3]):
                    try:
                        pred_str = safe_token_processing(pred[0], vocab, "RT_tokenized")
                        print(f"  Sample {i}: {pred_str[:100]}..." if len(pred_str) > 100 else f"  Sample {i}: {pred_str}")
                    except Exception as e:
                        print(f"  Sample {i}: [Error processing: {e}]")
            except Exception as e:
                print(f"  Error in detailed analysis: {e}")
        else:
            # Optional: Quick summary on non-10 epochs
            print(f"  (Detailed analysis shown every 10 epochs)")
        
        if generation_validity > 0.8 and val_acc > 0.7:
            print(f"🎯 Excellent generation quality achieved! Validity: {generation_validity:.1%}, Acc: {val_acc:.3f}")
        elif generation_validity < 0.1 and epoch > 30:
            print(f"⚠️  Poor generation quality detected. Consider adjusting hyperparameters.")

    # FIXED: Check and save best model with proper logging (only when validation runs)
    if (epoch + 1) % args.validation_freq == 0:
        model_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_dict': train_loss_dict,
            'val_loss_dict': val_loss_dict,
            'model_config': model_config,
            'global_step': global_step,
            'monotonic_step': monotonic_step,
        }
        # Save best loss model (traditional)
        loss_improved = earlystopping_loss(val_loss, model_dict)
        if loss_improved:
            print(f"💾 [INFO] New best loss model saved: {val_loss:.5f}")
        
        # Save best combined model (loss + validity)
        combined_improved = earlystopping(val_loss, model_dict, generation_validity)
        if combined_improved:
            print(f"🎯 [INFO] New best combined model saved")

    # Use combined metric for actual early stopping
    if global_step >= len(beta_schedule) and earlystopping.early_stop:
        print("Early stopping triggered.")
        break

    if math.isnan(train_loss):
        print("Network diverged! Training aborted.")
        break

    print("-" * 70)
    if (epoch + 1) % args.validation_freq == 0:
        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.5f} | Train KLD: {train_kld_loss:.5f} | Val Loss: {val_loss:.5f} | Val KLD: {val_kld_loss:.5f}")
        print(f"Train Acc: {train_acc:.5f} | Train MSE: {train_mse:.5f} | Val Acc: {val_acc:.5f} | Val MSE: {val_mse:.5f}")
        if generation_validity > 0:
            print(f"🧪 Format Validity: {generation_validity:.1%} | Chemical Validity: {rdkit_validity:.1%}")
    else:
        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.5f} | Train KLD: {train_kld_loss:.5f} | [Validation skipped]")
        print(f"Train Acc: {train_acc:.5f} | Train MSE: {train_mse:.5f}")
    alpha_str = f"{model.alpha:.5f}" if hasattr(model, 'alpha') else "N/A"
    print(f"Current Beta: {model.beta:.5f} | Current Alpha: {alpha_str}")
    print("-" * 70)

    # Store loss dicts
    train_loss_dict[epoch] = (train_total_losses, train_kld_losses, train_accs)
    val_loss_dict[epoch] = (val_total_losses, val_kld_losses, val_accs)

    # Save epoch metrics (only when validation runs)
    if (epoch + 1) % args.validation_freq == 0:
        train_metrics = {'loss': train_loss, 'kld': train_kld_loss, 'acc': train_acc, 'mse': train_mse}
        val_metrics = {'loss': val_loss, 'kld': val_kld_loss, 'acc': val_acc, 'mse': val_mse}
        save_epoch_metrics_to_csv(epoch + 1, train_metrics, val_metrics, directory_path, resume_from_checkpoint, generation_validity, rdkit_validity)

    # Save loss dictionaries periodically to avoid data loss
    if (epoch + 1) % args.checkpoint_freq == 0 or epoch == epochs - 1:
        save_loss_dicts(train_loss_dict, val_loss_dict, directory_path)

# Final save of loss dictionaries
save_loss_dicts(train_loss_dict, val_loss_dict, directory_path)

print('Done!\n')
print(f'Model trained to predict {property_count} properties: {property_names}')
print(f'Checkpoints saved to: {directory_path}')
print(f'Model files saved:')
print(f'  - model_best_loss.pt (traditional best validation loss)')
print(f'  - model_best_combined.pt (best combined loss + generation quality)')
print(f'  - model_latest.pt (most recent checkpoint)')
print(f'  - model_validity_*.pt (validity-based checkpoints)')
print(f'Final training configuration:')
print(f'  - Dropout rate: {args.dropout_rate}')
print(f'  - Weight decay: {args.weight_decay}')
print(f'  - Gradient clipping: {args.gradient_clip_threshold}')
print(f'  - Learning rate decay factor: {args.lr_decay_factor}')
print(f'  - Validation frequency: every {args.validation_freq} epoch(s)')
print(f'  - Checkpoint frequency: every {args.checkpoint_freq} epoch(s)')

# Add diagnosis function
def quick_generation_test(model, vocab, device, num_samples=10):
    """Quick test of generation quality"""
    print("\n🧪 Quick Generation Test...")
    model.eval()
    
    # Temporarily set higher beta for generation
    original_beta = model.beta
    model.beta = max(0.1, model.beta)
    
    with torch.no_grad():
        z = torch.randn(num_samples, model.embedding_dim, device=device)
        result = model.inference(data=z, device=device, sample=False)
        predictions = result[0]
        
        valid_format = 0
        for i, pred in enumerate(predictions[:5]):
            tokens = tokenids_to_vocab(pred[0], vocab)
            smiles = combine_tokens(tokens, tokenization="RT_tokenized")
            
            if '[*:' in smiles and '|' in smiles:
                valid_format += 1
            
            if i < 3:
                print(f"  Sample {i+1}: {smiles[:80]}...")
    
    print(f"  Format validity: {valid_format}/{min(5, num_samples)} samples")
    model.beta = original_beta
    model.train()
    return valid_format > 0

# Run quick test
if quick_generation_test(model, vocab, device):
    print("✅ Generation capability confirmed!")
else:
    print("⚠️  Generation needs improvement - check beta value and training")

#experiment.end()
