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


def train(dict_train_loader, global_step, monotonic_step, gradient_clip_threshold):
    # shuffle batches every epoch

    order_batches = list(range(len(dict_train_loader)))
    random.shuffle(order_batches)

    ce_losses = []
    total_losses = []
    kld_losses = []
    accs = []
    mses = []

    model.train()
    # Iterate in batches over the training dataset.
    for i, batch in enumerate(order_batches):
        # CRITICAL FIX: Clear decoder state completely before each batch
        if hasattr(model, 'Decoder'):
            model.Decoder.clear_decoder_state_completely()
        
        # Clear any accumulated gradients
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
            
            # Check for unstable loss values before backpropagation
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"WARNING: NaN or Inf detected in loss at batch {i}")
                print(f"Loss: {loss.item()}, Recon: {recon_loss.item()}, KLD: {kl_loss.item()}")
                continue  # Skip this batch

            # Check if KLD spike indicates instability
            if i > 0 and kl_loss.item() > 5 * np.mean(kld_losses[-min(10, len(kld_losses)):]):
                print(f"WARNING: KLD spike detected at batch {i}")
                print(f"Current KLD: {kl_loss.item()}, Recent mean: {np.mean(kld_losses[-min(10, len(kld_losses)):]):.2f}")

            # 🔥 CRITICAL FIX: Add gradient penalty to force gradients through context attention
            loss_with_penalty = add_gradient_penalty_to_loss(model, loss)
            # Add this debug print:
            if i == 0:  # First batch only
                print(f"🔍 DEBUG: Original loss: {loss.item():.6f}")
                print(f"🔍 DEBUG: Loss with penalty: {loss_with_penalty.item():.6f}")
                print(f"🔍 DEBUG: Penalty amount: {(loss_with_penalty - loss).item():.9f}")
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
            optimizer.step()
            
            # Store metrics
            ce_losses.append(recon_loss.item())
            total_losses.append(loss.item())
            kld_losses.append(kl_loss.item())
            accs.append(acc.item())
            mses.append(mse.item())
            
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

def save_epoch_metrics_to_csv(epoch, train_metrics, val_metrics, directory_path, resume_from_checkpoint=False, generation_validity=None):
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
                'val_loss_mean', 'val_kld_mean', 'val_acc_mean', 'val_mse_mean', 'generation_validity'
            ])
        # Always write the data row
        writer.writerow([
            epoch,
            train_metrics['loss'], train_metrics['kld'], train_metrics['acc'], train_metrics['mse'],
            val_metrics['loss'], val_metrics['kld'], val_metrics['acc'], val_metrics['mse'],
            generation_validity if generation_validity is not None else 0.0
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
    """Test generation quality during training with improved error handling"""
    model.eval()
    valid_count = 0
    
    with torch.no_grad():
        try:
            z_random = torch.randn(num_samples, model.embedding_dim, device=device)
            
            # Call inference on the full model, not just the decoder
            result = model.inference(data=z_random, device=device, sample=False, log_var=None)
            
            # Handle the return values properly based on model type
            if len(result) >= 4:
                predictions, _, _, _ = result[:4]
            else:
                predictions = result[0] if result else None
            
            if predictions is None:
                print("Warning: No predictions returned from model inference")
                return 0.0
            
            # Process predictions with improved error handling
            for i in range(min(len(predictions), num_samples)):
                try:
                    if len(predictions[i]) > 0 and hasattr(predictions[i][0], '__iter__'):
                        pred_tokens = predictions[i][0]
                        
                        # Use the safe token processing function
                        smiles_string = safe_token_processing(pred_tokens, vocab, "RT_tokenized")
                        
                        # Check for basic G2S format validity
                        if (len(smiles_string) > 10 and 
                            '|' in smiles_string and  # Must have pipe separators
                            '[*:' in smiles_string and  # Must have attachment points
                            smiles_string.count('(') == smiles_string.count(')') and
                            smiles_string.count('[') == smiles_string.count(']')):
                            valid_count += 1
                except Exception as e:
                    # Skip this sample and continue
                    continue
                    
        except Exception as e:
            print(f"Generation validation failed: {e}")
            return 0.0
    
    validity_rate = valid_count / num_samples if num_samples > 0 else 0.0
    model.train()
    return validity_rate

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

print(f"DEBUG: Constructed vocab file path: {vocab_file_path}")
print(f"DEBUG: File exists: {os.path.exists(vocab_file_path)}")
print(f"DEBUG: Current working directory: {os.getcwd()}")
print(f"DEBUG: Absolute path: {os.path.abspath(vocab_file_path)}")

vocab = load_vocab(vocab_file_path)

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
dict_train_loader = torch.load(data_path_prefix.format('train'))
dict_val_loader = torch.load(data_path_prefix.format('val'))
dict_test_loader = torch.load(data_path_prefix.format('test'))

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

# Initialize model with property count
if args.ppguided:
    model_type = G2S_VAE_PPguided
else:
    model_type = G2S_VAE

model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dim,device,model_config, vocab, seed, loss_weights=class_weights, add_latent=add_latent)
model.to(device)

# 🔧 CRITICAL FIX: Reset dead context attention components
model = reset_context_attention(model)

# 🎯 NEW: Add attention regularization
model = add_attention_regularization(model)

print(model)

# Use configurable warmup epochs
n_iter = int(20 * num_train_graphs/batch_size) # 20 epochs
# Beta scheduling function from Optimus paper 
def frange_cycle_zero_linear(n_iter, start=0.0, stop=model_config['max_beta'],  n_cycle=5, ratio_increase=0.5, ratio_zero=0.3): #, beginning_zero=0.1):
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
    ## beginning zero
    if args.AE_Warmup:
        B = np.zeros(int(args.warmup_epochs*num_train_graphs/batch_size)) # configurable warmup epochs
        L = np.append(B,L)
    return L 

if model_config['beta'] == "schedule":
    beta_schedule = frange_cycle_zero_linear(n_iter=n_iter)
elif model_config['beta'] == "normalVAE":
    beta_schedule = np.ones(1)

if model_config['alpha'] == "schedule":
    alpha_schedule = frange_cycle_zero_linear(n_iter=n_iter, start=0.0, stop=model_config['max_alpha'], n_cycle=5, ratio_increase=0.5, ratio_zero=0.3)
elif model_config['alpha'] == "fixed":
    alpha_schedule = np.ones(1)

# %%# %% Train

# Enhanced optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Enhanced learning rate scheduler with configurable parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_factor, 
                              patience=args.scheduler_patience, verbose=True, min_lr=args.min_lr)

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

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

es_patience = model_config['es_patience']
earlystopping = EarlyStopping(dir=directory_path, patience=es_patience)

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
    model, train_ce_losses, train_total_losses, train_kld_losses, train_accs, train_mses, global_step, monotonic_step = train(dict_train_loader, global_step, monotonic_step, args.gradient_clip_threshold)
    t2 = time.time()

    epoch_time = t2 - t1
    hours = int(epoch_time // 3600)
    minutes = int((epoch_time % 3600) // 60)
    seconds = epoch_time % 60
    time_str = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"

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

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch time: {time_str}")
    print(f"Current learning rate: {current_lr:.6f}")

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
        model_saved = earlystopping(val_loss, model_dict)
        if model_saved:
            print(f"🎯 [INFO] New best model saved with validation loss: {val_loss:.5f}")
    
    # Test generation validity every epoch (or at least when validation runs)
    generation_validity = 0.0
    if (epoch + 1) % args.validation_freq == 0:  # Test when validation runs
        print(f"🧪 Testing generation quality...")
        generation_validity = validate_generation_quality(model, vocab, device, num_samples=50)
        print(f"📊 Generation validity: {generation_validity:.1%}")
        
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
            print(f"🧪 Generation Validity: {generation_validity:.1%}")
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
        save_epoch_metrics_to_csv(epoch + 1, train_metrics, val_metrics, directory_path, resume_from_checkpoint, generation_validity)

    # Save loss dictionaries periodically to avoid data loss
    if (epoch + 1) % args.checkpoint_freq == 0 or epoch == epochs - 1:
        save_loss_dicts(train_loss_dict, val_loss_dict, directory_path)

# Final save of loss dictionaries
save_loss_dicts(train_loss_dict, val_loss_dict, directory_path)

print('Done!\n')
print(f'Model trained to predict {property_count} properties: {property_names}')
print(f'Checkpoints saved to: {directory_path}')
print(f'Final training configuration:')
print(f'  - Dropout rate: {args.dropout_rate}')
print(f'  - Weight decay: {args.weight_decay}')
print(f'  - Gradient clipping: {args.gradient_clip_threshold}')
print(f'  - Learning rate decay factor: {args.lr_decay_factor}')
print(f'  - Validation frequency: every {args.validation_freq} epoch(s)')
print(f'  - Checkpoint frequency: every {args.checkpoint_freq} epoch(s)')
#experiment.end()
