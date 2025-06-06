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


def train(dict_train_loader, global_step, monotonic_step):
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
        if model_config['beta']=="schedule":
            # determine beta at time step t
            if global_step >= len(beta_schedule):
                #if model.beta <=1:
                #    beta_t = 1.0 +0.001*monotonic_step
                #    monotonic_step+=1
                #else: beta_t = model.beta #stays the same
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

        # Perform a single forward pass.
        loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        
        # Check for unstable loss values before backpropagation
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"WARNING: NaN or Inf detected in loss at batch {i}")
            print(f"Loss: {loss.item()}, Recon: {recon_loss.item()}, KLD: {kl_loss.item()}")
            continue  # Skip this batch

        # Check if KLD spike indicates instability
        if i > 0 and kl_loss.item() > 5 * np.mean(kld_losses[-min(10, len(kld_losses)):]):
            print(f"WARNING: KLD spike detected at batch {i}")
            print(f"Current KLD: {kl_loss.item()}, Recent mean: {np.mean(kld_losses[-min(10, len(kld_losses)):]):.2f}")

        optimizer.zero_grad()
        loss.backward()
        
        # Monitor gradient norms before clipping
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        if total_grad_norm > 10.0:  # Warning threshold
            print(f"WARNING: Large gradient norm detected: {total_grad_norm:.2f}")
        
        # TODO: do we need the clip_grad_norm?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        ce_losses.append(recon_loss.item())
        total_losses.append(loss.item())
        kld_losses.append(kl_loss.item())
        accs.append(acc.item())
        mses.append(mse.item())
        if i % 10 == 0:
            print(f"\nBatch [{i:4d} / {len(order_batches):4d}]")
            print("-" * 70)
            print(f"Recon: {recon_loss.item():.6f} | Total: {loss.item():.6f} | KLD: {kl_loss.item():.6f} | Acc: {acc.item():.6f} | MSE: {mse.item():.6f} | Beta: {model.beta:.6f} | Alpha: {model.alpha:.6f}")
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

            # Perform a single forward pass.
            loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

            ce_losses.append(recon_loss.item())
            total_losses.append(loss.item())
            kld_losses.append(kl_loss.item())
            accs.append(acc.item())
            mses.append(mse.item())
        
    return ce_losses, total_losses, kld_losses, accs, mses

def save_epoch_metrics_to_csv(epoch, train_metrics, val_metrics, directory_path, resume_from_checkpoint=False):
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
                'val_loss_mean', 'val_kld_mean', 'val_acc_mean', 'val_mse_mean'
            ])
        # Always write the data row
        writer.writerow([
            epoch,
            train_metrics['loss'], train_metrics['kld'], train_metrics['acc'], train_metrics['mse'],
            val_metrics['loss'], val_metrics['kld'], val_metrics['acc'], val_metrics['mse']
        ])

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
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties to train the model to predict")
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
    'property_names': property_names
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

# Initialize model with property count
if args.ppguided:
    model_type = G2S_VAE_PPguided
else:
    model_type = G2S_VAE_PPguideddisabled

model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dim,device,model_config, vocab, seed, loss_weights=class_weights, add_latent=add_latent)
model.to(device)

print(model)

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
        B = np.zeros(int(5*num_train_graphs/batch_size)) # for 5 epochs
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

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Add learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, verbose=True, min_lr=1e-5)

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

# Optional: reset CSV flag if training from scratch
if not resume_from_checkpoint:
    flag_file = os.path.join(directory_path, '.csv_initialized')
    if os.path.exists(flag_file):
        print("[INFO] Removing old .csv_initialized to allow clean training log overwrite.")
        os.remove(flag_file)

print(f'STARTING TRAINING')
print(f'Model will predict {property_count} properties: {property_names}')
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

# Reset CSV flag if starting from scratch
if not resume_from_checkpoint:
    flag_file = os.path.join(directory_path, '.csv_initialized')
    if os.path.exists(flag_file):
        print("[INFO] Removing old .csv_initialized to allow clean training log overwrite.")
        os.remove(flag_file)

# Load the checkpoint if one was found
if checkpoint_file is not None:
    print(f"Loading model from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_cp = checkpoint['epoch']
    train_loss_dict = checkpoint['loss_dict']
    val_loss_dict = checkpoint['val_loss_dict']
    if model_config['beta'] == "schedule":
        global_step = checkpoint['global_step']
        monotonic_step = checkpoint['monotonic_step']
        model.beta = model_config['max_beta']
    resume_from_checkpoint = True
else:
    train_loss_dict = {}
    val_loss_dict = {}
    epoch_cp = 0
    global_step = 0
    monotonic_step = 0
    resume_from_checkpoint = False

for epoch in range(epoch_cp, epochs):
    print(f"Epoch {epoch + 1}\n" + "-" * 30)

    t1 = time.time()
    model, train_ce_losses, train_total_losses, train_kld_losses, train_accs, train_mses, global_step, monotonic_step = train(dict_train_loader, global_step, monotonic_step)
    t2 = time.time()

    epoch_time = t2 - t1
    hours = int(epoch_time // 3600)
    minutes = int((epoch_time % 3600) // 60)
    seconds = epoch_time % 60
    time_str = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"

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
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch time: {time_str}")
    print(f"Current learning rate: {current_lr:.6f}")

    # Save checkpoint
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

    # FIXED: Check and save best model with proper logging
    model_saved = earlystopping(val_loss, model_dict)
    if model_saved:
        print(f"🎯 [INFO] New best model saved with validation loss: {val_loss:.5f}")

    if global_step >= len(beta_schedule) and earlystopping.early_stop:
        print("Early stopping triggered.")
        break

    if math.isnan(train_loss):
        print("Network diverged! Training aborted.")
        break

    print("-" * 70)
    print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.5f} | Train KLD: {train_kld_loss:.5f} | Val Loss: {val_loss:.5f} | Val KLD: {val_kld_loss:.5f}")
    print(f"Train Acc: {train_acc:.5f} | Train MSE: {train_mse:.5f} | Val Acc: {val_acc:.5f} | Val MSE: {val_mse:.5f}")
    print(f"Current Beta: {model.beta:.5f} | Current Alpha: {model.alpha:.5f}")
    print("-" * 70)

    # Store loss dicts
    train_loss_dict[epoch] = (train_total_losses, train_kld_losses, train_accs)
    val_loss_dict[epoch] = (val_total_losses, val_kld_losses, val_accs)

    # Save epoch metrics
    train_metrics = {'loss': train_loss, 'kld': train_kld_loss, 'acc': train_acc, 'mse': train_mse}
    val_metrics = {'loss': val_loss, 'kld': val_kld_loss, 'acc': val_acc, 'mse': val_mse}
    save_epoch_metrics_to_csv(epoch + 1, train_metrics, val_metrics, directory_path, resume_from_checkpoint)

# Save the training loss values - only overwrite if starting fresh
file_mode = 'wb'  # Always use write mode - we're saving the full dictionaries
with open(os.path.join(directory_path,'train_loss.pkl'), file_mode) as file:
    pickle.dump(train_loss_dict, file)
 
# Save the validation loss values
with open(os.path.join(directory_path,'val_loss.pkl'), file_mode) as file:
    pickle.dump(val_loss_dict, file)

print('Done!\n')
print(f'Model trained to predict {property_count} properties: {property_names}')
print(f'Checkpoints saved to: {directory_path}')
#experiment.end()
