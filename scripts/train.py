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
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            torch.save(model_dict, os.path.join(self.save_dir,"model_best_loss.pt"))
            #torch.save(model.state_dict(), self.save_dir + "/model_best_loss.pth")
            self.counter = 0


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
            print(f"\nBatch [{i} / {len(order_batches)}]")
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


def save_epoch_metrics_to_csv(epoch, train_metrics, val_metrics, directory_path):
    """Save training and validation metrics for each epoch to CSV"""
    csv_file = os.path.join(directory_path, 'training_log.csv')
    
    # Create the CSV file and write headers if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss_mean', 'train_kld_mean', 'train_acc_mean', 'train_mse_mean',
                'val_loss_mean', 'val_kld_mean', 'val_acc_mean', 'val_mse_mean'
            ])
    
    # Append the current epoch's metrics
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
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



args = parser.parse_args()

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
vocab_file_path = main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
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
}
batch_size = model_config['batch_size']
epochs = model_config['epochs']
hidden_dimension = model_config['hidden_dimension']
embedding_dim = model_config['embedding_dim']
loss = model_config['loss']

# %% Call data
dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
dict_val_loader = torch.load(main_dir_path+'/data/dict_val_loader_'+augment+'_'+tokenization+'.pt')
dict_test_loader = torch.load(main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')

num_train_graphs = len(list(dict_train_loader.keys())[
    :-2])*batch_size + dict_train_loader[list(dict_train_loader.keys())[-1]][0].num_graphs
num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

assert dict_train_loader['0'][0].num_graphs == batch_size, 'Batch_sizes of data and model do not match'

# %% Create an instance of the G2S model
# only for wce loss we calculate the token weights from vocabulary
if model_config['loss']=="wce":
    vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    class_weights = token_weights(vocab_file)
    class_weights = torch.FloatTensor(class_weights)
if model_config['loss']=="ce":
    class_weights=None

# Initialize model
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
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'

# Use custom save directory if provided, otherwise use default path
if args.save_dir is not None:
    directory_path = args.save_dir
else:
    directory_path = os.path.join(main_dir_path,'Checkpoints/', model_name)

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

es_patience = model_config['es_patience']
earlystopping = EarlyStopping(dir=directory_path, patience=es_patience)

print(f'STARTING TRAINING')
# Prepare dictionaries for training or load checkpoint

checkpoint_file = None

# If resume_from_checkpoint is specified, use that checkpoint
if args.resume_from_checkpoint is not None:
    if os.path.exists(args.resume_from_checkpoint):
        checkpoint_file = args.resume_from_checkpoint
    else:
        print(f"Warning: Specified checkpoint {args.resume_from_checkpoint} does not exist. Starting from scratch.")

# Otherwise, try to load best model first, then latest from the default directory
elif os.path.exists(directory_path):
    if os.path.exists(os.path.join(directory_path, "model_best_loss.pt")):
        checkpoint_file = os.path.join(directory_path, "model_best_loss.pt")
    elif os.path.exists(os.path.join(directory_path, "model_latest.pt")):
        checkpoint_file = os.path.join(directory_path, "model_latest.pt")

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
else:
    print("Starting training from scratch")
    train_loss_dict = {}
    val_loss_dict = {}
    epoch_cp = 0
    global_step = 0
    monotonic_step = 0

for epoch in range(epoch_cp, epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    t1 = time.time()
    model, train_ce_losses, train_total_losses, train_kld_losses, train_accs, train_mses, global_step, monotonic_step = train(dict_train_loader, global_step, monotonic_step)
    t2 = time.time()
    print(f'epoch time: {t2-t1}\n')
    val_ce_losses, val_total_losses, val_kld_losses, val_accs, val_mses = test(dict_val_loader)
    train_loss = mean(train_total_losses)
    val_loss = mean(val_total_losses)
    train_kld_loss = mean(train_kld_losses)
    val_kld_loss = mean(val_kld_losses)
    train_acc = mean(train_accs)
    val_acc = mean(val_accs)
    train_mse = mean(train_mses)
    val_mse = mean(val_mses)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")
    
    # Early stopping check, but only if the cyclical annealing schedule is already done
    if global_step >= len(beta_schedule):
        model_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_dict': train_loss_dict,
        'val_loss_dict': val_loss_dict,
        'model_config':model_config,
        'global_step':global_step,
        'monotonic_step':monotonic_step,
        }
        earlystopping(val_loss, model_dict)
        if earlystopping.early_stop:
            print("Early stopping!")
            break

    
    # Save the latest checkpoint (overwrites every time)
    model_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_dict': train_loss_dict,
        'val_loss_dict': val_loss_dict,
        'model_config':model_config,
        'global_step':global_step,
        'monotonic_step':monotonic_step,
    }
    torch.save(model_dict, os.path.join(directory_path, "model_latest.pt"))

    # Print checkpoint status
    print(f"Saved latest checkpoint at epoch {epoch}")
    if hasattr(earlystopping, 'best_score') and earlystopping.best_score == val_loss:
        print(f"New best model saved with validation loss: {val_loss:.5f}")
    
    if math.isnan(train_loss):
        print("Network diverged!")
        break

    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} | Train KLD: {train_kld_loss:.5f} \n\
                         | Val Loss: {val_loss:.5f} | Val KLD: {val_kld_loss:.5f}\n")
    train_loss_dict[epoch] = (train_total_losses, train_kld_losses, train_accs)
    val_loss_dict[epoch] = (val_total_losses, val_kld_losses, val_accs)
    
    # Save metrics to CSV
    train_metrics = {
        'loss': train_loss,
        'kld': train_kld_loss,
        'acc': train_acc,
        'mse': train_mse
    }
    val_metrics = {
        'loss': val_loss,
        'kld': val_kld_loss,
        'acc': val_acc,
        'mse': val_mse
    }
    save_epoch_metrics_to_csv(epoch+1, train_metrics, val_metrics, directory_path)

# Save the training loss values
with open(os.path.join(directory_path,'train_loss.pkl'), 'wb') as file:
    pickle.dump(train_loss_dict, file)
 
# Save the validation loss values
with open(os.path.join(directory_path,'val_loss.pkl'), 'wb') as file:
    pickle.dump(val_loss_dict, file)

print('Done!\n')
#experiment.end()
