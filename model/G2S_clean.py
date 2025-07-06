from typing import Mapping
import numpy as np
import networkx as nx
#import igraph
import torch
#from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical
from torch_geometric.nn import MessagePassing, global_mean_pool

try:
    from rdkit import Chem
    from rdkit import RDLogger
    # Suppress RDKit warnings/errors
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available, chemical validation disabled")

import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear
import torch.nn.functional as F
#from torch.nn import Parameter
#from torch.nn.init import xavier_uniform
from torch import embedding_renorm_, scatter 

from torch_geometric.data import Data
#from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.utils import to_dense_adj

#from onmt.decoders import TransformerDecoder
from model.transformer_mod import TransformerDecoder, TransformerLMDecoder
#from onmt.modules.embeddings import Embeddings
from model.embeddings_mod import Embeddings
from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
from data_processing.data_utils import *


class Lin_layer_MP(torch.nn.Module):
    '''
    Linear NN used in the weighted edge centered MP from scratch
    '''

    def __init__(self, in_channels=300, out_channels=300):
        super(Lin_layer_MP, self).__init__()

        # Define linear layer and relu
        self.lin = Linear(in_channels, out_channels)
        self.relu = ReLU()

    def forward(self, h0, weighted_sum):
        x = self.lin(weighted_sum)
        h1 = self.relu(h0 + x)

        return h1


class vec_scratch_MP_layer(torch.nn.Module):
    '''
    This is a vectorized implementation of the edge centered message passing
    '''

    def __init__(self, in_channels=300, out_channels=300):
        super(vec_scratch_MP_layer, self).__init__()

        # Define linear layer and relu
        self.lin = Linear(in_channels, out_channels)
        self.relu = ReLU()

    def forward(self, h0, dest_is_origin_matrix, dev):
        # pass weighted sum through a NN to obtain new featurization of that edge
        weighted_sum = torch.sparse.mm(
            dest_is_origin_matrix.to(dev), h0.to(dev))
        x = self.lin(weighted_sum)
        h_next = self.relu(h0+x)

        return h_next


class Lin_layer_node(torch.nn.Module):
    '''
    Linear NN used in the weighted atom updater
    '''

    def __init__(self, in_channels, out_channels=300):
        super(Lin_layer_node, self).__init__()

        # Define linear layer and relu
        self.lin = Sequential(Linear(in_channels, out_channels),
                              ReLU())

    def forward(self, concat_atom_edges):
        atom_hidden = self.lin(concat_atom_edges)

        return atom_hidden


class vec_atom_updater(torch.nn.Module):
    '''
    This is a vectorized version of the atom update step
    '''

    def __init__(self, in_channels, out_channels=300, output_layer=False):
        super(vec_atom_updater, self).__init__()

        # Define linear layer and relu
        if not output_layer:
            self.lin = Sequential(Linear(in_channels, out_channels),
                              ReLU())
        else: 
            self.lin = Linear(in_channels, out_channels)

    def forward(self, nodes, h, inc_edges_to_atom_matrix, device):
        sum_inc_edges = torch.sparse.mm(inc_edges_to_atom_matrix.to(device), h)
        atom_embeddings = torch.cat((nodes.to(device), sum_inc_edges), dim=1)
        # pass through NN
        atom_updates = self.lin(atom_embeddings)
        return atom_updates

# %% Define models


class Wdmpnn_Conv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, device, first_linear=True):
        super(Wdmpnn_Conv, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.first_linear = first_linear    

        self.lin1 = Sequential(Linear(node_dim + edge_dim, hidden_dim),
                                ReLU(),
                                ).to(device)
        
        self.lin2 = Sequential(Linear(hidden_dim + edge_dim, hidden_dim),
                                ReLU(),
                                ).to(device)
        

        # define edge message passing layers
        self.vec_scratch_MP_layer1 = vec_scratch_MP_layer(
            in_channels=hidden_dim, out_channels=hidden_dim).to(device)
        self.vec_scratch_MP_layer2 = vec_scratch_MP_layer(
            in_channels=hidden_dim, out_channels=hidden_dim).to(device)
        self.vec_scratch_MP_layer3 = vec_scratch_MP_layer(
            in_channels=hidden_dim, out_channels=hidden_dim).to(device)

        # define node message passing layer
        self.vec_atom_updater = vec_atom_updater(
            in_channels=node_dim+hidden_dim, out_channels=hidden_dim).to(device)
        
        # define node message passing layer2
        self.vec_atom_updater2 = vec_atom_updater(
            in_channels=hidden_dim+hidden_dim, out_channels=hidden_dim, output_layer=True).to(device)

    def forward(self, graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
        
        # only in the first network with shared layers
        if self.first_linear:
            nodes = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            # Repeat the node features for each edge
            nodes_to_edge = nodes[edge_index[0]]
            # Initialize the edge features with the concatenation of the node and edge features
            h0 = torch.cat([nodes_to_edge, edge_attr], dim=1)
        
            # Pass this through a NN to compute the initialize hidden features
            h0 = self.lin1(h0)

            # pass the messages along edges
            h1 = self.vec_scratch_MP_layer1(h0, dest_is_origin_matrix, device)
            h2 = self.vec_scratch_MP_layer2(h1, dest_is_origin_matrix, device)
            h3 = self.vec_scratch_MP_layer3(h2, dest_is_origin_matrix, device)

            # get atom embeddings by summing over all incoming edges and concatenating with original atom features
            atom_embeddings = self.vec_atom_updater(
                nodes, h3, inc_edges_to_atom_matrix, device)
            # atom embeddings 
            
            return atom_embeddings
        
        else:
            nodes = graph.shared_output
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            # Repeat the node features for each edge
            nodes_to_edge = nodes[edge_index[0]]
            # Initialize the edge features with the concatenation of the node and edge features
            h0 = torch.cat([nodes_to_edge, edge_attr], dim=1)

            h0 = self.lin2(h0)
            # pass the messages along edges
            h1 = self.vec_scratch_MP_layer1(h0, dest_is_origin_matrix, device)
            h2 = self.vec_scratch_MP_layer2(h1, dest_is_origin_matrix, device)
            h3 = self.vec_scratch_MP_layer3(h2, dest_is_origin_matrix, device)

            # get atom embeddings by summing over all incoming edges and concatenating with original atom features
            atom_embeddings = self.vec_atom_updater2(
                nodes, h3, inc_edges_to_atom_matrix, device)
            
            return atom_embeddings

class GraphEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, device, model_config):
        super(GraphEncoder, self).__init__()

        self.tconv1 = Wdmpnn_Conv(node_dim, edge_dim, hidden_dim, device)
        
        self.mu = Wdmpnn_Conv(node_dim, edge_dim, hidden_dim, device, first_linear=False)
        self.logvar = Wdmpnn_Conv(node_dim, edge_dim, hidden_dim, device, first_linear=False)
    
    def forward(self, graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):

        atom_weights = graph.W_atoms

        shared_output = self.tconv1(graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

        graph.shared_output = shared_output

        mu = self.mu(graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        logvar = self.logvar(graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

        mu = global_mean_pool(mu * atom_weights.view(-1, 1), graph.batch).to(device)
        logvar = global_mean_pool(logvar * atom_weights.view(-1, 1), graph.batch).to(device)
        #print(torch.mean(mu), torch.mean(logvar))

        return mu, logvar


## Transformer decoder
class SequenceDecoder(nn.Module):
    def __init__(self, model_config, vocab, loss_weights, add_latent):
        """Implementation of transformer decoder
    
        Args:
            model_config (Dict): model config settings
            data_config (Dict): data config settings
            vocab (Dict): complete vocab of tokens
        """
        super().__init__()
        
        self.ndim= model_config['embedding_dim']
        self.config = model_config
        
        self.max_n = 512  # Changed from 256 to accommodate stoichiometry + connectivity
        
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.beam_size = 1
        self.add_latent = add_latent
    
        # 🔧 DEBUG: Add debug information before embeddings creation
        print(f"🔧 DEBUG: Creating embeddings with vocab size: {len(self.vocab)}")
        print(f"🔧 DEBUG: feat_padding_idx will be: []")
        print(f"🔧 DEBUG: feat_vocab_sizes will be: []")
        print(f"🔧 DEBUG: embedding_dim: {model_config['embedding_dim']}")
    
        self.decoder_embeddings = Embeddings(
            word_vec_size=model_config['embedding_dim'],
            word_vocab_size=len(self.vocab),
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            position_encoding_type="SinusoidalInterleaved",
            feat_merge="concat",
            feat_vec_exponent=0.7,
            feat_vec_size=-1,
            feat_padding_idx=[],          # ← CRITICAL: Empty list
            feat_vocab_sizes=[],          # ← CRITICAL: Empty list - no feature embeddings
            dropout=0.3,
            sparse=False,
            freeze_word_vecs=False
        )
    
        # 🔧 VALIDATION: Test embeddings configuration immediately after creation
        print("🧪 Validating embeddings configuration...")
        try:
            # CRITICAL FIX: Don't unsqueeze for validation test
            test_input = torch.tensor([[self.vocab["_SOS"]]], device='cpu').unsqueeze(-1)  # Shape: [1, 1, 1]
            test_output = self.decoder_embeddings(test_input)
            print(f"✅ Embeddings validation passed: input shape {test_input.shape} -> output shape {test_output.shape}")
        except Exception as e:
            print(f"❌ Embeddings validation failed: {e}")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Input dtype: {test_input.dtype}")
            print(f"   Input value range: [{test_input.min().item()}, {test_input.max().item()}]")
            print(f"   Vocab size: {len(self.vocab)}")
            if hasattr(self.decoder_embeddings, 'make_embedding'):
                print(f"   make_embedding type: {type(self.decoder_embeddings.make_embedding)}")
                if hasattr(self.decoder_embeddings.make_embedding, '__len__'):
                    print(f"   make_embedding length: {len(self.decoder_embeddings.make_embedding)}")
            print("   This confirms the Elementwise error. Check feat_padding_idx and feat_vocab_sizes parameters.")
            raise

        if self.add_latent: 
            d_model=model_config['embedding_dim']*2
        else: 
            d_model=model_config['embedding_dim']
        #TransformerDecoder (with EDATT) or TransformerLMDecoder (without EDAtt)
        self.Decoder = TransformerDecoder(num_layers=model_config['decoder_num_layers'], \
            d_model=d_model, heads=model_config['num_attention_heads'], \
            d_ff=2048, copy_attn=False, self_attn_type="scaled-dot", dropout=0.3, attention_dropout=0.3, \
            embeddings=self.decoder_embeddings, max_relative_positions=4, aan_useffn=False, \
            full_context_alignment=False, alignment_layer=-3, alignment_heads=0
        )

        self.output_layer = nn.Linear(d_model, len(self.vocab), bias=True)  # This should already be correct (no +1)

        if self.config['loss']=="ce" or self.config['loss']=="wce":
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.vocab["_PAD"],
                reduction="mean",
                weight=loss_weights
            )
        elif self.config['loss']=="focal":
            self.criterion = FocalLoss(
                gamma = 1,
                ignore_index=self.vocab["_PAD"],
                reduction="none"
            )


    def fix_polymer_format(self, smiles):
        """Fix common format errors in generated SMILES"""
        if not smiles:
            return smiles
        
        # Ensure polymer format marker
        if '|' not in smiles:
            # Add default stoichiometry and connectivity
            smiles += '|1.000|1.000|<1-1:0.500:0.500>'
        
        # Fix attachment points
        if '[*]' in smiles:
            smiles = smiles.replace('[*]', '[*:1]')
        
        # Ensure at least two attachment points
        if '[*:1]' in smiles and '[*:2]' not in smiles:
            # Find a suitable position for second attachment
            if 'c' in smiles:  # Aromatic carbon
                smiles = smiles.replace('c', 'c[*:2]', 1)
            elif 'C' in smiles:  # Aliphatic carbon
                smiles = smiles.replace('C', 'C[*:2]', 1)
        
        # Balance parentheses
        open_count = smiles.count('(')
        close_count = smiles.count(')')
        if open_count > close_count:
            smiles += ')' * (open_count - close_count)
        
        # Balance brackets
        open_brackets = smiles.count('[')
        close_brackets = smiles.count(']')
        if open_brackets > close_brackets:
            smiles += ']' * (open_brackets - close_brackets)
        
        # Ensure proper pipe count
        pipe_count = smiles.count('|')
        if pipe_count < 4:
            parts = smiles.split('|')
            while len(parts) < 4:
                if len(parts) == 1:
                    parts.append('1.000')  # Default stoich 1
                elif len(parts) == 2:
                    parts.append('1.000')  # Default stoich 2
                elif len(parts) == 3:
                    parts.append('<1-1:0.500:0.500>')  # Default connectivity
            smiles = '|'.join(parts)
        
        return smiles
        
    def constrained_beam_search(self, z, beam_size=3, temperature=0.8):
        """Generate with format constraints and chemical validity"""
        batch_size = z.size(0)
        device = z.device
        
        # Initialize beams
        beams = [[([self.vocab["_SOS"]], 0.0)] for _ in range(batch_size)]
        finished = [[] for _ in range(batch_size)]
        
        # Prepare encoder output
        enc_output = z.unsqueeze(1)
        src_lengths = torch.ones(batch_size, device=device).long()
        
        for step in range(self.max_n):
            new_beams = [[] for _ in range(batch_size)]
            
            for batch_idx in range(batch_size):
                for seq, score in beams[batch_idx]:
                    # Check if sequence is complete
                    if seq[-1] == self.vocab["_EOS"] or len(seq) > 400:
                        finished[batch_idx].append((seq, score))
                        continue
                    
                    # Prepare input
                    input_seq = torch.tensor([seq], device=device).unsqueeze(-1)
                    
                    # Get predictions
                    with torch.no_grad():
                        dec_out, _ = self.Decoder(
                            tgt=input_seq[-1:, :, :], 
                            enc_out=enc_output[batch_idx:batch_idx+1], 
                            src_len=src_lengths[batch_idx:batch_idx+1], 
                            step=len(seq), 
                            add_latent=self.add_latent
                        )
                        dec_out = dec_out.transpose(0, 1)  # [1, b, h] -> [b, 1, h]
                        logits = self.output_layer(dec_out)  # [b, 1, vocab_size]
                        probs = F.softmax(logits / temperature, dim=-1)
                    
                    # Apply constraints
                    probs = self.apply_generation_constraints(seq, probs)
                    
                    # Get top k tokens
                    top_k_probs, top_k_indices = torch.topk(probs, min(beam_size * 2, probs.size(-1)))
                    
                    for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                        new_seq = seq + [idx.item()]
                        new_score = score + torch.log(prob).item()
                        new_beams[batch_idx].append((new_seq, new_score))
                
                # Keep top beams
                new_beams[batch_idx] = sorted(new_beams[batch_idx], key=lambda x: x[1], reverse=True)[:beam_size]
                if not new_beams[batch_idx]:  # If all beams ended
                    new_beams[batch_idx] = beams[batch_idx][:1]  # Keep at least one
            
            beams = new_beams
            
            # Check if all batches have at least one finished sequence
            all_have_finished = all(len(f) > 0 for f in finished)
            if all_have_finished and step > 100:
                break
        
        # Select best sequences
        final_sequences = []
        for batch_idx in range(batch_size):
            all_seqs = beams[batch_idx] + finished[batch_idx]
            # Prefer finished sequences with valid format
            valid_seqs = [(seq, score) for seq, score in all_seqs 
                          if self.check_sequence_completeness(seq)]
            if valid_seqs:
                best_seq = max(valid_seqs, key=lambda x: x[1])
            else:
                best_seq = max(all_seqs, key=lambda x: x[1])
            final_sequences.append(best_seq[0])
        
        return final_sequences
        
    def sample_with_temperature(self, logits, temperature=1.0, top_k=50, top_p=0.95):
        """Sample from logits with temperature and top-k/top-p filtering"""
        # Apply temperature
        logits = logits / max(temperature, 0.1)  # Prevent division by zero
        
        # Top-k filtering
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, 
                               torch.full_like(logits, float('-inf')), 
                               logits)
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
        
    def apply_generation_constraints(self, current_seq, probs):
        """Apply hard constraints to generation probabilities"""
        # Convert current sequence to tokens
        current_tokens = [self.inv_vocab.get(t, '') for t in current_seq[1:]]  # Skip SOS
        current_string = ''.join(current_tokens)
        
        # Apply polymer-specific constraints first
        probs = self.polymer_specific_constraints(current_seq, probs)
        
        # Mask invalid tokens
        for token_id, token in self.inv_vocab.items():
            if not self.is_valid_next_token(current_tokens, token_id):
                probs[0, token_id] = 0.0
        
        # Renormalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback to uniform if all masked
            probs = torch.ones_like(probs) / probs.size(-1)
        
        return probs
    
    def polymer_specific_constraints(self, current_seq, probs):
        """Enhanced constraints for polymer SMILES"""
        current_tokens = [self.inv_vocab.get(t, '') for t in current_seq[1:]]
        current_string = ''.join(current_tokens)
        
        # Boost probability of polymer-specific tokens
        polymer_tokens = ['[*:', '|', '<', '-', ':', '.']
        for token_id, token in self.inv_vocab.items():
            if any(pt in token for pt in polymer_tokens):
                probs[0, token_id] *= 1.5  # Boost polymer tokens
        
        # Force proper sequence after certain patterns
        if current_string.endswith('[*'):
            # Must be followed by :1] or :2]
            for token_id, token in self.inv_vocab.items():
                if token not in [':1]', ':2]', ':']:
                    probs[0, token_id] = 0
        
        # Renormalize to ensure valid probability distribution
        if probs.sum() > 0:
            probs = probs / probs.sum()
        
        return probs
    
    def validate_smiles_during_generation(self, token_ids, vocab):
        """Validate SMILES during generation to prevent chemical invalidity"""
        try:
            # Convert tokens to string
            tokens = tokenids_to_vocab(token_ids, vocab)
            if not tokens:
                return True
                
            current_smiles = combine_tokens(tokens, tokenization="RT_tokenized")
            
            # Extract SMILES part (before first |)
            if '|' in current_smiles:
                smiles_part = current_smiles.split('|')[0]
            else:
                smiles_part = current_smiles
            
            # Check parentheses balance
            open_count = smiles_part.count('(')
            close_count = smiles_part.count(')')
            
            # Check ring closure balance
            ring_numbers = {}
            for char in smiles_part:
                if char.isdigit() and char != '0':
                    ring_numbers[char] = ring_numbers.get(char, 0) + 1
            
            # Check if we're in a valid state
            parentheses_balanced = open_count >= close_count  # Allow temporary imbalance during generation
            rings_properly_paired = all(count <= 2 for count in ring_numbers.values())
            
            return parentheses_balanced and rings_properly_paired
            
        except Exception:
            return True  # If validation fails, allow the token

    
    def validate_basic_chemistry(self, smiles_string):
        """Validate basic chemical structure of SMILES using RDKit"""
        try:
            if not RDKIT_AVAILABLE:
                return True  # Skip validation if RDKit not available
                
            if not smiles_string or len(smiles_string) < 5:
                return False
            
            # Extract SMILES part (before first |)
            smiles_part = smiles_string.split('|')[0] if '|' in smiles_string else smiles_string
            
            # Check chemical validity with RDKit
            if '.' in smiles_part:
                # Handle copolymer (multiple monomers)
                monomers = smiles_part.split('.')
                for monomer in monomers:
                    if monomer.strip():
                        mol = Chem.MolFromSmiles(monomer)
                        if mol is None:
                            return False
                return True
            else:
                # Single monomer
                mol = Chem.MolFromSmiles(smiles_part)
                return mol is not None
                
        except Exception:
            return False
            
    def get_current_tokens_from_predictions(self, predictions):
        """Extract current tokens from beam search predictions"""
        tokens = []
        try:
            if hasattr(predictions, '__iter__'):
                for token_id in predictions:
                    if isinstance(token_id, torch.Tensor):
                        token_id = token_id.item()
                    
                    if token_id == self.vocab.get("_EOS", -1):
                        break
                    
                    token = self.inv_vocab.get(token_id, '')
                    if token not in ['_SOS', '_PAD', '_UNK']:
                        tokens.append(token)
        except:
            pass
        
        return tokens

    def check_sequence_completeness(self, tokens):
        """Check if generated sequence has complete G2S polymer format"""
        try:
            # Convert tokens to string
            current_tokens = self.get_current_tokens_from_predictions(tokens)
            sequence = ''.join(current_tokens)
            
            if not sequence or len(sequence) < 20:
                return False
            
            # Quick checks for G2S components
            has_attachment_points = '[*:' in sequence and ']' in sequence
            has_pipes = sequence.count('|') >= 3  # Need SMILES|stoich1|stoich2|connectivity
            has_connectivity_start = '<' in sequence
            has_connectivity_format = ':' in sequence and '-' in sequence
            
            # Must have all essential G2S components
            return has_attachment_points and has_pipes and has_connectivity_start and has_connectivity_format
            
        except Exception:
            return False

    def is_valid_next_token(self, current_tokens, next_token_id):
        """Enhanced validation to prevent chemical invalidity"""
        try:
            next_token_str = self.inv_vocab.get(next_token_id, '')
            
            # Skip validation for special tokens
            if next_token_str in ['_SOS', '_EOS', '_PAD', '_UNK']:
                return True
            
            # Allow more freedom in early generation
            if len(current_tokens) < 20:  # Increased from 10
                return True
            
            current_sequence = ''.join(current_tokens)
            test_sequence = current_sequence + next_token_str
            
            # 🔧 ENHANCED: Strict parentheses checking
            open_parens = test_sequence.count('(')
            close_parens = test_sequence.count(')')
            
            # Never allow more closing than opening parentheses
            if close_parens > open_parens:
                return False
       
            # 🔧 ENHANCED: Ring closure validation
            ring_numbers = {}
            for char in test_sequence:
                if char.isdigit() and char != '0':  # Ring numbers 1-9
                    ring_numbers[char] = ring_numbers.get(char, 0) + 1
            
            # Each ring number should appear exactly twice (open and close)
            for ring_num, count in ring_numbers.items():
                if count > 2:  # More than 2 occurrences is invalid
                    return False
            
            # 🔧 ENHANCED: Bracket validation for attachment points
            open_brackets = test_sequence.count('[')
            close_brackets = test_sequence.count(']')
            
            # Never allow more closing than opening brackets
            if close_brackets > open_brackets:
                return False
            
            # Don't allow too many unclosed brackets
            if open_brackets - close_brackets > 3:
                return False
            
            # 🔧 ENHANCED: G2S connectivity patterns (keep existing logic)
            if ':' in next_token_str and current_sequence.endswith(':0.500'):
                last_pattern_start = current_sequence.rfind('<')
                if last_pattern_start != -1:
                    pattern_fragment = current_sequence[last_pattern_start:]
                    colon_count = pattern_fragment.count(':')
                    if colon_count >= 2:
                        return False
            
            # 🔧 ENHANCED: Length checking
            if len(test_sequence) > 300:  # Reasonable SMILES length limit
                return False
            
            # 🔧 ENHANCED: Valid chemistry atoms only
            if next_token_str and len(next_token_str) == 1:
                valid_atoms = set('CHONPSFIBrcnofpsibl()[]1234567890=#-+*.:')
                if next_token_str not in valid_atoms:
                    return False
            
            return True
            
        except Exception:
            return True  # If validation fails, allow the token

    def filter_invalid_tokens_optimized(self, decode_strategy, log_probs):
        """Prevent malformed G2S connectivity patterns"""
        batch_size, vocab_size = log_probs.shape
        filtered_log_probs = log_probs.clone()
        
        # Only check top tokens for performance
        TOP_K_TOKENS = 30
        top_k_probs, top_k_indices = torch.topk(log_probs, min(TOP_K_TOKENS, vocab_size), dim=1)
        
        for batch_idx in range(batch_size):
            try:
                # Get current tokens
                if hasattr(decode_strategy, 'alive_seq') and batch_idx < len(decode_strategy.alive_seq):
                    current_predictions = decode_strategy.alive_seq[batch_idx]
                elif hasattr(decode_strategy, 'current_predictions') and batch_idx < len(decode_strategy.current_predictions):
                    current_predictions = decode_strategy.current_predictions[batch_idx]
                else:
                    continue
                
                current_tokens = self.get_current_tokens_from_predictions(current_predictions)
                
                # Validate top-k tokens
                top_k_batch_indices = top_k_indices[batch_idx]
                for token_id in top_k_batch_indices:
                    if not self.is_valid_next_token(current_tokens, token_id.item()):
                        filtered_log_probs[batch_idx, token_id] = float('-inf')
                        
            except Exception:
                continue
        
        return filtered_log_probs
    
    def reset_decoder_cache(self):
        """Reset the decoder cache to prevent size mismatches"""
        if hasattr(self.Decoder, 'state') and 'cache' in self.Decoder.state:
            self.Decoder.state['cache'] = None
        
        # Reset layer caches in transformer layers with proper tensor initialization
        if hasattr(self.Decoder, 'transformer_layers'):
            for layer in self.Decoder.transformer_layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_cache'):
                    try:
                        # Get device from layer parameters
                        device = next(iter(layer.parameters())).device
                        # Initialize with empty tensors instead of None to prevent numel() errors
                        empty_tensor = torch.empty(0, device=device)
                        layer.self_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
                    except (StopIteration, AttributeError):
                        # Fallback to CPU if no parameters found
                        empty_tensor = torch.empty(0)
                        layer.self_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
    
    def clear_decoder_state_completely(self):
        """Completely clear decoder state during training to prevent graph retention"""
        # Clear main decoder state
        if hasattr(self.Decoder, 'state'):
            self.Decoder.state.clear()
        
        # Reset all layer caches with proper structure
        if hasattr(self.Decoder, 'transformer_layers'):
            for layer in self.Decoder.transformer_layers:
                if hasattr(layer, 'self_attn'):
                    # Initialize attention cache with proper structure
                    if hasattr(layer.self_attn, 'layer_cache'):
                        try:
                            device = next(iter(layer.parameters())).device
                            empty_tensor = torch.empty(0, device=device)
                            layer.self_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
                        except (StopIteration, AttributeError):
                            empty_tensor = torch.empty(0)
                            layer.self_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
                    
                    # Clear any stored attention weights
                    if hasattr(layer.self_attn, 'attn'):
                        layer.self_attn.attn = None
                
                if hasattr(layer, 'context_attn'):
                    # Initialize context attention cache with proper structure
                    if hasattr(layer.context_attn, 'layer_cache'):
                        try:
                            device = next(iter(layer.parameters())).device
                            empty_tensor = torch.empty(0, device=device)
                            layer.context_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
                        except (StopIteration, AttributeError):
                            empty_tensor = torch.empty(0)
                            layer.context_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
                    
                    if hasattr(layer.context_attn, 'attn'):
                        layer.context_attn.attn = None

    def forward(self, graph_batch, z, loss_weights=None, teacher_forcing_ratio=1.0):
        """Forward pass of decoder with scheduled teacher forcing
    
        Args:
            graph_batch (Data): Data of correct graphs
            z (Tensor): Latent space embedding [b, h]
            loss_weights: Optional loss weights
            teacher_forcing_ratio (float): Probability of using teacher forcing (1.0 = always use ground truth)
    
        Returns:
            Tensor, Tensor, Tensor, Tensor: Reconstruction loss, accuracy, predictions, target
        """
        # CRITICAL FIX: Reset decoder cache at the beginning of each forward pass
        self.reset_decoder_cache()
        
        z_length = 1  # Latent sequence length is always 1
        src_lengths = torch.ones(z.size(0), device=z.device).long()  # All 1s
        
        # prepare target
        target = torch.tensor(np.array(graph_batch.tgt_token_ids), device=z.device)[:, :-1]
        m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"]) #pads SOS token left side (beginning of sequences)
        target = m(target)
        
        # CRITICAL FIX: Don't unsqueeze when no feature embeddings are used
        # The Elementwise module expects the input dimensions to match the number of embedding modules
        # Since we have feat_vocab_sizes=[], we should NOT add the extra dimension
        # target = target.unsqueeze(-1)  # REMOVE THIS LINE
    
        # CRITICAL FIX: Clear any existing decoder state to prevent graph retention
        if hasattr(self.Decoder, 'state'):
            self.Decoder.state.clear()
        self.Decoder.state = {}
        
        enc_output = z.unsqueeze(1)
        self.Decoder.state["src"] = enc_output
        
        # Implement scheduled sampling
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        
        if use_teacher_forcing or not self.training:
            # Normal teacher forcing (use ground truth as input)
            target_3d = target.unsqueeze(-1)  # Add dimension for OpenNMT: [b, t] -> [b, t, 1]
            target_3d = target_3d.transpose(0, 1)  # Convert to sequence-first: [b, t, 1] -> [t, b, 1]
            dec_outs, _ = self.Decoder(
                tgt=target_3d,
                enc_out=enc_output, 
                src_len=src_lengths, 
                step=target.size(1), 
                add_latent=self.add_latent
            )
            dec_outs = dec_outs.transpose(0, 1)  # Convert back to batch-first: [t, b, h] -> [b, t, h]
            dec_outs = self.output_layer(dec_outs)  # [b, t, h] => [b, t, v]  (FIXED COMMENT)
            dec_outs = dec_outs.permute(0, 2, 1)    # [b, t, v] => [b, v, t]  (FIXED COMMENT)
        else:
            # Use model's own predictions (scheduled sampling)
            batch_size = z.size(0)
            max_len = target.size(1)
            vocab_size = len(self.vocab)
            
            # Initialize output tensor
            dec_outs = torch.zeros(batch_size, vocab_size, max_len, device=z.device)
            
            # Start with SOS token - keep in [b, 1] format
            current_input = target[:, :1]  # [b, 1]
            
            for t in range(max_len):
                # Decode one step - convert to 3D and sequence-first for OpenNMT
                current_input_3d = current_input.unsqueeze(-1).transpose(0, 1)  # [b, 1] -> [b, 1, 1] -> [1, b, 1]
                dec_out, _ = self.Decoder(
                    tgt=current_input_3d,
                    enc_out=enc_output, 
                    src_len=src_lengths, 
                    step=t+1, 
                    add_latent=self.add_latent
                )
                dec_out = dec_out.transpose(0, 1)  # [1, b, h] -> [b, 1, h]
                
                # Get logits for current step
                logits = self.output_layer(dec_out)  # [b, 1, vocab_size]
                dec_outs[:, :, t] = logits.squeeze(1)  # Store in output tensor
                
                # Get prediction for next input
                pred = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # [b, 1]
                current_input = pred  # Already [b, 1], no reshape needed
                
                # Optionally mix with ground truth (curriculum learning)
                if t < max_len - 1:
                    # Random sampling per batch element
                    use_gt = torch.rand(batch_size, 1, device=z.device) < teacher_forcing_ratio
                    ground_truth = target[:, t+1:t+2]  # Next ground truth token [b, 1]
                    
                    # Both are [b, 1], so where operation is straightforward
                    current_input = torch.where(use_gt, ground_truth, current_input)
    
        # evaluate
        target = torch.tensor(np.array(graph_batch.tgt_token_ids), device=z.device)
        recon_loss = self.criterion(
            input=dec_outs, 
            target=target.long()
        )
                
        predictions = torch.argmax(dec_outs.transpose(1,0), dim=0)                             # [b, t]
        mask = (target != self.vocab["_PAD"]).long()
        accs = (predictions == target).float()
        accs = accs * mask
        acc = accs.sum() / mask.sum()
    
        # 🔧 ENHANCED: Add chemical validity penalty during training with RDKit
        validity_penalty = 0
        if RDKIT_AVAILABLE and self.training:
            for sample in range(min(len(predictions), 10)):  # Check subset for efficiency
                try:
                    sample_tokens = predictions[sample].cpu().numpy()
                    # Convert to SMILES and validate with RDKit
                    tokens = tokenids_to_vocab(sample_tokens, self.vocab)
                    smiles_string = combine_tokens(tokens, tokenization="RT_tokenized")
                    
                    # NEW: Format validity REWARDS (negative penalties = rewards)
                    if '|' in smiles_string and smiles_string.count('|') >= 3:
                        validity_penalty -= 0.05  # REWARD for proper pipe structure
                    if '[*:' in smiles_string:
                        validity_penalty -= 0.05  # REWARD for attachment points
                    if '<' in smiles_string and ':' in smiles_string.split('|')[-1]:
                        validity_penalty -= 0.05  # REWARD for connectivity notation
                    
                    # Keep the penalty for invalid chemistry
                    if not self.validate_basic_chemistry(smiles_string):
                        validity_penalty += 0.1  # Penalty for RDKit-invalid chemistry
                except Exception:
                    validity_penalty += 0.05  # Penalty for unparseable sequences
        else:
            # Fallback to basic validation if RDKit not available
            for sample in range(min(len(predictions), 10)):
                try:
                    sample_tokens = predictions[sample].cpu().numpy()
                    tokens = tokenids_to_vocab(sample_tokens, self.vocab)
                    smiles_string = combine_tokens(tokens, tokenization="RT_tokenized")
                    
                    # NEW: Format validity REWARDS even without RDKit
                    if '|' in smiles_string and smiles_string.count('|') >= 3:
                        validity_penalty -= 0.03  # REWARD
                    if '[*:' in smiles_string:
                        validity_penalty -= 0.03  # REWARD
                    if '<' in smiles_string:
                        validity_penalty -= 0.03  # REWARD
                        
                    # Basic validation penalty
                    if not self.validate_smiles_during_generation(sample_tokens, self.vocab):
                        validity_penalty += 0.05
                except Exception:
                    pass
        
        # Add validity penalty to loss with increased weight
        if validity_penalty != 0:
            validity_penalty = torch.tensor(validity_penalty / min(10, len(predictions)), device=z.device)
            recon_loss = recon_loss + validity_penalty * 0.5  # Increased weight from 0.1 to 0.5
    
        return recon_loss, acc, predictions, target
        
    def safe_map_state(self, fn_map_state):
        """Safely apply map_state only if cache is properly initialized"""
        try:
            # Check if all layer caches are properly initialized
            if hasattr(self.Decoder, 'transformer_layers'):
                for layer in self.Decoder.transformer_layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_cache'):
                        cache = layer.self_attn.layer_cache
                        if (isinstance(cache, tuple) and len(cache) > 1 and 
                            isinstance(cache[1], dict) and 'keys' in cache[1]):
                            keys = cache[1]['keys']
                            # Check if keys is None or not a tensor
                            if keys is None or not hasattr(keys, 'numel'):
                                return False  # Skip map_state
            
            # If we get here, cache is properly initialized
            self.Decoder.map_state(fn_map_state)
            return True
        except Exception as e:
            print(f"Warning: map_state failed safely: {e}")
            return False

    def inference(self, z, temperature=0.9, use_beam_search=False):
        # CRITICAL FIX: Reset decoder cache before inference
        self.reset_decoder_cache()
        
        batch_size = z.size(0)
        device = z.device
        
        # Use sampling-based generation for better diversity
        if not use_beam_search:
            # Initialize
            enc_output = z.unsqueeze(1)
            src_lengths = torch.ones(batch_size, device=device).long()
            self.Decoder.state["src"] = enc_output
            
            # Start with SOS
            generated = torch.full((batch_size, 1), self.vocab["_SOS"], device=device)
            
            for step in range(self.max_n):
                # Prepare input
                input_seq = generated[:, -1:].unsqueeze(-1).transpose(0, 1)
                
                # Decode
                dec_out, _ = self.Decoder(
                    tgt=input_seq,
                    enc_out=enc_output,
                    src_len=src_lengths,
                    step=step+1,
                    add_latent=self.add_latent
                )
                dec_out = dec_out.transpose(0, 1)
                logits = self.output_layer(dec_out.squeeze(1))
                
                # Apply validity constraints softly
                for b in range(batch_size):
                    current_tokens = generated[b].tolist()
                    current_token_strings = [self.inv_vocab.get(t, '') for t in current_tokens]
                    for token_id in range(logits.size(-1)):
                        if not self.is_valid_next_token(current_token_strings, token_id):
                            logits[b, token_id] -= 5.0  # Soft penalty instead of -inf

                # Apply light format guidance even in regular inference
                current_tokens = [self.inv_vocab.get(t, '') for t in generated[b].tolist()]
                current_string = ''.join(current_tokens[1:])
                
                # Gentle boosts for format requirements
                if len(current_string) > 10 and '[*:' not in current_string:
                    for token_id, token in self.inv_vocab.items():
                        if '[*:' in token:
                            logits[b, token_id] += 0.5  # Gentle boost
                
                if '[*:' in current_string and '|' not in current_string:
                    pipe_id = self.vocab.get('|', -1)
                    if pipe_id >= 0:
                        logits[b, pipe_id] += 0.5
                        
                # Sample next token with temperature
                next_token = self.sample_with_temperature(
                    logits, temperature=temperature, top_k=50, top_p=0.95)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if (next_token == self.vocab["_EOS"]).all() or step > 300:
                    break
            
            # Convert to expected format
            predictions = []
            for b in range(batch_size):
                seq = generated[b].tolist()
                # Apply post-processing fixes
                tokens = tokenids_to_vocab(seq, self.vocab)
                smiles = combine_tokens(tokens, tokenization="RT_tokenized")
                if hasattr(self, 'fix_polymer_format'):
                    smiles = self.fix_polymer_format(smiles)
                predictions.append((seq, 0.0))
            
            return predictions
        
        else:
            # Fall back to original beam search if requested
            return self.constrained_beam_search(z, beam_size=3, temperature=temperature)

    def generate_with_format_guidance(self, z, max_length=400, temperature=0.7):
        """Generation with strong format guidance - NEW METHOD"""
        batch_size = z.size(0)
        device = z.device
        
        # Initialize
        enc_output = z.unsqueeze(1)
        src_lengths = torch.ones(batch_size, device=device).long()
        
        # Reset decoder cache
        self.reset_decoder_cache()
        self.Decoder.state["src"] = enc_output
        
        # Start with SOS
        generated = torch.full((batch_size, 1), self.vocab["_SOS"], device=device)
        
        # Format checkpoints
        format_stages = {
            'need_polymer_start': True,
            'need_first_pipe': False,
            'need_stoich1': False,
            'need_second_pipe': False,
            'need_stoich2': False,
            'need_third_pipe': False,
            'need_connectivity': False
        }
        
        for step in range(max_length):
            # Decode one step
            input_seq = generated[:, -1:].unsqueeze(-1).transpose(0, 1)
            dec_out, _ = self.Decoder(
                tgt=input_seq,
                enc_out=enc_output,
                src_len=src_lengths,
                step=step+1,
                add_latent=self.add_latent
            )
            dec_out = dec_out.transpose(0, 1)
            logits = self.output_layer(dec_out.squeeze(1))
            
            # Apply format-aware constraints
            current_tokens = [self.inv_vocab.get(t.item(), '') for t in generated[0]]
            current_string = ''.join(current_tokens[1:])  # Skip SOS
            
            # Boost probabilities based on format stage
            if format_stages['need_polymer_start'] and len(current_string) > 10:
                # Boost attachment point tokens
                for token_id, token in self.inv_vocab.items():
                    if '[*:' in token:
                        logits[:, token_id] += 2.0
                        
            if '[*:' in current_string and format_stages['need_first_pipe']:
                # Boost pipe token
                pipe_id = self.vocab.get('|', -1)
                if pipe_id >= 0:
                    logits[:, pipe_id] += 3.0
                    
            # Temperature-adjusted sampling
            next_token = self.sample_with_temperature(
                logits, temperature=temperature, top_k=50, top_p=0.95
            )
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update format stages
            current_string = ''.join([self.inv_vocab.get(t.item(), '') for t in generated[0][1:]])
            if '[*:' in current_string:
                format_stages['need_polymer_start'] = False
                format_stages['need_first_pipe'] = True
            if current_string.count('|') >= 1:
                format_stages['need_first_pipe'] = False
                format_stages['need_stoich1'] = True
            
            # Check for EOS or completion
            if (next_token == self.vocab["_EOS"]).all() or (
                current_string.count('|') == 3 and '>' in current_string
            ):
                break
        
        return generated

    def inference_with_retries(self, z, max_retries=5, temperature_range=(0.6, 1.0)):
        """Try multiple temperatures to get valid SMILES - NEW METHOD"""
        best_predictions = []
        best_validity = 0
        
        for batch_idx in range(z.size(0)):
            z_single = z[batch_idx:batch_idx+1]
            best_pred = None
            best_score = 0
            
            for retry in range(max_retries):
                temp = np.random.uniform(*temperature_range)
                
                # Try format-guided generation
                if retry % 2 == 0 and hasattr(self, 'generate_with_format_guidance'):
                    generated = self.generate_with_format_guidance(z_single, temperature=temp)
                    pred = (generated[0].tolist(), 0.0)
                else:
                    # Use existing inference
                    preds = self.inference(z_single, temperature=temp)
                    pred = preds[0] if preds else None
                
                if pred:
                    # Quick validity check
                    tokens = tokenids_to_vocab(pred[0], self.vocab)
                    smiles = combine_tokens(tokens, tokenization="RT_tokenized")
                    
                    score = 0
                    if '|' in smiles:
                        score += 1
                    if '[*:' in smiles:
                        score += 1
                    if smiles.count('|') == 3:
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_pred = pred
                    
                    if score >= 4:  # Good enough
                        break
            
            best_predictions.append(best_pred if best_pred else ([], 0.0))
        
        return best_predictions
        
    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.Decoder.state["cache"] is not None:
            _recursive_map(self.Decoder.state["cache"])    

    def reset_layer_cache(self):
        """After inference, layer cache needs to be reset"""
        if hasattr(self.Decoder, 'transformer_layers'):
            for layer in self.Decoder.transformer_layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_cache'):
                    try:
                        # Get device from layer parameters  
                        device = next(iter(layer.parameters())).device
                        # Initialize with empty tensors instead of None
                        empty_tensor = torch.empty(0, device=device)
                        layer.self_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
                    except (StopIteration, AttributeError):
                        # Fallback to CPU if no parameters found
                        empty_tensor = torch.empty(0)
                        layer.self_attn.layer_cache = (False, {'keys': empty_tensor, 'values': empty_tensor})
                                    
    def conditional_position_weights(self, batch):
        batch=batch.cpu()
        weights = np.ones_like(batch)    
        for nr, sample in enumerate(batch):

            # Define the preceding sequences and their corresponding weight multipliers
            preceding_sequences = [
                ["|", "0_0", "."], # stoichiometry decision
                ["|", "<", "1", "-"] ,# first decision connectivity
                ["|", "<", "1", "-", "3",":","0_0","."] ,# second decision connectivity
                
            ]
            # preceding sequences token ids
            preceding_sequences = [list(get_seq_features_from_line(preceding_sequence, vocab=self.vocab, max_tgt_len=len(preceding_sequence)+1)[0][:-1]) for preceding_sequence in preceding_sequences ]

            weight_multipliers = [2, 2]  # Weight multipliers for each preceding sequence
            # if preceding sequence, double the weight
            for i in range(len(sample)):
                for sequence, multiplier in zip(preceding_sequences, weight_multipliers):
                    if i >= len(sequence) and list(sample[i - len(sequence):i]) == sequence:
                        weights[nr,i] *= multiplier

        return weights
    
    def forward(self, property_predictions):
        """
        Args:
            property_predictions: dict of {property_name: tensor}
        Returns:
            predicted target property value
        """
        # Gather source property values
        source_values = []
        for prop in self.source_props:
            if prop in property_predictions:
                source_values.append(property_predictions[prop])
            else:
                raise ValueError(f"Source property {prop} not found in predictions")
        
        # Apply equation
        result = self.equation_lambda(source_values)
        return result

def parse_property_relationships(relationship_strings):
    """Parse command line relationship strings into structured format"""
    relationships = {}
    if relationship_strings is None:
        return relationships
        
    for rel_str in relationship_strings:
        # Format: "target=equation" e.g., "bandgap=abs(EA-IP)"
        if '=' not in rel_str:
            print(f"Warning: Invalid relationship format: {rel_str}")
            continue
            
        target, equation = rel_str.split('=', 1)
        target = target.strip()
        equation = equation.strip()
        
        # Extract source properties from equation
        # This is a simple extraction - looks for uppercase property names
        import re
        # Match property names (uppercase letters potentially followed by lowercase)
        potential_props = re.findall(r'\b[A-Z][A-Za-z0-9_]*\b', equation)
        source_props = [p for p in potential_props if p not in ['abs', 'exp', 'log', 'sqrt']]
        
        relationships[target] = {
            'equation': equation,
            'sources': source_props
        }
    
    return relationships

class G2S_VAE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, embedding_dim, device, model_config, vocab, seed, loss_weights=None, add_latent=True):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.seed=seed
        self.eps = model_config['epsilon']
        try: 
            self.embedding_dim = model_config['embedding_dim']
        except:
            self.embedding_dim = embedding_dim
        # in case beta is schedule the value will be specified in train.py
        if not model_config["beta"] =="schedule":
            self.beta=1.0
        self.config = model_config
        self.vocab = vocab
        self.alpha = 0.0
        #self.max_n=data_config['max_num_nodes']
        #if model_config['pooling']=='custom':
        #    self.Encoder = GraphEncoder_GMT(node_dim, edge_dim, hidden_dim, device, model_config)
        #elif model_config['pooling']=='mean':
        self.Encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, device, model_config)
        self.Decoder = SequenceDecoder(model_config, vocab, loss_weights, add_latent=add_latent)
        if not self.hidden_dim==self.embedding_dim:
            self.lincompress = Linear(self.hidden_dim, self.embedding_dim).to(device)

    def sample(self, mean, log_var, eps_scale=1):
        
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean  
              
    def sample_inference(self, mean, log_var, eps_scale=1):
        
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mean)   

    def forward(self, batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device, teacher_forcing_ratio=1.0):
        # encode
        h_G_mean, h_G_var = self.Encoder(batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not self.hidden_dim==self.embedding_dim:
            h_G_mean = self.lincompress(h_G_mean)
            h_G_var = self.lincompress(h_G_var)
        z = self.sample(h_G_mean, h_G_var, eps_scale=self.eps)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())/(len(batch_list.ptr-1))
    
        # decode with teacher forcing
        recon_loss, acc, predictions, target = self.Decoder(batch_list, z, teacher_forcing_ratio=teacher_forcing_ratio)
    
        return recon_loss + self.beta*kl_loss, recon_loss, kl_loss, acc, predictions, target, z
    

    def inference(self, data, device, dest_is_origin_matrix=None, inc_edges_to_atom_matrix=None, sample=False, log_var=None):
        if isinstance(data, torch.Tensor): # tensor with latent representations
            if data.size(-1) != self.embedding_dim: #tensor input needs to be embedding/hidden size
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.embedding_dim))
            if data.dim() == 1: # is the case if data is only one sample
                mean = data.unsqueeze(0) #dimension for batch size
            else:
                mean = data
        elif isinstance(data, Data): # batch list of graphs
            mean, log_var = self.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            if not self.hidden_dim==self.embedding_dim:
                mean = self.lincompress(mean)
                log_var = self.lincompress(log_var)
           
        if sample:
            z= self.sample_inference(mean, log_var, eps_scale=self.eps)
        else:
            z= mean
            log_var = 0
       
        predictions = self.Decoder.inference(z)
               
        return predictions, mean, log_var, z
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))

class PropertyRelationshipModule(nn.Module):
    """Module to handle property relationships via equations"""
    
    def __init__(self, equation_str, source_props, target_prop, device):
        super().__init__()
        self.equation_str = equation_str
        self.source_props = source_props
        self.target_prop = target_prop
        self.device = device
        
        # Parse equation to identify operations
        self.parse_equation()
        
    def parse_equation(self):
        """Parse equation string to create computation graph"""
        # Store original equation for reference
        self.original_equation = self.equation_str
        
        # Simple parser - can be extended for more complex equations
        # Supports: +, -, *, /, abs(), exp(), log(), sqrt(), power()
        self.equation_lambda = self._create_equation_function()
        
    def _create_equation_function(self):
        """Create a function from equation string"""
        # Replace property names with indexed variables for eval
        equation = self.equation_str
        for i, prop in enumerate(self.source_props):
            equation = equation.replace(prop, f"props[{i}]")
        
        # Create safe evaluation function
        def equation_func(props):
            # Safe functions that can be used in equations
            safe_dict = {
                'abs': torch.abs,
                'exp': torch.exp,
                'log': torch.log,
                'sqrt': torch.sqrt,
                'pow': torch.pow,
                'min': torch.min,
                'max': torch.max,
                'mean': torch.mean,
                'tanh': torch.tanh,
                'sigmoid': torch.sigmoid,
            }
            
            # Add props to local scope
            local_dict = {'props': props}
            local_dict.update(safe_dict)
            
            try:
                # Safe evaluation with limited scope
                result = eval(equation, {"__builtins__": {}}, local_dict)
                return result
            except Exception as e:
                print(f"Error evaluating equation {self.equation_str}: {e}")
                return torch.zeros_like(props[0])
        
        return equation_func
    
    def forward(self, property_predictions):
        """
        Args:
            property_predictions: dict of {property_name: tensor}
        Returns:
            predicted target property value
        """
        # Gather source property values
        source_values = []
        for prop in self.source_props:
            if prop in property_predictions:
                source_values.append(property_predictions[prop])
            else:
                raise ValueError(f"Source property {prop} not found in predictions")
        
        # Apply equation
        result = self.equation_lambda(source_values)
        return result

def parse_property_relationships(relationship_strings):
    """Parse command line relationship strings into structured format"""
    relationships = {}
    if relationship_strings is None:
        return relationships
        
    for rel_str in relationship_strings:
        # Format: "target=equation" e.g., "bandgap=abs(EA-IP)"
        if '=' not in rel_str:
            print(f"Warning: Invalid relationship format: {rel_str}")
            continue
            
        target, equation = rel_str.split('=', 1)
        target = target.strip()
        equation = equation.strip()
        
        # Extract source properties from equation
        # This is a simple extraction - looks for uppercase property names
        import re
        # Match property names (uppercase letters potentially followed by lowercase)
        potential_props = re.findall(r'\b[A-Z][A-Za-z0-9_]*\b', equation)
        source_props = [p for p in potential_props if p not in ['abs', 'exp', 'log', 'sqrt']]
        
        relationships[target] = {
            'equation': equation,
            'sources': source_props
        }
    
    return relationships
    
class G2S_VAE_PPguided(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, embedding_dim, device, model_config, vocab, seed, loss_weights=None, add_latent=True):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim    
        self.hidden_dim=hidden_dim
        self.device=device
        self.seed=seed
        self.eps = model_config['epsilon']

        # Property relationship modules
        self.property_relationships = model_config.get('property_relationships', {})
        self.relationship_weight = model_config.get('relationship_weight', 0.1)
        self.relationship_modules = nn.ModuleDict()
        
        # Store property names for relationship module
        self.property_names = model_config.get('property_names', [])
        
        # Create relationship modules if specified
        if self.property_relationships:
            for target_prop, rel_info in self.property_relationships.items():
                self.relationship_modules[target_prop] = PropertyRelationshipModule(
                    rel_info['equation'],
                    rel_info['sources'],
                    target_prop,
                    device
                )

        try: 
            self.embedding_dim = model_config['embedding_dim']
        except:
            self.embedding_dim = embedding_dim
        # in case beta is schedule the value will be specified in train.py
        if not model_config["beta"] =="schedule":
            self.beta=1.0
        self.config = model_config
        self.vocab = vocab
        
        # Get property count from model config - default to 2 for backward compatibility
        self.property_count = model_config.get('property_count', 2)
        
        #self.max_n=data_config['max_num_nodes']
        #if model_config['pooling']=='custom':
        #    self.Encoder = GraphEncoder_GMT(node_dim, edge_dim, hidden_dim, device, model_config)
        #elif model_config['pooling']=='mean':
        self.Encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, device, model_config)
        self.Decoder = SequenceDecoder(model_config, vocab, loss_weights, add_latent=add_latent)
        if not self.hidden_dim==self.embedding_dim:
            self.lincompress = Linear(self.hidden_dim, self.embedding_dim).to(device)
        
        self.pp_ffn_hidden = 56
        self.alpha = model_config['max_alpha'] if model_config['alpha'] == "fixed" else 0.0
        #self.alpha=0.1
        #self.max_n=data_config['max_num_nodes']
        self.PP_lin1 = Sequential(Linear(embedding_dim, self.pp_ffn_hidden), ReLU(), ).to(device)
        # Make property prediction layer flexible
        self.PP_lin2 = Sequential(Linear(self.pp_ffn_hidden, self.property_count)).to(device)
        self.dropout = nn.Dropout(0.2)

    def sample(self, mean, log_var, eps_scale=0.01):
        
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean

    def sample_inference(self, mean, log_var, eps_scale=1):
        
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mean)   

    def forward(self, batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device, teacher_forcing_ratio=1.0):
        # encode
        h_G_mean, h_G_var = self.Encoder(batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not self.hidden_dim==self.embedding_dim:
            h_G_mean = self.lincompress(h_G_mean)
            h_G_var = self.lincompress(h_G_var)
        z = self.sample(h_G_mean, h_G_var, eps_scale=self.eps)
        
        # Calculate raw KLD
        kl_loss_raw = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())
        kl_loss = kl_loss_raw / (len(batch_list.ptr-1))
        
        # Soft annealing instead of hard clamp
        if kl_loss > 100:
            # Gradually reduce instead of hard cutoff
            kl_loss = 100 + torch.tanh((kl_loss - 100) / 100) * 20  # Soft ceiling around 120
        
        # Debug print every 100 batches
        if hasattr(self, '_batch_counter'):
            self._batch_counter += 1
        else:
            self._batch_counter = 0
            
        if self._batch_counter % 100 == 0:
            kl_loss_unclamped = kl_loss_raw / (len(batch_list.ptr-1))
            print(f"KLD Debug - Raw: {kl_loss_unclamped.item():.2f}, Annealed: {kl_loss.item():.2f}")
    
        # Property predictions with flexible number of properties
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        pp_hidden = self.dropout(pp_hidden)
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, property_count]
        
        # 🔧 FIXED: Dynamically handle property targets based on available properties
        y_true_list = []
        
        # Check what properties actually exist in the batch
        available_properties = []
        for prop_name in ['y1', 'y2', 'y3', 'y4']:  # Check up to 4 properties
            if hasattr(batch_list, prop_name):
                available_properties.append(prop_name)
        
        # Build y_true based on available properties
        for i in range(self.property_count):
            if i < len(available_properties):
                prop_attr = available_properties[i]
                y_prop = torch.unsqueeze(getattr(batch_list, prop_attr).float(), 1)
            else:
                # If property doesn't exist, create NaN tensor (will be masked out)
                y_prop = torch.full((batch_list.y1.size(0), 1), float('nan'), device=device)
            y_true_list.append(y_prop)
        
        y_true = torch.cat(y_true_list, dim=1)
        mse = self.masked_mse(y_true, y)
    
        # decode
        recon_loss, acc, predictions, target = self.Decoder(batch_list, z, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # ADD: Validity loss
        validity_penalty = self.compute_validity_loss(predictions)
        
        # ENHANCED: Increase validity weight significantly after warmup
        validity_weight = 0.1 if hasattr(self, '_batch_counter') and self._batch_counter < 1000 else 0.5
        
        # Calculate relationship loss if applicable
        relationship_loss = torch.tensor(0.0, device=device)
        if self.property_relationships and self.training:
            # Get all property predictions
            property_preds = {}
            for i, prop_name in enumerate(self.property_names):
                if i < y.shape[1]:  # Ensure we don't go out of bounds
                    property_preds[prop_name] = y[:, i]
            
            # Calculate relationship losses
            for target_prop, rel_module in self.relationship_modules.items():
                if target_prop in property_preds:
                    # Check if all source properties are available
                    rel_info = self.property_relationships[target_prop]
                    if all(src in property_preds for src in rel_info['sources']):
                        # Predict target using relationship
                        predicted_target = rel_module(property_preds)
                        actual_target = property_preds[target_prop]
                        
                        # Only calculate loss for non-NaN values
                        mask = ~torch.isnan(actual_target)
                        if mask.any():
                            rel_loss = F.mse_loss(predicted_target[mask], actual_target[mask])
                            relationship_loss += rel_loss
        
        # Modified total loss calculation with relationship loss
        total_loss = recon_loss + self.beta*kl_loss + self.alpha*mse + validity_weight*validity_penalty + self.relationship_weight*relationship_loss
        
        # Return with additional relationship loss
        return total_loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y, relationship_loss

    def compute_validity_loss(self, predictions):
        """Enhanced validity loss with stronger penalties and rewards"""
        validity_penalty = 0.0
        batch_size = len(predictions)
        
        for pred in predictions:
            try:
                # Convert prediction to string
                pred_tokens = pred[0] if isinstance(pred, tuple) else pred
                if hasattr(pred_tokens, 'cpu'):
                    pred_tokens = pred_tokens.cpu().numpy()
                
                tokens = tokenids_to_vocab(pred_tokens, self.vocab)
                if tokens:
                    smiles_string = combine_tokens(tokens, tokenization="RT_tokenized")
                    
                    # ENHANCED PENALTIES with graduated severity
                    # Critical format requirements (highest penalty)
                    if '|' not in smiles_string:
                        validity_penalty += 2.0  # Increased from 1.0
                    elif smiles_string.count('|') < 3:
                        validity_penalty += 1.5  # Must have all pipe sections
                        
                    if '[*:' not in smiles_string:
                        validity_penalty += 2.0  # Increased from 1.0
                        
                    # Connectivity requirements
                    if '<' not in smiles_string:
                        validity_penalty += 1.5
                    if ':' not in smiles_string.split('|')[-1] if '|' in smiles_string else smiles_string:
                        validity_penalty += 1.0
                        
                    # Balance requirements
                    if smiles_string.count('(') != smiles_string.count(')'):
                        validity_penalty += 1.0  # Increased from 0.5
                    if smiles_string.count('[') != smiles_string.count(']'):
                        validity_penalty += 1.0  # Increased from 0.5
                        
                    # REWARDS for good format (negative penalty)
                    if len(smiles_string) > 30 and '|' in smiles_string:
                        validity_penalty -= 0.3  # Reward reasonable length
                    if smiles_string.count('|') == 3:  # Exact format
                        validity_penalty -= 0.5
                    if '[*:1]' in smiles_string and '[*:2]' in smiles_string:
                        validity_penalty -= 0.5  # Both attachment points
                        
            except:
                validity_penalty += 1.0  # Penalty for unparseable
                
        return torch.tensor(validity_penalty / batch_size, device=self.device, requires_grad=True)

    def masked_mse(self, y_true, y_pred):
        # Create a mask where the true values are not NaN
        mask = ~torch.isnan(y_true)
        
        # Only calculate MSE for non-NaN values
        if mask.any():
            # Calculate MSE only for non-missing values
            mse = F.mse_loss(y_pred[mask], y_true[mask], reduction='mean')
            return mse
        else:
            # If all values are NaN, return zero loss
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
    def inference(self, data, device, dest_is_origin_matrix=None, inc_edges_to_atom_matrix=None, sample=False, log_var=None):
        #TODO: Function arguments (test batch?, single graph?, latent representation?), right encoder call
        if isinstance(data, torch.Tensor): # tensor with latent representations
            if data.size(-1) != self.embedding_dim: #tensor input needs to be embedding/hidden size
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.embedding_dim))
            if data.dim() == 1: # is the case if data is only one sample
                mean = data.unsqueeze(0) #dimension for batch size
            else:
                mean = data
        elif isinstance(data, Data): # batch list of graphs
            mean, log_var = self.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            if not self.hidden_dim==self.embedding_dim:
                mean = self.lincompress(mean)
                log_var = self.lincompress(log_var)
           
        if sample:
            z= self.sample_inference(mean, log_var, eps_scale=self.eps)
        else:
            z= mean
            log_var = 0
       
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, property_count]

        predictions = self.Decoder.inference(z)
        # Property predictions 
               
        return predictions, mean, log_var, z, y 
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

    
class G2S_VAE_PPguideddisabled(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, embedding_dim, device, model_config, vocab, seed, loss_weights=None, add_latent=True):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.seed=seed
        self.eps = model_config['epsilon']

        try: 
            self.embedding_dim = model_config['embedding_dim']
        except:
            self.embedding_dim = embedding_dim
        # in case beta is schedule the value will be specified in train.py
        if not model_config["beta"] =="schedule":
            self.beta=1.0
        self.config = model_config
        self.vocab = vocab
        
        # Get property count from model config - default to 2 for backward compatibility
        self.property_count = model_config.get('property_count', 2)
        
        #self.max_n=data_config['max_num_nodes']
        #if model_config['pooling']=='custom':
        #    self.Encoder = GraphEncoder_GMT(node_dim, edge_dim, hidden_dim, device, model_config)
        #elif model_config['pooling']=='mean':
        self.Encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, device, model_config)
        self.Decoder = SequenceDecoder(model_config, vocab, loss_weights, add_latent=add_latent)
        if not self.hidden_dim==self.embedding_dim:
            self.lincompress = Linear(self.hidden_dim, self.embedding_dim).to(device)
        
        self.pp_ffn_hidden = 56
        self.alpha = model_config['max_alpha'] if model_config['alpha'] == "fixed" else 0.0
        #self.max_n=data_config['max_num_nodes']
        self.PP_lin1 = Sequential(Linear(embedding_dim, self.pp_ffn_hidden), ReLU(), ).to(device)
        # Make property prediction layer flexible
        self.PP_lin2 = Sequential(Linear(self.pp_ffn_hidden, self.property_count)).to(device)
        self.dropout = nn.Dropout(0.2)

    def sample(self, mean, log_var, eps_scale=0.01):
        
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean        

    def forward(self, batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
        # encode
        h_G_mean, h_G_var = self.Encoder(batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not self.hidden_dim==self.embedding_dim:
            h_G_mean = self.lincompress(h_G_mean)
            h_G_var = self.lincompress(h_G_var)
        z = self.sample(h_G_mean, h_G_var, eps_scale=self.eps)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())/(len(batch_list.ptr-1))

        # Property predictions with flexible number of properties
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        pp_hidden = self.dropout(pp_hidden)
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, property_count]
        
        # Dynamically handle property targets based on property count
        y_true_list = []
        for i in range(self.property_count):
            if i == 0:
                y_prop = torch.unsqueeze(batch_list.y1.float(), 1)
            elif i == 1:
                y_prop = torch.unsqueeze(batch_list.y2.float(), 1)
            else:
                # For additional properties beyond y1 and y2, check if they exist
                prop_attr = f'y{i+1}'
                if hasattr(batch_list, prop_attr):
                    y_prop = torch.unsqueeze(getattr(batch_list, prop_attr).float(), 1)
                else:
                    # If property doesn't exist, create NaN tensor
                    y_prop = torch.full((batch_list.y1.size(0), 1), float('nan'), device=device)
            y_true_list.append(y_prop)
        
        y_true = torch.cat(y_true_list, dim=1)
        mse = self.masked_mse(y_true, y)

        # decode
        recon_loss, acc, predictions, target = self.Decoder(batch_list, z)
        
        # Notice that the MSE explicitely is not used in the aggregated overall loss, so the loss does not contribute to changing the parameters of encoder and decoder. 
        return recon_loss + self.beta*kl_loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y

    def masked_mse(self, y_true, y_pred):
        # Create a mask where the true values are not NaN
        mask = ~torch.isnan(y_true)
        
        # Only calculate MSE for non-NaN values
        if mask.any():
            # Calculate MSE only for non-missing values
            mse = F.mse_loss(y_pred[mask], y_true[mask], reduction='mean')
            return mse
        else:
            # If all values are NaN, return zero loss
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
    def inference(self, data, device, dest_is_origin_matrix=None, inc_edges_to_atom_matrix=None, sample=False, log_var=None):
        #TODO: Function arguments (test batch?, single graph?, latent representation?), right encoder call
        if isinstance(data, torch.Tensor): # tensor with latent representations
            if data.size(-1) != self.embedding_dim: #tensor input needs to be embedding/hidden size
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.embedding_dim))
            if data.dim() == 1: # is the case if data is only one sample
                mean = data.unsqueeze(0) #dimension for batch size
            else:
                mean = data
        elif isinstance(data, Data): # batch list of graphs
            mean, log_var = self.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            if not self.hidden_dim==self.embedding_dim:
                mean = self.lincompress(mean)
                log_var = self.lincompress(log_var)
           
        if sample:
            z= self.sample(mean, log_var, eps_scale=self.eps)
        else:
            z= mean
            log_var = 0
       
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, property_count]

        predictions = self.Decoder.inference(z)
        # Property predictions 
               
        return predictions, mean, log_var, z, y 
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

class G2S_VAE_Transfer(nn.Module):
    """Transfer learning version that can load pretrained components and adapt to new properties"""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, embedding_dim, device, model_config, vocab, seed, 
                 loss_weights=None, add_latent=True, pretrained_model=None, freeze_components=None):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.eps = model_config['epsilon']
        self.embedding_dim = model_config.get('embedding_dim', embedding_dim)
        self.config = model_config
        self.vocab = vocab
        
        # Property configuration for transfer learning
        self.source_properties = model_config.get('source_properties', ['EA', 'IP'])
        self.target_properties = model_config.get('target_properties', ['bandgap'])
        self.property_count = len(self.target_properties)
        
        if freeze_components is None:
            freeze_components = {'encoder': False, 'decoder': False}
        
        # Load pretrained components if provided
        if pretrained_model is not None:
            print(f"Loading pretrained components from model trained on {self.source_properties}")
            self.Encoder = pretrained_model.Encoder
            self.Decoder = pretrained_model.Decoder
            
            # Optionally freeze components
            if freeze_components['encoder']:
                print("Freezing encoder weights")
                for param in self.Encoder.parameters():
                    param.requires_grad = False
                    
            if freeze_components['decoder']:
                print("Freezing decoder weights")
                for param in self.Decoder.parameters():
                    param.requires_grad = False
        else:
            # Initialize new components (for stage 1)
            self.Encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, device, model_config)
            self.Decoder = SequenceDecoder(model_config, vocab, loss_weights, add_latent=add_latent)
        
        # Always create new property predictor for target properties
        self.pp_ffn_hidden = 128  # Larger for better transfer
        self.alpha = model_config['max_alpha'] if model_config['alpha'] == "fixed" else 0.0
        self.beta = 1.0 if model_config["beta"] != "schedule" else 1.0
        
        # Enhanced property predictor for transfer learning
        self.PP_lin1 = Sequential(
            Linear(embedding_dim, self.pp_ffn_hidden),
            nn.LayerNorm(self.pp_ffn_hidden),  # Add normalization
            ReLU(),
            nn.Dropout(0.3)
        ).to(device)
        
        self.PP_lin2 = Sequential(
            Linear(self.pp_ffn_hidden, self.pp_ffn_hidden // 2),
            nn.LayerNorm(self.pp_ffn_hidden // 2),
            ReLU(),
            nn.Dropout(0.2),
            Linear(self.pp_ffn_hidden // 2, self.property_count)
        ).to(device)
        
        # Compression layer if needed
        if not self.hidden_dim == self.embedding_dim:
            self.lincompress = Linear(self.hidden_dim, self.embedding_dim).to(device)
    
    # Use the same forward, sample, and inference methods as G2S_VAE_PPguided
    def sample(self, mean, log_var, eps_scale=0.01):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean
    
    def forward(self, batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
        # Encode
        h_G_mean, h_G_var = self.Encoder(batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not self.hidden_dim == self.embedding_dim:
            h_G_mean = self.lincompress(h_G_mean)
            h_G_var = self.lincompress(h_G_var)
        z = self.sample(h_G_mean, h_G_var, eps_scale=self.eps)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())/(len(batch_list.ptr-1))
    
        # Property predictions with better architecture
        pp_hidden = self.PP_lin1(z)
        y = self.PP_lin2(pp_hidden)
        
        # Get true values for target properties
        y_true_list = []
        for i, prop_name in enumerate(self.target_properties):
            # Map property names to batch attributes
            prop_attr_map = {
                'EA': 'y1', 'IP': 'y2', 'bandgap': 'y3',  # Adjust based on your setup
                # Add more mappings as needed
            }
            
            prop_attr = prop_attr_map.get(prop_name, f'y{i+1}')
            if hasattr(batch_list, prop_attr):
                y_prop = torch.unsqueeze(getattr(batch_list, prop_attr).float(), 1)
            else:
                y_prop = torch.full((batch_list.y1.size(0), 1), float('nan'), device=device)
            y_true_list.append(y_prop)
        
        y_true = torch.cat(y_true_list, dim=1)
        mse = self.masked_mse(y_true, y)
    
        # Decode
        recon_loss, acc, predictions, target = self.Decoder(batch_list, z)
    
        return recon_loss + self.beta*kl_loss + self.alpha*mse, recon_loss, kl_loss, mse, acc, predictions, target, z, y
    
    def masked_mse(self, y_true, y_pred):
        mask = ~torch.isnan(y_true)
        if mask.any():
            mse = F.mse_loss(y_pred[mask], y_true[mask], reduction='mean')
            return mse
        else:
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
    def inference(self, data, device, dest_is_origin_matrix=None, inc_edges_to_atom_matrix=None, sample=False, log_var=None):
        # Same as G2S_VAE_PPguided
        if isinstance(data, torch.Tensor):
            if data.size(-1) != self.embedding_dim:
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.embedding_dim))
            if data.dim() == 1:
                mean = data.unsqueeze(0)
            else:
                mean = data
        elif isinstance(data, Data):
            mean, log_var = self.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            if not self.hidden_dim == self.embedding_dim:
                mean = self.lincompress(mean)
                log_var = self.lincompress(log_var)
        
        if sample:
            z = self.sample_inference(mean, log_var, eps_scale=self.eps)
        else:
            z = mean
            log_var = 0
        
        pp_hidden = self.PP_lin1(z)
        y = self.PP_lin2(pp_hidden)
        
        predictions = self.Decoder.inference(z)
        
        return predictions, mean, log_var, z, y
    
    def sample_inference(self, mean, log_var, eps_scale=1):
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mean)

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets
     from https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b '''

    def __init__(self, gamma, scale_loss=10, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma
        self.scale_loss = scale_loss

    def forward(self, input, target):
        cross_entropy = super().forward(input, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        loss = loss*self.scale_loss
        if self.reduction == 'mean': 
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
