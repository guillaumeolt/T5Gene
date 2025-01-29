import os
import argparse
import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import pandas as pd
import json
from model import VAE
from model import SelfiesDataset
from model import frange_cycle_linear
from model import get_optim_params

# Global variables for hardcoded model parameters
LATENT_DIM = 256
#NOISE_DIM = 1000  
#CONDITION_RANDOM_SEED = 42
#MAX_SMILES_LENGTH = 120
MAX_SELFIE_LENGTH = 160 # 148+2
#WEIGHTS_PATH = None  # os.path.join(WEIGHTS_PATH, 'autoencoder.h5')
#NUM_TOKENS = len(CHARSET)

PATH_SMILES_CHEMBL="/home/guillaumeolt/CMPLI/Projet_ED_GAN/src/CPMolGAN/data/ChEMBL_standardized_smiles.csv"
PATH_SELFIES_VOCABULARY="/home/gui"
N_SMILES = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of each GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("CUDA is not available. Using CPU.")

def main(args, latent_dim = LATENT_DIM):
    ##############
    # Load data chemicals
    mat_smiles_chembl = pd.read_csv('../Data/chemical_data/mat_smiles_chembl_clean.csv',sep="\t")
    # Load selfies vocabulary
    json_file_path = '../Data/chemical_data/chembl_vocab.json'
    with open(json_file_path, 'r') as json_file:
        vocab_selfies = json.load(json_file)
    vocab_selfies = vocab_selfies
    vocab = {token: idx for idx, token in enumerate(vocab_selfies)}
    vocab_size = len(vocab)
    
    # Set up data
    max_length = MAX_SELFIE_LENGTH
    data_train = mat_smiles_chembl["SELFIES"][(mat_smiles_chembl["SELFIES"].notnull()) & (mat_smiles_chembl["len_selfies"] < max_length-5)].tolist()[:args.n_smiles]
    data_val = mat_smiles_chembl["SELFIES"][(mat_smiles_chembl["SELFIES"].notnull()) & (mat_smiles_chembl["len_selfies"] < max_length-5)].tolist()[args.n_smiles:args.n_smiles+500]
    data_test = mat_smiles_chembl["SELFIES"][(mat_smiles_chembl["SELFIES"].notnull()) & (mat_smiles_chembl["len_selfies"] < max_length-5)].tolist()[:100]

    # add <BOS> and <EOS> tokens
    data_train = ['[BOS]' + selfies + '[EOS]' for selfies in data_train]
    data_val = ['[BOS]' + selfies + '[EOS]' for selfies in data_val]
    data_test = ['[BOS]' + selfies + '[EOS]' for selfies in data_test]
    # Set up data loading
    batch_size = args.batch_size
    dataset_train = SelfiesDataset(data_train, vocab_selfies, max_length)  # Initialize your custom dataset
    dataset_val = SelfiesDataset(data_val, vocab_selfies, max_length)  # Initialize your custom dataset
    dataset_test = SelfiesDataset(data_test, vocab_selfies, max_length)  # Initialize your custom dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    # Initialize the VAE model
    epochs = 10
    embedding_dim = 150  # Dimensionality of your input data
    latent_dim = args.latent_dim  # Dimensionality of the latent space
    hidden_dim = 512 # Dimensionality of the hidden state
    kl_weight = 0 # Weight of the KL divergence term in the loss function
    kl_weight_beta_np_cyc = frange_cycle_linear(0.0, 1, epochs, 1, 1.0)
    clip_grad = 50
    vae = VAE(vocab, vocab_size, embedding_dim, hidden_dim, latent_dim, max_length)
    vae.to(device)
        
    # Define optimizer
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # Set dic history
    dic_history = {
        'epoch': [],
        'loss_train': [],
        'kl_loss_train': [],
        'reconstruction_loss_train': [],
        'loss_val': [],
    }

    # Training vae
    for epoch in range(epochs):
        total_loss = 0
        vae.train()  # Set the model to train mode
        for batch_idx, data in tqdm(enumerate(dataloader_train)):
            inputs = data.to(device)  # Assuming you are using GPU
            kl_loss, recon_loss = vae(inputs)

            # Get the current annealed value for kl_weight
            kl_weight = kl_weight_beta_np_cyc[epoch]

            loss = kl_weight * kl_loss + recon_loss
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            clip_grad_norm_(get_optim_params(vae), clip_grad)
            optimizer.step()
        
        # Validation phase
        vae.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # No need to compute gradients during validation
            for data in tqdm(dataloader_val, desc="Validation"):
                inputs = data.to(device)
                kl_loss, recon_loss = vae(inputs)
                loss = kl_weight * kl_loss + recon_loss
                val_loss += loss.item()
        print('Training - Epoch {}, Loss: {:.4f}, kl_loss: {:.4f}, reconstruction_loss: {:.4f}, kl_weight: {}'.format(epoch+1,
                                                                                                                    total_loss / len(dataloader_train),
                                                                                                                    kl_loss, recon_loss,
                                                                                                                    kl_weight))
        print('Validation - Epoch {}, Loss: {:.4f}'.format(epoch+1, val_loss / len(dataloader_val)))
        
        # Test phase
        # Reconstruction accuracy
        """
        with torch.no_grad():  # No need to compute gradients during validation
            num = 0
            for batch_idx, data in enumerate(dataloader_test): 
                inputs = data.to(device)  # Assuming you are using GPU
                for ech in inputs:
                    ech_selfie = vae.tensor2selfies(ech)
                    res_lst = []
                    for i in tqdm(range(500)):
                        res = vae.encode_decode(ech.unsqueeze(0))
                        res_lst.extend(res)
                    if ech in res:
                        num += 1
                    print("recons num: ", num)
        print('Test - Epoch {}, Acc {}'.format(epoch+1, num*1.0/100))        
        """




        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': total_loss / len(dataloader_train),
            'kl_loss_train': kl_loss,
            'reconstruction_loss_train': recon_loss,
            'loss_val': val_loss / len(dataloader_val),
            }, args.loss_output_path)
        
        # Save the model
        torch.save(vae, args.weights_path)

        # Convert tensors to numpy arrays before saving
        train_loss = total_loss / len(dataloader_train)
        train_kl_loss = kl_loss.item()
        train_recon_loss = recon_loss.item()
        val_loss_avg = val_loss / len(dataloader_val)

        # Append training history to dictionary
        dic_history['epoch'].append(epoch)
        dic_history['loss_train'].append(train_loss)
        dic_history['kl_loss_train'].append(train_kl_loss)
        dic_history['reconstruction_loss_train'].append(train_recon_loss)
        dic_history['loss_val'].append(val_loss_avg)
    print(dic_history)
    with open(args.loss_output_path+".json", 'w') as f:
        json.dump(dic_history, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE model for SMILES representation.')
    parser.add_argument('--data_path', type=str, default=PATH_SMILES_CHEMBL, help='Path to the data file.', required=True)
    parser.add_argument('--n_smiles', type=int, default=N_SMILES, help='Number of SMILES to use for training. If number == -1 then all SMILES are used.')
    parser.add_argument('--max_smiles_length', type=int, default=120, help='Maximum length of SMILES sequences.')
    parser.add_argument('--max_selfie_length', type=int, default=MAX_SELFIE_LENGTH, help='Maximum length of SELFIES sequences.')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='Dimensionality of the latent space.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--weights_path', type=str, default="../Results/model_vae_TEST/vae_pytorch_model.pt", help='Path to save model weights.')
    parser.add_argument('--loss_output_path', type=str, default='../Results/model_vae_TEST/loss_values', help='Path to save loss values.')
    args = parser.parse_args()
    
    main(args)
