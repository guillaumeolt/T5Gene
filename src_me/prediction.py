import os
import argparse
import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
from model import VAE
from model import SelfiesDataset
import subprocess
from utils import write_xls_vae_prediction_test, plot_maccs_morgan_similarity
import selfies as sf

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

def main_vae(args):
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
    data_test = mat_smiles_chembl["SELFIES"][(mat_smiles_chembl["SELFIES"].notnull()) & (mat_smiles_chembl["len_selfies"] < max_length-5)].tolist()[:args.n_smiles]
    # add <BOS> and <EOS> tokens
    data_test_bos_eos = ['[BOS]' + selfies + '[EOS]' for selfies in data_test]
    # Set up data loading
    batch_size = 128
    dataset_test = SelfiesDataset(data_test_bos_eos, vocab_selfies, max_length)  # Initialize your custom dataset
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    # Load the model
    # Initialize the VAE model
    vae = torch.load("../Results/model_vae_TEST/vae_pytorch_model_14_05_2024.pt")
    vae.eval()  # Set the model to evaluation mode

    # Pedict smiles
    reconstructed_selfies = []
    for batch_idx, data in enumerate(dataloader_test): 
        inputs = data.to(device)  # Assuming you are using GPU
        reconstructed_selfies = reconstructed_selfies + vae.encode_decode(inputs)

    # Save results
    original_smiles = [sf.decoder(selfies) for selfies in data_test]
    decoded_smiles = [sf.decoder(selfies) for selfies in reconstructed_selfies]
    dt_test_data_decoded_smiles = [(i, p) for i,p in zip(original_smiles, decoded_smiles) if i != p]

    # Convert lists to DataFrame
    df_test_data_decoded_smiles = pd.DataFrame(dt_test_data_decoded_smiles, columns=['Original_SMILES', 'Decoded_SMILES'])

    # Save DataFrame to CSV
    df_test_data_decoded_smiles.to_csv(os.path.join(args.output_path, 'vae_decoded_smiles.csv'), index=False)

    print("CSV file saved successfully.")

    # Save DataFrame original_molecule and decoded_molecules with their respective images and tanimoto similarity (MACCS and morgan)
    write_xls_vae_prediction_test(dt_test_data_decoded_smiles, args.output_path_vae_prediction_xls, args.output_path_images_mols)

    # Plot MACCS and Morgan similarity
    plot_maccs_morgan_similarity(dt_test_data_decoded_smiles, args.output_path)

    # Launch MOSES metrics
    script_path = "path/to/your/script.py"
    script_file = "moses_metrics.py"
    subprocess.call(["conda", "run","-n","moses", "python3",
                        script_file,
                        "--data_path", os.path.join(args.output_path, 'vae_decoded_smiles.csv'),
                        "--output_path", os.path.join(args.output_path, 'vae_moses_metrics.csv')]
                        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test VAE models for SMILES representation.')
    parser.add_argument('--data_path', type=str, default=PATH_SMILES_CHEMBL, help='Path to the data file.', required=True)
    parser.add_argument('--n_smiles', type=int, default=500, help='Number of SMILES to use to test. If number == -1 then all SMILES are used.')
    parser.add_argument('--padding', type=str, default='right', choices=['left', 'right', 'none'], help='Padding direction for SMILES preprocessing.')
    parser.add_argument('--max_smiles_length', type=int, default=120, help='Maximum length of SMILES sequences.')
    parser.add_argument('--max_selfie_length', type=int, default=131, help='Maximum length of SELFIES sequences.')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimensionality of the latent space.')
    parser.add_argument('--weights_path_vae', type=str, default=None, help='Path to pre-trained weights.')
    parser.add_argument('--output_path', type=str, default='../Results/model_vae_TEST/', help='Path to save output results values.')

    parser.add_argument('--output_path_vae_prediction_xls', type=str, default='../Results/model_vae_TEST/vae_prediction_test.xlsx', help='Path to save output results values.')
    parser.add_argument('--output_path_images_mols', type=str, default='../Results/model_vae_TEST/img_mols_prediction/', help='Path to save output results values.')
    args = parser.parse_args()
    
    main_vae(args)

# python prediction.py --data_path "/home/guillaumeolt/CMPLI/Projet_ED_GAN/src/CPMolGAN/data/ChEMBL_standardized_smiles.csv" --n_smiles 10000 --weights_path_vae
