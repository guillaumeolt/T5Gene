import os
import argparse
from moses.metrics import get_all_metrics
import pandas as pd

PATH_DATA="/home/guillaumeolt/CMPLI/Projet_ED_GAN/Results/model_vae_TEST/vae_decoded_smiles.csv"
PATH_OUT="/home/guillaumeolt/CMPLI/Projet_ED_GAN/Results/model_vae_TEST/vae_moses_metrics.csv"

def main(args):

    # Load data
    data = pd.read_csv(args.data_path)
    metrics = get_all_metrics(gen=data["Decoded_SMILES"],
                              test=data["Original_SMILES"], k=3)
    dt_metrics = pd.DataFrame([metrics])
    print(dt_metrics)
    dt_metrics.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute moses metrics.')
    parser.add_argument('--data_path', type=str, default=PATH_DATA, help='Path to the data file.', required=True)
    parser.add_argument('--output_path', type=str, default=PATH_OUT, help='Path to output the moses metrics.', required=True)
    args = parser.parse_args()
    
    main(args) 
