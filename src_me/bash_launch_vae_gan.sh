#!/bin/bash

# VAE train (test)
python vae.py --data_path "/home/guillaumeolt/CMPLI/Projet_ED_GAN/src/CPMolGAN/data/ChEMBL_standardized_smiles.csv" --n_smiles 10000 --weights_path ../Results/model_vae_TEST/vae_test_all_original_test.weights.h5

# VAE train ("oscar parameters")
python vae.py --data_path "/home/guillaumeolt/CMPLI/Projet_ED_GAN/src/CPMolGAN/data/ChEMBL_standardized_smiles.csv" --n_smiles -1 --batch_size 512 --epochs 10

# VAE test
python prediction.py --data_path "/home/guillaumeolt/CMPLI/Projet_ED_GAN/src/CPMolGAN/data/ChEMBL_standardized_smiles.csv" --n_smiles 1000 --weights_path_vae ../Results/model_vae_TEST/vae_test_all_original_bis.weights.h5
python prediction.py --data_path "/home/guillaumeolt/CMPLI/Projet_ED_GAN/src/CPMolGAN/data/ChEMBL_standardized_smiles.csv" --n_smiles 1000 --weights_path_vae ../Results/model_vae_TEST/vae_test_all_original.weights.h5

python prediction.py --data_path "/home/guillaumeolt/CMPLI/Projet_ED_GAN/src/CPMolGAN/data/ChEMBL_standardized_smiles.csv" --n_smiles 1000 --weights_path_vae ../Results/model_vae_TEST/vae_test_all_original_10_05_2024.weights.h5

# GAN train
GAN.py --weights_path_vae ../Results/model_vae_TEST/vae_test_all_original_bis.weights.h5

# GAN test

## T5 GAN

# lincs
python model_with_t5chem_train.py --method lincs_frogs_mols --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --embedding_size 978
# lincs_frogs_mols
python model_with_t5chem_train.py --method lincs_frogs_mols --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_frogs --output_dir ../Results/model_t5chem_gene_lincs_frogs_mols
# lincs_frogs_mols up down
python model_with_t5chem_train.py --method lincs_frogs_mols --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_frogs_up_down --output_dir ../Results/model_t5chem_gene_lincs_frogs_mols_up_down --embedding_size 1024
# lincs_text_mols
python model_with_t5chem_train.py --method lincs_text_mols --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_text --output_dir ../Results/model_t5chem_lincs_text_mols
# lincs_text_mols positional
python model_with_t5chem_train.py --method lincs_text_mols --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_text_positional --output_dir ../Results/model_t5chem_lincs_text_positional_mols --vocab ../Data/data_t5chem/MCF7_24h_10um/z_score_vocab.txt


## T5 GAN Validation
# lincs
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978
# lincs_frogs_mols
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_frogs --output_dir ../Results/model_t5chem_gene_lincs_frogs_mols --model_dir ../Results/model_t5chem_gene_lincs_frogs_mols/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_frogs_mols
# lincs_frogs_mols up down
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_frogs_up_down --output_dir ../Results/model_t5chem_gene_lincs_frogs_mols_up_down --model_dir ../Results/model_t5chem_gene_lincs_frogs_mols_up_down/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_frogs_mols --embedding_size 1024
# lincs_text_mols
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_text --output_dir ../Results/model_t5chem_lincs_text_mols --model_dir ../Results/model_t5chem_lincs_text_mols/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_text_mols



# T5 GAN validation EDC

# lincs
## Train
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file ./Data/data_t5chem/datasets_MCF7/datasets_MCF7/lincs/train_ED.csv --num_beams 1 --num_preds 1
## Val
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file ./Data/data_t5chem/datasets_MCF7/datasets_MCF7/lincs/val_ED.csv --num_beams 1 --num_preds 1
## Test
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file ./Data/data_t5chem/datasets_MCF7/datasets_MCF7/lincs/test_ED.csv --num_beams 1 --num_preds 1














