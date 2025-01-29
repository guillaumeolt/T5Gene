#!/bin/bash
#SBATCH --partition=ipop-up
#SBATCH --job-name=t5chem
#SBATCH --account=ed-screen
#SBATCH --output=log_t5chem_gene_test_pytorch.log
#SBATCH --mem 64GB #256GB
#SBATCH --cpus-per-task 16
#SBATCH --gres=gpu:1 #a100_7g.80gb:1

#module load python-pytorch-tensorflow/3.9-1.11.0-2.6.2
echo TEST RUNNING
#python model_with_t5chem_gene.py
#python model_with_t5chem_test.py

<<comment
# lincs
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --num_beams 1000 --num_preds 1 --batch_size 1

# lincs permutation
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down_permutation --method lincs_mols --embedding_size 978 --num_beams 1000 --num_preds 1 --batch_size 1

# lincs_frogs_mols
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_frogs --output_dir ../Results/model_t5chem_gene_lincs_frogs_mols --model_dir ../Results/model_t5chem_gene_lincs_frogs_mols/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_frogs_mols --num_beams 1000 --num_preds 100 --batch_size 1
# lincs_frogs_mols up down
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_frogs_up_down --output_dir ../Results/model_t5chem_gene_lincs_frogs_mols_up_down --model_dir ../Results/model_t5chem_gene_lincs_frogs_mols_up_down/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_frogs_up_down_mols --embedding_size 1024 --num_beams 1000 --num_preds 100 --batch_size 1
# lincs_text_mols
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_text --output_dir ../Results/model_t5chem_lincs_text_mols --model_dir ../Results/model_t5chem_lincs_text_mols/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_text_mols --num_beams 1000 --num_preds 100 --batch_size 1
# lincs_text_mols positional
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs_text_positional --output_dir ../Results/model_t5chem_lincs_text_positional_mols --model_dir ../Results/model_t5chem_lincs_text_positional_mols/best_cp-10000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_text_positional_mols --vocab ../Data/data_t5chem/MCF7_24h_10um/z_score_vocab.txt --num_beams 1000 --num_preds 100 --batch_size 1

#### TEST EDC ####
# T5 GAN validation EDC

# lincs
## EDC
## Train
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file train_ED_active_ED --num_beams 100 --num_preds 1 --batch_size 8
## Val
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file val_ED_active_ED --num_beams 100 --num_preds 1 --batch_size 8
## Test
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file test_ED_active_ED --num_beams 100 --num_preds 1 --batch_size 8

comment
## ER
## Train
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file train_ED_active_ER --num_beams 100 --num_preds 1 --batch_size 1
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file train_ED_inactive_ER --num_beams 100 --num_preds 1 --batch_size 1
## Val
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file val_ED_active_ER --num_beams 100 --num_preds 1 --batch_size 1
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file val_ED_inactive_ER --num_beams 100 --num_preds 1 --batch_size 1
## Test
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file test_ED_active_ER --num_beams 100 --num_preds 1 --batch_size 1

python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file test_ED_inactive_ER --num_beams 100 --num_preds 1 --batch_size 1
<<comment
## AR
## Train
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file train_ED_active_AR --num_beams 100 --num_preds 1 --batch_size 8
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file train_ED_inactive_AR --num_beams 100 --num_preds 1 --batch_size 8
## Val
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file val_ED_active_AR --num_beams 100 --num_preds 1 --batch_size 8
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file val_ED_inactive_AR --num_beams 100 --num_preds 1 --batch_size 8
## Test
python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file test_ED_active_AR --num_beams 100 --num_preds 1 --batch_size 8

python model_with_t5chem_test.py --data_dir ../Data/data_t5chem/datasets_MCF7/lincs --output_dir ../Results/model_t5chem_gene_lincs --model_dir ../Results/model_t5chem_gene_lincs/best_cp-5000/ --data_dir_gene_knockdown ../Data/data_t5chem/datasets_MCF7/gene_knocked_down --method lincs_mols --embedding_size 978 --data_file test_ED_inactive_AR --num_beams 100 --num_preds 1 --batch_size 8

comment


