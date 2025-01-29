#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from transformers import T5ForConditionalGeneration

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, DataStructs
from rdkit.Contrib.SA_Score import sascorer
from fcd import get_fcd

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from functools import partial

from model_with_t5chem_tokenizer import MyGeneTokenizer
from model_with_t5chem_model import T5GeneToMol
from model_with_t5chem_utils import TaskDataset, data_collator, EarlyStopTrainer, TaskLincsDataset, data_collator_lincs

def standize(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except:
        return ''
def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0
def eval_validity(smiles_gen):
    valid = 0
    invalid = 0
    valid_smiles_gen = []
    for smi in smiles_gen:
        if Chem.MolFromSmiles(smi) and smi != "":
            valid += 1
            valid_smiles_gen.append(smi)
        else:
            invalid += 1
    return valid/len(smiles_gen), valid_smiles_gen
def eval_internal_diversity(smiles_gen, p=1):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_gen if Chem.MolFromSmiles(smi)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    similarities = []
    for i, fp1 in enumerate(fps):
        for j, fp2 in enumerate(fps):
            if i < j:
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                similarities.append(similarity)
    IntDiv = 1-p*((sum(similarities)**p)/(len(similarities)**2)**0.5)
    return IntDiv
def eval_uniqueness(smiles_gen):
    return len(set(smiles_gen))/len(smiles_gen)
def canonicalize_smiles(smiles_list):
    canonical_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            canonical_smiles.append(canonical_smi)
    return canonical_smiles
def eval_novelty(smiles_gen, smiles_train):
    smiles_gen_canonical = canonicalize_smiles(smiles_gen)
    smiles_train_canonical = canonicalize_smiles(smiles_train)
    return len(set(smiles_gen_canonical) - set(smiles_train_canonical)) / len(smiles_gen_canonical)

def eval_property_molecular_weight(smiles_gen):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_gen if Chem.MolFromSmiles(smi)]
    mw = [Chem.Descriptors.ExactMolWt(m) for m in mols]
    return mw
def eval_property_logp(smiles_gen):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_gen if Chem.MolFromSmiles(smi)]
    logp = [Chem.Crippen.MolLogP(m) for m in mols]
    return logp
def eval_property_synthetic_accessibility(smiles_gen):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_gen if Chem.MolFromSmiles(smi)]
    sa = []
    for m in mols:
        try:
            sa.append(sascorer.calculateScore(m))
        except:
            sa.append(None)
    return sa
def eval_property_qed(smiles_gen):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_gen if Chem.MolFromSmiles(smi)]
    qed = [Chem.QED.qed(m) for m in mols]
    return qed
def eval_property_fcd(smiles_gen, smiles_train):
    return get_fcd(smiles_gen, smiles_train)
# Function to compute Morgan fingerprint
def compute_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return None
# Function to compute Tanimoto similarity
def compute_tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Function to check if two SMILES strings represent the same molecule
def is_equal_molecule(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1, isomericSmiles=False)
        mol2 = Chem.MolFromSmiles(smiles2, isomericSmiles=False)
        if mol1 is None or mol2 is None:
            return False
        # Compare the canonical SMILES, which accounts for molecular equivalence
        return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)
    except:
        return False

def add_args(parser):
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../Data/data_t5chem/datasets_MCF7/lincs_frogs",
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../Results/model_t5chem_gene",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../Results/lincs_text_mols/best_cp-25000",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--pretrain",
        default='t5chem_model/models/pretrain/simple',
        help="Path to a pretrained model. If not given, we will train from scratch",
    )
    parser.add_argument(
        "--data_dir_gene_knockdown",
        type=str,
        default="../Data/data_t5chem/datasets_MCF7/datasets_MCF7/gene_knocked_down",
        required=True,
        help="The input data dir.",
    )  
    parser.add_argument(
        "--vocab",
        default="../Data/data_t5chem/MCF7_24h_10um/vocab.txt",
        help="Vocabulary file to load.",
    )
    parser.add_argument(
        "--random_seed",
        default=850,
        type=int,
        help="The random seed for model initialization",
    )
    parser.add_argument(
        "--log_step",
        default=5000,
        type=int,
        help="Logging after every log_step",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--max_source_length",
        default=1000,
        type=int,
        help="The maximum length of the input source text",
    )
    parser.add_argument(
        "--max_target_length",
        default=200,
        type=int,
        help="The maximum length of the target text",
    )
    parser.add_argument(
        "--embedding_size",
        default=512,
        type=int,
        help="The size of the embedding layer",
    )
    parser.add_argument(
        "--method",
        default="lincs_frogs_mols",
        type=str,
        help="The method the model is trained on. Options are lincs_frogs_mols, lincs_mols, lincs_text_mols, lincs_text_positional_mols",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--num_preds",
        default=1, 
        type=int,
        help="Number of predictions to generate",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../Data/data_t5chem/datasets_MCF7/datasets_MCF7/chemicals.txt",
        help="The input data file.",
    )

if __name__ == "__main__":
    #data_dir = "../Data/data_t5chem/MCF7_24h_10um/"
    #pretrain = 't5chem_model/models/pretrain/simple'
    #model_dir = "../Results/model_t5chem_gene/best_cp-25000/"
    #max_source_length = 1000
    #max_target_length = 200
    #prediction = ""
    #prefix = ""
    #num_beams = 1
    #num_preds = 1
    #batch_size = 32

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.isfile(args.data_dir):
        data_dir, base = os.path.split(args.data_dir)
        base = base.split('.')[0]
    else:
        base = "test"
    
    # ### load pretrain tokenizer

    tokenizer = MyGeneTokenizer(args.vocab)

    # ### load pretrain model
    if args.method == "lincs_frogs_mols" or args.method == "lincs_frogs_up_down_mols":
        model = T5GeneToMol.from_pretrained(args.pretrain, new_input_size=args.embedding_size)
    if args.method == "lincs_mols":
        model = T5GeneToMol.from_pretrained(args.pretrain, new_input_size=args.embedding_size)
    if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
        model = T5ForConditionalGeneration.from_pretrained(args.pretrain)

    # change embedding layer
    embedding_dim = model.shared.weight.size(1)  # Keep the same embedding dimension

    # Reinitialize the embedding layer
    model.shared = nn.Embedding(tokenizer.vocab_size, embedding_dim)

    # If you have tied embeddings (e.g., in T5), you might need to update other parts as well
    model.encoder.embed_tokens = model.shared
    model.decoder.embed_tokens = model.shared

    # If using tied weights, you need to tie them again
    model.lm_head.weight = model.shared.weight

    print(model)
        
    state_dict = torch.load(args.model_dir + "pytorch_model.bin")
    model.load_state_dict(state_dict)

    model.eval()
    model = model.to(device)

    ## TEST
    if False:
        if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
            testset = TaskLincsDataset(
                tokenizer=tokenizer, 
                data_dir=args.data_dir,
                type_path="test",
                max_target_length=args.max_target_length,
            )
            data_collator_padded = partial(
                data_collator_lincs, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))
        if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
            testset = TaskDataset(
                tokenizer, 
                data_dir=args.data_dir,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                type_path="test",
            )
            data_collator_padded = partial(
                data_collator, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

        test_loader = DataLoader(
            testset, 
            batch_size=args.batch_size,
            collate_fn=data_collator_padded,
        )
        
        predictions = [[] for i in range(args.num_preds)]
        for batch in tqdm(test_loader, desc="prediction"):
            for k, v in batch.items():
                batch[k] = v.to(device)
            del batch['labels']
            with torch.no_grad():
                if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
                    outputs = model.generate(**batch, early_stopping=True,
                                                max_length=args.max_target_length,
                                                num_beams=args.num_beams, 
                                                num_return_sequences=args.num_preds,
                                                decoder_start_token_id=tokenizer.pad_token_id)
                if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
                    #batch['inputs_embeds'] = model.linear(batch['inputs_embeds'])
                    #batch['inputs_embeds'] = batch['inputs_embeds'].unsqueeze(1)  # Shape: [32, 1, 256]
                    outputs = model.generate(   inputs_embeds=batch['inputs_embeds'],
                                                max_length=args.max_target_length,
                                                num_beams=args.num_beams, 
                                                num_return_sequences=args.num_preds,
                                                decoder_start_token_id=tokenizer.pad_token_id)   
            for i,pred in enumerate(outputs):
                prod = tokenizer.decode(pred, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                predictions[i % args.num_preds].append(prod)
        
        #### GET TARGETS ####
        targets = []
        if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
            target_path = os.path.join(args.data_dir, base+"_target.csv")
        if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
            target_path = os.path.join(args.data_dir, base+".target")
        with open(target_path) as rf:
            for line in rf:
                targets.append(standize(line.strip()[:args.max_target_length]))
        test_df = pd.DataFrame(targets, columns=['target'])
        #### GET RANK ####
        for i, preds in enumerate(predictions):
            test_df['prediction_{}'.format(i + 1)] = preds
            test_df['prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(standize)
        test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', args.num_preds), axis=1)


        correct = 0
        invalid_smiles = 0
        results = []
        list_columns = []

        for i in range(1, args.num_preds + 1):
            list_columns.append('prediction_{}'.format(i))

            # Check if predictions are valid molecules and equal to the target molecule
            test_df["is_correct"] = test_df[list_columns].apply(
                lambda x: is_equal_molecule(x, test_df['target']), axis=1
            )

            correct = test_df["is_correct"].sum()

            # Count invalid SMILES in the predictions
            invalid_smiles += test_df['prediction_{}'.format(i)].apply(lambda x: Chem.MolFromSmiles(x) is None).sum()

            # Calculate percentages
            top_i_percentage = correct / len(test_df) * 100
            invalid_percentage = invalid_smiles / len(test_df) * 100

            # Print and store results
            print('Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, top_i_percentage, invalid_percentage))
            result_line = 'Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, top_i_percentage, invalid_percentage)
            results.append(result_line)

        # Write results to file
        output_file_path = os.path.join(args.output_dir, 
            'results_ranks_nbeam_' + str(args.num_beams) + '_npreds_' + str(args.num_preds) + '_' + base + '_predictions.txt')

        with open(output_file_path, 'w') as wf:
            wf.write('\n'.join(results))



        #### SAVE RANK TOP PERCENT ####


        #### SAVE PREDICTIONS ####

        test_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

        #### EVALUATION PROPERTY ####

        smiles_gen = test_df['prediction_1'].tolist()
        smiles_test = test_df['target'].tolist()

        eval_validity, valid_smiles_gen = eval_validity(smiles_gen)

        print("Validity: ", eval_validity)
        print("Internal Diversity: ", eval_internal_diversity(valid_smiles_gen))
        print("Uniqueness: ", eval_uniqueness(valid_smiles_gen))
        print("Novelty: ", eval_novelty(valid_smiles_gen, smiles_test))
        print("Molecular Weight: ", eval_property_molecular_weight(valid_smiles_gen))
        print("LogP: ", eval_property_logp(valid_smiles_gen))
        print("Synthetic Accessibility: ", eval_property_synthetic_accessibility(valid_smiles_gen))
        print("QED: ", eval_property_qed(valid_smiles_gen))
        print("FCD: ", eval_property_fcd(valid_smiles_gen, smiles_test))

        # save evaluation property using a dictionary
        evaluation_property = {
            "Validity": eval_validity,
            "Internal Diversity": eval_internal_diversity(valid_smiles_gen),
            "Uniqueness": eval_uniqueness(valid_smiles_gen),
            "Novelty": eval_novelty(valid_smiles_gen, smiles_test),
            "Molecular Weight": eval_property_molecular_weight(valid_smiles_gen),
            "LogP": eval_property_logp(valid_smiles_gen),
            "Synthetic Accessibility": eval_property_synthetic_accessibility(valid_smiles_gen),
            "QED": eval_property_qed(valid_smiles_gen),
            "FCD": eval_property_fcd(valid_smiles_gen, smiles_test)
        }
        # save as json
        with open(os.path.join(args.output_dir, 'evaluation_property.json'), 'w') as wf:
            json.dump(evaluation_property, wf)

    #### EVALUATION GENE KNOCK DOWN ####
    
    if False:
        path_input_gene_knockdown = os.path.join(args.data_dir_gene_knockdown, "data_" + args.method + ".txt")

        # Read the file
        with open(path_input_gene_knockdown, 'r') as f:
            lines = f.readlines()

        # Create the matrix
        data = []
        for line in lines:
            if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
                gene_knockdown, other_column = line.strip().split(',')
                data.append([gene_knockdown, other_column])
            if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
                gene_knockdown, other_column = line.strip().split('\t')[0], line.strip().split('\t')[1:]
                other_column = list(map(float, other_column))
                data.append([gene_knockdown, other_column])

        # Create the dataframe
        df = pd.DataFrame(data, columns=['Gene Knockdown', 'Input Profile'])
        df.set_index('Gene Knockdown', inplace=True)

        num_beams_geneknockdown = 1
        num_preds_geneknockdown = 1

        outputs = dict()
        for index, row in df.iterrows():
            if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
                input_tk = tokenizer(row["Input Profile"],
                                    max_length=args.max_target_length,
                                    padding="do_not_pad",
                                    truncation=True,
                                    return_tensors='pt',
                                    )
                # generate id predictions
                for k, v in input_tk.items():
                    input_tk[k] = v.to(device)
                preds = model.generate(input_ids=input_tk['input_ids'],
                                attention_mask=input_tk['attention_mask'],
                                early_stopping=True,
                                max_length=args.max_target_length,
                                num_beams=args.num_beams, 
                                num_return_sequences=args.num_preds,
                                pad_token_id=tokenizer.pad_token_id,
                                decoder_start_token_id=tokenizer.pad_token_id)
            if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
                inputs_embeds_geneknockdown = row["Input Profile"]
                # list to tensor
                inputs_embeds_geneknockdown = torch.tensor(inputs_embeds_geneknockdown).to(device)
                inputs_embeds_geneknockdown = inputs_embeds_geneknockdown.unsqueeze(0)  # Shape: [1, 256]
                preds = model.generate(inputs_embeds=inputs_embeds_geneknockdown,
                                max_length=args.max_target_length,
                                early_stopping=True,
                                num_beams=args.num_beams, 
                                num_return_sequences=args.num_preds,
                                pad_token_id=tokenizer.pad_token_id,
                                decoder_start_token_id=tokenizer.pad_token_id)

            # convert to smiles
            outputs[index] = []
            for pred in preds:
                filtered_pred = [token_id for token_id in pred if token_id != tokenizer.pad_token_id] # remove padding   
                output_smile = tokenizer.decode(filtered_pred, skip_special_tok=True, clean_up_tokenization_spaces=False) # convert to smiles
                outputs[index].append(output_smile)
        
        list_gene_knockdown = ["AKT1", "AKT2", "AURKB", "CTSK", "EGFR", "HDAC1", "MTOR", "PIK3CA", "SMAD3", "TP53"]
        log_list_result_gene_knockdown = []

        for gene_knockdown in list_gene_knockdown:
            mat = pd.read_csv("../Data/validation_chemicals/ExCAPE-DB/"+gene_knockdown+".txt",sep="\t")
        
            # Example: Compute Tanimoto similarity between first compound and all others
            max_tanimoto = 0
            query_smile_max_tanimoto = ""
            target_smile_max_tanimoto = ""
            mat['fingerprint'] = mat['SMILES'].map(compute_morgan_fingerprint)
            
            for key in outputs.keys():
                if key.split(':')[1] == gene_knockdown:
                    for query_smile in outputs[key]:
                        # Compute fingerprints for query and dataset
                        query_fp = compute_morgan_fingerprint(query_smile)
                        if query_fp is None:
                            continue
                        mat['tanimoto_similarity_map'] = mat['fingerprint'].map(lambda fp: compute_tanimoto_similarity(query_fp, fp) if fp is not None else None)
                        query_max_tanimoto = np.max(mat['tanimoto_similarity_map'])
                        if query_max_tanimoto > max_tanimoto:
                            max_tanimoto = query_max_tanimoto
                            query_smile_max_tanimoto = query_smile
                            target_smile_max_tanimoto = mat.loc[mat['tanimoto_similarity_map'].idxmax()]['SMILES']
                    print("Gene :", gene_knockdown, "", max_tanimoto," Query :", query_smile_max_tanimoto, " Target :", target_smile_max_tanimoto)
                    log_result_gene_knockdown = "Gene : "+gene_knockdown+" "+str(max_tanimoto)+" Query : "+query_smile_max_tanimoto+" Target : "+target_smile_max_tanimoto
                    log_list_result_gene_knockdown.append(log_result_gene_knockdown)
                else:
                    continue
        # save evaluation gene knockdown
        with open(os.path.join(args.output_dir, 'evaluation_gene_knockdown.txt'), 'w') as wf:
            for log_result_gene_knockdown in log_list_result_gene_knockdown:
                wf.write(log_result_gene_knockdown+"\n")


    #### EVALUATION GENERAL GENERATOR ####

    if True:
        path_input_file = os.path.join(args.data_file + args.data_file)
        
        if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
            testset = TaskLincsDataset(
                tokenizer=tokenizer, 
                data_dir=args.data_dir,
                type_path=args.data_file,
                max_target_length=args.max_target_length,
            )
            data_collator_padded = partial(
                data_collator_lincs, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))
        if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
            testset = TaskDataset(
                tokenizer, 
                data_dir=args.data_dir,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                type_path=args.data_file,
            )
            data_collator_padded = partial(
                data_collator, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

        test_loader = DataLoader(
            testset, 
            batch_size=args.batch_size,
            collate_fn=data_collator_padded,
        )
        
        predictions = [[] for i in range(args.num_preds)]
        for batch in tqdm(test_loader, desc="prediction"):
            for k, v in batch.items():
                batch[k] = v.to(device)
            del batch['labels']
            with torch.no_grad():
                if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
                    outputs = model.generate(**batch, early_stopping=True,
                                                max_length=args.max_target_length,
                                                num_beams=args.num_beams, 
                                                num_return_sequences=args.num_preds,
                                                decoder_start_token_id=tokenizer.pad_token_id)
                if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
                    #batch['inputs_embeds'] = model.linear(batch['inputs_embeds'])
                    #batch['inputs_embeds'] = batch['inputs_embeds'].unsqueeze(1)  # Shape: [32, 1, 256]
                    outputs = model.generate(   inputs_embeds=batch['inputs_embeds'],
                                                max_length=args.max_target_length,
                                                num_beams=args.num_beams, 
                                                num_return_sequences=args.num_preds,
                                                decoder_start_token_id=tokenizer.pad_token_id)   
            for i,pred in enumerate(outputs):
                prod = tokenizer.decode(pred, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                predictions[i % args.num_preds].append(prod)

        #### GET TARGETS ####
        targets = []
        if args.method == "lincs_frogs_mols" or args.method == "lincs_mols" or args.method == "lincs_frogs_up_down_mols":
            target_path = os.path.join(args.data_dir, args.data_file+"_target.csv")
        if args.method == "lincs_text_mols" or args.method == "lincs_text_positional_mols":
            target_path = os.path.join(args.data_dir, args.data_file+".target")
        with open(target_path) as rf:
            for line in rf:
                targets.append(standize(line.strip()[:args.max_target_length]))
        test_df = pd.DataFrame(targets, columns=['target'])

        #### CREATE DATAFRAME ####
        for i, preds in enumerate(predictions):
            test_df['prediction_{}'.format(i + 1)] = preds
            test_df['prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(standize)
        #### SAVE PREDICTIONS ####

        test_df.to_csv(os.path.join(args.output_dir, "results_nbeam_" + str(args.num_beams) + "_npreds_" + str(args.num_preds) + "_" + args.data_file + '_predictions.csv'), index=False)
