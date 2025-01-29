import os

# write output file
import xlsxwriter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
from rdkit.Chem.Draw import MolToFile
import matplotlib.pyplot as plt
import seaborn as sns

def compute_maccs_similarity(smiles1, smiles2):
    """
    Compute MACCS fingerprints and Tanimoto similarity
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string")
    
    # Generate MACCS fingerprints
    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)
    
    # Compute Tanimoto similarity
    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return similarity

def compute_morgan_similarity(smiles1, smiles2, radius=2, nbits=1024):
    """
    Compute MORGAN fingerprints and Tanimoto similarity
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string")
    
    # Generate Morgan fingerprints
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
    
    # Compute Tanimoto similarity
    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return similarity

def save_2d_image_PNG_list(mol_list, path_output, name = "_Name"):
    """
    Save list of molecules to png files
    """
    for mol in mol_list:
        mol.UpdatePropertyCache()
        Chem.rdDepictor.Compute2DCoords(mol)
        mol = Chem.RemoveHs(mol)
        path_out = os.path.join(path_output, mol.GetProp(name) + ".png")
        MolToFile(mol, path_out, size=(200, 200))
        try:
            MolToFile(mol, path_out, size=(200, 200))
        except:
            pass

def write_xls_vae_prediction_test(dt_original_decoded_smiles, path_out_vae_prediction_xls, path_images_mols):
    """ ADD IMAGES TO CSV FILE """
    
    workbook = xlsxwriter.Workbook(path_out_vae_prediction_xls)
    worksheet = workbook.add_worksheet('sheet1')
    worksheet.write(0 , 0, "Original_SMILES")
    worksheet.write(0 , 1, "Decoded_SMILES")
    worksheet.write(0 , 2, "Original_Molecule")
    worksheet.write(0 , 3, "Decoded_Molecule")
    worksheet.write(0 , 4, "Similarity MACCS")
    worksheet.write(0 , 5, "Similarity Morgan")

    row_i = 1
    id=0
    for original_molecule, decoded_molecule in dt_original_decoded_smiles:
        try:
            # Original molecule image
            m = Chem.MolFromSmiles(original_molecule)
            m.SetProp("_Name","mol_original_"+str(id))
            save_2d_image_PNG_list([m],path_images_mols)
            worksheet.insert_image(row_i, 2, os.path.join(path_images_mols, "mol_original_"+str(id)+".png"), {'x_scale': 0.3, 'y_scale': 0.3})
        except:
            pass
            #print("no image")
        m = Chem.MolFromSmiles(original_molecule)
        m.SetProp("_Name","mol_original_"+str(id))
        save_2d_image_PNG_list([m],path_images_mols)
        try:
            # Decoded molecule image
            m = Chem.MolFromSmiles(decoded_molecule)
            m.SetProp("_Name","mol_decoded_"+str(id))
            save_2d_image_PNG_list([m],path_images_mols)
            worksheet.insert_image(row_i, 3, os.path.join(path_images_mols, "mol_decoded_"+str(id)+".png"), {'x_scale': 0.3, 'y_scale': 0.3})
        except:
            pass
            #print("no image")
        # Write original and decoded molecules smiles
        worksheet.write(row_i , 0, original_molecule)
        worksheet.write(row_i , 1, decoded_molecule)

        try:
            maccs_similarity = compute_maccs_similarity(original_molecule, decoded_molecule)
            morgan_similarity = compute_morgan_similarity(original_molecule, decoded_molecule)
        except:
            maccs_similarity = None
            morgan_similarity = None
        worksheet.write(row_i , 4, maccs_similarity)
        worksheet.write(row_i , 5, morgan_similarity)
        row_i+=1
        id+=1
        
    workbook.close()

def plot_maccs_morgan_similarity(dt_original_decoded_smiles, path_out):
    """
    Plot MACCS and morgan similarity distribution
    """
    # Get maccs and morgan similarity
    list_maccs_similarity = []
    list_morgan_similarity = []
    for i in range(len(dt_original_decoded_smiles)):
        try:
            maccs_similarity = compute_maccs_similarity(dt_original_decoded_smiles[i][0], dt_original_decoded_smiles[i][1])
            morgan_similarity = compute_morgan_similarity(dt_original_decoded_smiles[i][0], dt_original_decoded_smiles[i][1])
            list_maccs_similarity.append(maccs_similarity)
            list_morgan_similarity.append(morgan_similarity)
        except:
            pass
    # Create density plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.kdeplot(list_maccs_similarity, label='MACCS Similarity', fill=True)
    sns.kdeplot(list_morgan_similarity, label='Morgan Similarity', fill=True)

    plt.title('Density Plot of MACCS and Morgan Similarities')
    plt.xlabel('Similarity')
    plt.ylabel('Density')

    plt.legend()
    plt.savefig(os.path.join(path_out,'density_plot_prediction_vae_maccs_morgan.png'), dpi=600)  # Adjust dpi as needed for quality