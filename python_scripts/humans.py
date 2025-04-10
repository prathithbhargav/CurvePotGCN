import torch
import pandas as pd
import numpy as np
from scipy.spatial import distance
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random


def get_humans_data(owner,in_device):

        
    human_positives = pd.read_csv('PATH_TO_human_positive_cleaned_dataset.csv' ) 
    human_positives = human_positives[['interactor_A', 'interactor_B']]
    
    human_negatives = pd.read_csv('PATH_TO_human_negative_cleaned_dataset.csv' ) 
    human_negatives = human_negatives[['interactor_A', 'interactor_B']]

    base = 'PATH_TO_FILES_WITH_CURVATURE_AND_POTENTIAL'

    device = torch.device(in_device)
    
    
    
    
    
    CUTOFF_DIST = 12
    list_of_all_data = list()
        
    for i in range(len(human_negatives)):
        a,b = human_negatives.iloc[i].values
        ligand = pd.read_csv(base + a + '.csv')
        receptor = pd.read_csv(base + b + '.csv')
        
        #extracting and making a matrix
        ligand_feats = np.asarray(ligand[['curvature','average_potential']])
        ligand = ligand[['x','y','z']]
        receptor_feats = np.asarray(receptor[['curvature','average_potential']])
        receptor = receptor[['x','y','z']]
        
        #making a matrix of distances between each node with another node for ligand and receptor
        distance_matrix_ligand = distance.cdist(ligand,ligand,'euclidean')
        distance_matrix_receptor = distance.cdist(receptor,receptor,'euclidean')
        
        copy_ligand = distance_matrix_ligand
        copy_receptor = distance_matrix_receptor
        
        
        adj_ligand = np.where((copy_ligand > 0)&(copy_ligand <= CUTOFF_DIST), 1, 0)
        adj_receptor = np.where((copy_receptor > 0)&(copy_receptor <= CUTOFF_DIST), 1, 0)
        # print(adj_ligand.shape,adj_receptor.shape, '  ', counter)
        # counter+=1
        
        #making diagonal adjcency matrix of ligand and receptors. [[ligand, zeros],[zeros,receptor]]
        adj_together = np.block([
                                [adj_ligand,np.zeros((adj_ligand.shape[0],adj_receptor.shape[1]))],
                                [np.zeros((adj_receptor.shape[0],adj_ligand.shape[1])),adj_receptor]
        ])
        
        
        # coordinate form data about the adjcency matrix formed above, pytorch does this for efficiency reasons.
        coo_together = np.asarray(np.where(adj_together == 1))
        
        #verticall stacking features of ligands and receptors
        together_feats = np.vstack((ligand_feats,receptor_feats))
        
        
        #create a data object
        data = Data(x =torch.tensor(together_feats).float().to(device),edge_index = torch.tensor(coo_together).int().to(device),y=torch.tensor(np.asarray([0])).type(torch.LongTensor).to(device))
        
        list_of_all_data.append(data)

        
    random.shuffle(list_of_all_data)
    list_of_all_data = list_of_all_data
    print('negative_shape', len(list_of_all_data))
    
    for i in range(len(human_positives)):
        a,b = human_positives.iloc[i].values
        ligand = pd.read_csv(base + a + '.csv')
        receptor = pd.read_csv(base + b + '.csv')
            
        #extracting and making a matrix
        ligand_feats = np.asarray(ligand[['curvature','average_potential']])
        ligand = ligand[['x','y','z']]
        receptor_feats = np.asarray(receptor[['curvature','average_potential']])
        receptor = receptor[['x','y','z']]
        
        #making a matrix of distances between each node with another node for ligand and receptor
        distance_matrix_ligand = distance.cdist(ligand,ligand,'euclidean')
        distance_matrix_receptor = distance.cdist(receptor,receptor,'euclidean')
        
        copy_ligand = distance_matrix_ligand
        copy_receptor = distance_matrix_receptor
        
        
        adj_ligand = np.where((copy_ligand > 0)&(copy_ligand <= CUTOFF_DIST), 1, 0)
        adj_receptor = np.where((copy_receptor > 0)&(copy_receptor <= CUTOFF_DIST), 1, 0)
        # print(adj_ligand.shape,adj_receptor.shape, '  ', counter)
        # counter+=1
        
        #making diagonal adjcency matrix of ligand and receptors. [[ligand, zeros],[zeros,receptor]]
        adj_together = np.block([
                                [adj_ligand,np.zeros((adj_ligand.shape[0],adj_receptor.shape[1]))],
                                [np.zeros((adj_receptor.shape[0],adj_ligand.shape[1])),adj_receptor]
        ])
        
        
        # coordinate form data about the adjcency matrix formed above, pytorch does this for efficiency reasons.
        coo_together = np.asarray(np.where(adj_together == 1))
        
        #verticall stacking features of ligands and receptors
        together_feats = np.vstack((ligand_feats,receptor_feats))
        
        
        #create a data object
        data = Data(x = torch.tensor(together_feats).float().to(device),edge_index = torch.tensor(coo_together).int().to(device),y=torch.tensor(np.asarray([1])).type(torch.LongTensor).to(device))
        
        list_of_all_data.append(data)
    random.shuffle(list_of_all_data)

    return list_of_all_data