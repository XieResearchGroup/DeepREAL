import pandas as pd
import numpy as np
# import pickle
import pickle5 as pickle
import json
from rdkit import Chem
from torch_geometric.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch
import torch
from typing import Sequence, Tuple, List, Union
from ligand_graph_features import *
# from fingerprint.graph import *
##### tsv modules #####


def save_interaction_tsv(p,tsvname,datalist):
    with open(os.path.join(p,tsvname), 'wt',newline='') as f_output:
        tsv_output=csv.writer(f_output, delimiter='\t')
        for c in datalist:
            tsv_output.writerow(c)
def read_interaction_tsv(path):
  lines = []
  with open(path, 'r') as f:
    for line in f:
      lines.append(line.split('\n')[0].split('\t'))
  return lines

##### JSON modules #####
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4)

def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data

##### pickle modules #####
def save_dict_pickle(data,filename):
  with open(filename,'wb') as handle:
    pickle.dump(data,handle, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
  with open(path, 'rb') as f:
    dict = pickle.load(f)
  return  dict


##### DTI #####

#------------------
#  read data
#------------------

def load_training_data(exp_path,debug_ratio,balanced=False):
    def load_data(exp_path,file,debug_ratio):
        dataset = pd.read_csv(exp_path +file)
        cut = int(dataset.shape[0] * debug_ratio)
        print(file[:-3] + ' size:', cut)
        return dataset.iloc[:cut,:]

    train = load_data(exp_path,'train.csv',debug_ratio)
    # dev   = load_data(exp_path,'dev.csv',debug_ratio)
    if balanced:
        print('balanced test data')
        test = load_data(exp_path, 'test_balanced.csv', debug_ratio)
    else:
        test  = load_data(exp_path,'test.csv',debug_ratio)

    return train,  test

def get_repr_DTI(batch_data,tokenizer,chem_dict,protein_dict,prot_descriptor_choice,chem_option):
    #  . . . .  chemicals  . . . .
    chem_smiles = chem_dict[batch_data['ikey'].values.tolist()].values.tolist()
    chem_graph_list = []
    for smiles in chem_smiles:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_graph_data_obj_simple(mol)
        chem_graph_list.append(graph)
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=batch_data.shape[0],
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch
    uniprot_list = batch_data['uniprot'].values.tolist()
    protein_tokenized = torch.tensor([tokenizer.encode(protein_dict[uni]) for uni in uniprot_list  ])
    return chem_graphs, protein_tokenized



