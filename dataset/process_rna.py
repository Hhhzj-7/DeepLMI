import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def process_rna_file_to_df(input_file_path):    
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()

    names = []
    sequences = []
    
    for i in range(0, len(lines), 2):
        name = lines[i].strip()
        sequence = lines[i + 1].strip()

        names.append(name[1:])
        sequences.append(sequence)

    df = pd.DataFrame({
        'RNA_id': names,
        'Sequence': sequences
    })
    
    return df


class miRNA_dataset(InMemoryDataset):
    def __init__(self,
                 dataset_name='mirna',
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 is_merged=False):

        self.dataset_name = dataset_name
        self.is_merged = is_merged

        if self.is_merged:
            self.emb_folder_path = 'dataset/merged_mirna_emb/representations'
            self.concat_folder_path = 'dataset/merged_mirna_contact'
            self.df = process_rna_file_to_df('dataset/mirna.fasta')
            root = 'dataset/merged_mirna' if root is None else root
        else:
            self.emb_folder_path = 'dataset/mirna_emb/representations'
            self.concat_folder_path = 'dataset/mirna_contact'
            self.df = process_rna_file_to_df('dataset/mirna.fasta')
            root = 'dataset/mirna' if root is None else root

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        



    @property
    def processed_file_names(self):
        return "data_rna.pt"


    def process(self):
        data_list = []
        for index, row in self.df.iterrows():
            id_value = row['RNA_id']
            sequence = row['Sequence']


            file_path = os.path.join(self.concat_folder_path, f"{id_value}.prob_single")

            if os.path.exists(file_path):
                matrix = np.loadtxt(file_path)[:640,:640]
                matrix[matrix < 0.5] = 0
                matrix[matrix > 0.5] = 1
                edges = np.argwhere(matrix == 1)
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                print(edge_index.size())
            else:
                print('error', file_path)

                
            one_hot_sequence = [char_to_one_hot(char) for char in sequence]
            x = torch.tensor(one_hot_sequence, dtype=torch.float32)
            rna_len = x.size()[0]
            
            # read language model embedding        
            emb_file_path = os.path.join(self.emb_folder_path, f"{id_value}.npy")
            if os.path.exists(emb_file_path):
                rna_emb = torch.tensor(np.load(emb_file_path))[:640,:]
            else:
                print('bad', emb_file_path)
                continue
            if len(row['Sequence']) > 640:
                print("too long!!")
                print(rna_emb.size())
                continue
            data = Data(id=id_value, x=x, edge_index=edge_index, emb=rna_emb, rna_len=rna_len)

            if x.size(0) != rna_emb.size(0):
                print('error!!', x.size(0), rna_emb.size(0))
            data_list.append(data)

        
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

def char_to_one_hot(char):
    if char == 'T':
        print("T")
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'T':1, 'X':4, 'Y':5}
    return [mapping[char]]
