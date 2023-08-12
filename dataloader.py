import pandas as pd
import torch
from torch.utils.data import Dataset
import tqdm
from copy import deepcopy



# Define a dictonary to convert amino acids to binary vectors

aa_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
aa_dict = {}
binary_vector = [0] * 20
for aa in aa_index:
    aa_vector = deepcopy(binary_vector)
    aa_vector[aa_index[aa]] = 1
    aa_dict[aa] = aa_vector

def proteinread(filename):
    # Read the CSV file with sequences
    data = pd.read_csv(filename, header=None)
    binary_sequence_matrices = []
    max_sequence_length = 0
    
    # Iterate through each row in the dataset
    for index, row in data.iterrows():
        sequence = row[0]
        binary_sequence_matrix = [aa_dict[amino] for amino in sequence]
        binary_sequence_matrices.append(binary_sequence_matrix)
    
    if len(binary_sequence_matrix) > max_sequence_length:
        max_sequence_length = len(binary_sequence_matrix)

    # Pad sequences to the same length
    padded_binary_sequence_matrices = [seq + [[0] * 20] * (max_sequence_length - len(seq)) for seq in binary_sequence_matrices]

    # Convert the padded matrices to a tensor
    binary_sequence_tensor = torch.tensor(padded_binary_sequence_matrices, dtype=torch.float32)
    return binary_sequence_tensor

# Define a custom dataset class
class ProteinDataset(Dataset):
    def __init__(self, data_path):
        
        self.data = proteinread(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

