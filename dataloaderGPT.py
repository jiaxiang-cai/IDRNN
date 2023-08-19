import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Dictionary to map amino acids to integers
amino_acid_to_int = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
                     'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
                     'V': 17, 'W': 18, 'Y': 19}

# Copy from ChatGPT
class AminoAcidDataset(Dataset):
    def __init__(self, csv_file, max_sequence_length, num_amino_acids):
        self.data = pd.read_csv(csv_file)
        self.max_sequence_length = max_sequence_length
        self.num_amino_acids = num_amino_acids
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence_str = self.data.iloc[idx, 0]  # Assuming the sequence is in the first column
        
        # Convert amino acid sequence string to a list of integers (e.g., using a dictionary)
        sequence = [amino_acid_to_int[aa] for aa in sequence_str]
        sequence_length = [len(sequence_str)]
        
        # Perform one-hot encoding and post-padding
        padded_sequence = np.zeros((self.max_sequence_length, self.num_amino_acids), dtype=np.float32)
        for i, amino_acid in enumerate(sequence):
            padded_sequence[i][amino_acid] = 1.0
        
        return torch.tensor(padded_sequence, dtype=torch.float32).transpose(0, 1), torch.tensor(sequence_length, dtype=torch.long)
#


# Path to the CSV file containing amino acid sequences
# csv_file_path = 'path/to/your/csv/file.csv'

# Set the maximum sequence length and number of amino acids
# max_sequence_length = 10
# num_amino_acids = 20

# Create an instance of the AminoAcidDataset
# dataset = AminoAcidDataset(csv_file_path, max_sequence_length, num_amino_acids)

# # Create a DataLoader
# batch_size = 2
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Iterate through the DataLoader
# for batch in dataloader:
#     print("Batch shape:", batch.shape)
#     print("Batch data:", batch)
