import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Read the CSV file with sequences
data = pd.read_csv("data.csv", header=None)

# Define a function to convert amino acids to binary vectors
def amino_to_binary(amino):
    amino_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
                      'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                      'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    
    index = amino_to_index.get(amino, -1)
    if index == -1:
        raise ValueError(f"Invalid amino acid: {amino}")
    
    binary_vector = [0] * 20
    binary_vector[index] = 1
    return binary_vector
  # Convert sequences to binary matrices
binary_sequence_matrices = []
max_sequence_length = 0

for index, row in data.iterrows():
    sequence = row[0]
    binary_sequence_matrix = [amino_to_binary(amino) for amino in sequence]
    binary_sequence_matrices.append(binary_sequence_matrix)
    
    if len(binary_sequence_matrix) > max_sequence_length:
        max_sequence_length = len(binary_sequence_matrix)

# Pad sequences to the same length
padded_binary_sequence_matrices = [seq + [[0] * 20] * (max_sequence_length - len(seq)) for seq in binary_sequence_matrices]

# Convert the padded matrices to a tensor
binary_sequence_tensor = torch.tensor(padded_binary_sequence_matrices, dtype=torch.float32)
# Split the data into a training set and a validation set
train_size = 0.8  # You can adjust this ratio
train_data, val_data = train_test_split(binary_sequence_tensor, train_size=train_size, random_state=42)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create dataset instances for training and validation
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)

# Create data loaders for training and validation
batch_size = 64  # You can adjust this batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
