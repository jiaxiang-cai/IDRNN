import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

amino_acid_mapping = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5,
    "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11,
    "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17,
    "W": 18, "Y": 19
}
class ProteinDataset(Dataset):
    def __init__(self, input_file):
        # Read the CSV file into a pandas DataFrame
        self.df = pd.read_csv(input_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the protein label and sequence at the specified index
        protein_label = str(self.df.loc[idx, "Label"]).strip()
        protein_sequence = str(self.df.loc[idx, "Sequence"]).strip()

        protein_numbers = [amino_acid_mapping[amino_acid] for amino_acid in protein_sequence]
        protein_numbers = torch.tensor(protein_numbers)

        # Return the protein label and sequence as tensors
        return protein_numbers, protein_label


if __name__ == "__main__":
    # Replace this path with the actual path to your input CSV file
    input_csv_file = "data.csv"

    # Create an instance of the ProteinDataset
    dataset = ProteinDataset(input_csv_file)

    # Define the validation set size (20% of the total data)
    validation_size = int(0.2 * len(dataset))
    training_size = len(dataset) - validation_size

    # Split the dataset into training and validation sets
    training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])

    # Define batch size (adjust according to your requirements)
    batch_size = 10

    # Create DataLoaders for training and validation sets
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


