import torch
import datasets
from .config import Config

class FailurePredDataset(torch.utils.data.Dataset):
    def __init__(self, ds:datasets.Dataset, cfg:Config, send_output_sequences=False) -> None:
        self.labels = ds['labels']
        self.sentence_embeddings = ds['sentence_embeddings']
        if send_output_sequences:
            self.output_sequences = ds['output_sequences']
        self.send_output_sequences = send_output_sequences

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.send_output_sequences:
            return self.sentence_embeddings[idx], self.output_sequences[idx], self.labels[idx]
        else:
            return self.sentence_embeddings[idx], self.labels[idx]
        

class SequencePredDataset(torch.utils.data.Dataset):
    def __init__(self, log_window_ds:datasets.Dataset, cfg:Config) -> None:
        self.labels = log_window_ds['labels']
        self.sentence_embeddings = log_window_ds['sentence_embeddings']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sentence_embeddings[idx], self.labels[idx]
    

def collate_for_sequence_pred(batch):
    x, y = zip(*batch)
    batch_inputs = torch.cat([sentence_embeddings.squeeze(1).unsqueeze(0) for sentence_embeddings in x])
    return batch_inputs, torch.tensor(y)