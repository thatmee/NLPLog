import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from nlplog.config import Config

class NeuralLogSingleModel(nn.Module):
    def __init__(self, hidden_size, config:Config) -> None:
        super().__init__()
        self.config = config
        self.linear_1 = nn.Linear(hidden_size, 32)
        self.linear_2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(config.dropout)
        self.dropout_2 = nn.Dropout(config.dropout)
        
    def forward(self, sentence_embeddings, labels=None):
        """
        Parameters:
            sentence_embeddings:
                a batch of sentence_embeddings calculated by language models.
                the shape should be [batch_size, lm_hidden_size].
        """
        
        outputs = self.dropout_1(sentence_embeddings)
        outputs = self.linear_1(sentence_embeddings)
        outputs = self.relu(outputs)
        outputs = self.dropout_2(outputs)
        logits = self.linear_2(outputs)  # [batch_size, 2]
        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }