import torch
import torch.nn as nn
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding.type(torch.float32)


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_encoding = positional_encoding(max_len,
                                                embed_dim)
 
    def call(self, x):
        seq_len = x.shape[1]
        x += self.pos_encoding[:, :seq_len, :]
        return x


class NeuralLog(nn.Module):
    def __init__(self, d_model=768, nhead=12, dim_feedforward=2048, dropout=0.1):
        super(NeuralLog, self).__init__()
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.embedding = PositionEmbedding(100, 2000, d_model)
        self.linear_1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(32, 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, labels=None):
        outputs = self.embedding(x)
        outputs = self.transformer_block(outputs)
        outputs = torch.mean(outputs, dim=1)
        outputs = self.dropout_1(outputs)
        outputs = self.linear_1(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout_2(outputs)
        outputs = self.linear_2(outputs)

        loss = None
        if labels is not None:
            labels = labels.to(outputs.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs, labels)

        return {
            "logits": outputs,
            "loss": loss,
        }
    
