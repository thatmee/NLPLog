import math
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from nlplog.config import Config

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, config:Config):
        super(SinusoidalPositionalEmbedding, self).__init__()

        pe = torch.zeros(config.window_size, d_model) # max_len代表句子中最多有几个词 
        position = torch.arange(0, config.window_size).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # d_model即公式中的d 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x): 
        x = x + self.pe[:, :x.size(1)]  # 原向量加上计算出的位置信息才是最终的embedding return self.dropout(x)

        
class SequencePredModel(nn.Module):
    def __init__(self, llm_hidden_state, config:Config) -> None:
        super().__init__()
        self.config = config
        self.embedding_norm = LlamaRMSNorm(llm_hidden_state)
        self.position_embedding = SinusoidalPositionalEmbedding(llm_hidden_state, config)
        self.sequence_encoder = nn.TransformerEncoderLayer(
            d_model=llm_hidden_state,
            nhead=config.nhead,
            dim_feedforward=config.sequence_transformer_hidden_size,
            dropout=config.dropout,
            batch_first=True)
        self.hidden_layer = nn.Linear(llm_hidden_state, config.sequence_linear_hidden_size, bias=False)
        self.sequence_pred_head = nn.Linear(config.sequence_linear_hidden_size, 2, bias=False)
        self.norm1 = nn.BatchNorm1d(llm_hidden_state)
        self.norm2 = nn.BatchNorm1d(config.sequence_linear_hidden_size)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, labels=None):
        """
        Parameters:
            inputs:
                a batch of sentence_embeddings. 
                shape should be [batch_size, window_size, llm_hidden_size].
        """
        # 1. add position embedding and encode the sequence
        # inputs = self.embedding_norm(inputs)
        outputs = self.position_embedding(inputs)  # [batch_size, window_size, llm_hidden_size]
        outputs = self.sequence_encoder(inputs)  # [batch_size, window_size, llm_hidden_size]

        # 2. get the windows embedding by average pooling
        outputs = outputs.mean(dim=1)  # [batch_size, llm_hidden_size]

        # 3. hidden layer 1
        outputs = self.norm1(outputs)
        outputs = self.hidden_layer(outputs)  # [batch_size, sequence_linear_hidden_size]
        outputs = self.relu(outputs)
        # outputs = self.dropout2(outputs)

        # 4. predict the result
        outputs = self.norm2(outputs)
        logits = self.sequence_pred_head(outputs)  # [batch_size, 2]

        # 5. calculate the loss
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }