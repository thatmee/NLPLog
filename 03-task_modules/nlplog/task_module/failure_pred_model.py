import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from nlplog.config import Config


class FailurePredModel(nn.Module):
    def __init__(self, llm_hidden_size, config:Config) -> None:
        """
        Parameters:
            model (AutoModelForCausalLMWithValueHead):
                the RLHFed LLM model, used for generating descriptions of raw logs
        """
        super().__init__()
        self.config = config
        self.norm1 = LlamaRMSNorm(llm_hidden_size)
        self.norm2 = LlamaRMSNorm(config.single_linear_hidden_size)
        self.hidden_layer = nn.Linear(llm_hidden_size, config.single_linear_hidden_size, bias=False)
        self.failure_pred_head = nn.Linear(config.single_linear_hidden_size, 2, bias=False)
        # self.failure_pred_head = nn.Linear(llm_hidden_size, 2, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, sentence_embeddings, labels=None, output_sequences=None):
        """
        Parameters:
            sentence_embeddings:
                a batch of sentence_embedding. 
                shape should be [batch_size, llm_hidden_size].
        """
        # 1. predict the failure probability of the expanded log descriptions
        outputs = self.norm1(sentence_embeddings)
        outputs = self.hidden_layer(sentence_embeddings)
        # outputs = self.relu(outputs)
        outputs = self.norm2(outputs)
        logits = self.failure_pred_head(outputs)  # [batch_size, 2]
    
        # 2. output the expanded log descriptions
        if self.config.show_HR_output:
            if output_sequences is None:
                self.config.logger.warning("If you want to show the output, please send the output_sequences to the model.")
            else:
                self.config.logger.info(self.config.tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
        
        # 3. calculate the loss
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }