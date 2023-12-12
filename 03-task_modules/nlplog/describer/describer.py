import torch
import torch.nn as nn
from transformers.generation import BeamSearchDecoderOnlyOutput, GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, BeamSampleDecoderOnlyOutput
from transformers.generation import LogitsProcessorList, ForcedEOSTokenLogitsProcessor
from trl import AutoModelForCausalLMWithValueHead
from ..config import Config


class Describer(nn.Module):
    def __init__(self, llm_model:AutoModelForCausalLMWithValueHead, config:Config) -> None:
        """
        Parameters:
            model (AutoModelForCausalLMWithValueHead):
                the RLHFed LLM model, used for generating descriptions of raw logs
        """
        super().__init__()
        self.config = config
        self.describer = llm_model.pretrained_model
        
    @torch.no_grad()
    def forward(self, inputs):

        # 1. generate the expanded log descriptions (texts after increasing dimension)
        # generate func is decorated by @torch.no_grad(), so the desriber model won't be updated during training
        # generate returns a tuple(tuple, tuple(tensor[batch_size, generated_length, hidden_size], tensor, ...), ...)
        outputs = self.describer.generate(
            **inputs, 
            return_dict_in_generate=True, 
            output_scores=True, 
            output_hidden_states=True, 
            logits_processor=LogitsProcessorList([
                ForcedEOSTokenLogitsProcessor(
                    max_length=self.config.max_input_length + self.config.generation_kwargs['max_new_tokens'],
                    eos_token_id=self.config.tokenizer.eos_token_id)]),
            **self.config.generation_kwargs)

        # 2. get the sentence embeddings and output sequences of expanded log descriptions
        # the last_hidden_states of the eos token is the sentence embedding
        eos_token_id = self.config.tokenizer.eos_token_id
        if isinstance(outputs, GreedySearchDecoderOnlyOutput):
            eos_pos = torch.where(outputs.sequences == eos_token_id)[1]\
                           .view(inputs['input_ids'].shape[0], -1)[:, 1] \
                           - outputs.sequences.shape[1]
            # every element in sentence_embeddings is [1 (generated_length), hidden_size]
            sentence_embeddings = [outputs.hidden_states[p][-1][i] for i, p in enumerate(eos_pos)]  # [batch_size, hidden_size]

            # get the output sequences
            start_pos = self.config.max_input_length
            output_sequences = [outputs.sequences[i][start_pos:]\
                                .unsqueeze(dim=0) for i in range(inputs['input_ids'].shape[0])]
            
        elif isinstance(outputs, BeamSearchDecoderOnlyOutput):
            # shape is [batch_size*num_beams*num_return_sequences, generated_length(1), hidden_size]
            # sentence_embedding = last_hidden_states.pool?
            #todo
            self.config.logger.warning(f"Beam search is not supported yet.")
            raise NotImplementedError
        
        elif isinstance(outputs, BeamSampleDecoderOnlyOutput):
            # size of outputs.hidden_states: 
            # (generated_token, layer_nums, [batch_size*num_beams, generated_length, hidden_size])
            num_return_sequences = self.config.generation_kwargs.get("num_return_sequences", 1)
            batch_size = inputs['input_ids'].shape[0]
            max_input_length = self.config.max_input_length

            start_pos = self.config.max_input_length
            eos_pos = torch.where(outputs.sequences == eos_token_id)[1]\
                           .view(batch_size * num_return_sequences, -1)[:, 1] \
                           - outputs.sequences.shape[1]
            beam_indices = outputs.beam_indices[torch.arange(batch_size*num_return_sequences), eos_pos-max_input_length]
            
            sentence_embeddings = []
            output_sequences =[]
            for i in range(batch_size):
                hd_for_one_input = []
                sequences_for_one_input = []
                for j in range(num_return_sequences):
                    p = eos_pos[i]
                    beam_idx = beam_indices[i*num_return_sequences+j]
                    hd_for_one_input.append(outputs.hidden_states[p][-1][beam_idx])
                    sequences_for_one_input.append(outputs.sequences[i*num_return_sequences+j][start_pos:])
                hd_for_one_input = torch.cat(hd_for_one_input, dim=0)
                sequences_for_one_input = torch.cat(sequences_for_one_input, dim=0)
                sentence_embeddings.append(hd_for_one_input)
                output_sequences.append(sequences_for_one_input)
            
        elif isinstance(outputs, SampleDecoderOnlyOutput):
            self.config.logger.warning(f"Sample decoding is not supported yet.")
            # todo
            raise NotImplementedError

        else:
            self.config.logger.warning(f"Unknown decoding method.")
            raise NotImplementedError
        
        return sentence_embeddings, output_sequences