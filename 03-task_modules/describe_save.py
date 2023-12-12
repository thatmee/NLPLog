import time
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from trl import AutoModelForCausalLMWithValueHead
from transformers import LlamaTokenizer
from transformers.generation import BeamSearchDecoderOnlyOutput, GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, BeamSampleDecoderOnlyOutput
from transformers.generation import LogitsProcessorList, ForcedEOSTokenLogitsProcessor
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from datasets import Dataset
from nlplog.describer.describer import Describer
from nlplog.config import Config

# 1 for abnormal and 0 for normal
ABNORMAL = 1
NORMAL = 0


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# shared preprocessing
def get_tokenized_inputs(example, tokenizer, config:Config):
    log_message = example['log_message']
    input_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a log analysis expert. Most of the raw logs are concise, so based on your extensive experience in log analysis, you should expand and explain the original logs. The better the information you provide, the more helpful it will be for downstream log anomaly detection tasks and operations engineers. To provide better information, you need to focus on the following aspects:

1. Convert abbreviations to full words or use parentheses to explain them.
2. Convert unnatural string to natural language. For example, convert "2005-06-03-15.42.50.363779" into "timestamp(2005-06-03-15.42.50.363779)"
3. Pay attention to the content after the last colon of log message. Explain what is it and why does it happen and what will it cause potentially.
4. Pay attention to the parameter part and the number part. For example, exit code, indicating 0 is normal, otherwise it is an exception. 0 and 1 may represent down and up in command. For file path parameters, explain the meaning of every sub-path. For other strings, explaining their meaning.
5. For very short logs without colons nor parameters, focus on providing as much relevant information as possible related to that particular log entry.
6. Do not summarize

### Input:
expand and explain this log : "{log_message}"

### Response:"""
    example['inputs'] = tokenizer(input_text, return_tensors='pt', max_length=config.max_input_length, padding='max_length', truncation=True)
    if 'labels' in example.keys():
        pass
    elif 'label_info' not in example.keys():
        example['labels'] = None
    else:
        example['labels'] = NORMAL if example['label_info'] == '-' else ABNORMAL
    
    return example


def describe_log(batch, describer:Describer):
    batch_inputs = batch['inputs']
    device = describer.describer.device
    if device.type == 'cpu':
        print("Warning: the model is running on cpu, which is very slow.")
    inputs = {'input_ids': torch.cat([i['input_ids'] for i in batch_inputs], dim=0).to(device), 
              'attention_mask': torch.cat([i['attention_mask'] for i in batch_inputs], dim=0).to(device)}

    sentence_embeddings, output_sequences = describer.forward(inputs)
    
    batch['sentence_embeddings'] = [i.to('cpu') for i in sentence_embeddings]
    batch['output_sequences'] = [i.to('cpu') for i in output_sequences]

    return batch

# START generate test datasets
CFG_gen_test = Config(
    batch_size=8,
    epochs=100,
    lr=1e-5,
    train_ratio=1,
    early_stop=10,
    lr_scheduler = 'none',
    wandb_enable=False,
    tqdm_disable=True,
    confusion_matrix_enable=False,
    wandb_project='sequence_anomaly_detection',
)

## preprocess
unique_test_df = pd.read_csv('/data/user/nyf/LAB/NLPLog/data/bgl/test20_bgl_structured_fillall.csv')
unique_test_df = unique_test_df.drop_duplicates(subset=['log_message'])
unique_test_df = unique_test_df.sample(20000, random_state=CFG_gen_test.random_state)

unique_test_ds = Dataset.from_pandas(unique_test_df, preserve_index=False)
sentence_emb_map_ds = unique_test_ds.map(get_tokenized_inputs, num_proc=CFG_gen_test.num_workers, fn_kwargs={'tokenizer':CFG_gen_test.tokenizer, 'config':CFG_gen_test})
sentence_emb_map_ds = sentence_emb_map_ds.with_format('torch')

llm = AutoModelForCausalLMWithValueHead.from_pretrained(
    CFG_gen_test.pretrained_llama_dir,
    load_in_8bit=True,
    device_map='sequential',
)
llm.v_head.cuda()

describer_model = Describer(llm, CFG_gen_test)
describer_model.eval()

sentence_emb_map_ds = sentence_emb_map_ds.map(describe_log, batched=True, batch_size=CFG_gen_test.batch_size, fn_kwargs={'describer':describer_model}, cache_file_name=f'{CFG_gen_test.cache_dir}cache-bgl_test_20000.arrow', new_fingerprint='bgl_test_20000')
sentence_emb_map_ds.save_to_disk('../data_hf/BGL/single/test_20000')