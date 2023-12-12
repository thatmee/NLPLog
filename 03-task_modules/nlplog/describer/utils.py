import torch
from nlplog import Config
from .describer import Describer

NORMAL = 0
ABNORMAL = 1

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