import torch
import datasets
from nlplog import Config
from transformers import AutoModel

NORMAL = 0
ABNORMAL = 1

def tokenize_logs(example, config:Config):
    log_message = example['log_message']
    example['inputs'] = config.tokenizer(log_message, return_tensors='pt', max_length=config.max_input_length, padding='max_length', truncation=True)
    if 'labels' in example.keys():
        pass
    elif 'label_info' not in example.keys():
        example['labels'] = None
    else:
        example['labels'] = NORMAL if example['label_info'] == '-' else ABNORMAL
    
    return example

@torch.no_grad()
def get_log_embeddings(batch, config:Config):
    batch_inputs = batch['inputs']
    inputs = {'input_ids': torch.cat([i['input_ids'] for i in batch_inputs], dim=0).to(config.device), 
            'attention_mask': torch.cat([i['attention_mask'] for i in batch_inputs], dim=0).to(config.device)}
    sentence_embeddings = config.embed_model.forward(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)['last_hidden_state']
    # sentence_embeddings = config.embed_model.forward(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[1]
    # print(sentence_embeddings.shape)
    sentence_embeddings = sentence_embeddings.mean(dim=1)
    
    batch['sentence_embeddings'] = [i.to('cpu') for i in sentence_embeddings]

    return batch


class SingleLogEmbedDataset(torch.utils.data.Dataset):
    def __init__(self, ds:datasets.Dataset, cfg:Config) -> None:
        cfg.logger.info('tokenizing logs')
        ds = ds.map(tokenize_logs, num_proc=1, fn_kwargs={'config':cfg})
        ds = ds.with_format('torch')

        cfg.logger.info('get log embeddings')
        cfg.embed_model.eval()
        ds = ds.map(get_log_embeddings, batched=True, batch_size=cfg.batch_size, fn_kwargs={'config':cfg})
        self.sentence_embeddings = ds['sentence_embeddings']
        self.labels = ds['labels']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sentence_embeddings[idx], self.labels[idx]