import time
import torch
import wandb
import datasets
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from nlplog import Config
from .scorer import calculate_score_by_log
from .utils import get_collate_and_bsz, get_lr_scheduler


def train_fn(
        train_dataloader:torch.utils.data.DataLoader,
        model,
        optimizer,
        epoch:int,
        device,
        train_len:int,
        config:Config
    ):
    # LOGGER.info(f'Epoch {epoch+1} - start train')
    model.train()
    
    total_loss_train = 0
    train_label, mask, input_id, output, batch_loss = None, None, None, None, None
    
    try:
        with tqdm(train_dataloader, desc=f'Epoch {epoch}-train', disable=config.tqdm_disable) as t:
            step = 1
            length = len(train_dataloader)
            for train_inputs, train_labels in t:
                # print(train_inputs['input_ids'].shape, train_labels.shape)
                train_inputs = train_inputs.squeeze(1).to(device)
                train_labels = train_labels.to(device)
                output = model.forward(train_inputs, labels=train_labels)

                # loss
                batch_loss = output['loss']
                total_loss_train += batch_loss.item()

                # wandb
                t.set_description(f"loss: {batch_loss}")
                if model.config.wandb_enable:
                    wandb.log({"train_loss": batch_loss}, step=epoch*length+step)

                # model update
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                step += 1
    except:
        t.close()
        del train_label, mask, input_id, output, batch_loss
        torch.cuda.empty_cache()
        raise
    t.close()

    return (total_loss_train / train_len)


def valid_fn(
        val_dataloader:torch.utils.data.DataLoader,
        model,
        device,
        val_len,
        config:Config,
        epoch=None,
    ):
    # LOGGER.info(f'Epoch {epoch+1} - start valid')
    model.eval()

    total_loss_val = 0
    predictions = []
    logits = []

    output, batch_loss = None, None

    with torch.no_grad():
        try:
            with tqdm(val_dataloader, desc=f'Epoch {epoch}-eval', disable=config.tqdm_disable) as t:
                for val_inputs, val_labels in t:
                    val_inputs = val_inputs.squeeze(1).to(device)
                    val_labels = val_labels.to(device)
                    output = model.forward(val_inputs, labels=val_labels)
                    predictions.append(output['logits'].argmax(dim=1).to('cpu').numpy())
                    logits.extend(output['logits'].to('cpu').numpy().tolist())
                    batch_loss = output['loss']
                    total_loss_val += batch_loss.item()
                    t.set_description(f"loss: {batch_loss}")
        except:
            t.close()
            del output, batch_loss
            torch.cuda.empty_cache()
            raise
        t.close()

    predictions = np.concatenate(predictions, axis = 0)
    return (total_loss_val / val_len),  predictions, logits



def train_and_eval(
        model,
        dataset_builder:torch.utils.data.Dataset,
        train_data:datasets.Dataset,
        config:Config
    ):
    # split train and val (stratify)
    if config.train_ratio == 1:
        val_data = train_data
    else:
        train_data = train_data.train_test_split(train_size = config.train_ratio, stratify_by_column='labels', seed=config.random_state)
        train_data, val_data = train_data['train'], train_data['test']

    # dataset
    config.logger.info('load dataset')
    train_set, val_set = dataset_builder(train_data, config), dataset_builder(val_data, config)

    # dataloader
    config.logger.info('load dataloader')
    collate_fn, batch_size = get_collate_and_bsz(dataset_builder, config)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)

    # criterion & optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)

    # scheduler
    scheduler = get_lr_scheduler(optimizer, config)

    model.to(config.device)
    
    best_f1_score = 0
    early_stop_cnt = 0
    for epoch_num in range(config.epochs):
        start_time = time.time()
        
        avg_train_loss = train_fn(train_dataloader, model, optimizer, epoch_num, config.device, len(train_data), config=config)
        avg_val_loss, val_predictions, _ = valid_fn(val_dataloader, model, config.device, len(val_data), config=config, epoch=epoch_num)

        # compute precision, recall and f1-score
        report = calculate_score_by_log(val_predictions, val_data, config)

        if config.lr_scheduler == 'reduceonplateau':
            scheduler.step(f1_score)
        elif config.lr_scheduler != 'none':
            scheduler.step()

        elapsed = time.time() - start_time

        # wandb
        current_lr = optimizer.param_groups[0]['lr']
        if config.wandb_enable:
            wandb.log({"epoch": epoch_num,
                    "learning_rate": current_lr,
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                    #    "train_precision": train_precision,
                    #    "train_recall": train_recall,
                    #    "train_f1_score": train_f1_score,
                    "report": report,
                    "elapsed_time": elapsed,
                    "val_predictions": val_predictions})
        else:
            pass
            # config.logger.info(f"""Epoch {epoch_num+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  elapsed_time: {elapsed:.4f}  lr: {current_lr:.4e}""")
            # if config.confusion_matrix_enable:
            #     config.logger.info(f"\tconfusion_matrix:\n{report['confusion_matrix']}")
            # for report_type in config.f1_report:
            #     precision, recall, f1 = report[report_type]['precision'], report[report_type]['recall'], report[report_type]['f1']
            #     config.logger.info(f"\t[{report_type}] precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f}")

        f1_score = report[config.best_model_metric]['f1']
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            early_stop_cnt = 0
            model_type = model.__class__.__name__
            print(f'Save best {config.best_model_metric}-f1 score: {best_f1_score:.4f} {model_type}')
            torch.save(
                    obj={f'model': model.state_dict(), 'predictions': val_predictions}, 
                    f=f"{config.model_save_dir}{model_type}.pth"
                )
        elif f1_score != 0:
            early_stop_cnt += 1
            if config.early_stop != 0 and early_stop_cnt >= config.early_stop:
                config.logger.info('early stop and exit.')
                break