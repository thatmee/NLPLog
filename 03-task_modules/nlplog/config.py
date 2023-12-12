import os
import time
import torch
import wandb
import numpy as np
from tqdm import tqdm
from logging import getLogger, INFO, Formatter, StreamHandler, FileHandler
from typing import Union
from transformers import LlamaTokenizer

class Config:
    def __init__(
            self,
            output_dir = 'out/',
            pretrained_llama_dir = '/data/user/nyf/LAB/llama-trl/checkpoints/instruction+log/tuning_llama_rl/step_10',
            max_input_length = 512,
            weight_metric = False,
            train_ratio = 0.9,
            num_workers = 16,
            batch_size = 16,
            early_stop = 0,
            lr = 0.001,
            lr_scheduler = 'none',  # 'reduceonplateau'
            warmup_steps = 0,
            epochs = 50,
            dropout = 0.1,
            logger_file = None,
            random_state = 1006,
            single_linear_hidden_size = 32,
            sequence_transformer_hidden_size = 768,
            sequence_linear_hidden_size = 32,
            nhead = 8,
            wandb_enable = False,
            tqdm_disable = False,
            confusion_matrix_enable = True,
            wandb_project = 'failure_prediction',
            train_data_file = '/data/user/nyf/LAB/llama-trl/data/train_600.csv',
            test_data_file = '/data/user/nyf/LAB/LogTurbo/data/bgl/test20_bgl_structured_fillall.csv',
            f1_report:Union[list, str] = 'binary',
            best_model_metric = 'binary',
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            task_type = None,
        ) -> None:

        # tools
        self.labels = {'abnormal':1, 'normal':0}
        self.device = device
        self.task_type = task_type

        # wandb
        self.tqdm_disable = tqdm_disable
        self.confusion_matrix_enable = confusion_matrix_enable
        self.wandb_enable = wandb_enable
        self.wandb_project = wandb_project  # or 'sequence_anomaly_detection'

        # parameters about saving and loading
        self.output_dir = output_dir
        self.model_save_dir = f'{output_dir}model/'
        self.predictions_save_dir = f'{output_dir}predictions/'
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.predictions_save_dir):
            os.makedirs(self.predictions_save_dir)
        self.pretrained_llama_dir = pretrained_llama_dir
        if pretrained_llama_dir is not None:
            self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_llama_dir, padding_side="left")
        else:
            self.tokenizer = None
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.cache_dir = '/data/user/nyf/LAB/llama-trl/.cache/hf-datasets/'
        
        # parameters about generation
        self.max_input_length = max_input_length
        self.show_HR_output = False
        self.generation_kwargs = {
            # "num_beams": 2,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": False,  # default use Greedy Search
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer is not None else 0,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer is not None else 2,
            "min_new_tokens": 16,  # 如果不设置，模型可能一直输出pad_token(对于greedy search)
                                   # 或者输出一个 eos 就结束（对 beam_search）
            "max_new_tokens": 512,  # 几乎不可能达到的长度，输出是否停止是通过 eos 控制的
                                    # 对一些特殊日志，模型的输出很长，此处设置一个很大的值可保证其不被截断
            # "num_return_sequences": 2,
        }

        # hyper-parameters
        self.batch_size = batch_size
        self.window_size = 20
        self.step_size = 20  # todo merge sliding window fn into this
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.lr_scheduler = lr_scheduler
        self.single_linear_hidden_size = single_linear_hidden_size
        self.sequence_transformer_hidden_size = sequence_transformer_hidden_size
        self.sequence_linear_hidden_size = sequence_linear_hidden_size
        self.nhead = nhead

        # parameters about metrics and others
        self.train_ratio = train_ratio
        self.weight_metric = weight_metric
        self.num_workers = num_workers
        self.random_state = random_state
        self.early_stop = early_stop

        if isinstance(f1_report, str):
            f1_report = [f1_report]
        if best_model_metric not in f1_report:
            f1_report.append(best_model_metric)
        
        for report in f1_report:
            if report not in ['binary', 'micro', 'macro']:
                raise ValueError(f'config.f1_report "{report}" is none of [binary, micro, macro]')
        self.f1_report = f1_report
        self.best_model_metric = best_model_metric

        # set logger
        if logger_file is None:
            # default logger file name is the time when the program starts
            time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
            if not os.path.exists(f'{self.output_dir}train_log/'):
                os.makedirs(f'{self.output_dir}train_log/')
            self.set_logger(f'{self.output_dir}train_log/{time_str}.csv')
        else:
            self.set_logger(f'{self.output_dir}train_log/{logger_file}')

    def set_logger(self, log_file='train.log'):
        # if not os.path.exists(log_file):
        #     os.system(r"touch {}".format(log_file))
        self.logger = getLogger('/data/user/nyf/LAB/llama-trl/cache_and_task_modules.ipynb')
        self.logger.setLevel(INFO)
        if not self.logger.handlers:
            handler1 = StreamHandler()
            handler1.setFormatter(Formatter("%(message)s"))
            handler2 = FileHandler(filename=f"{log_file}")
            handler2.setFormatter(Formatter("%(message)s"))
            self.logger.addHandler(handler1)
            self.logger.addHandler(handler2)
        self.logger.parent.handlers.clear()