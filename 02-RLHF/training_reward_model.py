import os

import torch
import evaluate
import numpy as np
import torch.nn as nn
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import PaddingStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

os.environ["LOCAL_RANK"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[int] = field(default=0.001)
    seed: Optional[int] = field(default=1103)
    max_length: Optional[int] = field(default=1024)
    model_name: Optional[str] = field(
        default="decapoda-research/llama-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub or local."
        },
    )
    dataset_name: Optional[str] = field(
        default="../data/rlhf/log_supplement_comparison_data.json",
        metadata={"help": "The dataset name"},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        # metadata={
        #     "help":
        # },
    )
    num_train_epochs: Optional[int] = field(
        default=10,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    output_dir: Optional[str] = field(default="./checkpoints/training_reward_model/", metadata={"help": "n steps to save the model"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

# Load the dataset for tuning the reward model.
data_path = script_args.dataset_name
if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=data_path, split="train")
else:
    dataset = load_dataset(data_path, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=script_args.seed)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

if script_args.train_subset > 0:
    train_dataset = train_dataset.select(range(script_args.train_subset))
if script_args.eval_subset > 0:
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]
output_name = (
    f"{model_name_split}_peft_gpt-4-llm_rm_{script_args.train_subset}_{script_args.learning_rate}"
)

training_args = TrainingArguments(
    output_dir=os.path.join(script_args.output_dir, output_name),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    fp16=script_args.fp16,
    logging_strategy="steps",
    logging_steps=1,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    report_to="wandb"
)

if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1":
    script_args.model_name = '/data/user/nyf/.cache/huggingface/modules/opt-1.3b'

# Load the value-head model and tokenizer.
config = AutoConfig.from_pretrained(script_args.model_name)
if "decapoda" in script_args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# model = LlamaForSequenceClassification(config)
model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    # llm_int8_skip_modules=["score"],
    # device_map='auto'
    # device_map={"": Accelerator().local_process_index},
)

# model = AutoModelForSequenceClassification.from_pretrained(
#     script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
# )

# model.int8()

# model = AutoModelForCausalLM.from_pretrained(
#         script_args.model_name,
#         load_in_8bit=True,
#         device_map={"": Accelerator().local_process_index},
#     )

# weights_location = hf_hub_download(script_args.model_name, 'pytorch_model.bin')
# print(weights_location)
# with init_empty_weights():
#     model = AutoModelForSequenceClassification.from_config(config)
# model.tie_weights()
# device_map = infer_auto_device_map(model)
# print(device_map)
# print(config)
# model = load_checkpoint_and_dispatch(
#     model, weights_location, device_map='auto', load_in_8bit=True
# )



model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = script_args.gradient_checkpointing
num_proc = None  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for log_i, response_j, response_k in zip(examples["input_log"], examples["response_a"],
                                                examples["response_b"]):
        question = f"""### Input:{log_i}
### Response:"""

        tokenized_j = tokenizer(question + response_j, truncation=True, max_length=script_args.max_length)
        tokenized_k = tokenizer(question + response_k, truncation=True, max_length=script_args.max_length)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


# preprocess the dataset and filter out QAs that are longer than max_length
train_dataset = train_dataset.map(
    preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)

eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

print("init trainer")
# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)

print("start training")
trainer.train(resume_from_checkpoint=False)

print("Saving last checkpoint of the model")
model.save_pretrained(script_args.output_dir + "peft_last_checkpoint")
