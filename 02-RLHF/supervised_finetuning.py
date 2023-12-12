import os
import argparse

from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    TrainingArguments,
    logging,
    set_seed
)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from utils.merge import merge_llm_with_lora

# os.environ["LOCAL_RANK"] = "1"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="../data/rlhf/train80_chatgpt_sft.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--no_bf16", action="store_false", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/supervised_llama/")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--save_total_limit", default=3, type=int)
    parser.add_argument("--run_name", default="llama-supervised-finetuned", type=str)
    parser.add_argument("--merge_lora", action="store_true", default=False)

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens, max_token_len = 0, 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        # print(text)
        total_characters += len(text)
        if tokenizer.is_fast:
            length = len(tokenizer(text).tokens())
            total_tokens += length
            max_token_len = max(max_token_len, length)
        else:
            length = len(tokenizer.tokenize(text))
            total_tokens += length
            max_token_len = max(max_token_len, length)

    return total_characters / total_tokens, max_token_len


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(data_point):
    """Prepare the text from a sample of the dataset."""


    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a log analysis expert. Most of the raw logs are concise, so based on your extensive experience in log analysis, you should expand and explain the original logs. The better the information you provide, the more helpful it will be for downstream log anomaly detection tasks and operations engineers. To provide better information, you need to focus on the following aspects:

1. Convert abbreviations to full words or use parentheses to explain them.
2. Convert unnatural string to natural language. For example, convert "2005-06-03-15.42.50.363779" into "timestamp(2005-06-03-15.42.50.363779)"
3. Pay attention to the content after the last colon of log message. Explain what is it and why does it happen and what will it cause potentially.
4. Pay attention to the parameter part and the number part. For example, exit code, indicating 0 is normal, otherwise it is an exception. 0 and 1 may represent down and up in command. For file path parameters, explain the meaning of every sub-path. For other strings, explaining their meaning.
5. For very short logs without colons nor parameters, focus on providing as much relevant information as possible related to that particular log entry.
6. Do not summarize

### Input:
expand and explain this log : "{data_point["log_message"]}"

### Response:
{data_point["response"]}
"""


def create_datasets(tokenizer, args):
    data_path = args.dataset_name
    data_kwargs = {
        "split": args.split,
        "num_proc": args.num_workers if not args.streaming else None,
        "streaming": args.streaming,
    }
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, **data_kwargs)
    elif data_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=data_path, **data_kwargs)
    else:
        dataset = load_dataset(data_path, **data_kwargs)

    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token, max_token_lenth = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    print(f"The max token length of the dataset is: {max_token_lenth}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data, tokenizer=None):
    print("Loading the model")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.no_bf16,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        run_name=args.run_name,
        report_to="wandb",
        # ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None,
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        device_map={"": Accelerator().local_process_index},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        max_seq_length=args.seq_length,
        packing=True,
    )

    print_trainable_parameters(model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    final_model_path = os.path.join(args.output_dir, "final_checkpoint/")
    trainer.model.save_pretrained(final_model_path)

    if args.merge_lora:
        merge_llm_with_lora(args.base_model, final_model_path, args.output_dir)


def main(args):
    if "decapoda" in args.base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
                "pad_token": "</s>",
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset, tokenizer)


if __name__ == "__main__":
    args = get_args()
    assert args.base_model != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
