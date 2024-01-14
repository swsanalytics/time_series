import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer

model_name = "mistralai/Mixtral-8x7B-v0.1"
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left'

def format_ultrachat(ds):
  text = []
  for row in ds:
    if len(row['messages']) > 2:
      text.append("### Human: "+row['messages'][0]['content']+"### Assistant: "+row['messages'][1]['content']+"### Human: "+row['messages'][2]['content']+"### Assistant: "+row['messages'][3]['content'])
    else: #not all tialogues have more than one turn
      text.append("### Human: "+row['messages'][0]['content']+"### Assistant: "+row['messages'][1]['content'])
  ds = ds.add_column(name="text", column=text)
  return ds
dataset_train_sft = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
dataset_test_sft = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:5%]")

dataset_test_sft = format_ultrachat(dataset_test_sft)
dataset_train_sft = format_ultrachat(dataset_train_sft)


compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}
)
model = prepare_model_for_kbit_training(model)
#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching



peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj']
)


training_arguments = TrainingArguments(
        output_dir="./results_mixtral_sft/",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_steps=50,
        logging_steps=50,
        learning_rate=2e-5,
        eval_steps=50,
        max_steps=300,
        warmup_steps=30,
        lr_scheduler_type="linear",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train_sft,
        eval_dataset=dataset_test_sft,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
)

trainer.train()

