import modal
import pathlib
from modal import Image, App, Secret, Volume

app = App("hf-train-dora")
image = Image.from_dockerfile("Dockerfile", gpu='l4')

image = image.add_local_file("composer_aim_logger.py", "/root/composer_aim_logger.py")
image = image.add_local_file("aim_remote_uploader.py", "/root/aim_remote_uploader.py")

MODEL_CHECKPOINT_VOLUME = Volume.from_name("lrg-model-checkpoints", create_if_missing=True)
MODEL_CHECKPOINT_VOLUME_MOUNT_PATH = pathlib.Path("/model-checkpoints")

@app.function(gpu="A100-80GB", image=image, timeout=12*3600, secrets=[Secret.from_name("LRG")],
             concurrency_limit=1, volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME})
def _train():
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from composer.utils.reproducibility import seed_all

    seed=17
    seed_all(seed)

    wandb.login()

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=False,
        use_dora=True,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        peft_type="LORA"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def process_example(question, answer, tokenizer, max_length=2048):
        question_text = f"Question: {question}\n"
        answer_text = f"Answer: {answer}"
        
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        
        input_ids = tokenizer.build_inputs_with_special_tokens(question_tokens, answer_tokens)
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        attention_mask = [1] * len(input_ids)
        
        question_len = len(tokenizer.build_inputs_with_special_tokens(question_tokens))
        question_len = min(question_len, max_length)
        
        labels = [-100] * question_len
        answer_part = input_ids[question_len:max_length] if question_len < max_length else []
        labels.extend(answer_part)
        
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([-100] * padding_length)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def preprocess(examples, tokenizer, max_length=2048):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for question, answer in zip(examples["query"], examples["response"]):
            processed = process_example(question, answer, tokenizer, max_length)
            
            results["input_ids"].append(processed["input_ids"])
            results["attention_mask"].append(processed["attention_mask"])
            results["labels"].append(processed["labels"])
        
        return results

    def load_and_process_dataset(tokenizer, dataset_name="meta-math/MetaMathQA", split="train", max_length=2048):
        dataset = load_dataset(dataset_name, split=split)
        
        processed_dataset = dataset.map(
            lambda examples: preprocess(examples, tokenizer, max_length),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return processed_dataset
    

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer.pad_token = tokenizer.eos_token

    print("Processing dataset...")
    tokenized_dataset = load_and_process_dataset(
        tokenizer,
        split="train",
        max_length=2048
    )

    print(f"Processed dataset size: {len(tokenized_dataset)}")
    print(f"Input IDs shape: {tokenized_dataset[0]['input_ids'].shape}")
    print(tokenized_dataset)

    training_args = TrainingArguments(
        output_dir="/content",
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        max_steps=5000,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=1e-4,
        report_to="wandb",   
        logging_steps=1, 
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    wandb.init(
        entity="local-research-group",
        project="hf-trainer-smollm2-135m",
        name="smollm2-135m-dora-5000ba",
    )
    
    trainer.train()
    wandb.finish()
    #model.push_to_hub("LocalResearchGroup/HF-Trainer-smollm2-135m-DoRA-5000ba")

@app.local_entrypoint()
def main():
    _train.remote()
