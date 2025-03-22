import modal
import pathlib
from modal import Image, App, Secret, Volume

app = App("weight-changes-composer-dora")
image = Image.from_dockerfile("Dockerfile", gpu='l4')

image = image.add_local_file("composer_aim_logger.py", "/root/composer_aim_logger.py")
image = image.add_local_file("aim_remote_uploader.py", "/root/aim_remote_uploader.py")

MODEL_CHECKPOINT_VOLUME = Volume.from_name("lrg-model-checkpoints", create_if_missing=True)
MODEL_CHECKPOINT_VOLUME_MOUNT_PATH = pathlib.Path("/model-checkpoints")

@app.function(gpu="A100-80GB", image=image, timeout=12*3600, secrets=[Secret.from_name("LRG")],
             concurrency_limit=1, volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME})
def _train():
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from composer import Trainer
    from composer.models import HuggingFaceModel
    from composer.metrics import LanguageCrossEntropy
    from composer.optim import DecoupledAdamW
    import types
    from peft.tuners.lora.layer import Linear
    from peft.utils.integrations import dequantize_module_weight
    from composer_aim_logger import AimLogger
    from composer.utils.reproducibility import seed_all

    seed=17
    seed_all(seed)

    model_nm = "HuggingFaceTB/SmolLM2-135M"

    aim_logger = AimLogger(repo=".aim", experiment_name="vishal_composer_dora_smollm2-135m_5000ba")

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
    
    tokenizer = AutoTokenizer.from_pretrained(model_nm)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Processing dataset...")
    tokenized_dataset = load_and_process_dataset(
        tokenizer,
        split="train",
        max_length=2048
    )
    
    print(f"Processed dataset size: {len(tokenized_dataset)}")
    print(f"Input IDs shape: {tokenized_dataset[0]['input_ids'].shape}")

    train_dataloader = DataLoader(tokenized_dataset, batch_size=10, shuffle=True)

    # Add monitoring for DoRA scaling factors
    def add_dora_diagnostics(model):
        original_forward = Linear.forward
        
        def patched_forward(self, x, *args, **kwargs):
            result = original_forward(self, x, *args, **kwargs)
            if hasattr(self, 'use_dora') and any(self.use_dora.values()):
                adapter_name = list(self.use_dora.keys())[0]
                if hasattr(self, 'lora_magnitude_vector') and adapter_name in self.lora_magnitude_vector:
                    mag_vector = self.lora_magnitude_vector[adapter_name].weight
                    
                    step = getattr(self, 'step_counter', 0)
                    if step % 100 == 0:  # Print every 100 steps
                        # Get module name
                        module_name = "unknown"
                        for name, module in model.named_modules():
                            if module is self:
                                module_name = name
                                break
                        
                        # Calculate mag_norm_scale
                        weight = dequantize_module_weight(self.get_base_layer())
                        weight_norm = self.lora_magnitude_vector[adapter_name].get_weight_norm(
                            weight, 
                            self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight, 
                            scaling=1
                        ).detach()
                        mag_norm_scale = (mag_vector / weight_norm).mean().item()
                        
                        print(f"Step {step}, {module_name}: mag_norm_scale = {mag_norm_scale}")
                    
                    self.step_counter = step + 1
            return result
        
        Linear.forward = patched_forward
        return original_forward

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_nm)
    
    # Setup PEFT config
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=False,
        use_dora=True,
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    
    # Add diagnostics
    original_forward = add_dora_diagnostics(model)
    
    # Create Composer model
    print("Building model...")
    composer_model = HuggingFaceModel(
        model,
        tokenizer=tokenizer,
        metrics=[LanguageCrossEntropy()],
        use_logits=True,
        peft_config=peft_config
    )
    
    # Create trainer
    # save_folder: /model-checkpoints/smollm2-135m_lora-20250305_114026/native_checkpoints
    # Model path: /model-checkpoints/smollm2-135m_lora-20250305_114026
    # model_path = Path(model_path).name = smollm2-135m_lora-20250305_114026
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        max_duration="5000ba",
        save_interval="5000ba",
        optimizers=DecoupledAdamW(composer_model.parameters(), lr=1e-4),
        device="gpu",
        precision="amp_bf16",
        loggers=[aim_logger],
        save_folder="/model-checkpoints/smollm2-135m_dora_composer-20250305-160000/native_checkpoints",
        save_filename="ep0-ba5000-rank0.pt",
        seed=seed
    )
    
    print("Starting DoRA training with HuggingFace dataset...")
    trainer.fit()
    
    # Restore original forward
    Linear.forward = original_forward

@app.local_entrypoint()
def main():
    _train.remote()
