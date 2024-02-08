import torch
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

import pickle
import transformers
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model


from baichuan.modeling_baichuan import BaichuanForCausalLM


# config begin
model_name_or_path = "baichuan-inc/Baichuan2-7B-Chat"
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["W_pack", "o_proj", "gate_proj", "down_proj", "up_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
)
# config end


def main():
    parser = transformers.HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()

    with open("corpus.pkl", 'rb') as f:
        train_dataset = pickle.load(f)

    # creating model
    model = BaichuanForCausalLM.from_pretrained(
        model_name_or_path,
        use_cache=False,
        torch_dtype=torch.bfloat16).train()

    # pre normalize lm_head
    model.lm_head(torch.empty((0, model.lm_head.weight.size(1)), dtype=torch.bfloat16))

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    # Start training and save the model and trainer state
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
