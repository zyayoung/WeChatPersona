import torch
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

from baichuan.generation_utils import build_chat_input
from baichuan.modeling_baichuan import BaichuanForCausalLM
from transformers import AutoTokenizer, TextStreamer
from peft import PeftModel


# config begin
base_model = "baichuan-inc/Baichuan2-7B-Chat"
lora_model = "checkpoints/wechat_v3"
remark = '小明'
prefix = '我：a6000好慢啊\n我：oh好像不慢 是nfs慢\n'
# config end

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
model = BaichuanForCausalLM.from_pretrained(
    base_model,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    use_cache=True
).eval()

print('Loading LoRA weights...')
model = PeftModel.from_pretrained(model, lora_model, device_map="cuda:0")
print('Merging LoRA weights...')
model = model.merge_and_unload()

messages = [
    {"role": "user", "content": f"生成与{remark}的微信聊天记录"},
    {"role": "assistant", "content": prefix},
]

with torch.inference_mode():
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    input_ids = build_chat_input(model, tokenizer, messages, max_new_tokens=1024)
    outputs = model.generate(input_ids, streamer=streamer, temperature=0.8, max_new_tokens=512)
