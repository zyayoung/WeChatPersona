import pickle
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)

with open("corpus.pkl", 'rb') as f:
    train_dataset = pickle.load(f)

    for data in train_dataset:
        print(tokenizer.decode(data['input_ids']))
