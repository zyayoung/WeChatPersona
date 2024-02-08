from collections import defaultdict
import csv
import pickle
import re
import sqlite3
from tqdm import tqdm


# config begin
src = "/mnt/Data/Share/messages.csv"  # export from https://github.com/LC044/WeChatMsg
StoreEmotion = None  # (optional) e.g. StoreEmotion = "/mnt/Data/Share/StoreEmotion.db"
# config end


use_emotion_desc = StoreEmotion is not None
if use_emotion_desc:
    con = sqlite3.connect(StoreEmotion)
    cur = con.execute("SELECT MD5,Desc FROM StoreEmotionDesc")
    emotion_map = {md5.lower(): desc[6:].decode().partition('\t')[0] for md5, desc in cur.fetchall()}
else:
    emotion_map = {}


ALT_TYPES = {
    '3': '图片',
    '34': '语音',
    '42': '名片',
    '43': '视频',
    # '47': '表情包',
    '48': '位置',
    '4903': '音乐与音频',
    '4906': '文件',
    '4905': '分享卡片',
    '492000': '转账',
    '50': '音视频通话',
    # '10000': '拍一拍等系统消息',
}

record_by_remark = defaultdict(list)

# Open the CSV file
with open(src, encoding='utf-8') as f:
    csvreader = csv.reader(f)

    # Skip the header row
    next(csvreader)

    # Read the rows one by one
    for row in csvreader:
        localId, TalkerId, Type, SubType, IsSender, CreateTime, Status, StrContent, StrTime, Remark, NickName, Sender = row
        if Type == '47':
            md5 = re.search(r'md5="([0-9abcdef]+?)"', StrContent)[1].lower()
            if md5 in emotion_map:
                StrContent = f"<{emotion_map[md5]}>"
            else:
                StrContent = f"<表情>"
        elif Type == '1':
            pass
        elif Type in ALT_TYPES:
            StrContent = f"[{ALT_TYPES[Type]}]"
        elif Type in {'49', '11000', '10000'}:
            continue
        else:
            breakpoint()
        assert ':' not in Sender
        if ':' in StrContent:
            continue
        if '\t' in StrContent:
            continue
        if Sender != '我':
            Sender = '对方：'
        else:
            Sender = '我：'
        # record_by_remark[Remark].append(f"{Sender}{StrContent}")
        record_by_remark[Remark].append((Sender, StrContent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
cfg = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")
corpus = []
cur_batch = []

for remark, records in record_by_remark.items():
    system_tokens = [cfg.user_token_id, *tokenizer(f'生成与{remark}的微信聊天记录').input_ids, cfg.assistant_token_id]
    cur_batch.extend(system_tokens)
    for sender, rec in tqdm(records):
        input_ids = [*tokenizer(sender).input_ids, *tokenizer(rec).input_ids, 5]
        cur_batch.extend(input_ids)
        if len(cur_batch) >= 1024:
            corpus.append(cur_batch[:1024])
            cur_batch = system_tokens + input_ids

emoji_tok = tokenizer("<表情>").input_ids

dataset = []
for input_ids in tqdm(corpus):
    labels:list = input_ids.copy()
    is_sys = False
    for i, id in enumerate(input_ids):
        if id == cfg.user_token_id:
            is_sys = True
        if is_sys or (use_emotion_desc and id == emoji_tok[1] and input_ids[i-1] == emoji_tok[0]):
            labels[i] = -100
        if id == cfg.assistant_token_id:
            is_sys = False
    dataset.append({
        'input_ids': input_ids,
        'labels': labels,
    })

with open("corpus.pkl", 'wb') as f:
    pickle.dump(dataset, f)
