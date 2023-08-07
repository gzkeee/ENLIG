from util import get_rel2idx, get_ent2text, save_file
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification

# 3. 将三元组转换为可训练数据
def convert_triple_to_trainable_data(data):
    rel2idx = get_rel2idx()
    ent2text = get_ent2text()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []

    for idx, triple in enumerate(tqdm(list(data))):
        h, r, t = triple
        h = ent2text.get(h, [])
        t = ent2text.get(t, [])
        if len(h) == 0 or len(t) == 0:
            continue
        else:
            h = h[0]
            t = t[0]

        if r not in rel2idx.keys():
            # print(r)
            continue

        encoding = tokenizer(h, t, max_length=32,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        input_ids.append(encoding.input_ids)
        token_type_ids.append(encoding.token_type_ids)
        attention_mask.append(encoding.attention_mask)
        labels.append(rel2idx[r])

    data_save = [torch.cat(input_ids),
                 torch.cat(token_type_ids),
                 torch.cat(attention_mask),
                 torch.tensor(labels)]

    print(len(input_ids))
    save_file(data_save, f'../data/train_data')