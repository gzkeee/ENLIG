from util import load_json, get_cur, sql_ent
from util import get_rel2idx, get_ent2text, save_file, load_file, check_model_grad
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification
import lora
import torch.utils.data as Data
import evaluate



# 冻结除了embedding以外的所有参数
# def train_setting(model):
#     for n, p in model.named_parameters():
#         if 'embedding' in n:
#             p.requires_grad = True

# 1. 收集fewrel的实体
def collect_ent():
    all_ent = []
    files = ['train', 'dev', 'test']
    for f in files:
        data = load_json(f'../data/fewrel/{f}.json')
        for item in data:
            head = item['ents'][0][0]
            tail = item['ents'][1][0]
            all_ent.append(head)
            all_ent.append(tail)
    all_ent = list(set(all_ent))
    return all_ent



# 2. 获取与fewrel相关的所有三元组
def collect_related_triples(all_ent):
    related_triples = []
    cur = get_cur()
    for e in all_ent:
        triples = sql_ent(cur, e)
        related_triples += triples
    return related_triples


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


# embedding 和 映射层 插入lora
def get_inset_lora_model(r):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          problem_type="single_label_classification",
                                                          num_labels=825)
    # save_file(model.state_dict(), '../param/pretrain')
    embedding = model.bert.get_input_embeddings()
    model.bert.set_input_embeddings(lora.Embedding(embedding.num_embeddings, embedding.embedding_dim, 1))
    encoder = model.bert.encoder
    layers = encoder.layer
    for lay in layers:
        d = lay.output.dense
        lay.output.dense = lora.Linear(d.in_features, d.out_features, r)

    model.load_state_dict(load_file('../param/pretrain'), strict=False)
    return model


def get_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          problem_type="single_label_classification",
                                                          num_labels=825)
    return model

# 4. train model
def train(model):
    device = 'cuda'
    model = model.to(device)
    # train_setting(model.bert)
    # model.load_state_dict(load_file('../param/rel_feature_extract_1'))
    # lora.mark_only_lora_as_trainable(model.bert)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    train_data = load_file(f'../data/trainable triple/train_data')
    loss_print = 0
    ground_truth = []
    prediction = []
    accuracy_metric = evaluate.load("accuracy")
    divide = 1

    check_model_grad(model)

    for epc in range(divide):
        dataset = Data.TensorDataset(*train_data)
        accumulation_steps = 1
        load = Data.DataLoader(dataset, batch_size=128, shuffle=True)
        for idx, item in enumerate(tqdm(load)):
            item = [i.to(device) for i in item]
            input_ids, token_type_ids, attention_mask, label = item
            res = model(input_ids, attention_mask, token_type_ids, labels=label)
            loss = res.loss / accumulation_steps
            loss_print += loss
            pred = res.logits

            prediction.append(torch.argmax(pred, dim=1))
            ground_truth.append(label)
            loss.backward()

            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % 256 == 0:
                ground_truth = torch.cat(ground_truth)
                prediction = torch.cat(prediction)
                results = accuracy_metric.compute(references=ground_truth, predictions=prediction)
                print(results)
                print(f'loss:{loss_print}')
                prediction = []
                ground_truth = []
                loss_print = 0
        save_file(model.bert.get_input_embeddings().state_dict(), f'../param/knowledge_embedding_{epc}')


if __name__ == '__main__':
    # ents = collect_ent()
    # triples = collect_related_triples(ents)
    # convert_triple_to_trainable_data(triples)
    train(get_model())