from util import load_json, get_cur, sql_ent
from util import get_rel2idx, get_ent2text, save_file, load_file, check_model_grad
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification
import lora
from torch.utils.data import TensorDataset, DataLoader
import evaluate
import sqlite3


def get_cur():
    db_path = '../data/wikidata_5m.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    return cur


def sql_triple_with_rel(cur, rel):
    sql_query_entity = '''
        select * from triples where Rel == '{}'
        LIMIT 1000
        '''
    cur.execute(sql_query_entity.format(rel))
    res = cur.fetchall()
    return res


def get_train_triple_from_db():
    cur = get_cur()
    labels = collect_rel()
    all_triples = []
    for l in labels:
        all_triples += sql_triple_with_rel(cur, l)
    return all_triples


def collect_rel():
    all_rel = []
    data = load_json(f'../data/fewrel/train.json')
    for item in data:
        rel = item['label']
        all_rel.append(rel)
    return list(set(all_rel))


def collect_train_ent():
    all_triple = []
    data = load_json(f'../data/fewrel/train.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        rel = item['label']
        all_triple.append((head, rel, tail))
    return all_triple


def collect_test_ent():
    all_triple = []
    data = load_json(f'../data/fewrel/test.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        rel = item['label']
        all_triple.append((head, rel, tail))
    return all_triple


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
    return data_save


def get_inset_lora_model(r):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          problem_type="single_label_classification",
                                                          num_labels=825)
    # save_file(model.state_dict(), '../param/pretrain')
    embedding = model.bert.get_input_embeddings()
    model.bert.set_input_embeddings(lora.Embedding(embedding.num_embeddings, embedding.embedding_dim, r))
    encoder = model.bert.encoder
    layers = encoder.layer
    for lay in layers:
        d = lay.output.dense
        lay.output.dense = lora.Linear(d.in_features, d.out_features, r)

    model.load_state_dict(load_file('../param/pretrain'), strict=False)
    return model


def train(model, train_data):
    device = 'cuda'
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
    loss_print = 0
    ground_truth = []
    prediction = []
    accuracy_metric = evaluate.load("accuracy")
    divide = 1


    model.train()
    for epc in range(divide):
        dataset = TensorDataset(*train_data)
        accumulation_steps = 1
        load = DataLoader(dataset, batch_size=128, shuffle=True)
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

            if (idx + 1) % 512 == 0:
                ground_truth = torch.cat(ground_truth)
                prediction = torch.cat(prediction)
                results = accuracy_metric.compute(references=ground_truth, predictions=prediction)
                print(results)
                print(f'loss:{loss_print}')
                prediction = []
                ground_truth = []
                loss_print = 0
                # break
        save_file(model.state_dict(), f'../param/fewrel_triple_lab/{epc}')


def eval(model, data):
    device = 'cuda'
    bsz = 32
    model = model.to(device)
    accuracy_metric = evaluate.load("accuracy")
    dataset = TensorDataset(*data)
    load = DataLoader(dataset, batch_size=bsz)
    ground_truth = []
    prediction = []
    model.eval()
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, token_type_ids, attention_mask, label = item
        res = model(input_ids, attention_mask, token_type_ids, labels=label)

        predict = res.logits.argmax(-1)
        ground_truth += label.cpu().numpy().tolist()
        prediction += predict.cpu().numpy().tolist()

    result = accuracy_metric.compute(references=ground_truth, predictions=prediction)
    print(result)


def train_setting(model):
    for idx, (n, p) in enumerate(model.named_parameters()):
        if idx <= 180:
            p.requires_grad = False

# 仅通过三元组组成的数据训练模型
if __name__ == '__main__':
    # train_data = convert_triple_to_trainable_data(get_train_triple_from_db())
    train_data = convert_triple_to_trainable_data(collect_train_ent()+collect_test_ent())
    eval_data = convert_triple_to_trainable_data(collect_test_ent())
    model = get_inset_lora_model(8)
    model.load_state_dict(load_file('../param/fewrel_triple_lab/0'))
    lora.mark_only_lora_as_trainable(model.bert)
    check_model_grad(model)
    for i in range(10):
        train(model, train_data)
        eval(model, eval_data)

