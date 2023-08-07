import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, BertModel
from util import load_json, check_model_grad
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def get_rel2id():
    kg = open('../data/wiki/wikidata5m_inductive_train.txt')
    rel2id = {}
    count = 1
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split()
        if r not in rel2id.keys():
            rel2id[r] = count
            count += 1
    return rel2id


def get_ent2rel(rel2id):
    kg = open('../data/wiki/wikidata5m_inductive_train.txt')
    rel_num = len(rel2id)
    ent2rel = {}

    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split()

        if h not in ent2rel.keys():
            ent2rel[h] = []
        ent2rel[h].append(rel2id[r])

        if t not in ent2rel.keys():
            ent2rel[t] = []
        ent2rel[t].append(rel2id[r]+rel_num)

    for k, v in ent2rel.items():
        ent2rel[k] = list(set(v))
    return ent2rel


def check(ent2rel, rel2id):
    kg = open('../data/wiki/wikidata5m_inductive_train.txt')
    rel_num = len(rel2id)
    result = []
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split()

        hr = ent2rel[h]
        tr = ent2rel[t]
        hr = [r+rel_num for r in hr]
        # print(hr)
        # print(tr)
        result.append(len(set(hr)&set(tr)))
    return result





def convert_triple_to_trainable_data(data, rel2idx, ent2rel):
    head_ids = []
    tail_ids = []
    labels = []
    id_max_len = 32
    for idx, triple in enumerate(tqdm(list(data))):
        h, r, t = triple

        if h not in ent2rel.keys():
            print(h)
            continue
        if t not in ent2rel.keys():
            print(t)
            continue
        h_id = ent2rel[h]
        if len(h_id) > id_max_len:
            h_id = h_id[:id_max_len]
        else:
            h_id = [0]*(id_max_len-len(h_id)) + h_id

        t_id = ent2rel[t]
        if len(t_id) > id_max_len:
            t_id = t_id[:id_max_len]
        else:
            t_id = [0] * (id_max_len - len(t_id)) + t_id


        head_ids.append(torch.tensor(h_id))
        tail_ids.append(torch.tensor(t_id))
        labels.append(rel2idx[r])

    data_save = [torch.stack(head_ids),
                 torch.stack(tail_ids),
                 torch.tensor(labels)]

    print(len(head_ids))
    return data_save


import evaluate
def train(model, train_data):
    device = 'cuda'
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2)
    loss_print = 0
    ground_truth = []
    prediction = []
    accuracy_metric = evaluate.load("accuracy")
    divide = 1


    model.train()
    dataset = TensorDataset(*train_data)
    accumulation_steps = 1
    load = DataLoader(dataset, batch_size=32, shuffle=True)
    print(torch.sum(torch.abs(model.rel_embedding.weight)))
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        head_id, tail_id, label = item
        res = model(head_id, tail_id, label)
        loss = res.loss / accumulation_steps
        loss_print += loss

        pred = res.logits
        prediction.append(torch.argmax(pred, dim=1))
        ground_truth.append(label)
        loss.backward()

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (idx + 0) % 128 == 0:
            ground_truth = torch.cat(ground_truth)
            prediction = torch.cat(prediction)
            results = accuracy_metric.compute(references=ground_truth, predictions=prediction)
            print(results)
            print(f'loss:{loss_print}')
            print(torch.sum(torch.abs(model.rel_embedding.weight)))
            prediction = []
            ground_truth = []
            loss_print = 0


def collect_test_ent():
    all_triple = []
    data = load_json(f'../data/fewrel/test.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        rel = item['label']
        all_triple.append((head, rel, tail))
    return all_triple


def get_triple():
    kg = open('../data/wiki/wikidata5m_inductive_train.txt')
    all_triple = []
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split()
        all_triple.append((h, r, t))
    return all_triple[::5]

from util import load_file

class TrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="single_label_classification",
                                                          num_labels=825)
        rel_num = 822
        # self.rel_embedding = nn.Embedding(rel_num*2+1+1, 768, padding_idx=0)
        self.rel_embedding = nn.Embedding.from_pretrained(load_file('../param/embedding/rel_embed'))
        self.rel_embedding.weight = torch.nn.Parameter(self.rel_embedding.weight)
        self.token_embedding = self.model.bert.get_input_embeddings()
        # print(torch.sum(torch.abs(self.rel_embedding.weight)))
        # self.token_embedding = self.model.bert.get_input_embeddings()


    def forward(self, h_id, t_id, labels):
        # h_id += 1000
        # t_id += 1000
        # h_id[:, 0] = 822*2+1
        # t_id[:, 0] = 822*2+1
        cls_embed = self.token_embedding(torch.ones((len(labels), 1)).long().cuda()*103)
        head_embed = self.rel_embedding(h_id)
        tail_embed = self.rel_embedding(t_id)
        head_embed = torch.sum(head_embed, dim=1).unsqueeze(dim=1)
        tail_embed = torch.sum(tail_embed, dim=1).unsqueeze(dim=1)
        input_embed = torch.cat([head_embed, tail_embed], dim=1)
        # print(input_embed.size())
        # pos_id = torch.zeros(input_embed.size()[:2]).long()
        res = self.model(inputs_embeds=input_embed, labels=labels)
        return res


def train_setting(model):
    for idx, (n, p) in enumerate(model.named_parameters()):
        if idx <= 198:
            p.requires_grad = False

if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # print(tokenizer.encode('[CLS]'))
    rel2idx = load_json('../data/rel2id_wiki.json')
    ent2rel = load_json('../data/ent2rel.json')
    # triple = get_triple()
    triple = collect_test_ent()
    data = convert_triple_to_trainable_data(triple, rel2idx, ent2rel)

    model = TrainModel()
    train_setting(model)
    check_model_grad(model)
    for i in range(100):
        train(model, data)


