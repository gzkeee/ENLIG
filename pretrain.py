import dgl
import torch
from tqdm import tqdm
import evaluate
from util import save_file, load_file, load_json, check_model_grad
from torch.utils.data import TensorDataset, DataLoader
from data import get_graph
from model import TrainModel


def train_setting(model):
    for idx, (n, p) in enumerate(model.named_parameters()):
        if idx <= 198:
            p.requires_grad = False


def data_transfer(ents, ent2graph):
    g_batch = []
    for e in ents:
        g = ent2graph[e]
        if g is None:
            g = get_graph(e)
        g_batch.append(g)
    return dgl.batch(g_batch)


def data_transfer(ents, ent2graph):
    g_batch = []
    for e in ents:
        g = ent2graph[e]
        if g is None:
            g = get_graph(e)
        g_batch.append(g)
    return dgl.batch(g_batch)


def train(model, train_data, ent2graph):
    device = 'cuda'
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loss_print = 0
    ground_truth = []
    prediction = []
    accuracy_metric = evaluate.load("accuracy")

    model.train()
    dataset = TensorDataset(*train_data)
    accumulation_steps = 1
    load = DataLoader(dataset, batch_size=32, shuffle=True)
    # print(torch.sum(torch.abs(model.rel_embedding.weight)))
    for i in range(10):
        for idx, item in enumerate(tqdm(load)):
            item = [i.to(device) for i in item]
            head_id, tail_id, label = item
            head_id = data_transfer(head_id, ent2graph).to(device)
            tail_id = data_transfer(tail_id, ent2graph).to(device)
            res = model(head_id, tail_id, label)
            loss = res['loss'] / accumulation_steps
            pred = res['logits']
            # loss = res.loss / accumulation_steps
            # pred = res.logits
            loss_print += loss

            prediction.append(torch.argmax(pred, dim=1))
            ground_truth.append(label)
            loss.backward()

            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 0) % 1024 == 0:
                ground_truth = torch.cat(ground_truth)
                prediction = torch.cat(prediction)
                results = accuracy_metric.compute(references=ground_truth, predictions=prediction)
                print(results)
                print(f'loss:{loss_print}')
                # print(torch.sum(torch.abs(model.rel_embedding.weight)))
                prediction = []
                ground_truth = []
                loss_print = 0

    save_file(model.ent_embedding.state_dict(), './param/graph_encoder')


def get_train_data():
    kg = open('./data/wiki/wikidata5m_inductive_train.txt')
    rel2id = load_json('./data/rel2id_wiki.json')
    ent2id = load_json('./data/ent2id.json')
    head = []
    tail = []
    rel = []
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split()
        head.append(ent2id[h])
        tail.append(ent2id[t])
        rel.append(rel2id[r])

    head = torch.tensor(head)
    tail = torch.tensor(tail)
    rel = torch.LongTensor(rel)
    return head, tail, rel


if __name__ == '__main__':
    ent2graph = load_file('./data/ent2graph')
    model = TrainModel()
    train_setting(model)
    check_model_grad(model)
    dataset = get_train_data()
    train(model, dataset, ent2graph)