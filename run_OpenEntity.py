import random
from util import load_file, save_file, load_json, get_csr, get_csc, save_json
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import OpenEntityTrainModel
from line_graph_util import get_graph
import dgl
import argparse

def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        for i in range(len(x1)):
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def loose_macro(true, pred):
    num_entities = len(true)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in zip(true, pred):
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1(precision, recall)

def loose_micro(true, pred):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in zip(true, pred):
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)


# fewrel测评任务
# 输入一个模型 测试该模型在fewrel任务上的结果
def get_train_data():
    return load_file('./data/OpenEntity/train')


def get_eval_data():
    return load_file('./data/OpenEntity/test')


def train(model, data, ent_data, ent2graph):
    device = 'cuda'
    bsz = 32
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    dataset = TensorDataset(*data, ent_data)
    load = DataLoader(dataset, batch_size=bsz, shuffle=True)
    model.train()
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, attention_mask, token_type_ids, label, ent = item
        ent = data_transfer(ent, ent2graph, label).to(device)
        res = model(input_ids, token_type_ids, attention_mask, label, ent)
        loss = res.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def collect_train_ent(threshold, ent2id):
    # ent2id = load_json('./data/ent2id.json')
    train = load_json('./data/OpenEntity/train.json')
    sel_ents = []
    for item in train:
        start = item['start']
        end = item['end']
        ents = item['ents']

        target_ent = 'Q0'
        for e in ents:
            if e[1] >= start or e[2] <= end and e[3] > threshold:
                target_ent = e[0]
        sel_ents.append(ent2id.get(target_ent, -1))

    print(len(sel_ents))
    return sel_ents


def collect_test_ent(threshold, ent2id):
    # ent2id = load_json('./data/ent2id.json')
    train = load_json('./data/OpenEntity/test.json')
    sel_ents = []
    for item in train:
        start = item['start']
        end = item['end']
        ents = item['ents']

        target_ent = 'Q0'
        for e in ents:
            if e[1] >= start or e[2] <= end and e[3] > threshold:
                target_ent = e[0]
        sel_ents.append(ent2id.get(target_ent, -1))

    print(len(sel_ents))
    return sel_ents


def eval_data(model, data, ent_data, ent2graph):
    device = 'cuda'
    bsz = 16
    model = model.to(device)
    dataset = TensorDataset(*data, ent_data)
    load = DataLoader(dataset, batch_size=bsz)
    pred = []
    true = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # model.eval()
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, attention_mask, token_type_ids, labels, ent = item
        ent = data_transfer(ent, ent2graph, labels).to(device)
        res = model(input_ids, token_type_ids, attention_mask, labels, ent)

        logits = res.logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        tmp_eval_accuracy, tmp_pred, tmp_true = accuracy(logits, labels)
        pred.extend(tmp_pred)
        true.extend(tmp_true)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_accuracy': eval_accuracy,
              'micro': loose_micro(true, pred),
              'macro':loose_macro(true, pred)
              }
    print(result)


def ent2link_graph(ents):
    csr = get_csr()
    csc = get_csc()

    ents = list(set(ents))
    ent2graph = {}
    for e in tqdm(ents):
        ent2graph[e] = get_graph(e, csr, csc)

    save_file(ent2graph, './data/OpenEntity/ent2graph_OpenEntity')
    return ent2graph




# 根据ent的id将实体转换为线图
def data_transfer(ents, ent2graph, labels):
    # print(labels)
    # r = random.random()
    # if r < 0.25:
    #     for i in range(len(ents)):
    #         if labels[i][0] == 1:
    #             ents[i] = ent2id['Q6892981']
    #         if labels[i][1] == 1:
    #             ents[i] = list(ent2graph.keys())[90]
    #         if labels[i][2] == 1:
    #             ents[i] = list(ent2graph.keys())[100]
    #         if labels[i][3] == 1:
    #             ents[i] = list(ent2graph.keys())[110]
    #         if labels[i][4] == 1:
    #             ents[i] = list(ent2graph.keys())[120]
    #         if labels[i][5] == 1:
    #             ents[i] = list(ent2graph.keys())[130]
    #         if labels[i][6] == 1:
    #             ents[i] = list(ent2graph.keys())[140]
    #         if labels[i][7] == 1:
    #             ents[i] = list(ent2graph.keys())[150]

    # exit()
    g_batch = []
    for e in ents:
        g = ent2graph[int(e)]
        g_batch.append(g)
    return dgl.batch(g_batch)


def collect_Q():
    train = load_json('./data/OpenEntity/train.json')
    sel_ents = []
    for item in train:
        start = item['start']
        end = item['end']
        ents = item['ents']

        target_ent = 'Q0'
        for e in ents:
            if e[1] >= start or e[2] <= end and e[3] > 0:
                target_ent = e[0]
        sel_ents.append(target_ent)

    test = load_json('./data/OpenEntity/test.json')
    for item in test:
        start = item['start']
        end = item['end']
        ents = item['ents']

        target_ent = 'Q0'
        for e in ents:
            if e[1] >= start or e[2] <= end and e[3] > 0:
                target_ent = e[0]
        sel_ents.append(target_ent)

    print(len(sel_ents))
    return sel_ents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--threshold', type=float, default=.3)

    args = parser.parse_args()

    ent2id = load_json('./data/ent2id_.json')
    # print(ent2id[-1])
    # exit()
    model = OpenEntityTrainModel()
    thre = 0.
    ent_train = collect_train_ent(thre, ent2id)
    ent_test = collect_test_ent(thre, ent2id)
    ent2link_graph(ent_train+ent_test)
    ent2graph = load_file('./data/OpenEntity/ent2graph_OpenEntity')
    data_train = get_train_data()
    data_eval = get_eval_data()


    ent_fewrel = set(load_file('./data/fewrel_ents'))

    ent_open = collect_Q()

    label_train = data_train[-1]
    label_eval = data_eval[-1]
    labels = torch.cat([label_train, label_eval])
    print(labels.size())
    save_json(ent2id, './ent2id.json')

    ent_train = collect_train_ent(thre, ent2id)
    ent_test = collect_test_ent(thre, ent2id)
    ent_train = torch.tensor(ent_train).unsqueeze(dim=1)
    ent_test = torch.tensor(ent_test).unsqueeze(dim=1)

    epoch = 15
    for i in range(epoch):
        train(model, data_train, ent_train, ent2graph)
        eval_data(model, data_eval, ent_test, ent2graph)
        eval_data(model, data_eval, ent_test, ent2graph)


