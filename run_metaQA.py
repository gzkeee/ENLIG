import networkx as nx
from util import save_json, save_file, load_file, load_json
from tqdm import tqdm
import torch
from transformers import BertForTokenClassification, set_seed, BertTokenizer
import torch.utils.data as Data
import evaluate
from model import GraphEncoder
from torch import nn
import argparse
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from line_graph_util import get_graph, get_graph_2h
import dgl

def f1():
    f = open('../kb_entity_dict.txt', 'r')
    ent2id = {}
    id2ent = {}
    for item in f:
        item = item.strip().split('\t')
        # 将0作为空位
        ent2id[item[1]] = int(item[0])+1
        id2ent[int(item[0])+1] = item[1]
    save_json(ent2id, '../ent2id.json')
    save_json(id2ent, '../id2ent.json')


# 创建rel2id以及id2rel
def f2():
    f = open('../kb.txt', 'r')
    rel2id = {}
    id2rel = {}
    all_relations = []
    for item in f:
        item = item.strip().split('|')
        all_relations.append(item[1])

    all_relations = list(set(all_relations))
    for idx, rel in enumerate(all_relations):
        # 将0作为空位
        rel2id[rel] = idx+1
        id2rel[idx+1] = rel
    save_json(rel2id, '../rel2id.json')
    save_json(id2rel, '../id2rel.json')


# 将知识图谱转化为图
# 50%的KG
def f3():
    f = open('./data/kb.txt', 'r')
    biG = nx.MultiDiGraph()
    for item in list(f)[::2]:
        item = item.strip().split('|')
        item[1] = item[1].replace('_', ' ')
        biG.add_edge(item[0], item[2], relation=item[1])
    save_file(biG, './data/biG_f_0.5')
    # nx.write_gml(biG, './data/metaQA/biG')


# 将实体按关系进行分类
def f4():
    f = open('./data/metaQA/kb.txt')
    rel2head = {}
    rel2tail = {}
    for triple in f:
        triple = triple.strip()
        triple = triple.split('|')
        rel2head[triple[1]] = rel2head.get(triple[1], [])+[triple[0]]
        rel2tail[triple[1]] = rel2tail.get(triple[1], [])+[triple[2]]
        # print(triple)
        # exit()
    rel2head = [list(set(v)) for k, v in rel2head.items()]
    rel2tail = [list(set(v)) for k, v in rel2tail.items()]
    # save_json(rel2head, './data/metaQA/rel2head.json')
    # save_json(rel2tail, './data/metaQA/rel2tail.json')
    save_json(rel2tail+rel2head, './data/metaQA/ent_cls.json')


# 创建id2triple
def f5():
    f = open('./kb.txt', 'r')
    f = list(f)
    id2triple = {}
    triple2id = {}
    for idx, item in enumerate(f):
        item = item.strip().split('|')
        item[1] = item[1].replace('_', ' ')
        id2triple[idx] = (item[0], item[1], item[2])
        triple2id[' '.join([item[0], item[1], item[2]])] = idx

    save_json(id2triple, './id2triple.json')
    save_json(triple2id, './triple2id.json')


# 创建ent2rel
def f6():
    kg = open('../kb.txt')
    rel2id = load_json('../rel2id.json')
    ent2id = load_json('../ent2id.json')
    rel_num = len(rel2id)
    ent2rel = {}

    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split('|')

        if h not in ent2rel.keys():
            ent2rel[h] = []
        ent2rel[h].append(rel2id[r])

        if t not in ent2rel.keys():
            ent2rel[t] = []
        ent2rel[t].append(rel2id[r] + rel_num)

    for k, v in ent2rel.items():
        ent2rel[k] = list(set(v))
    print(ent2rel)

    res = torch.zeros((len(ent2rel)+1, rel_num*2+1))

    # 返回索引
    for k, v in ent2rel.items():
        for idx, r in enumerate(v):
            res[ent2id[k], idx] = r
    print(res)

    return res



def extract_ent(text):
    start = text.find("[") + 1
    end = text.find("]")
    return text[start:end]


def ret_triple_with_ent(ent, graph):
    know_in = graph.in_edges(ent, data=True)
    know_in = [(k[0], k[2]['relation'], k[1]) for k in know_in]
    know_out = graph.out_edges(ent, data=True)
    know_out = [(k[0], k[2]['relation'], k[1]) for k in know_out]
    know = know_in + know_out
    return know


def encode_text_hop1(tag='train'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ent2id = load_json('./data/ent2id.json')
    flist = list(open(f'./data/qa_{tag}.txt', 'r'))
    graph = load_file('./data/biG_f')
    # num_labels = len(ent2id.keys())
    # 假设一个问题最多只有max_k个triple
    max_k = 64
    max_len = 128
    all_labels = torch.ones(len(flist), max_len)*-100
    all_ents = torch.ones(len(flist), max_k)*-1
    all_question = []
    all_knowledge = []
    for idx, qa in tqdm(enumerate(flist)):
        qa = qa.strip()
        q, a = qa.split('\t')
        labels = a.split('|')

        ent = extract_ent(qa)
        know_in = graph.in_edges(ent, data=True)
        know_in = [k[0] for k in know_in]
        know_out = graph.out_edges(ent, data=True)
        know_out = [k[1] for k in know_out]
        know = know_in + know_out

        # 仅使用50%的kg
        # know = know[::2]
        # 获取所有one-hop邻居
        for i in range(len(know)):
            if i >= 64:
                # print(i)
                continue
            all_ents[idx, i] = ent2id[know[i]]

        # 为所有one-hop邻居打上标签
        for iidx, can in enumerate(know):
            if iidx >= 64:
                # print(iidx)
                continue
            if can in labels:
                # [CLS]
                all_labels[idx, iidx + 1] = 1
            else:
                all_labels[idx, iidx + 1] = 0

        all_question.append(q)
        # 补位
        all_knowledge.append('@ '*max_k)

    input = tokenizer(all_knowledge, all_question,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt')

    return [input.input_ids, input.attention_mask, input.token_type_ids, all_labels, all_ents]



def encode_text_hop2(tag='train'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ent2id = load_json('./data/ent2id.json')
    flist = list(open(f'./data/qa_{tag}.txt', 'r'))
    graph = load_file('./data/biG_f')
    # num_labels = len(ent2id.keys())
    # 假设一个问题最多只有max_k个triple
    max_k = 192
    max_len = 256
    all_labels = torch.ones(len(flist), max_len)*-100
    all_ents = torch.ones(len(flist), max_k)*-1
    all_question = []
    all_knowledge = []
    for idx, qa in tqdm(enumerate(flist)):
        qa = qa.strip()
        q, a = qa.split('\t')
        labels = a.split('|')

        ent = extract_ent(qa)
        know_in = graph.in_edges(ent, data=True)
        know_in = [k[0] for k in know_in]
        know_out = graph.out_edges(ent, data=True)
        know_out = [k[1] for k in know_out]
        know = know_in + know_out

        know_2h = []
        for k in know:
            know_in = graph.in_edges(k, data=True)
            know_in_2 = [k[0] for k in know_in]
            know_out = graph.out_edges(k, data=True)
            know_out_2 = [k[1] for k in know_out]
            know_2h += know_in_2
            know_2h += know_out_2

        know = list(set(know_2h))

        # 仅使用50%的kg
        # know = know[::2]
        # 获取所有one-hop邻居
        for i in range(len(know)):
            if i >= max_k:
                # print(i)
                continue
            all_ents[idx, i] = ent2id[know[i]]

        # 为所有one-hop邻居打上标签
        for iidx, can in enumerate(know):
            if iidx >= max_k:
                # print(iidx)
                continue
            if can in labels:
                # [CLS]
                all_labels[idx, iidx + 1] = 1
            else:
                all_labels[idx, iidx + 1] = 0

        all_question.append(q)
        # 补位
        all_knowledge.append('@ '*max_k)

    input = tokenizer(all_knowledge, all_question,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt')

    return [input.input_ids, input.attention_mask, input.token_type_ids, all_labels, all_ents]


class TrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.ent_embedding = GraphEncoder()
        self.token_embedding = self.model.bert.get_input_embeddings()


    # head # id=1001
    # tail $ id=1002
    def forward(self, input_ids, attention_mask, token_type_ids, labels, ent_id):
        # print(tokenizer.decode(input_ids[0]))
        # exit()
        bsz = input_ids.size()[0]
        input_embed = self.token_embedding(input_ids)
        ent_embed = self.ent_embedding(ent_id)
        ent_embed = ent_embed.view(bsz, -1, 768)
        # print(ent_embed.size())
        # exit()
        ent_num = ent_embed.size()[1]

        input_embed[:, 1:ent_num+1] = ent_embed
        res = self.model(inputs_embeds=input_embed, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, labels=labels)

        return res


def train(model, dataset, ent2graph):
    device = 'cuda'
    bsz = 4
    accumulation_steps = 32/bsz
    dataset = Data.TensorDataset(*dataset)
    load = Data.DataLoader(dataset, batch_size=bsz, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss_print = 0
    accuracy_metric = evaluate.load("accuracy")
    hit1 = 0
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, attention_mask, token_type_ids, labels, ents = item
        ents = data_transfer(ents, ent2graph).to(device)
        res = model(input_ids, attention_mask, token_type_ids, labels.long(), ents)
        loss = res.loss / accumulation_steps
        loss_print += loss
        logits = res.logits

        # print(labels)
        for i in range(len(input_ids)):
            sel = labels[i] != -100
            if torch.sum(sel) == 0:
                continue
            temp = logits[i, sel, 1]
            h = torch.argmax(temp)
            lab = labels[i, sel]
            hit1 += lab[h]

        loss.backward()

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (idx+1) % 256 == 0:
            print(f'Hit@1:{hit1/(256*bsz)}')
            print(loss_print)
            hit1 = 0
            loss_print = 0


def create_coo(kg, rel2id, ent2id):
    row = []
    col = []
    data = []
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split('|')
        row.append(ent2id[h])
        col.append(ent2id[t])
        data.append(rel2id[r])

    coo = coo_matrix((data, (row, col)), shape=(len(ent2id)+1, len(ent2id)+1))
    return coo


def compressed_csr(coo):
    row = []
    col = []
    data = []
    csr = csr_matrix(coo)
    ent_num = coo.shape[0]
    for e in tqdm(range(ent_num)):
        row_e = csr.getrow(e)
        rel = []
        for i in range(len(row_e.indices)):
            if row_e.data[i] not in rel:
                row.append(e)
                col.append(row_e.indices[i])
                data.append(row_e.data[i])
                rel.append(row_e.data[i])
    coo = coo_matrix((data, (row, col)), shape=(ent_num, ent_num))
    compressed_csr = csr_matrix(coo)
    return compressed_csr


def compressed_csc(coo):
    row = []
    col = []
    data = []
    csc = csc_matrix(coo)
    ent_num = coo.shape[0]
    for e in tqdm(range(ent_num)):
        col_e = csc.getcol(e)
        rel = []
        for i in range(len(col_e.indices)):
            if col_e.data[i] not in rel:
                col.append(e)
                row.append(col_e.indices[i])
                data.append(col_e.data[i])
                rel.append(col_e.data[i])
    coo = coo_matrix((data, (row, col)), shape=(ent_num, ent_num))
    compressed_csc = csc_matrix(coo)
    return compressed_csc


def prepare():
    kg = open('./data/kb.txt')
    rel2id = load_json('./data/rel2id.json')
    ent2id = load_json('./data/ent2id.json')
    coo = create_coo(list(kg)[::2], rel2id, ent2id)
    csc = compressed_csc(coo)
    csr = compressed_csr(coo)
    save_file(coo, './data/coo_0.5')
    save_file(csc, './data/csc_0.5')
    save_file(csr, './data/csr_0.5')


def get_ent2graph():
    kg = open('./data/kb.txt')
    ent2id = load_json('./data/ent2id.json')
    csr = load_file('./data/csr')
    csc = load_file('./data/csc')


    ents = []
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split('|')
        ents.append(h)
        ents.append(t)

    ents = list(set(ents))
    ent2graph = {}
    for e in tqdm(ents):
        idx = ent2id.get(e, -1)
        ent2graph[idx] = get_graph(idx, csr, csc)
    idx = -1
    ent2graph[idx] = get_graph(idx, csr, csc)

    save_file(ent2graph, './ent2graph_metaQA')
    return ent2graph


def get_ent2graph_2h():
    kg = open('./data/kb.txt')
    ent2id = load_json('./data/ent2id.json')
    csr = load_file('./data/csr')
    csc = load_file('./data/csc')


    ents = []
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split('|')
        ents.append(h)
        ents.append(t)

    ents = list(set(ents))
    ent2graph = {}
    for e in tqdm(ents):
        idx = ent2id.get(e, -1)
        ent2graph[idx] = get_graph_2h(idx, csr, csc)
    idx = -1
    ent2graph[idx] = get_graph_2h(idx, csr, csc)

    save_file(ent2graph, './ent2graph_metaQA_2h')
    return ent2graph


def get_ent2graph_50():
    kg = list(open('./data/kb.txt'))[::2]
    ent2id = load_json('./data/ent2id.json')
    csr = load_file('./data/csr_0.5')
    csc = load_file('./data/csc_0.5')


    ents = []
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split('|')
        ents.append(h)
        ents.append(t)

    ents = list(set(ents))
    ent2graph = {}
    for e in tqdm(ents):
        idx = ent2id.get(e, -1)
        ent2graph[idx] = get_graph(idx, csr, csc)
    idx = -1
    ent2graph[idx] = get_graph(idx, csr, csc)

    save_file(ent2graph, './ent2graph_metaQA_0.5')
    return ent2graph


def data_transfer(ents, ent2graph):
    ents = ents.view(-1)
    g_batch = []
    for e in ents:
        g = ent2graph[int(e)]
        if g is None:
            g = get_graph(e)
        g_batch.append(g)
    return dgl.batch(g_batch)


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

    args = parser.parse_args()

    ent2graph = load_file('./ent2graph_metaQA')
    model = TrainModel()
    # model.apply(weight_load)
    # dataset = encode_text_hop1()
    dataset = encode_text_hop1()
    train(model, dataset, ent2graph)
