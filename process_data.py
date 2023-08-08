from util import save_json, save_file, load_file, load_json
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import torch
import dgl

coo = load_file('../../data/coo')
# csr = csr_matrix(coo)
# csc = csc_matrix(coo)
csr = load_file('./data/csr_compressed')
csc = load_file('./data/csc_compressed')
ent2triple = load_file('../../data/ent2triple')
rel2id = load_json('../../data/rel2id_wiki.json')
ent2id = load_json('../../data/ent2id.json')


def cal_ent_num_in_kg():
    ent2num = {}
    kg = open('../../data/wiki/wikidata5m_inductive_train.txt')
    for triple in tqdm(kg):
        triple = triple.strip()
        h, r, t = triple.split()
        num = ent2num.get(h, 0) + 1
        ent2num[h] = num

        num = ent2num.get(t, 0) + 1
        ent2num[t] = num
    return ent2num


def sql_related_triple(ent):
    triple_filter = []

    row = csr.getrow(ent)
    col = csc.getcol(ent)
    for i in range(len(row.indices)):
        triple_filter.append((ent, row.data[i], row.indices[i]))

    for i in range(len(col.indices)):
        triple_filter.append((col.indices[i], col.data[i], ent))

    return triple_filter


# 将one-hop三元组转换为线图
def triples2line_graph(triples):
    num = len(triples)
    head = []
    tail = []

    pats = []
    rels_idx = []
    # print(num)
    for i in range(num):
        rels_idx.append(triples[i][1])
        # print(rels_idx)
        # for j in range(i, num):
        for j in range(num):
            if i != j:
                # print((i, j))
                pat = cal_pattern(triples[i], triples[j])
                pats.append(pat)
                head.append(i)
                tail.append(j)

                # pats.append(pat)
                # head.append(j)
                # tail.append(i)

    if len(head) == 0:
        g = dgl.graph((head, tail), idtype=torch.int32, num_nodes=1)
    else:
        g = dgl.graph((head, tail), idtype=torch.int32)

    return g, rels_idx, pats

# 根据两个三元组计算关系连接模式
def cal_pattern(triple1, triple2):
    pat2id = {'None': 0,
              'H-T': 1,
              'T-T': 2,
              'H-H': 3,
              'T-H': 4,
              'PARA': 5,
              'LOOP': 6, }

    if triple1[0] == triple2[0] and triple1[2] != triple2[2]:
        return pat2id['H-H']
    if triple1[2] == triple2[2] and triple1[0] != triple2[0]:
        return pat2id['T-T']
    if triple1[0] == triple2[2] and triple1[2] != triple2[0]:
        return pat2id['H-T']
    if triple1[2] == triple2[0] and triple1[0] != triple2[2]:
        return pat2id['T-H']
    if triple1[0] == triple2[2] and triple1[2] == triple2[0]:
        return pat2id['LOOP']
    if triple1[0] == triple2[0] and triple1[2] == triple2[2]:
        return pat2id['PARA']

    return pat2id['None']

# 根据实体周边的关系创建线图
def get_graph(ent):
    ent = ent2id[ent]
    triples = sql_related_triple(ent)
    g, rels_idx, pats = triples2line_graph(triples)
    g.ndata['idx'] = torch.tensor(rels_idx, dtype=torch.int32)
    g.edata['idx'] = torch.tensor(pats, dtype=torch.int8)
    return g


if __name__ == '__main__':
    kg = open('../../data/wiki/wikidata5m_inductive_train.txt')
    ent_filter = set()
    for triple in tqdm(list(kg)[::200]):
        triple = triple.strip()
        h, r, t = triple.split()
        # get_graph(h)
        # get_graph(t)
        ent_filter.add(h)
        ent_filter.add(t)

    ent2num = cal_ent_num_in_kg()
    ent2graph = [None for _ in range(len(ent2num)+1)]

    for e in tqdm(list(ent_filter)):
        ent2graph[ent2id[e]] = get_graph(e)
