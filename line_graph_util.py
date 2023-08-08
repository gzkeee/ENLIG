from tqdm import tqdm
import torch
import dgl


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


def sql_related_triple(ent, csr, csc):
    triple_filter = []
    row = csr.getrow(ent)
    col = csc.getcol(ent)
    rel = []
    for i in range(len(row.indices)):
        if row.data[i] not in rel:
            triple_filter.append((ent, row.data[i], row.indices[i]))
            rel.append(row.data[i])

    rel = []
    for i in range(len(col.indices)):
        if col.data[i] not in rel:
            triple_filter.append((col.indices[i], col.data[i], ent))
            rel.append(col.data[i])

    return triple_filter


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


# 将one-hop三元组转换为线图
def triples2line_graph(triples):
    num = len(triples)
    head = []
    tail = []
    pats = []
    rels_idx = []
    for i in range(num):
        rels_idx.append(triples[i][1])
        for j in range(i, num):
            if i != j:
                pat = cal_pattern(triples[i], triples[j])
                pats.append(pat)
                head.append(i)
                tail.append(j)

                pats.append(pat)
                head.append(j)
                tail.append(i)

    if len(head) == 0:
        g = dgl.graph((head, tail), idtype=torch.int32, num_nodes=1)
    else:
        g = dgl.graph((head, tail), idtype=torch.int32)
    return g, rels_idx, pats


# 根据实体周边的关系创建线图
def get_graph(ent, csr, csc):
    # 返回空图
    if ent < 0:
        g = dgl.graph(([], []), idtype=torch.int32, num_nodes=1)
        g.ndata['idx'] = torch.tensor([0], dtype=torch.int32)
        g.edata['idx'] = torch.tensor([], dtype=torch.int8)
    else:
        triples = sql_related_triple(ent, csr, csc)
        g, rels_idx, pats = triples2line_graph(triples)
        g.ndata['idx'] = torch.tensor(rels_idx, dtype=torch.int32)
        g.edata['idx'] = torch.tensor(pats, dtype=torch.int8)
    return g


def sql_related_triple_2h(ent, csr, csc):
    triple_filter = []
    row = csr.getrow(ent)
    col = csc.getcol(ent)

    nei = []
    for i in range(len(row.indices)):
            triple_filter.append((ent, row.data[i], row.indices[i]))
            nei.append(row.indices[i])

    for i in range(len(col.indices)):
            triple_filter.append((col.indices[i], col.data[i], ent))
            nei.append(col.indices[i])

    triple_2h = []
    for e in nei:
        triple_2h += sql_related_triple(e, csr, csc)
    triple = list(set(triple_2h+triple_filter))

    return triple


# 根据实体周边的关系创建线图
def get_graph_2h(ent, csr, csc):
    # 返回空图
    if ent < 0:
        g = dgl.graph(([], []), idtype=torch.int32, num_nodes=1)
        g.ndata['idx'] = torch.tensor([0], dtype=torch.int32)
        g.edata['idx'] = torch.tensor([], dtype=torch.int8)
    else:
        triples = sql_related_triple_2h(ent, csr, csc)
        g, rels_idx, pats = triples2line_graph(triples)
        g.ndata['idx'] = torch.tensor(rels_idx, dtype=torch.int32)
        g.edata['idx'] = torch.tensor(pats, dtype=torch.int8)
    return g