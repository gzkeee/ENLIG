from dgl.data import DGLDataset
import torch
import dgl
from util import get_cur, sql_ent, sql_triple_with_tail, load_json, load_file
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

# biG = load_file('../../data/wiki/biG_f')
coo = load_file('../../data/coo')
csr = csr_matrix(coo)
csc = csc_matrix(coo)
ent2triple = load_file('../../data/ent2triple')
rel2id = load_json('../../data/rel2id_wiki.json')
ent2id = load_json('../../data/ent2id.json')
ent2graph = load_file('../../data/ent2triple')


# 返回经过过滤后的实体相关三元组
def sql_related_triple(ent):
    triple_filter = ent2triple.get(ent, [])
    if len(triple_filter) > 0:
        return triple_filter

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


# 将one-hop三元组转换为线图
def triples2line_graph(triples):
    num = len(triples)
    head = []
    tail = []
    pats = []

    for i in range(num):
        for j in range(i, num):
            pat = cal_pattern(triples[i], triples[j])
            pats.append(pat)
            head.append(triples[i][1])
            tail.append(triples[j][1])

            pats.append(pat)
            head.append(triples[j][1])
            tail.append(triples[i][1])

    # 将文本转化为数字
    rels = list(set(head+tail))
    # rels_idx = [rel2id[r] for r in rels]
    rels_idx = rels
    rel2id = {}
    for idx, r in enumerate(rels):
        rel2id[r] = idx

    head = [rel2id[r] for r in head]
    tail = [rel2id[r] for r in tail]

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

    if triple1[0] == triple2[2] and triple1[2] == triple2[0]:
        return pat2id['LOOP']
    if triple1[0] == triple2[0] and triple1[2] == triple2[2]:
        return pat2id['PARA']
    if triple1[0] == triple2[0] and triple1[2] != triple2[2]:
        return pat2id['H-H']
    if triple1[2] == triple2[2] and triple1[0] != triple2[0]:
        return pat2id['T-T']
    if triple1[0] == triple2[2] and triple1[2] != triple2[0]:
        return pat2id['H-T']
    if triple1[2] == triple2[0] and triple1[0] != triple2[2]:
        return pat2id['T-H']
    return pat2id['None']


# 根据实体周边的关系创建线图
def get_graph(ent):
    # ent = ent2id[ent]
    triples = sql_related_triple(ent)
    g, rels_idx, pats = triples2line_graph(triples)
    g.ndata['idx'] = torch.tensor(rels_idx)
    g.edata['idx'] = torch.tensor(pats)
    return g


def get_triple():
    kg = open('../../data/wiki/wikidata5m_inductive_train.txt')
    all_triple = []
    for triple in tqdm(list(kg)[::10000]):
        triple = triple.strip()
        h, r, t = triple.split()
        all_triple.append((h, r, t))
    return all_triple


class KGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='KG')

    def process(self):
        rel2id = load_json('../../data/rel2id_wiki.json')
        self.head_graphs = []
        self.tail_graphs = []
        self.labels = []
        all_triples = get_triple()

        for i in tqdm(all_triples):
            g = ent2graph.get(i[0], None)
            if g is None:
                g = get_graph(i[0])
            self.head_graphs.append(g)

            g = ent2graph.get(i[2], None)
            if g is None:
                g = get_graph(i[2])
            self.tail_graphs.append(g)

            self.labels.append(rel2id[i[1]])

        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.head_graphs[i], self.tail_graphs[i], self.labels[i]

    def __len__(self):
        return len(self.head_graphs)


if __name__ == '__main__':
    pass