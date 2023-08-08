import pickle
import json
import sqlite3
from multiprocessing import Pool
from transformers import set_seed
import random
import torch
from tqdm import tqdm


def save_file(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_json(path):
    f = open(path, 'r', encoding='utf-8')
    return json.load(f)


def save_json(obj, path):
    b = json.dumps(obj)
    f2 = open(path, 'w')
    f2.write(b)
    f2.close()


def get_cur():
    db_path = 'C:\\Users\\Gezk\\Desktop\\representation learning\\data\\wikidata_5m.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    return cur


def sql_ent(cur, ent):
    sql_query_entity = '''
        select * from triples where Head == '{}'
        '''
    # print(sql_query_entity)
    cur.execute(sql_query_entity.format(ent))
    res = cur.fetchall()
    return res


def sql_triple_with_tail(cur, ent):
    sql_query_entity = '''
        select * from triples where Tail == '{}'
        '''
    # print(sql_query_entity)
    cur.execute(sql_query_entity.format(ent))
    res = cur.fetchall()
    return res


def sql_triple_with_rel(cur, rel):
    sql_query_entity = '''
        select * from triples where Rel == '{}'
        LIMIT 10000
        '''
    cur.execute(sql_query_entity.format(rel))
    res = cur.fetchall()
    return res


# 获取rel2idx的映射
def get_rel2idx():
    return load_json('../data/wiki/rel2id.json')


def get_ent2text():
    return load_json('../data/ent2text.json')


def check_model_grad(model):
    for idx, (n, p) in enumerate(model.named_parameters()):
        print(f'{idx}::{n}:{p.size()}:{p.requires_grad}')


def sql_head_tail(cur, head, tail):
    sql_query_entity = '''
        select * from triples where Head == '{}' and Tail == '{}'
        '''
    cur.execute(sql_query_entity.format(head, tail))
    # print(sql_query_entity.format(head, tail))
    res = cur.fetchall()
    rel = [r[1] for r in res]
    return rel


def cal_ent_num_in_kg():
    ent2num = {}
    kg = open('../../data/wiki/wikidata5m_inductive_train.txt')
    for triple in kg:
        triple = triple.strip()
        h, r, t = triple.split()
        num = ent2num.get(h, 0) + 1
        ent2num[h] = num

        num = ent2num.get(t, 0) + 1
        ent2num[t] = num


def get_csr():
    return load_file('./data/csr_compressed')


def get_csc():
    return load_file('./data/csc_compressed')


def create_coo():
    rel2id = load_json('../../data/rel2id_wiki.json')
    ent2id = load_json('../../data/ent2id.json')
    row = []
    col = []
    data = []
    kg = open('../../data/wiki/wikidata5m_inductive_train.txt')
    for triple in tqdm(list(kg)):
        triple = triple.strip()
        h, r, t = triple.split()
        row.append(ent2id[h])
        col.append(ent2id[t])
        data.append(rel2id[r])

    coo = coo_matrix((data, (row, col)))

if __name__ == '__main__':
    cur = get_cur()
    cur.execute('''
    CREATE INDEX tail
        on triples (Tail)
    ''')