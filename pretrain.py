import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from data import KGDataset
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import evaluate
from util import save_file, load_file, load_json, check_model_grad
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import dgl.function as fn
from data import get_graph




def train_setting(model):
    for idx, (n, p) in enumerate(model.named_parameters()):
        if idx <= 198:
            p.requires_grad = False

class SAGEConv(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, rel, pattern):
        with g.local_scope():
            g.ndata['h'] = rel
            g.edata['p'] = pattern
            # update_all is a message passing API.
            g.update_all(message_func=fn.u_mul_e('h', 'p', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            # h_total = torch.cat([rel, h_N], dim=1)
            return h_N+rel

class ConvAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvAT, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z1 = edges.src["h"]*edges.data['p']
        z2 = torch.cat([z1, edges.dst["h"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a), 'm': z1}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"e": edges.data["e"], "m": edges.data["m"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["m"], dim=1)
        return {"h": h}

    def forward(self, g, rel, pattern):
        with g.local_scope():
            g.ndata['h'] = rel
            g.edata['p'] = pattern
            g.apply_edges(self.edge_attention)
            # update_all is a message passing API.
            g.update_all(self.message_func, self.reduce_func)
            # h_N = g.ndata['h_N']
            # h_total = torch.cat([rel, h_N], dim=1)
            return g.ndata.pop("h")

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(ConvAT(in_dim, out_dim))

    def forward(self, g, rel, pattern):
        head_outs = [attn_head(g, rel, pattern) for attn_head in self.heads]
        # merge using average
        return torch.mean(torch.stack(head_outs))


# 输入graph 得到graph的向量表示
class Model(nn.Module):
    def __init__(self, in_feats=768, hidden_size=768):
        super(Model, self).__init__()
        # self.conv1 = ConvAT(in_feats, hidden_size)
        self.conv1 = MultiHeadGATLayer(in_feats, hidden_size, 1)
        self.rel_embedding = nn.Embedding(825, hidden_size, padding_idx=0)
        self.pattern_embedding = nn.Embedding(10, hidden_size)
        nn.init.constant_(self.rel_embedding.weight, 0.)
        nn.init.constant_(self.pattern_embedding.weight, 0.)
        # self.act = nn.ReLU()

    def forward(self, g):
        rel_h = self.rel_embedding(g.ndata['idx'].long())
        pat_h = self.pattern_embedding(g.edata['idx'].long())
        h_1 = self.conv1(g, rel_h, pat_h)
        h_1 = F.leaky_relu(h_1)

        g.ndata['f'] = rel_h+h_1
        e_h = dgl.sum_nodes(g, 'f')
        return e_h


class Cls(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(Cls, self).__init__()
        self.num_labels = 800
        self.conv1 = SAGEConv(in_feats, hidden_size)
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.rel_embedding = nn.Embedding(825, hidden_size)
        self.pattern_embedding = nn.Embedding(10, hidden_size)
        # nn.init.constant_(self.rel_embedding.weight, 0.)
        # nn.init.constant_(self.pattern_embedding.weight, 0.)

        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, g_h, g_t, labels):
        rel_h = self.rel_embedding(g_h.ndata['idx'].long())
        pat_h = self.pattern_embedding(g_h.edata['idx'].long())
        # h = self.conv1(g_h, rel_h, pat_h)
        g_h.ndata['f'] = rel_h
        e_h = dgl.sum_nodes(g_h, 'f')



        rel_t = self.rel_embedding(g_t.ndata['idx'].long())
        pat_t = self.pattern_embedding(g_t.edata['idx'].long())
        # h = self.conv1(g_t, rel_t, pat_t)
        g_t.ndata['f'] = rel_t
        e_t = dgl.sum_nodes(g_t, 'f')

        e = e_h-e_t
        logits = self.classifier(e)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {'logits': logits, 'loss': loss}
        # return e_h


class TrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="single_label_classification",
                                                          num_labels=825)
        rel_num = 822
        self.ent_embedding = Model(768, 768)
        # self.rel_embedding = nn.Embedding.from_pretrained(load_file('../param/embedding/rel_embed'))
        # self.rel_embedding = nn.Embedding(rel_num*2+1+1, 768, padding_idx=0)
        # self.rel_embedding.weight = torch.nn.Parameter(torch.zeros_like(self.rel_embedding.weight))
        self.token_embedding = self.model.bert.get_input_embeddings()


    def forward(self, h_graph, t_graph, labels):
        head_embed = self.ent_embedding(h_graph)
        tail_embed = self.ent_embedding(t_graph)
        # print(head_embed.size())
        input_embed = torch.stack([head_embed, tail_embed], dim=1)
        # print(input_embed.size())
        res = self.model(inputs_embeds=input_embed, labels=labels)
        return res


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


def train(model, train_data):
    # ent2graph = load_file('../../data/ent2graph_list_0.05')
    device = 'cuda'
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
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

    save_file(model.ent_embedding.rel_embedding.weight, './param/rel_embed_0.05_0h')
    save_file(model.ent_embedding.pattern_embedding.weight, './param/pat_embed_0.05_0h')




def get_train_data():
    kg = open('../../data/wiki/wikidata5m_inductive_train.txt')
    rel2id = load_json('../../data/rel2id_wiki.json')
    ent2id = load_json('../../data/ent2id.json')
    head = []
    tail = []
    rel = []
    for triple in tqdm(list(kg)[::20]):
        triple = triple.strip()
        h, r, t = triple.split()
        head.append(ent2id[h])
        tail.append(ent2id[t])
        rel.append(rel2id[r])

    head = torch.tensor(head)
    tail = torch.tensor(tail)
    rel = torch.LongTensor(rel)
    return head, tail, rel

def init_weight(m):
    nn.init.constant_(m, 0.)


if __name__ == '__main__':
    # dataset = KGDataset()
    # save_file(dataset, './data/kg_data')
    # dataset = load_file('./data/kg_data')
    model = TrainModel()
    # model = Cls(768, 768)
    # model.apply(init_weight)
    train_setting(model)
    check_model_grad(model)
    dataset = get_train_data()
    train(model, dataset)