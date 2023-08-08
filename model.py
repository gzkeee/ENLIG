import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, hidden_size):
        super(Conv, self).__init__()
        self.pattern_embedding = nn.Embedding(10, hidden_size)

    def forward(self, g, rel):
        with g.local_scope():
            g.ndata['h'] = rel
            g.edata['p'] = self.pattern_embedding(g.edata['idx'].long())
            g.update_all(message_func=fn.u_mul_e('h', 'p', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            return h_N+rel


class ConvAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvAT, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z1 = self.fc(edges.src["h"]*edges.data['p'])
        dst = self.fc(edges.dst["h"])
        z2 = torch.cat([z1, dst], dim=1)
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
class GraphEncoder(nn.Module):
    def __init__(self, in_feats=768, hidden_size=768):
        super(GraphEncoder, self).__init__()
        # self.conv1 = ConvAT(in_feats, hidden_size)
        self.conv1 = MultiHeadGATLayer(in_feats, hidden_size, 4)
        self.rel_embedding = nn.Embedding(825, hidden_size, padding_idx=0)
        self.pattern_embedding = nn.Embedding(10, hidden_size)
        nn.init.constant_(self.rel_embedding.weight, 0.)
        nn.init.constant_(self.pattern_embedding.weight, 0.)
        self.act = nn.ReLU()

    def forward(self, g):
        rel_h = self.rel_embedding(g.ndata['idx'].long())
        pat_h = self.pattern_embedding(g.edata['idx'].long())
        h_1 = self.conv1(g, rel_h, pat_h)
        h_1 = self.act(h_1)

        g.ndata['f'] = rel_h+h_1
        e_h = dgl.sum_nodes(g, 'f')
        return e_h


class TrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="single_label_classification",
                                                          num_labels=825)
        self.ent_embedding = GraphEncoder()

    def forward(self, h_graph, t_graph, labels):
        head_embed = self.ent_embedding(h_graph)
        tail_embed = self.ent_embedding(t_graph)
        input_embed = torch.stack([head_embed, tail_embed], dim=1)
        res = self.model(inputs_embeds=input_embed, labels=labels)
        return res


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class FewRelTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="single_label_classification",
                                                          num_labels=825)
        self.token_embedding = self.model.bert.get_input_embeddings()
        self.ent_embedding = GraphEncoder()
        self.config = self.model.config

    # head # id=1001
    # tail $ id=1002
    def forward(self, input_ids, token_type_ids, attention_mask, labels, h_id, t_id, h_pos, t_pos):
        bsz = input_ids.size()[0]
        # print(tokenizer.decode(input_ids[0]))
        pos_id = torch.arange(1, 126).expand((bsz, -1)).to('cuda')
        pos_cls = torch.zeros((bsz, 1)).to('cuda')
        pos_id = torch.cat([pos_cls, h_pos, t_pos, pos_id], dim=1).long()
        # print(pos_id.size())
        input_embed = self.token_embedding(input_ids)
        head_embed = self.ent_embedding(h_id)
        tail_embed = self.ent_embedding(t_id)

        # print(head_embed.size())
        input_embed[:, 1] = head_embed
        input_embed[:, 2] = tail_embed
        res = self.model(inputs_embeds=input_embed, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=pos_id, labels=labels)
        # exit()
        return res


class OpenEntityTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=9,
                                                                   problem_type="multi_label_classification")
        self.token_embedding = self.model.bert.get_input_embeddings()
        self.ent_embedding = GraphEncoder()

    # head # id=1001
    # tail $ id=1002
    def forward(self, input_ids, token_type_ids, attention_mask, labels, h_id):
        input_embed = self.token_embedding(input_ids)
        head_embed = self.ent_embedding(h_id)
        # print(head_embed.size())
        input_embed[:, 1] = head_embed
        res = self.model(inputs_embeds=input_embed, attention_mask=attention_mask,
            token_type_ids=token_type_ids, labels=labels)
        return res


def weigth_init(m):
    if isinstance(m, GraphEncoder):
        nn.init.constant_(m.rel_embedding.weight, 0.)
        nn.init.constant_(m.pattern_embedding.weight, 0.)


if __name__ == '__main__':
    model = TrainModel()
    model.apply(weigth_init)
    # model.apply(weight_load)



