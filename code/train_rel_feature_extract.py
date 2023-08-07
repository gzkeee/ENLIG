from util import load_json, save_json, save_file, load_file
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import lora
import evaluate
from data_process import convert_triple_to_trainable_data


# 获取全部的三元组
def get_triples():
    tf = open('../data/wiki/wikidata5m_inductive_train.txt')
    triples = []
    for t in tqdm(list(tf)):
        t = t.strip()
        h, r, t = t.split('\t')
        triples.append((h, r, t))
    return triples


def get_ent_num_statistic():
    return load_json('../data/ent2num_in_wiki')


# 获取rel2idx的映射
def get_rel2idx():
    return load_json('../data/wiki/rel2id.json')


def get_ent2text():
    return load_json('../data/ent2text.json')


def get_high_freq_triple():
    return load_json('../data/wiki/high_freq_triple')


# 将出现次数太少的三元组过滤掉
def filter_triple(triple, ent_num, threshold=500):
    triple_filter = []
    for tri in tqdm(triple):
        h, r, t = tri
        if ent_num.get(h, 0) < threshold and ent_num.get(t, 0) < threshold:
            triple_filter.append(tri)
    return triple_filter



def get_filter_triple():
    triples = get_triples()
    ent2num = get_ent_num_statistic()
    triples = filter_triple(triples, ent2num, 50)
    save_json(triples, '../data/wiki/freq_triple')
    return triples





def get_train_data():
    return load_file(f'../data/trainable triple/train_data_higher than 100')

def get_inset_lora_model(r):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          problem_type="single_label_classification",
                                                          num_labels=825)
    # save_file(model.state_dict(), '../param/pretrain')
    encoder = model.bert.encoder
    layers = encoder.layer
    for lay in layers:
        d = lay.output.dense
        lay.output.dense = lora.Linear(d.in_features, d.out_features, r)

    model.load_state_dict(load_file('../param/pretrain'), strict=False)
    return model




# 训练模型
# freeze embedding and main model
def train():
    device = 'cuda'
    model = get_inset_lora_model(8)
    model = model.to(device)
    lora.mark_only_lora_as_trainable(model.bert)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    train_data = get_train_data()
    loss_print = 0
    ground_truth = []
    prediction = []
    accuracy_metric = evaluate.load("accuracy")
    divide = 1
    for epc in range(divide):
        dataset = TensorDataset(*train_data)
        accumulation_steps = 1
        load = DataLoader(dataset, batch_size=128, shuffle=True)
        for idx, item in enumerate(tqdm(load)):
            item = [i.to(device) for i in item]
            input_ids, token_type_ids, attention_mask, label = item
            res = model(input_ids, attention_mask, token_type_ids, labels=label)
            loss = res.loss / accumulation_steps
            loss_print += loss
            pred = res.logits

            prediction.append(torch.argmax(pred, dim=1))
            ground_truth.append(label)
            loss.backward()

            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % 256 == 0:
                ground_truth = torch.cat(ground_truth)
                prediction = torch.cat(prediction)
                results = accuracy_metric.compute(references=ground_truth, predictions=prediction)
                print(results)
                print(f'loss:{loss_print}')
                prediction = []
                ground_truth = []
                loss_print = 0
        save_file(model.state_dict(), f'../param/rel_feature_extract_{epc}')






if __name__ == '__main__':
    # triple = get_filter_triple()
    # convert_triple_to_trainable_data(triple)
    train()