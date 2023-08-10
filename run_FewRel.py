from util import load_json, check_model_grad
from tqdm import tqdm
import torch
import evaluate
from torch.utils.data import TensorDataset, DataLoader
import dgl
from model import FewRelTrainModel
from line_graph_util import get_graph
from util import save_file, load_file, get_csr, get_csc
import json
import os
from transformers import AutoTokenizer
import argparse
from transformers import set_seed


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, h_pos, t_pos, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.h_pos = h_pos
        self.t_pos = t_pos
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_list = sorted(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h, t = example.text_a[1]
        h_name = ex_text_a[h[1]:h[2]]
        h_idx = h[0]
        t_name = ex_text_a[t[1]:t[2]]
        t_idx = t[0]
        # Add [HD] and [TL], which are "#" and "$" respectively.
        if h[1] < t[1]:
            ex_text_a = ex_text_a[:h[1]] + "# " + h_name + " #" + ex_text_a[
                                                                  h[2]:t[1]] + "$ " + t_name + " $" + ex_text_a[t[2]:]
            h_pos = len(tokenizer.encode(ex_text_a[:h[1]] + "# " + h_name))+2
            t_pos = len(tokenizer.encode(ex_text_a[:h[1]] + "# " + h_name + " #" + ex_text_a[
                                                                  h[2]:t[1]] + "$ " + t_name))+2
            # print(h_pos)
            # exit()
        else:
            ex_text_a = ex_text_a[:t[1]] + "$ " + t_name + " $" + ex_text_a[
                                                                  t[2]:h[1]] + "# " + h_name + " #" + ex_text_a[h[2]:]
            h_pos = len(tokenizer.encode(ex_text_a[:t[1]] + "$ " + t_name + " $" + ex_text_a[
                                                                  t[2]:h[1]] + "# " + h_name))+2
            t_pos = len(tokenizer.encode(ex_text_a[:t[1]] + "$ " + t_name))+2

        ex_text_a = ' entity entity '+ex_text_a
        input = tokenizer(ex_text_a, max_length=128, padding='max_length', truncation=True)
        label_id = label_map[example.label]
        features.append(
            InputFeatures(input_ids=input.input_ids,
                          input_mask=input.attention_mask,
                          segment_ids=input.token_type_ids,
                          h_pos=h_pos,
                          t_pos=t_pos,
                          label_id=label_id))
    # exit()
    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())


class FewrelProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
            text_a = (line['text'], line['ents'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_triple_to_trainable_data(data, ent2id, rel2idx):
    head_ids = []
    tail_ids = []
    labels = []
    for idx, triple in enumerate(tqdm(list(data))):
        h, r, t = triple
        head_ids.append(torch.tensor(ent2id.get(h, -1)))
        tail_ids.append(torch.tensor(ent2id.get(t, -1)))
        labels.append(torch.tensor(rel2idx[r]))

    data_save = [torch.stack(head_ids),
                 torch.stack(tail_ids),
                 torch.stack(labels)]

    print(torch.stack(head_ids).size())
    return data_save


# 根据ent的id将实体转换为线图
def data_transfer(ents, ent2graph):
    g_batch = []
    for e in ents:
        g = ent2graph[int(e)]
        if g is None:
            g = get_graph(e)
        g_batch.append(g)
    return dgl.batch(g_batch)


def train(model, train_data, ent_data, ent2graph):
    device = 'cuda'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss_print = 0
    ground_truth = []
    prediction = []
    accuracy_metric = evaluate.load("accuracy")
    divide = 1


    model.train()
    dataset = TensorDataset(*train_data, *ent_data)
    accumulation_steps = 1
    load = DataLoader(dataset, batch_size=32, shuffle=True)
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, attention_mask, token_type_ids, h_pos, t_pos, _, head_id, tail_id, label = item
        head_id = data_transfer(head_id, ent2graph).to(device)
        tail_id = data_transfer(tail_id, ent2graph).to(device)

        # head_pos = torch.argwhere(input_ids == 1001)[:, 1][::2]-1
        # head_pos = head_pos.unsqueeze(dim=1)
        # tail_pos = torch.argwhere(input_ids == 1002)[:, 1][::2]-1
        # tail_pos = tail_pos.unsqueeze(dim=1)
        h_pos = h_pos.unsqueeze(dim=1)
        t_pos = t_pos.unsqueeze(dim=1)
        # exit()

        res = model(input_ids, token_type_ids, attention_mask, label, head_id, tail_id, h_pos, t_pos)
        loss = res.loss / accumulation_steps
        loss_print += loss

        pred = res.logits
        prediction.append(torch.argmax(pred, dim=1))
        ground_truth.append(label)
        loss.backward()

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (idx + 1) % 64 == 0:
            ground_truth = torch.cat(ground_truth)
            prediction = torch.cat(prediction)
            results = accuracy_metric.compute(references=ground_truth, predictions=prediction)
            print(results)
            print(f'loss:{loss_print}')
            # print(torch.sum(torch.abs(model.rel_embedding.weight)))
            prediction = []
            ground_truth = []
            loss_print = 0


def eval_data(model, data, ent_data, ent2graph):
    device = 'cuda'
    bsz = 16
    model = model.to(device)
    accuracy_metric = evaluate.load("accuracy")

    ground_truth = []
    prediction = []
    model.eval()
    dataset = TensorDataset(*data, *ent_data)
    load = DataLoader(dataset, batch_size=bsz)
    for idx, item in enumerate(tqdm(load)):
        with torch.no_grad():
            item = [i.to(device) for i in item]
            input_ids, attention_mask, token_type_ids, h_pos, t_pos, _, head_id, tail_id, label = item
            h_pos = h_pos.unsqueeze(dim=1)
            t_pos = t_pos.unsqueeze(dim=1)
            head_id = data_transfer(head_id, ent2graph).to(device)
            tail_id = data_transfer(tail_id, ent2graph).to(device)
            res = model(input_ids, token_type_ids, attention_mask, label, head_id, tail_id, h_pos, t_pos)

            predict = res.logits.argmax(-1)
            ground_truth += label.cpu().numpy().tolist()
            prediction += predict.cpu().numpy().tolist()

    result = accuracy_metric.compute(references=ground_truth, predictions=prediction)
    print(result)


def collect_train_triple():
    all_triple = []
    data = load_json(f'./data/fewrel/train.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        rel = item['label']
        all_triple.append((head, rel, tail))
    return all_triple


def collect_eval_triple():
    all_triple = []
    data = load_json(f'./data/fewrel/dev.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        rel = item['label']
        all_triple.append((head, rel, tail))
    return all_triple


def collect_test_triple():
    all_triple = []
    data = load_json(f'./data/fewrel/test.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        rel = item['label']
        all_triple.append((head, rel, tail))
    return all_triple


def collect_Q():
    all_ent = []
    data = load_json(f'./data/fewrel/train.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        all_ent.append(head)
        all_ent.append(tail)

    data = load_json(f'./data/fewrel/test.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        all_ent.append(head)
        all_ent.append(tail)
    return all_ent


def get_ent2graph():
    ent2id = load_json('../ent2id.json')
    ents = []
    csr = get_csr()
    csc = get_csc()
    data = load_json(f'./data/fewrel/train.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        ents.append(head)
        ents.append(tail)

    data = load_json(f'./data/fewrel/dev.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        ents.append(head)
        ents.append(tail)

    data = load_json(f'./data/fewrel/test.json')
    for item in data:
        head = item['ents'][0][0]
        tail = item['ents'][1][0]
        ents.append(head)
        ents.append(tail)

    ents = list(set(ents))
    ent2graph = {}
    for e in tqdm(ents):
        ent2graph[ent2id.get(e, -1)] = get_graph(ent2id.get(e, -1), csr, csc)

    save_file(ent2graph, './data/fewrel/ent2graph_fewrel')
    return ent2graph


def get_text_data(tag='train'):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = FewrelProcessor()
    if tag == 'train':
        train_examples, label_list = processor.get_train_examples('./data/fewrel/')
    if tag == 'eval':
        train_examples, label_list = processor.get_dev_examples('./data/fewrel/')
    if tag == 'test':
        train_examples, label_list = processor.get_test_examples('./data/fewrel/')
    train_features = convert_examples_to_features(train_examples, label_list, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_h_pos = torch.tensor([f.h_pos for f in train_features], dtype=torch.long)
    all_t_pos = torch.tensor([f.t_pos for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    text_data = [all_input_ids, all_input_mask, all_segment_ids,all_h_pos,all_t_pos,
                       all_label_ids]
    return text_data


import numpy as np


class F1metrics:
    def __init__(self, y_true:np.array, y_pred:np.array):
        self.TP,self.FP,self.TN,self.FN=0,0,0,0
        for y, y_hat in zip(y_true, y_pred):
            if y_hat == 1 and y_hat == y:
                self.TP += 1
            elif y_hat == 1 and y_hat != y:
                self.FP += 1
            elif y_hat == 0 and y_hat == y:
                self.TN += 1
            else:
                self.FN += 1
        if (self.TP + self.FP) == 0:
            self.precision = 0
        else:
            self.precision = self.TP /(self.TP + self.FP)

        if (self.TP + self.FN) == 0:
            self.recall = 0
        else:
            self.recall = self.TP/(self.TP + self.FN)

    def f1_score(self):
        if self.precision + self.recall == 0:
            f1_score = 0.0
            return f1_score, (self.TP,self.FP,self.TN,self.FN)
        else:
            f1_score = (2 * self.precision * self.recall)/(self.precision + self.recall)
            return f1_score, (self.TP,self.FP,self.TN,self.FN)


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

    print(args)
    print(args.data_dir)
    exit()


    model = FewRelTrainModel()

    rel2idx = load_json('./data/rel2id_wiki.json')
    ent2id = load_json('../ent2id.json')
    ent2graph = load_file('./data/fewrel/ent2graph_fewrel')

    triple = collect_train_triple()
    data = convert_triple_to_trainable_data(triple, ent2id, rel2idx)

    triple = collect_eval_triple()
    data_test = convert_triple_to_trainable_data(triple, ent2id, rel2idx)

    triple = collect_test_triple()
    data_eval = convert_triple_to_trainable_data(triple, ent2id, rel2idx)

    text_data_train = get_text_data('train')
    text_data_eval = get_text_data('eval')
    text_data_test = get_text_data('test')

    for i in range(15):
        train(model, text_data_train, data, ent2graph)
        eval_data(model, text_data_test, data_test, ent2graph)
        eval_data(model, text_data_eval, data_eval, ent2graph)


