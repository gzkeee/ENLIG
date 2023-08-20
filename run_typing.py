import json
import os
from util import load_file, save_file, load_json, get_csr, get_csc, save_json, get_graph
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import OpenEntityTrainModel
import dgl
import argparse
from transformers import AutoTokenizer, set_seed


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


def train(model, data, ent_data, ent2graph, args):
    device = 'cuda'
    bsz = args.train_batch_size
    lr = args.learning_rate
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(*data, ent_data)
    load = DataLoader(dataset, batch_size=bsz, shuffle=True)
    model.train()
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, attention_mask, token_type_ids, pos, label, ent = item
        ent = data_transfer(ent, ent2graph).to(device)
        res = model(input_ids, token_type_ids, attention_mask, label, ent, pos)
        loss = res.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_data(model, data, ent_data, ent2graph, args):
    device = 'cuda'
    bsz = args.train_batch_size
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
        input_ids, attention_mask, token_type_ids, pos, labels, ent = item
        ent = data_transfer(ent, ent2graph).to(device)
        res = model(input_ids, token_type_ids, attention_mask, labels, ent, pos)

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
              'macro': loose_macro(true, pred)
              }
    print(result)


def collect_ent(tag, threshold, ent2id, args):
    if tag == 'train':
        data = load_json(f'{args.data_dir}/train.json')
    if tag == 'dev':
        data = load_json(f'{args.data_dir}/dev.json')
    if tag == 'test':
        data = load_json(f'{args.data_dir}/test.json')

    sel_ents = []
    for item in data:
        start = item['start']
        end = item['end']
        ents = item['ents']
        target_ent = 'Q0'
        for e in ents:
            if e[1] >= start and e[2] <= end and e[3] > threshold:
                target_ent = e[0]
        sel_ents.append(ent2id.get(target_ent, -1))
    sel_ents = torch.tensor(sel_ents).unsqueeze(dim=1)
    return sel_ents


def ent2line_graph(ents):
    csr = get_csr()
    csc = get_csc()

    ents = list(set(ents))
    ent2graph = {}
    for e in tqdm(ents):
        ent2graph[e] = get_graph(e, csr, csc)
    save_file(ent2graph, './data/OpenEntity/ent2graph_OpenEntity')
    return ent2graph


# 根据ent的id将实体转换为线图
def data_transfer(ents, ent2graph):
    g_ = dgl.graph([])
    g_.ndata['idx'] = torch.tensor([], dtype=torch.int32)
    g_.edata['idx'] = torch.tensor([], dtype=torch.int8)
    g_batch = []
    for e in ents:
        g = ent2graph.get(int(e), g_)
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


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, h_pos, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.h_pos = h_pos
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_examples_to_features(examples, label_list, tokenizer, args):
    """Loads a data file into a list of `InputBatch`s."""
    label_list = sorted(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h = example.text_a[1][0]
        h_name = ex_text_a[h[1]:h[2]]
        # ex_text_a = ex_text_a[:h[1]] + "@ " + ex_text_a[h[1]:h[2]] + " @" + ex_text_a[h[2]:]
        begin, end = h[1:3]
        # h[1] += 2
        # h[2] += 2

        # ex_text_a = example.text_a[0]
        # h, t = example.text_a[1]
        # h_name = ex_text_a[h[1]:h[2]]
        # t_name = ex_text_a[t[1]:t[2]]
        # Add [HD] and [TL], which are "#" and "$" respectively.
        ex_text_a = ex_text_a[:h[1]] + "@ " + h_name + " @" + ex_text_a[h[2]:]
        h_pos = len(tokenizer.encode(ex_text_a[:h[1]] + "@ " + h_name))+2

        ex_text_a = ' entity '+ex_text_a
        input = tokenizer(ex_text_a, max_length=args.max_seq_length, padding='max_length', truncation=True)

        labels = [0] * 9
        for l in example.label:
            l = label_map[l]
            labels[l] = 1



        features.append(
            InputFeatures(input_ids=input.input_ids,
                          input_mask=input.attention_mask,
                          segment_ids=input.token_type_ids,
                          h_pos=h_pos,
                          label_id=labels))
    # print(label_list)
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
        with open(input_file, "r") as f:
            return json.load(f)


class TypingProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        d = {}
        for e in examples:
            for l in e.label:
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
        for k, v in d.items():
            d[k] = (len(examples) - v) * 1. /v
        return examples, list(d.keys()), d

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
        d = {}
        for e in examples:
            for l in e.label:
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
        for k, v in d.items():
            d[k] = (len(examples) - v) * 1. / v
        return examples, list(d.keys()), d

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")
        d = {}
        for e in examples:
            for l in e.label:
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
        for k, v in d.items():
            d[k] = (len(examples) - v) * 1. / v
        return examples, list(d.keys()), d

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = (line['sent'], [["SPAN", line["start"], line["end"]]])
            text_b = line['ents']
            label = line['labels']
            #if guid != 51:
            #    continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def get_text_data(tag, args):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = TypingProcessor()
    if tag == 'train':
        train_examples, _, _ = processor.get_train_examples(args.data_dir)
    if tag == 'dev':
        train_examples, _, _ = processor.get_dev_examples(args.data_dir)
    if tag == 'test':
        train_examples, _, _ = processor.get_test_examples(args.data_dir)
    label_list = processor.get_train_examples(args.data_dir)[1]
    train_features = convert_examples_to_features(train_examples, label_list, tokenizer, args)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_h_pos = torch.tensor([f.h_pos for f in train_features], dtype=torch.long).unsqueeze(dim=1)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    text_data = [all_input_ids, all_input_mask, all_segment_ids, all_h_pos,
                       all_label_ids]
    return text_data


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
                        type=int,
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
    thre = args.threshold
    epoch = args.num_train_epochs
    seed = args.seed

    set_seed(seed)
    print(args)

    ent2id = load_json('./data/ent2id_.json')

    model = OpenEntityTrainModel()
    model.ent_embedding.load_state_dict(load_file('./param/graph_encoder'))

    ent_train = collect_ent('train', thre, ent2id, args)
    ent_eval = collect_ent('dev', thre, ent2id, args)
    ent_test = collect_ent('test', thre, ent2id, args)
    print(ent_train.size())
    all_ent = torch.cat([ent_train, ent_eval, ent_test])
    ent2graph = ent2line_graph(all_ent)

    data_train = get_text_data('train', args)
    data_eval = get_text_data('dev', args)
    data_test = get_text_data('test', args)

    for i in range(epoch):
        train(model, data_train, ent_train, ent2graph, args)
        eval_data(model, data_eval, ent_eval, ent2graph, args)
        eval_data(model, data_test, ent_test, ent2graph, args)


