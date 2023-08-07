import torch
import random
from transformers import BertTokenizerFast, BertForSequenceClassification
from util import load_json, get_rel2idx, get_ent2text, get_cur, sql_triple_with_rel, save_file, check_model_grad, load_file
from tqdm import tqdm
import evaluate
import lora
from torch.utils.data import TensorDataset, DataLoader


# tokens = ["When", "they", "reached", "adulthood", ",", "Pelias", "and", "Neleus", "found", "Tyro", "and", "killed", "her", "stepmother", ",", "Sidero", ",", "for", "having", "mistreated", "their", "mother", "(", "Salmoneus", "married", "Sidero", "when", "Alkidike", ",", "his", "wife", "and", "the", "mother", "of", "Tyro", ",", "died", ")."]
# subj_start = 9
# subj_end = 9
# obj_start = 5
# obj_end = 5
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# text = ' '.join(tokens)
# chunk1 = ' '.join(tokens[:obj_start])
# chunk2 = ' '.join(tokens[obj_end+1:subj_start])
# chunk3 = ' '.join(tokens[subj_end+1:])
# ent1 = 'Ouarzazate'
# ent2 = 'Tajikistan'
# print(tokenizer.tokenize(text))
#
# print(chunk1)
# print(chunk2)
# print(chunk3)
#
# print(tokenizer(chunk1))

def train_setting(model):
    for idx, (n, p) in enumerate(model.named_parameters()):
        if idx <= 180:
            p.requires_grad = False


def tokenize_with_ent(chunk1, ent1, chunk2, ent2, chunk3):
    ent1_start = len(tokenizer.tokenize(chunk1))
    ent1_end = ent1_start + len(tokenizer.tokenize(ent1))

    ent2_start = len(tokenizer.tokenize(' '.join([chunk1, ent1, chunk2])))
    ent2_end = ent2_start + len(tokenizer.tokenize(ent2))

    full_text = ' '.join([chunk1, ent1, chunk2, ent2, chunk3])

    output = tokenizer(full_text, max_length=128,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
    input_ids = output.input_ids
    token_type_ids = output.token_type_ids
    attention_mask = output.attention_mask
    ent_pos = torch.zeros_like(input_ids)
    # [CLS] ...
    # ent_pos[ent1_start+1:ent1_end] = 1
    # ent_pos[ent2_start+1:ent2_end] = 1

    # print(torch.sum(token_type_ids))
    token_type_ids[0, ent1_start + 1:ent1_end + 1] = 1
    token_type_ids[0, ent2_start + 1:ent2_end + 1] = 1
    # print(torch.sum(token_type_ids))


    return input_ids, token_type_ids, attention_mask


def get_template():
    return load_json('../data/fewrel/chunk/dev.json')


def get_train_triple_from_db():
    cur = get_cur()
    labels = collect_rel()
    all_triples = []
    for l in tqdm(labels, desc='rel sql'):
        all_triples += sql_triple_with_rel(cur, l)
    return all_triples


def collect_rel():
    all_rel = []
    data = load_json(f'../data/fewrel/train.json')
    for item in data:
        rel = item['label']
        all_rel.append(rel)
    return list(set(all_rel))


def convert_triple_to_trainable_data(triple, all_template):
    rel2idx = get_rel2idx()
    ent2text = get_ent2text()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_labels = []

    for idx, triple in enumerate(tqdm(list(triple)[::2])):
        h, r, t = triple
        h = ent2text.get(h, [])
        t = ent2text.get(t, [])
        if len(h) == 0 or len(t) == 0:
            continue
        else:
            h = h[0]
            t = t[0]

        if r not in rel2idx.keys():
            # print(r)
            continue

        template = random.choice(all_template)
        input_ids, token_type_ids, attention_mask = \
            tokenize_with_ent(**template, ent1=' # '+h+' # ', ent2=' $ '+t+' $ ')
        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(rel2idx[r])

        input_ids, token_type_ids, attention_mask = \
            tokenize_with_ent(**template, ent2=' # ' + h + ' # ', ent1=' $ ' + t + ' $ ')
        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(rel2idx[r])

    data_save = [torch.cat(all_input_ids),
                 torch.cat(all_token_type_ids),
                 torch.cat(all_attention_mask),
                 torch.tensor(all_labels)]

    print(len(input_ids))
    return data_save


# 将input_id转化为embedding
def convert_arg_format(input_ids, attention_mask, token_type_ids, labels, token_embedding, know_embedding):
    inputs_embeds = token_embedding(input_ids)
    know_sel = input_ids.clone().detach()
    know_sel[token_type_ids != 1] = 0
    inputs_embeds = inputs_embeds + know_embedding(know_sel)
    token_type_ids = torch.zeros_like(token_type_ids)
    out = {'inputs_embeds': inputs_embeds,
           'attention_mask': attention_mask,
           'token_type_ids': token_type_ids,
           'labels': labels}
    return out


def train(model, train_data):
    device = 'cuda'
    model = model.to(device)
    # config = model.config
    # knowledge_embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size,
    #                                                          padding_idx=config.pad_token_id).to(device)
    # knowledge_embedding.weight = torch.nn.Parameter(torch.zeros_like(knowledge_embedding.weight))
    # token_embedding = model.bert.get_input_embeddings()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
    # optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': knowledge_embedding.parameters()}], lr=3e-3)
    loss_print = 0
    ground_truth = []
    prediction = []
    accuracy_metric = evaluate.load("accuracy")
    divide = 1


    model.train()
    for epc in range(divide):
        dataset = TensorDataset(*train_data)
        accumulation_steps = 1
        load = DataLoader(dataset, batch_size=32, shuffle=True)
        for idx, item in enumerate(tqdm(load)):
            item = [i.to(device) for i in item]
            input_ids, token_type_ids, attention_mask, label = item
            # out = convert_arg_format(input_ids, attention_mask, token_type_ids, label, token_embedding, knowledge_embedding)
            res = model(input_ids, attention_mask, token_type_ids, label)
            # res = model(input_ids, attention_mask, token_type_ids, labels=label)
            # res = model(**out)
            loss = res.loss / accumulation_steps
            loss_print += loss

            pred = res.logits
            prediction.append(torch.argmax(pred, dim=1))
            ground_truth.append(label)
            loss.backward()

            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % 128 == 0:
                ground_truth = torch.cat(ground_truth)
                prediction = torch.cat(prediction)
                results = accuracy_metric.compute(references=ground_truth, predictions=prediction)
                print(results)
                print(f'loss:{loss_print}')
                print(torch.sum(torch.abs(model.know_embedding.weight)))
                prediction = []
                ground_truth = []
                loss_print = 0
                # break
        save_file(model.know_embedding.weight, f'../param/embedding/{epc}')


class TrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="single_label_classification",
                                                          num_labels=825)
        config = self.model.config
        self.token_embedding = self.model.bert.get_input_embeddings()
        self.know_embedding = torch.nn.Embedding.from_pretrained(torch.zeros(config.vocab_size, config.hidden_size), padding_idx=config.pad_token_id)
        self.know_embedding.weight = torch.nn.Parameter(torch.zeros_like(self.know_embedding.weight))

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        inputs_embeds = self.token_embedding(input_ids)
        know_sel = input_ids.clone().detach()
        know_sel[token_type_ids != 1] = 0

        inputs_embeds = inputs_embeds + self.know_embedding(know_sel)
        token_type_ids = torch.zeros_like(token_type_ids)
        out = {'inputs_embeds': inputs_embeds,
               'attention_mask': attention_mask,
               'token_type_ids': token_type_ids,
               'labels': labels}
        res = self.model(**out)
        return res

if __name__ == '__main__':
    triple = get_train_triple_from_db()
    template = get_template()
    data = convert_triple_to_trainable_data(triple, template)
    model = TrainModel()
    train_setting(model)
    check_model_grad(model)
    train(model, data)
