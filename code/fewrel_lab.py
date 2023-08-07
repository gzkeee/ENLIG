from util import load_file, save_file, check_model_grad
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig
import evaluate
import lora


def get_inset_lora_model(r):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          problem_type="single_label_classification",
                                                          num_labels=80)
    save_file(model.state_dict(), '../param/pretrain_80')
    encoder = model.bert.encoder
    layers = encoder.layer
    for lay in layers:
        d = lay.output.dense
        lay.output.dense = lora.Linear(d.in_features, d.out_features, r)

    model.load_state_dict(load_file('../param/pretrain_80'), strict=False)
    return model


def get_inset_path_select_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          problem_type="single_label_classification",
                                                          num_labels=80)
    save_file(model.state_dict(), '../param/pretrain_80')
    encoder = model.bert.encoder
    layers = encoder.layer
    for lay in layers:
        lay.output = lora.BertOutput(BertConfig())
        lay.attention.output = lora.BertSelfOutput(BertConfig())

    model.load_state_dict(load_file('../param/pretrain_80'), strict=False)
    return model


# 测试通路搭配的影响




def get_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          problem_type="single_label_classification",
                                                          num_labels=80)

    # embedding = model.bert.get_input_embeddings()
    # model.bert.set_input_embeddings(lora.Embedding(embedding.num_embeddings, embedding.embedding_dim, 1))
    # model.bert.get_input_embeddings().load_state_dict(load_file('../param/knowledge_embedding_0'))
    return model


# fewrel测评任务
# 输入一个模型 测试该模型在fewrel任务上的结果
def get_train_data():
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = load_file('../data/fewrel/train')
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return train_data


def get_eval_data():
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = load_file('../data/fewrel/eval')
    eval_data = TensorDataset(all_input_ids[::8], all_input_mask[::8], all_segment_ids[::8], all_label_ids[::8])
    return eval_data


def train(model, data):
    device = 'cuda'
    bsz = 32
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    load = DataLoader(data, batch_size=bsz, shuffle=True)
    model.train()
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, attention_mask, token_type_ids, label = item
        res = model(input_ids, attention_mask, token_type_ids, labels=label)
        loss = res.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if (idx+1) % 240 == 0:
        #     for n, p in model.named_parameters():
        #         if 'path_' in n:
        #             print(p)


def eval_data(model, data):
    device = 'cuda'
    bsz = 16
    model = model.to(device)
    accuracy_metric = evaluate.load("accuracy")

    load = DataLoader(data, batch_size=bsz)
    ground_truth = []
    prediction = []
    # model.eval()
    for idx, item in enumerate(tqdm(load)):
        item = [i.to(device) for i in item]
        input_ids, attention_mask, token_type_ids, label = item
        res = model(input_ids, attention_mask, token_type_ids, labels=label)

        predict = res.logits.argmax(-1)
        ground_truth += label.cpu().numpy().tolist()
        prediction += predict.cpu().numpy().tolist()

    result = accuracy_metric.compute(references=ground_truth, predictions=prediction)
    print(result)


# 添加实体标记
def transfer_data(all_input_ids, all_segment_ids):
    pass


if __name__ == '__main__':
    model = get_inset_path_select_model()
    # model = get_model()
    lora.mark_only_path_as_trainable(model.bert)
    check_model_grad(model)
    # lora.mark_only_lora_as_trainable(model.bert)
    data_train = get_train_data()
    data_eval = get_eval_data()
    epoch = 15
    for i in range(epoch):
        train(model, data_train)
        eval_data(model, data_eval)
