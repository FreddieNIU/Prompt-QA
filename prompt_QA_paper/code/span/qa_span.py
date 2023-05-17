import json
import collections

import torch
from torch.utils import data
from transformers import AutoModelForQuestionAnswering

from config import *
from torch.utils.data import DataLoader
from transformers import AdamW
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm

no_deprecation_warning=True

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def read_dataset(path, have_template=True):
    contexts = []
    questions = []
    answers = []

    qa_dataset = json.load(open(path, 'r', encoding='utf8'))
    for qa in qa_dataset:
        if have_template:
            template = qa['template']
            context = qa['context'] + str([template[i] for i in range(len(template))])
        else:
            context = qa['context']
        question = qa['question']
        answer_start = qa['answer'][0]
        answer_end = qa['answer'][1]
        answer_text = context[answer_start: answer_end]
        answer = {'text': answer_text,
                  'answer_start': answer_start,
                  'answer_end': answer_end}

        contexts.append(context)
        questions.append(question)
        answers.append(answer)

    return contexts, questions, answers


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def train():
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

    optim = AdamW(model.parameters(), lr=config.lr)

    cur_best_f1 = 0
    batch_index = 0

    for epoch in range(config.num_epoch):
        for batch in tqdm(train_loader):
            batch_index += 1

            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            start_positions = batch['start_positions'].to(config.device)
            end_positions = batch['end_positions'].to(config.device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            

            loss = outputs[0]
            loss.backward()

            if batch_index % config.batch_accum == 0:
                optim.step()
                optim.zero_grad()
                batch_index = 0

            torch.cuda.empty_cache()

        dev_f1, dev_precision, dev_recall, dev_exact_match = test(model=model, config=config, tokenizer=tokenizer,
            test_loader=DataLoader(dev_dataset, batch_size=config.test_batch_size, shuffle=False))
        print('dev f1:', dev_f1, 'precision:', dev_precision, 'recall:', dev_recall, 'exact_match:', dev_exact_match)
        if dev_f1 > cur_best_f1:
            cur_best_f1 = dev_f1
            torch.save(model, config.model_save_path)
    return cur_best_f1


def compute_f1(gold_toks, pred_toks):
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks), 0, 0

    if num_same == 0:
        return 0, 0, 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def test(test_loader, model, config, tokenizer):
    model.eval()
    with torch.no_grad():
        f1_list = []
        precision_list = []
        recall_list = []
        exact_match_list = []

        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            start_positions = batch['start_positions'].to(config.device)
            end_positions = batch['end_positions'].to(config.device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            pred_start_idx = outputs['start_logits'].argmax(dim=1)
            pred_end_idx = outputs['end_logits'].argmax(dim=1)
            for pred_start, pred_end, gold_start, gold_end, input_id_list in zip(pred_start_idx.tolist(),
                                                                                 pred_end_idx.tolist(),
                                                                                 start_positions.tolist(),
                                                                                 end_positions.tolist(),
                                                                                 input_ids.tolist()):
                if pred_start < pred_end:
                    pred_answer_ids = input_id_list[pred_start: pred_end]
                    pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(pred_answer_ids))
                    gold_answer_ids = input_id_list[gold_start: gold_end]
                    gold_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(gold_answer_ids))

                    pred_toks = []
                    [pred_toks.append(toks) for toks in word_tokenize(pred_answer)]
                    gold_toks = []
                    [gold_toks.append(toks) for toks in word_tokenize(gold_answer)]
                    f1, precision, recall = compute_f1(gold_toks, pred_toks)
                    exact_match = 1 if gold_answer == pred_answer else 0
                else:
                    f1, precision, recall, exact_match = 0, 0, 0, 0
                f1_list.append(f1)
                precision_list.append(precision)
                recall_list.append(recall)
                exact_match_list.append(exact_match)

        f1_test = np.mean(f1_list)
        precision_test = np.mean(precision_list)
        recall_test = np.mean(recall_list)
        exact_match_test = np.mean(exact_match_list)
        return f1_test, precision_test, recall_test, exact_match_test


if __name__ == '__main__':
    configs = [
                BertConfig(), 
               # BertLargeConfig(),
               # RobertaConfig(), 
               # RobertaLargeConfig(),
               # RobertaSQuADConfig(), 
               # RobertaSQuADLargeConfig(),

               # SpanBertConfig(),
              ]

    for config in configs:
        print('model:', config.model)
        print('train data: ', config.train_data_path)
        print('dev data: ', config.dev_data_path)
        print('test data: ', config.test_data_path)
        
        train_contexts, train_questions, train_answers = read_dataset(config.train_data_path, have_template=config.have_template)
        dev_contexts, dev_questions, dev_answers = read_dataset(config.dev_data_path, have_template=config.have_template)
        test_contexts, test_questions, test_answers = read_dataset(config.test_data_path, have_template=config.have_template)

        tokenizer = config.tokenizer

        # output of tokenizer:
        # { 'input_ids': [[], [], []...], 'token_type_ids':, [[], [], []...], 'attention_mask': [[], [], []...] }

        train_encodings = tokenizer(train_questions, train_contexts, truncation=True, padding=True)
        dev_encodings = tokenizer(dev_questions, dev_contexts, truncation=True, padding=True)
        test_encodings = tokenizer(test_questions, test_contexts, truncation=True, padding=True)

        add_token_positions(train_encodings, train_answers)
        add_token_positions(dev_encodings, dev_answers)
        add_token_positions(test_encodings, test_answers)

        train_dataset = SquadDataset(train_encodings)
        dev_dataset = SquadDataset(dev_encodings)
        test_dataset = SquadDataset(test_encodings)

        c = 0
        for i in train_dataset:
            print(i['start_positions'])
            c += 1
            if c > 20:
                break

        # # training
        # model = AutoModelForQuestionAnswering.from_pretrained(config.model)
        # model.to(config.device)
        # print('training')
        # best_dev_f1 = train()

        # # test
        # print('testing')
        # model = torch.load(config.model_save_path)

        # train_f1, train_precision, train_recall, train_exact_match = test(model=model, config=config, tokenizer=tokenizer,
        #     test_loader=DataLoader(train_dataset, batch_size=config.test_batch_size, shuffle=False))
        # print('train f1:', train_f1, 'precision:', train_precision, 'recall:', train_recall, 'exact_match:', train_exact_match)

        # dev_f1, dev_precision, dev_recall, dev_exact_match = test(model=model, config=config, tokenizer=tokenizer,
        #     test_loader=DataLoader(dev_dataset, batch_size=config.test_batch_size, shuffle=False))
        # print('dev f1:', dev_f1, 'precision:', dev_precision, 'recall:', dev_recall, 'exact_match:', dev_exact_match)

        # test_f1, test_precision, test_recall, test_exact_match = test(model=model, config=config, tokenizer=tokenizer,
        #     test_loader=DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False))
        # print('test f1:', test_f1, 'precision:', test_precision, 'recall:', test_recall, 'exact_match:', test_exact_match)
