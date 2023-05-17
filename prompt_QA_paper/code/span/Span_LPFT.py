import json
import collections

import torch
from torch.utils import data
from transformers import AutoModelForQuestionAnswering


from torch.utils.data import DataLoader
from transformers import AdamW
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm

from MyRoberta import *
from config import *

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def read_dataset(path):
    contexts = []
    questions = []
    answers = []

    qa_dataset = json.load(open(path, 'r', encoding='utf8'))
    for qa in qa_dataset:
        template = qa['template']
        context = qa['context'] + str([template[i] for i in range(len(template))])
        # context = qa['context']
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


def train(model,lr, num_epoch=1, SAVE=True):
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

    optim = AdamW(model.parameters(), lr=lr)

    cur_best_f1 = 0
    batch_index = 0

    for epoch in range(num_epoch):
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

        dev_f1, dev_precision, dev_recall, dev_exact_match = test(model,
            test_loader=DataLoader(dev_dataset, batch_size=config.test_batch_size, shuffle=False))
        print('dev f1:', dev_f1, 'precision:', dev_precision, 'recall:', dev_recall, 'exact_match:', dev_exact_match)
        if dev_f1 > cur_best_f1:
            cur_best_f1 = dev_f1
            if SAVE:
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


def test(model, test_loader):
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
               # BertConfig(), 
               # BertLargeConfig(),
               # RobertaConfig(), 
               RobertaLargeConfig(),
               # SpanBertConfig(),
              ]

    for config in configs:
        print('model:', config.model)
        # print('data path:',config.base_path)
        print('train data: ', config.train_data_path)
        print('dev data: ', config.dev_data_path)
        print('test data: ', config.test_data_path)
        
        train_contexts, train_questions, train_answers = read_dataset(config.train_data_path)
        dev_contexts, dev_questions, dev_answers = read_dataset(config.dev_data_path)
        test_contexts, test_questions, test_answers = read_dataset(config.test_data_path)

        tokenizer = config.tokenizer

        # output of tokenizer:
        # { 'input_ids': [[], [], []...], 'token_type_ids':, [[], [], []...], 'attention_mask': [[], [], []...] }

        train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
        dev_encodings = tokenizer(dev_contexts, dev_questions, truncation=True, padding=True)
        test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)

        add_token_positions(train_encodings, train_answers)
        add_token_positions(dev_encodings, dev_answers)
        add_token_positions(test_encodings, test_answers)

        train_dataset = SquadDataset(train_encodings)
        dev_dataset = SquadDataset(dev_encodings)
        test_dataset = SquadDataset(test_encodings)

        # training
        if "roberta" in config.model:
            LP_model = MyRobertaForQuestionAnswering_LP.from_pretrained(config.model)
        else:
            LP_model = MyRobertaForQuestionAnswering_LP.from_pretrained(config.model)
        LP_model.to(config.device)
        print('training')
        
        # record previous parameters
        previous_Roberta = list(LP_model.named_parameters())[:-2]
        previous_Roberta_tensors = []
        for i in range(len(previous_Roberta)):
            previous_Roberta_tensors.append(previous_Roberta[i][1].clone())
        previous_head = list(LP_model.qa_outputs.named_parameters())[0][1].clone()
        
        # LP fine tune
        best_dev_f1 = train(LP_model, num_epoch=3, SAVE=False, lr=1e-6)
        
        # LP-FT fine tune
        Flag = 0
        for i in range(len(previous_Roberta_tensors)):
            a = (previous_Roberta_tensors[i]== list(LP_model.named_parameters())[:-2][i][1])
            if False in a:
                Flag = 1
                print('Roberta updated')
                break
        if Flag == 0:
            print('Roberta not updated')
        a = (previous_head == list(LP_model.qa_outputs.named_parameters())[0][1])
        if False in a:
            print('Head updated')
        else:
            print('Head not updated')  
        
        # head_param_init = {}
        # for name, param in LP_model.qa_outputs.named_parameters():
        #     head_param_init[name] = param
        head_param_init =  LP_model.qa_outputs.state_dict()
        print("LP Head parameter: ", head_param_init)
        
        LPFT_model = MyRobertaForQuestionAnswering_LPFT.from_pretrained(config.model)
        
        print('LPFT head parameter: ', list(LPFT_model.qa_outputs.named_parameters()))
        # for name, param in LPFT_model.qa_outputs.named_parameters():
        #     param = head_param_init[name]
        LPFT_model.qa_outputs.load_state_dict(head_param_init)
        print('LPFT head initialized: ', list(LPFT_model.qa_outputs.named_parameters()))
        
        LPFT_model.to(config.device)
        best_dev_f1 = train(LPFT_model, num_epoch=config.num_epoch, SAVE=True, lr=config.lr)



        # test
        print('testing')
        model = torch.load(config.model_save_path)

        train_f1, train_precision, train_recall, train_exact_match = test(model,
            test_loader=DataLoader(train_dataset, batch_size=config.test_batch_size, shuffle=False))
        print('train f1:', train_f1, 'precision:', train_precision, 'recall:', train_recall, 'exact_match:', train_exact_match)

        dev_f1, dev_precision, dev_recall, dev_exact_match = test(model,
            test_loader=DataLoader(dev_dataset, batch_size=config.test_batch_size, shuffle=False))
        print('dev f1:', dev_f1, 'precision:', dev_precision, 'recall:', dev_recall, 'exact_match:', dev_exact_match)

        test_f1, test_precision, test_recall, test_exact_match = test(model,
            test_loader=DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False))
        print('test f1:', test_f1, 'precision:', test_precision, 'recall:', test_recall, 'exact_match:', test_exact_match)

