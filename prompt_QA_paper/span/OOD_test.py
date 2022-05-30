import collections
import json

import numpy as np
from nltk.tokenize import word_tokenize
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from config import *
from qa_span import *


configs = [
               # BertConfig(), 
               # BertLargeConfig(),
               RobertaConfig(), 
               # RobertaLargeConfig(),
               # SpanBertConfig(),
              ]

for config in configs:
    # config = T5SmallConfig()

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


    print('model:', config.model_save_path)
    print('train data: ', config.train_data_path)
    print('dev data: ', config.dev_data_path)
    print('test data: ', config.test_data_path)

    print('testing')
    model = torch.load(config.model_save_path)

    # train_f1, train_precision, train_recall, train_exact_match = test(model=model,  config=config, tokenizer=tokenizer,
    #     test_loader=DataLoader(train_dataset, batch_size=config.test_batch_size, shuffle=False))
    # print('train f1:', train_f1, 'precision:', train_precision, 'recall:', train_recall, 'exact_match:', train_exact_match)

    # dev_f1, dev_precision, dev_recall, dev_exact_match = test(model=model,  config=config, tokenizer=tokenizer,
    #     test_loader=DataLoader(dev_dataset, batch_size=config.test_batch_size, shuffle=False))
    # print('dev f1:', dev_f1, 'precision:', dev_precision, 'recall:', dev_recall, 'exact_match:', dev_exact_match)

    test_f1, test_precision, test_recall, test_exact_match = test(model=model, config=config, tokenizer=tokenizer,
        test_loader=DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False))
    print('test f1:', test_f1, 'precision:', test_precision, 'recall:', test_recall, 'exact_match:', test_exact_match)