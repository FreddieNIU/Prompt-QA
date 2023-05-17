import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, BartTokenizerFast, AutoTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class BertConfig(object):
    # base_path = '../../dataset/qa/'
    # base_path = 'dataset_with_sentimentWord_and_Qtype/'
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    source_domain = 'consumer'
    target_domain = 'consumer'
    template = 'Qtype'
    
    base_path = 'Domain_without_duplicate_dataset/{}_template/'.format(template)
    # base_path = '/home/yingjie/Year_1/qa/Domain_without_duplicate_dataset/without_template/'


    train_data_path = base_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = base_path + 'qa_{}_dev.json'.format(source_domain)
    # test_data_path = base_path + 'qa_{}_test.json'.format(target_domain)
    test_data_path='Domain_without_duplicate_dataset/phrase_template/qa_{}_test.json'.format(target_domain)


    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'bert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 4
    

    # model_save_path = './ckp/' + model_name + '.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, template) + str(num_epoch) + 'epoch_LP_' + model_name + '.ckp'
    model_save_path = './ckp/' +base_path+ model + '.ckp'



class BertLargeConfig(object):
    # base_path = '../../dataset/qa/'
    base_path = '../dataset/'
    train_data_path = base_path + 'qa_dataset_train.json'
    dev_data_path = base_path + 'qa_dataset_dev.json'
    test_data_path = base_path + 'qa_dataset_test.json'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'bert-large-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 7

    model_save_path = './ckp/' + model + '.ckp'


class RobertaConfig(object):
    
    # base_path = '/home/yingjie/Year_1/qa/dataset/'
    # base_path = 'dataset_with_sentimentWord_and_Qtype/'
    # source_path = base_path
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    source_domain = 'technology'
    target_domain = 'technology'
    source_template = 'Qtype'
    target_template = 'Qtype'
    
    source_path = 'Domain_without_duplicate_dataset/{}_template/'.format(source_template)
    target_path = 'Domain_without_duplicate_dataset/{}_template/'.format(target_template)
    
    train_data_path = source_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = source_path + 'qa_{}_dev.json'.format(source_domain)
    test_data_path = target_path + 'qa_{}_test.json'.format(target_domain)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'roberta-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)
    
    summary_writer_path = '../template_QA/SummaryWriterPath/'

    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 3

    model_save_path = './ckp/all_dataset_LPFT_with_template/'+ str(num_epoch) + 'epoch_LPFT_'+ model + '.ckp'
    
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_' + model + '.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_LPFT_' + model + '.ckp'



    

class RobertaLargeConfig(object):
   
    
    # base_path = 'dataset_with_sentimentWord_and_Qtype/'
    # source_path = base_path
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    # industrials, consumer, technology
    source_domain = 'technology'
    target_domain = 'technology'
    source_template = 'Qtype'
    target_template = 'Qtype'
    
    source_path = 'Domain_without_duplicate_dataset/{}_template/'.format(source_template)
    target_path = 'Domain_without_duplicate_dataset/{}_template/'.format(target_template)
    
    train_data_path = source_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = source_path + 'qa_{}_dev.json'.format(source_domain)
    test_data_path = target_path + 'qa_{}_test.json'.format(target_domain)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'roberta-large'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)
    
    summary_writer_path = '../template_QA/SummaryWriterPath/'

    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 3

    # model_save_path = './ckp/all_dataset_LPFT_with_template/'+ str(num_epoch) + 'epoch_LPFT_'+ model + '.ckp'
    

class SpanBertConfig(object):
    
    source_domain = 'consumer'
    target_domain = 'consumer'
    source_template = 'no'
    target_template = 'no'
    source_path = 'Domain_without_duplicate_dataset/{}_template/'.format(source_template)
    target_path = 'Domain_without_duplicate_dataset/{}_template/'.format(target_template)
    
    # base_path = '/home/yingjie/Year_1/qa/Domain_without_duplicate_dataset/without_template/'
    train_data_path = source_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = source_path + 'qa_{}_dev.json'.format(source_domain)
    test_data_path = target_path + 'qa_{}_test.json'.format(target_domain)


    
    # base_path = 'dataset_with_sentimentWord_and_Qtype/'
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'mrm8488/spanbert-finetuned-squadv1'
    model_name = "spanbert-finetuned-squadv1"
    tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=512)
    
    summary_writer_path = '../template_QA/SummaryWriterPath/'


    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 7

    # model_save_path = './ckp/' +source_path+ model_name + '.ckp'
    model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_LP_' + model_name + '.ckp'

