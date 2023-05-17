import torch
from transformers import BartTokenizer, T5Tokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class BartConfig(object):
    # base_path = '/home/yingjie/Year_1/qa/dataset/'
    base_path = 'dataset_with_sentimentWord_and_Qtype/'
    source_path = base_path
    train_data_path = base_path + 'qa_dataset_train.json'
    dev_data_path = base_path + 'qa_dataset_dev.json'
    test_data_path = base_path + 'qa_dataset_test.json'
    
    source_domain = 'all'
    target_domain = 'all'
    source_template = 'sentimentNqtype'
    target_template = 'sentimentNqtype'
    
#     source_base_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(source_template)
#     target_bath_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(target_template)

#     train_data_path = source_base_path + 'qa_{}_train.json'.format(source_domain)
#     dev_data_path = source_base_path + 'qa_{}_dev.json'.format(source_domain)
#     test_data_path = target_bath_path + 'qa_{}_test.json'.format(target_domain)


    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'facebook/bart-base'
    model_name = 'bart-base'
    tokenizer = BartTokenizer.from_pretrained(model)
    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 14
    

    model_save_path = './ckp/all_dataset_LPFT_with_template/'+ str(num_epoch) + 'epoch_LPFT_'+ model_name + '_02.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) +str(num_epoch) + 'epoch' + model_name + '.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_LP_' + model_name + '.ckp'



class BartLargeConfig(object):
    # base_path = 'dataset_with_sentimentWord_and_Qtype/'
    # source_path = base_path
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    # industrials, consumer, technology
    source_domain = 'industrials'
    target_domain = 'industrials'
    source_template = 'phrase'
    target_template = 'phrase'
    
    source_base_path = 'Domain_without_duplicate_dataset/{}_template/'.format(source_template)
    target_bath_path = 'Domain_without_duplicate_dataset/{}_template/'.format(target_template)

    train_data_path = source_base_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = source_base_path + 'qa_{}_dev.json'.format(source_domain)
    test_data_path = target_bath_path + 'qa_{}_test.json'.format(target_domain)


    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'facebook/bart-large'
    model_name = 'bart-large'
    tokenizer = BartTokenizer.from_pretrained(model)
    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 14
    
    model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) +str(num_epoch) + 'epoch_LPFT_' + model_name + '.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) +str(num_epoch) + 'epoch_' + model + '.ckp'


class T5SmallConfig(object):
    # base_path = '../../dataset/qa/'
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    source_domain = 'technology'
    target_domain = 'industrials'
    source_template = 'Qtype'
    target_template = 'phrase'
    
    source_base_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(source_template)
    target_bath_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(target_template)

    train_data_path = source_base_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = source_base_path + 'qa_{}_dev.json'.format(source_domain)
    test_data_path = target_bath_path + 'qa_{}_test.json'.format(target_domain)


    summary_writer_path = 'SummaryWriterPath/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model)
    lr = 1e-7
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 4

    model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) +str(num_epoch) + 'epoch' + model + '.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_LP_' + model + '.ckp'



class T5Config(object):
    # base_path = '../../dataset/qa/'
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    source_domain = 'consumer'
    target_domain = 'consumer'
    source_template = 'Qtype'
    target_template = 'Qtype'
    
    source_base_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(source_template)
    target_bath_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(target_template)

    train_data_path = source_base_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = source_base_path + 'qa_{}_dev.json'.format(source_domain)
    test_data_path = target_bath_path + 'qa_{}_test.json'.format(target_domain)


    summary_writer_path = 'SummaryWriterPath/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model)
    lr = 1e-7
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 7

    # model_save_path = './ckp/' + model + '.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) +str(num_epoch) + 'epoch_' + model + '.ckp'
    model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_LPFT_' + model + '.ckp'



class T5LargeConfig(object):
    
    # base_path = '../../dataset/qa/'
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    source_domain = 'consumer'
    target_domain = 'consumer'
    source_template = 'Qtype'
    target_template = 'Qtype'
    
    source_base_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(source_template)
    target_bath_path = '/home/yingjie/Year_1/template_QA/Domain_without_duplicate_dataset/{}_template/'.format(target_template)

    train_data_path = source_base_path + 'qa_{}_train.json'.format(source_domain)
    dev_data_path = source_base_path + 'qa_{}_dev.json'.format(source_domain)
    test_data_path = target_bath_path + 'qa_{}_test.json'.format(target_domain)


    summary_writer_path = 'SummaryWriterPath/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 't5-large'
    tokenizer = T5Tokenizer.from_pretrained(model)
    lr = 1e-7
    train_batch_size = 1
    batch_accum = 32
    test_batch_size = 32
    num_epoch = 7

    # model_save_path = './ckp/' + model + '.ckp'
    # model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) +str(num_epoch) + 'epoch_' + model + '.ckp'
    model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_LPFT_' + model + '.ckp'

