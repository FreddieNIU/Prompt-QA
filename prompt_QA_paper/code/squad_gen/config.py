import torch
from transformers import BartTokenizer, T5Tokenizer
import transformers
transformers.logging.set_verbosity_error()
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class BartConfig(object):

    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/'
    have_template = False
    
    train_data_path = base_path + 'squad_train.json'
    dev_data_path = prompt_folder + 'data/NewsQA/qtype_newsqa/newsqa_dev.json'
    # test_data_path = base_path + 'squad_test.json'
    test_data_path = prompt_folder + 'data/NewsQA/qtype_newsqa/newsqa_test.json'
    # test_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_test.json'
    # test_data_path = prompt_folder + 'data/SQuAD_V2/reformulatedSQuAD/squad_test_OOD.json'


    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'facebook/bart-base'
    model_name = 'bart-base'
    tokenizer = BartTokenizer.from_pretrained(model)

    lr = 1e-5
    train_batch_size = 8
    batch_accum = 4
    test_batch_size = 32
    num_epoch = 3
    

    model_save_path = prompt_folder + 'models/domain_adaptation/'+ model_name +'_squad_news'+'.ckp'

class BartConfig_2(object):

    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/'
    have_template = False
    
    train_data_path = base_path + 'squad_train.json'
    dev_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_dev.json'
    # test_data_path = base_path + 'squad_test.json'
    test_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_test.json'
    # test_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_test.json'
    # test_data_path = prompt_folder + 'data/SQuAD_V2/reformulatedSQuAD/squad_test_OOD.json'


    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'facebook/bart-base'
    model_name = 'bart-base'
    tokenizer = BartTokenizer.from_pretrained(model)

    lr = 1e-5
    train_batch_size = 8
    batch_accum = 4
    test_batch_size = 32
    num_epoch = 3
    

    model_save_path = prompt_folder + 'models/domain_adaptation/'+ model_name +'_squad_causal'+'.ckp'

class BartPromptConfig(object):

    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/NewsQA/qtype_newsqa/'
    have_template = True
    
    train_data_path = base_path + 'newsqa_train.json'
    dev_data_path = base_path + 'newsqa_dev.json'
    # test_data_path = base_path + 'newsqa_test.json'
    # test_data_path = prompt_folder + 'data/NewsQA/processed_Qtype_newsQA.json'
    test_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_test.json'
    # test_data_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/squad_test_OOD.json'

    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'facebook/bart-base'
    model_name = 'bart-base'
    tokenizer = BartTokenizer.from_pretrained(model)

    lr = 1e-5
    train_batch_size = 8
    batch_accum = 4
    test_batch_size = 32
    num_epoch = 5
    

    model_save_path = prompt_folder + 'models/qtype_newsqa/'+ model_name +'_'+ str(lr) +'_LPFT'+'.ckp'

class BartLargeConfig(object):
    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/SQuAD_V2/reformulatedSQuAD/'
    have_template = False
    
    train_data_path = base_path + 'squad_train.json'
    dev_data_path = base_path + 'squad_dev.json'
    test_data_path = base_path + 'squad_test_OOD.json'

    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'facebook/bart-large'
    model_name = 'bart-large'
    tokenizer = BartTokenizer.from_pretrained(model)
    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 4
    
    model_save_path = prompt_folder + 'models/squad/'+ model_name +'_'+ str(lr) +'_'+'.ckp'

class BartLargePromptConfig(object):
    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/'
    have_template = False
    
    train_data_path = base_path + 'squad_train.json'
    dev_data_path = base_path + 'squad_dev.json'
    test_data_path = base_path + 'squad_test_OOD.json'

    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'facebook/bart-large'
    model_name = 'bart-large'
    tokenizer = BartTokenizer.from_pretrained(model)
    lr = 1e-5
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 3
    
    model_save_path = prompt_folder + 'models/qtype_squad/'+ model_name +'_'+ str(lr) +'_'+'.ckp'

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

