import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, BartTokenizerFast, AutoTokenizer
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    

class BertConfig(object):
    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/'
    have_template = True
    
    train_data_path = base_path + 'squad_train.json'
    dev_data_path = base_path + 'squad_dev.json'
    test_data_path = base_path + 'squad_test_OOD.json'
    
    summary_writer_path = 'SummaryWriterPath/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'bert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    lr = 1e-5
    train_batch_size = 24
    batch_accum = 2
    test_batch_size = 32
    num_epoch = 4
    
    
    model_save_path = prompt_folder + 'models/qtype_squad/'+ model +'_'+ str(lr) +'_top10'+ '.ckp'



class BertLargeConfig(object):
    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'

    base_path = prompt_folder + 'data/SQuAD_V2/reformulatedSQuAD/'
    train_data_path = base_path + 'squad_dev.json'
    dev_data_path = base_path + 'squad_test_ID.json'
    test_data_path = base_path + 'squad_test_ID.json'
    have_template = False

    summary_writer_path = 'SummaryWriterPath/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    lr = 1e-5
    train_batch_size = 8
    batch_accum = 8
    test_batch_size = 32
    num_epoch = 7

    model_save_path = prompt_folder + 'models/squad/'+ model + '.ckp'


class RobertaConfig(object):
    
    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/NewsQA/qtype_newsqa/'
    have_template = False
    
    train_data_path = base_path + 'newsqa_train.json'
    dev_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_dev.json'
    # test_data_path = base_path + 'newsqa_test.json'
    # test_data_path = prompt_folder + 'data/NewsQA/qtype_newsqa/newsqa_test.json'
    test_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_test.json'
    # test_data_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/squad_test_OOD.json'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'roberta-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)
    
    summary_writer_path = '../template_QA/SummaryWriterPath/'

    lr = 1e-5
    train_batch_size =24
    batch_accum = 2
    test_batch_size = 32
    num_epoch = 3

    model_save_path = prompt_folder + 'models/domain_adaptation/'+ model +'_news_causal'+'.ckp'

    
class RobertaPromptConfig(object):
    
    prompt_folder = '/home/yingjie_niu/Year_1/PromptQA/'
    base_path = prompt_folder + 'data/NewsQA/qtype_newsqa/'
    have_template = False
    
    train_data_path = base_path + 'newsqa_train.json'
    dev_data_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/squad_dev.json'
    # test_data_path = base_path + 'newsqa_test.json'
    # test_data_path = prompt_folder + 'data/NewsQA/qtype_newsqa/newsqa_test.json'
    # test_data_path = prompt_folder + 'data/CausalQA/Qtype_causal/causal_test.json'
    test_data_path = prompt_folder + 'data/SQuAD_V2/Qtype_SQUAD/squad_test_OOD.json'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'roberta-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)
    
    summary_writer_path = '../template_QA/SummaryWriterPath/'

    lr = 1e-5
    train_batch_size =24
    batch_accum = 2
    test_batch_size = 32
    num_epoch = 3

    model_save_path = prompt_folder + 'models/domain_adaptation/'+ model +'_news_squad'+'.ckp'

    

class RobertaLargeConfig(object):
   
    
    # base_path = 'dataset_with_sentimentWord_and_Qtype/'
    # source_path = base_path
    # train_data_path = base_path + 'qa_dataset_train.json'
    # dev_data_path = base_path + 'qa_dataset_dev.json'
    # test_data_path = base_path + 'qa_dataset_test.json'
    
    # industrials, consumer, technology
    source_domain = 'technology'
    target_domain = 'technology'
    # "Qtype", "phrase", "sentiment", "entity" ,"no"
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
    model_save_path = './ckp/{}_trained_with_{}_template/'.format(source_domain, source_template) + str(num_epoch) + 'epoch_' + model + '.ckp'

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

