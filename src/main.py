from utils.tokenize import tokenize_bert
from utils.helper import set_seed
from utils.config_model import load_method

from train.train_cl import train

from data_process.read_data import read_data
from data_process.task_seq import get_subset
import numpy as np
import torch
from transformers import BertForMaskedLM
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--enc_type', type=str, default='bert-base-cased')
parser.add_argument('--bsize', type=int, default=32)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--sequence', type=str, nargs='+', default=['snli'])
parser.add_argument('--freeze_pretrain', action="store_true", default=False)#Whether to freeze the pre-trained model
parser.add_argument('--freeze_all', action="store_true", default=False)
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--plotrep', action="store_true", default=False)
parser.add_argument('--ratio_train', type=float, default=0.1)
parser.add_argument('--erace', action="store_true", default=False)
parser.add_argument('--er', action="store_true", default=False)
parser.add_argument('--agem', action="store_true", default=False)
parser.add_argument('--neigh_layer_min', type=int, default=0)
parser.add_argument('--neigh_layer_max', type=int, default=12)
parser.add_argument('--scale', type=float, default=0.1)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--pf', action="store_true", default=False) #Probing first strategy
parser.add_argument('--method', type=str, default='wirefix') #Select from type: wirefix, wireneigh, clora and origin



args = parser.parse_args()

if __name__ == '__main__':
    device = "cuda"
    #Parameters
    seed = args.seed
    encoder_type = args.enc_type 
    sequence = args.sequence   
    freeze_pretrain = args.freeze_pretrain
    freeze_all = args.freeze_all
    max_epoch = args.epoch
    plotrep = args.plotrep
    lr = args.lr
    ratio_train = args.ratio_train
    erace = args.erace
    er = args.er
    agem = args.agem
    neigh_layer_min = args.neigh_layer_min 
    neigh_layer_max = args.neigh_layer_max
    scale = args.scale
    k = args.k
    pf = args.pf
    method = args.method

    if 'large' in encoder_type:
        hidden = 1024
        hidden_latent = 1024
        n_head = 16
        n_layer = 24
    else:
        hidden = 768
        hidden_latent = 768
        n_head = 12
        n_layer = 12
    
    set_seed(seed)
    model_type = 'bert'
    dataset_all = [] 
    word_to_global_label = {}
    data_index = np.random.permutation(len(sequence))
    #data_index = np.arange(len(sequence))

    #The model name to save (reflect the sequence information)
    for j in range(len(sequence)):
        dataset = sequence[data_index[j]]
        if j == 0:
            save_name = dataset
        else:
            save_name = save_name + '_' + dataset
        print("==============================") 
        print(dataset)
        print("==============================")
        
        subset = get_subset(dataset, shuffle = True)
                
        for sub in subset:
            print("==============================") 
            print(sub)
            print("==============================")
            data, batch_size, n_class, max_len, word_to_global_label = read_data(dataset, sub, args, word_to_global_label, ratio_train = ratio_train)
            print(word_to_global_label)
            
            sent_train, seg_train, spec_token, vocabulary = tokenize_bert(data['train'][0], data['train'][1], encoder_type, max_len)
            label_train = data['train'][2]
                        
            sent_valid, seg_valid, _, _ = tokenize_bert(data['valid'][0], data['valid'][1], encoder_type, max_len)
            label_valid = data['valid'][2]
            
            sent_test, seg_test, _, _ = tokenize_bert(data['test'][0], data['test'][1], encoder_type, max_len)
            label_test = data['test'][2]
            
            if sub is not None:
                name = sub
            else:
                name = dataset
            dataset_all.append({'train':(sent_train, seg_train, (np.array(label_train)).tolist()),
                              'dev':(sent_valid, seg_valid, (np.array(label_valid)).tolist()),
                              'test':(sent_test, seg_test, (np.array(label_test)).tolist()),
                              'local_dic':data['label_dic'],
                              'name':name}) 
            
    #Configuration of task-wise class mask
    class_all = len(word_to_global_label.keys())
    for i in range(len(dataset_all)):
        #all_class masks all classes seen before current task
        if i == 0:
            all_class = np.zeros(class_all)
        local_class = np.zeros(class_all)
        data_class_to_global_label = dataset_all[i]['local_dic']
        for key, value in data_class_to_global_label.items():
            local_class[value] = 1
        print('local class:')
        print(local_class)
        all_class = all_class + local_class * (1 - all_class)
        print('all class')
        print(all_class)
        #local_class = all_class
        dataset_all[i]['class'] = torch.Tensor(local_class).long().unsqueeze(0) #size: 1, n_class
        dataset_all[i]['all_class'] = torch.Tensor(all_class).long().unsqueeze(0) #size: 1, n_class

    kwargs = {'k': k, 'scale': scale, 'encoder_type': encoder_type, 'n_embd': hidden, 'n_layer': n_layer, 'n_head': n_head, "neigh_layer_min": neigh_layer_min, "neigh_layer_max": neigh_layer_max, "spec_token":spec_token}
    if_decoding = True
    config, model = load_method(method, vocabulary, class_all, kwargs, if_decoding, len(dataset_all))

    #Load the pre-trained encoder        
    LM_pretrained = BertForMaskedLM.from_pretrained(encoder_type)
    model.load_from_bert(LM_pretrained.bert)
    if if_decoding:
        model.load_decoder(LM_pretrained)
    del LM_pretrained
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("TRAINING")
    train(max_epoch, lr, model, dataset_all, batch_size, erace, er, agem, pf, freeze_pretrain, device, save_name, plotrep) 