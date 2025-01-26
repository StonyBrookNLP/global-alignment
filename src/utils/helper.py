import numpy as np
import random
import torch
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os 
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed, cuda = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_optimizer(model, lr, training_steps):
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'layernorm.bias', 'layernorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if ((not any(nd in n for nd in no_decay) and p.requires_grad))], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if ((any(nd in n for nd in no_decay) and p.requires_grad))], 'weight_decay': 0.00}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = lr, eps = 1e-6, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*training_steps), training_steps)
    return optimizer, scheduler

def visualize(x, label, filename, marker_style = 'o', edge_c = 'face'):
    n_class = 0
    for i in label:
        if i > n_class:
            n_class = i
    n_class = n_class + 1
    cmap = plt.cm.hsv(np.linspace(0, 0.9, n_class))
    rep = TSNE(n_components = 2, random_state = 1234).fit_transform(x)
    plt.scatter(x = rep[:,0], y = rep[:,1], marker=marker_style, edgecolors = edge_c, s = 1, c = [cmap[int(l)] for l in label])
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 5)
    #plt.set_size_inches(18.5, 10.5)
    plt.savefig(filename)
    plt.clf()

def batchify(sentence, segment, labels):
    lengths = np.array([len(x) for x in sentence])
    n_sent = len(sentence)
    max_len = int(np.max(lengths))
    
    batch_sentence = torch.zeros([n_sent, max_len], dtype = torch.long)
    batch_segment = torch.LongTensor(n_sent, max_len).zero_()
    batch_mask = torch.LongTensor(n_sent, max_len).zero_()
    batch_label = torch.LongTensor(n_sent, 1)

    for i in range(n_sent):
        
        sent_len = len(sentence[i])
        batch_sentence[i, :sent_len] = torch.Tensor(sentence[i]).long()
        batch_segment[i, :sent_len] = torch.Tensor(segment[i]).long()
        batch_mask[i, :sent_len] = 1
        batch_label[i, :] = torch.Tensor(labels[i]).long()
    return batch_sentence, batch_segment, batch_mask, batch_label