import codecs
import os
import numpy as np
from utils.tokenize import tokenize_bert
from utils.helper import set_seed, batchify
import torch
import os
import model.transformer_wireneigh as transformer_wireneigh
import model.transformer_clora as transformer_clora
import model.transformer_wirefix as transformer_wirefix
import model.transformer_original as transformer_original
from transformers import BertForMaskedLM, BertTokenizer
import argparse
###################################################
#Recall@20 after decoding the cls representation
#################################################
def loadFile(fpath):
  with codecs.open(fpath, 'rb', 'utf-8') as f:
    return [line for line in f.read().splitlines()]

def loadKey(fpath):
  keywords = []
  with codecs.open(fpath, 'rb', 'utf-8') as f:
    for line in f.read().splitlines():
      segments = line.rstrip().split(',')
      keywords.append(segments)
  return keywords
	
def readData(taskpath):
  s1 = loadFile(os.path.join(taskpath, 's1.dev'))
  s2 = loadFile(os.path.join(taskpath, 's2.dev'))
  keywords = loadKey(os.path.join(taskpath, 'keywords.dev'))
  label_dic = {0: 'entailment',  1: 'neutral',  2: 'contradiction'}    
  data = {'sentence': (s1, s2),
          'keywords': keywords, #rational tokens
          'label_dic': label_dic
          }    
  return data

#Convert rationale tokens to indexes
def keywords_index(tokenizer, keywords, unknown = '[UNK]'):
  unknown = tokenizer.convert_tokens_to_ids(unknown)
  index = []
  sent_index = []
  for i in range(len(keywords)):
    tmp = []
    key_index = tokenizer.convert_tokens_to_ids(keywords[i])
    #Filter the UNK
    for k in key_index:
      if k != unknown:
        tmp.append(k)
    if len(tmp) > 0:
      index.append(tmp)
      sent_index.append(i)  
  return index, sent_index
    
def keywords_recall(model, sentence, segment, keywords, label, batch_size, k = 20, device = "cuda", printsample = False):
  model.eval()
  recall = 0
  k_len = 0
  for i in range(0, len(sentence), batch_size):
    batch_sentence, batch_segment, batch_mask, batch_label = batchify(sentence[i:i + batch_size], segment[i:i + batch_size], label[i:i + batch_size])
    batch_sentence, batch_segment, batch_mask, batch_label = batch_sentence.to(device), batch_segment.to(device), batch_mask.to(device), batch_label.to(device)
    batch_keywords = keywords[i:i + batch_size]

    with torch.no_grad():
      x = model(batch_sentence, batch_mask, batch_segment) 
      x_cls = x[:, :1]
      cls_word = model.mlm(x_cls)
      mask_nearest = cls_word.data.topk(k, dim = 2)[1]

      for j in range(x.size(0)):
        nearest = mask_nearest[j].data.cpu().numpy()[0]
        k_len += len(batch_keywords[j])
        for key in batch_keywords[j]:
          norm = len(batch_keywords[j])
          if key in nearest:
            recall += 1 / norm
  return recall/len(sentence)*100
		
parser = argparse.ArgumentParser()
parser.add_argument('--enc_type', type=str, default='bert-base-cased')
parser.add_argument('--bsize', type=int, default=32)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--sequence', type=str, nargs='+', default=['snli'])
parser.add_argument('--neigh_layer_min', type=int, default=0)
parser.add_argument('--neigh_layer_max', type=int, default=12)
parser.add_argument('--scale', type=float, default=0.1)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--method', type=str, default='wirefix') #Select from type: wirefix, wireneigh, clora and origin
args = parser.parse_args()

if __name__ == '__main__':
    device = "cuda"
    #Parameters
    seed = args.seed#1234. 39, 1, 67, 123
    encoder_type = args.enc_type      
    neigh_layer_min = args.neigh_layer_min #3
    neigh_layer_max = args.neigh_layer_max
    sequence = args.sequence
    scale = args.scale
    k = args.k
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
	
    fpath = '../../data/ESNLI'
    data = readData(fpath)
    batch_size = 32
    max_len = 128
    
    sent, seg, spec_token, vocabulary = tokenize_bert(data['sentence'][0], data['sentence'][1], encoder_type, max_len)
    keywords = data['keywords'] 
    if 'uncased' in encoder_type:
        do_lower_case = True
    else:
        do_lower_case = False
    assert 'bert' in encoder_type
    tokenizer = BertTokenizer.from_pretrained(encoder_type, do_lower_case = do_lower_case) 

    keyw, sent_index = keywords_index(tokenizer, keywords)
    label = np.ones((len(keyw), 1))
    label_dic = data['label_dic']
    sent = np.array(sent, dtype=object)[sent_index].tolist()
    seg = np.array(seg, dtype=object)[sent_index].tolist()
    assert len(keyw) == len(sent)
    
    #Load model
    LM_pretrained = BertForMaskedLM.from_pretrained(encoder_type)
    mask_token, cls_token, sep_token, pad_token = spec_token['mask_token'], spec_token['cls_token'], spec_token['sep_token'], spec_token['pad_token']
    kwargs = {'k': k, 'scale': scale, 'encoder_type': encoder_type, 'n_embd': hidden, 'n_layer': n_layer, 'n_head': n_head, "neigh_layer_min": neigh_layer_min, "neigh_layer_max": neigh_layer_max, "spec_token":spec_token}    
    
    print("Loading the " + method + " model...")
    if method == 'wirefix':
        config = transformer_wirefix.TransformerConfig(vocabulary, 0, kwargs)
        model = transformer_wirefix.Transformer(config, class_head = False, mlm_head = True) 
    elif method == 'wireneigh':
        config = transformer_wireneigh.TransformerConfig(vocabulary, 0, kwargs)
        model = transformer_wireneigh.Transformer(config, class_head = False, mlm_head = True) 
    elif method == 'clora':
        config = transformer_clora.TransformerConfig(vocabulary, 0, kwargs)
        model = transformer_clora.Transformer(config, class_head = False, mlm_head = True) 
    else:
        #Load the original fine-tuning model
        config = transformer_original.TransformerConfig(vocabulary, 0, kwargs)
        model = transformer_original.Transformer(config, class_head = False, mlm_head = True)

    model.load_from_bert(LM_pretrained.bert)
    model.load_decoder(LM_pretrained)
    
    del LM_pretrained
    model = model.cuda()

    print("Evaluating")
    k = 20
    output_dir = './model_output'
    model_name = None
    for name in sequence:
      if model_name is not None:
        model_name = model_name + '_' + name
      else:
        model_name = name
    #################################################################################
    #Need to first save the model during training. Modified in train_cl.py line 115
    ################################################################################
    output_model_file = os.path.join(output_dir, model_name + "_class.pth")
    state_dict = torch.load(output_model_file)
    model.load_state_dict(state_dict, strict=False)
    
    recall = keywords_recall(model, sent, seg, keyw, label, batch_size, k, device)
    print("The recall of decoding CLS representation is:")
    print(recall)
    