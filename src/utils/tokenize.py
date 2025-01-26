from transformers import BertTokenizer
import torch
import numpy as np

def tokenize_bert(s1, s2, tokenizer_name, max_len = 512): 
    sentence = []
    segment = []
    if 'uncased' in tokenizer_name:
        do_lower_case = True
    else:
        do_lower_case = False
    assert 'bert' in tokenizer_name
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case = do_lower_case) 
    vocabulary = len(tokenizer)

    cls = ["[CLS]"]
    sep = ["[SEP]"]
    mask = ["[MASK]"]
    pad = ["[PAD]"]
    spec_token = {'pad_token': tokenizer.convert_tokens_to_ids(pad)[0], 'mask_token': tokenizer.convert_tokens_to_ids(mask)[0], 'sep_token': tokenizer.convert_tokens_to_ids(sep)[0], 'cls_token': tokenizer.convert_tokens_to_ids(cls)[0]}
     
    for i in range(len(s1)):
        tmp_s1 = tokenizer.tokenize(s1[i])
        if len(s2[i])>0:
            tmp_s2 = tokenizer.tokenize(s2[i])
            #If exceed the length limit, discard the longer sentence.
            if len(tmp_s1) + len(tmp_s2) > max_len-3:
                if len(tmp_s1) > len(tmp_s2):
                    tokenized_s1 = tmp_s1[:(max_len-len(tmp_s2)-3)]
                    tokenized_s2 = tmp_s2
                else:
                    tokenized_s2 = tmp_s2[:(max_len-len(tmp_s1)-3)]
                    tokenized_s1 = tmp_s1
            else:
                tokenized_s1 = tmp_s1
                tokenized_s2 = tmp_s2 
            # [CLS] s1 [SEP] s2 [SEP]
            sentence_i = cls + tokenized_s1 + sep + tokenized_s2 + sep
        else:
            # [CLS] s1 [SEP]
            tokenized_s1 = tmp_s1[:max_len-2]
            sentence_i = cls + tokenized_s1 + sep
        sentence_tmp = tokenizer.convert_tokens_to_ids(sentence_i)
        sentence.append(sentence_tmp)

        #Segment the sentence
        segment_sentence = np.zeros(len(sentence_i))
        segment_sentence[(len(tokenized_s1)+2):] = 1 
        segment.append(segment_sentence.tolist())
    return sentence, segment, spec_token, vocabulary

#Convert token-index to original tokens for visualizing task related tokens
def index_to_sentence(sequence, mask, tokenizer_name = 'bert-base-cased'):
    sent = []
    if 'uncased' in tokenizer_name:
        do_lower_case = True
    else:
        do_lower_case = False
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case = do_lower_case)
        
    lengths_sentence = torch.sum(mask, dim = -1)
    for i in range(sequence.size(0)):
        sent_list = sequence[i].data.cpu().numpy().tolist()
        sent_list = tokenizer.convert_ids_to_tokens(sent_list)
        sent_len = lengths_sentence[i]
        tmp = ""
        for j in range(sent_len):
            tmp = tmp + " " + sent_list[j]
        sent.append([tmp])
    return sent