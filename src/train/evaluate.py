import torch
import numpy as np
from utils.helper import batchify
from utils.tokenize import index_to_sentence
PRINT_DECODING = True

def evaluate_class(local_class, model, sentence, segment, label, batch_size, plot = False, device = "cuda"):
    model.eval()
    correct = 0
    correct_cil = 0
    rep = []
    
    for i in range(0, len(sentence), batch_size):
        batch_sentence, batch_segment, batch_mask, batch_label = batchify(sentence[i:i + batch_size], segment[i:i + batch_size], label[i:i + batch_size])
        batch_sentence, batch_segment, batch_mask, batch_label = batch_sentence.to(device), batch_segment.to(device), batch_mask.to(device), batch_label.to(device)
        local_class = local_class.to(device)
        
        with torch.no_grad():
            x = model(batch_sentence, batch_mask, batch_segment)
            x_cls = x[:, :1]
            predict_label = model.classify(x, local_class)
            predict_label_classil = model.classify(x, None)
            
            correct_til_tmp = predict_label.data.max(2)[1].long().eq(batch_label.data.long())
            correct_cil_tmp = predict_label_classil.data.max(2)[1].long().eq(batch_label.data.long())
            correct += correct_til_tmp.sum().cpu().numpy()
            correct_cil += correct_cil_tmp.sum().cpu().numpy()
            
            if plot:
                #Normalize x_cls
                x_norm = torch.sqrt(torch.sum(x_cls ** 2, dim=-1, keepdim=True))
                x_cls = x_cls/(x_norm + 1e-9)
                rep.extend(x_cls.squeeze(1).data.cpu().numpy())

            #Print tokens in the token space that is closest to cls representation
            if PRINT_DECODING and i == batch_size:
                #Print output
                sentence_origin = index_to_sentence(batch_sentence[0].unsqueeze(0), batch_mask[0].unsqueeze(0))
                print(sentence_origin)
                
                print("Nearest cls word after decoding:")
                rep_predict = model.mlm(x_cls)
                word = rep_predict[0].data.topk(50, dim = 1)[1]
                print_word = index_to_sentence(word, torch.ones(word.size(), dtype = torch.long))
                print(print_word) 
    return correct/len(sentence)*100, correct_cil/len(sentence)*100, rep