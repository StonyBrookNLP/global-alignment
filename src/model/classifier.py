import torch
import torch.nn as nn
import torch.nn.functional as F 

def stable_softmax(att, local_class = None):
    if local_class is not None:
        local_class = local_class.unsqueeze(-2).expand_as(att)
        att = att.masked_fill(local_class == 0, -1e9) 
    att_max = torch.max(att, dim = -1, keepdim = True)[0]
    att = att - att_max
    att = torch.exp(att)
    att = att/torch.sum(att, dim = -1, keepdim = True)  
    return att

class Classifier(nn.Module):
    def __init__(self, config):       
        super(Classifier, self).__init__() 
        self.classifier = nn.Linear(config.n_embd, config.n_class, bias=True)
    
    def forward(self, x, local_class = None): 
        x_cls = x[:, :1]
        class_predict = self.classifier((x_cls))#B, 1, n_class

        if local_class is None:
            # softmax over all classes
            class_prob = F.log_softmax(class_predict, dim=2)
        else:
            # softmax over local classes
            class_prob = torch.log(stable_softmax(class_predict, local_class) + 1e-9)
        return class_prob