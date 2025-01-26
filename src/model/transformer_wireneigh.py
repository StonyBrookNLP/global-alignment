import torch
import torch.nn as nn
from model.classifier import Classifier
import torch.nn.functional as F 
import logging 
import math
import numpy as np
logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
        
def stable_softmax(att, local_class = None):
    if local_class is not None:
        local_class = local_class.unsqueeze(-2).expand_as(att)
        att = att.masked_fill(local_class == 0, -1e9) 
    att_max = torch.max(att, dim = -1, keepdim = True)[0]
    att = att - att_max
    att = torch.exp(att)
    att = att/torch.sum(att, dim = -1, keepdim = True)  
    return att

class TransformerConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 12
    n_head = 12
    n_embd = 768
    max_position_embeddings = 512
    type_vocab_size = 2
    
    def __init__(self, vocab_size, n_class, kwargs):
        self.vocab_size = vocab_size
        self.n_class = n_class
        for k,v in kwargs.items():
            setattr(self, k, v)        
               
class SelfAttention(nn.Module):

    def __init__(self, config, if_neigh):
        super(SelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # drop out
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.layernorm = nn.LayerNorm(config.n_embd, eps = 1e-12)
        if if_neigh:
            # low rank parameters
            r = 8
            self.key_neigh_down = nn.Linear(config.n_embd, r)
            self.key_neigh_up = nn.Linear(r, config.n_embd)
            self.scale = config.scale #0.1
            #self.neigh_drop = nn.Dropout(0.1)
            self.init_wire()

        self.if_neigh = if_neigh

    def init_wire(self):
        self.key_neigh_up.weight.data.zero_()
        nn.init.kaiming_uniform_(self.key_neigh_down.weight.data, a=math.sqrt(5))
    
    #Same self_att as pre-trained model
    def self_att(self, x, mask):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x[:, 1:]) #Tokens besides the first token
        k = self.key(x)
        v = self.value(x)
        
        q = q.view(B, T-1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))   
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9)   

        att = stable_softmax(att)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)    
        y = y.transpose(1, 2).contiguous().view(B, T-1, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y # B, nh, T, T

    def cls_att(self, x, mask):
        B, T, C = x.size()
        q = self.query(x[:, :1]) #First token as [CLS]
        k = self.key_neigh_up(self.key_neigh_down(x))
        v = self.value(x)

        q = q.view(B, 1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, 1, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9) 

        att = stable_softmax(att)
        att = self.attn_drop(att)   
    

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)    
        y = y.transpose(1, 2).contiguous().view(B, 1, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y

    def neigh_att(self, x, neighbour_rep):
        candidate = torch.cat((x[:, 1:].unsqueeze(2), neighbour_rep), dim = -2) #For tokens other than [CLS], update their representations by neighbor attentions. x is the neighbor of its own
        B, T, k, C = candidate.size()

        q_neigh = self.query(candidate)
        k_neigh = self.key_neigh_up(self.key_neigh_down(candidate))#self.key(candidate_kv) 
        v_neigh = self.value(candidate) 

        q_neigh = q_neigh.view(B, T, k, self.n_head, C // self.n_head).transpose(2, 3)
        k_neigh = k_neigh.view(B, T, k, self.n_head, C // self.n_head).transpose(2, 3)
        v_neigh = v_neigh.view(B, T, k, self.n_head, C // self.n_head).transpose(2, 3)

        att_neigh = (q_neigh @ k_neigh.transpose(-2, -1)) * (1.0 / math.sqrt(q_neigh.size(-1)))  #(B, T, n_head, 1, k+1) 
        att_neigh = stable_softmax(att_neigh)
        att_neigh = self.attn_drop(att_neigh)

        v_neigh = att_neigh @ v_neigh #(B, T, n_head, k+1, k+1) x (B, T, n_head, k+1, n_hid)-> (B, T, n_head, k+1, n_hid)
        v_neigh = v_neigh.transpose(2, 3).contiguous().view(B, T, k, C)
        output = self.resid_drop(self.proj(v_neigh))

        y_neigh = output[:, :, 0, :]
        neighbour_rep = output[:, :, 1:, :]
        return y_neigh, neighbour_rep
    
    def forward(self, x, mask, neighbour_rep): 
        #For CLS token
        y_cls = self.cls_att(x, mask)

        #For tokens other than [CLS]
        y = self.self_att(x, mask)

        if self.if_neigh:
            assert (neighbour_rep is not None)
            y_neigh, neighbour_tmp = self.neigh_att(x, neighbour_rep)

            #neigh_scale = 1
            #neighbour_rep = self.layernorm(neigh_scale  * neighbour_tmp + (1 - neigh_scale) * y.data.unsqueeze(2) + neighbour_rep)
            y = self.scale  * y_neigh + (1 - self.scale) * y
            neighbour_rep = self.layernorm(neighbour_tmp + neighbour_rep)

        y = torch.cat((y_cls, y), dim = 1)
        y = self.layernorm(y + x)    
        return y, neighbour_rep

    def load_from_bert(self, bert_attention):
        self.key.weight.data.copy_(bert_attention.self.key.weight.data)
        self.key.bias.data.copy_(bert_attention.self.key.bias.data)

        self.query.weight.data.copy_(bert_attention.self.query.weight.data)
        self.query.bias.data.copy_(bert_attention.self.query.bias.data)

        self.value.weight.data.copy_(bert_attention.self.value.weight.data)
        self.value.bias.data.copy_(bert_attention.self.value.bias.data)

        self.proj.weight.data.copy_(bert_attention.output.dense.weight.data)
        self.proj.bias.data.copy_(bert_attention.output.dense.bias.data)

        self.layernorm.weight.data.copy_(bert_attention.output.LayerNorm.weight.data) 
        self.layernorm.bias.data.copy_(bert_attention.output.LayerNorm.bias.data) 
            
    def freeze_pretrain(self):
        for n, p in list(self.key.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.query.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.value.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.proj.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.layernorm.named_parameters()):
            p.requires_grad = False 

class Block(nn.Module):
    def __init__(self, config, if_neigh):
        super(Block, self).__init__()
        self.if_neigh = if_neigh
        self.attn = SelfAttention(config, if_neigh)
        self.n_head = config.n_head
        self.intermediate = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU()
        )
        self.output = nn.Sequential(
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        self.layernorm = nn.LayerNorm(config.n_embd, eps = 1e-12)
        
    def forward(self, x, mask, neighbour_rep):
        x, neighbour_rep = self.attn(x, mask, neighbour_rep)
        tmp = x
        tmp = self.intermediate(tmp) 
        tmp = self.output(tmp)
        x = self.layernorm(x + tmp)

        #Update the neighbor representations
        if self.if_neigh:
            tmp = self.intermediate(neighbour_rep) #B, T, k, C
            neighbour_rep = self.layernorm(neighbour_rep + self.output(tmp))
        return x, neighbour_rep
    
    def load_from_bert(self, bert_layer):
        self.attn.load_from_bert(bert_layer.attention)
        self.intermediate[0].weight.data.copy_(bert_layer.intermediate.dense.weight.data)
        self.intermediate[0].bias.data.copy_(bert_layer.intermediate.dense.bias.data)
        self.output[0].weight.data.copy_(bert_layer.output.dense.weight.data)
        self.output[0].bias.data.copy_(bert_layer.output.dense.bias.data)
        self.layernorm.weight.data.copy_(bert_layer.output.LayerNorm.weight.data)
        self.layernorm.bias.data.copy_(bert_layer.output.LayerNorm.bias.data)
            
    def freeze_pretrain(self):
        self.attn.freeze_pretrain()
        for n, p in list(self.intermediate.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.output.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.layernorm.named_parameters()):
            p.requires_grad = False

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.n_embd = config.n_embd
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.segment_emb = nn.Embedding(config.type_vocab_size, config.n_embd)  
        self.position_emb = nn.Embedding(config.max_position_embeddings, config.n_embd)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("vocab_ids", torch.arange(config.vocab_size).expand((1, -1)))
        self.layernorm = nn.LayerNorm(config.n_embd, eps = 1e-12)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.spec_token = config.spec_token
        self.sample_pool = 50 #the range of top-K neighbors
        self.sample_drop = nn.Dropout(config.embd_pdrop)

    def forward(self, x, token_type_ids):
        position_ids = self.position_ids[:, :x.size(1)].to(x.device)
        position = self.position_emb(position_ids)
        segment = self.segment_emb(token_type_ids)
        token = self.tok_emb(x)

        emb = token + segment + position
        emb = self.layernorm(emb)
        emb = self.drop(emb)
        return emb, position, segment
    
    def nearest_neighbour(self, emb, k, pos_emb, seg_emb):
        rep_matrix = self.tok_emb.weight.data #vocab * hidden
        #Calculate the cosine similarity
        norm_1 = torch.sum(emb * emb, dim = -1, keepdim = True) #B, T, 1
        norm_2 = torch.sum(rep_matrix * rep_matrix, dim = -1, keepdim = True).unsqueeze(0).repeat(emb.size(0), 1 ,1) #batch, vocab, 1
        norm_matrix = torch.sqrt(norm_1 @ norm_2.transpose(1, 2)) #B, T, vocab
        att = emb @ rep_matrix.transpose(0,1) * (1.0 / norm_matrix)
        #Drop for exploration
        att = self.sample_drop(att)
         
        if k > self.sample_pool:
            sample_pool = k
        else:
            sample_pool = self.sample_pool
        
        stride = np.ceil(sample_pool/(k + 1e-9))
        
        start = 0
        top_index = np.arange(start, sample_pool + start, stride, dtype=int)
        
        neighbour_index_all = torch.sort(att, dim = -1, descending = True)[1]
        neighbour_index = neighbour_index_all[:, :, top_index] #B, T, k, C

        emb_neigh = self.tok_emb(neighbour_index)
         
        if pos_emb is not None:
            emb_neigh = emb_neigh + pos_emb.unsqueeze(2) + seg_emb.unsqueeze(2)
         
        emb_neigh = self.drop(self.layernorm(emb_neigh)) #* nearest_att.unsqueeze(3)
        return emb_neigh
        
    def load_from_bert(self, bert_emb):
        self.tok_emb.weight.data.copy_(bert_emb.word_embeddings.weight.data) 
        self.segment_emb.weight.data.copy_(bert_emb.token_type_embeddings.weight.data) 
        self.position_emb.weight.data.copy_(bert_emb.position_embeddings.weight.data) 
        self.layernorm.weight.data.copy_(bert_emb.LayerNorm.weight.data)
        self.layernorm.bias.data.copy_(bert_emb.LayerNorm.bias.data)
            
    def freeze_pretrain(self):
        for n, p in list(self.tok_emb.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.segment_emb.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.position_emb.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.layernorm.named_parameters()):
            p.requires_grad = False  
    
class Transformer(nn.Module):
    def __init__(self, config, class_head=True, mlm_head = False):
        super(Transformer, self).__init__()
        # input embedding stem
        self.emb = BertEmbedding(config)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.class_head = class_head
        self.mlm_head = mlm_head
        self.class_no = config.n_class
        # transformer
        self.layer = config.n_layer
        self.neigh_layer_min = config.neigh_layer_min#3
        self.neigh_layer_max = config.neigh_layer_max#10 # was 11

        module = []
        for i in range(config.n_layer):
            #Add neighbor attention to specific range of layers: default all layers
            if i >= self.neigh_layer_min and i <= self.neigh_layer_max:
                if_neigh = True
            else:
                if_neigh = False
            module.append(Block(config, if_neigh))
        self.blocks = nn.ModuleList(module)

        self.k = config.k
        if class_head:
            self.classifier = Classifier(config)
        if mlm_head:
            self.MLM = nn.Sequential(nn.Linear(config.n_embd, config.n_embd), GELU(), nn.LayerNorm(config.n_embd, eps=1e-12))
            self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=True)
    
        self.vocab = config.vocab_size 
        self.method = 'wireneigh'
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
            
    def load_from_bert(self, bert_model):
        self.emb.load_from_bert(bert_model.embeddings)
        i = 0
        for layer in self.blocks:
            layer.load_from_bert(bert_model.encoder.layer[i]) 
            i += 1

    def load_decoder(self, model): 
        for n, p in list(self.MLM.named_parameters()):
            if '0' in n:
                if 'weight' in n:
                    p.data.copy_(model.cls.predictions.transform.dense.weight.data)
                elif 'bias' in n:
                    p.data.copy_(model.cls.predictions.transform.dense.bias.data)
            elif '2' in n:
                if 'weight' in n:
                    p.data.copy_(model.cls.predictions.transform.LayerNorm.weight.data)
                elif 'bias' in n:
                    p.data.copy_(model.cls.predictions.transform.LayerNorm.bias.data)
        self.decoder.weight.data.copy_(model.cls.predictions.decoder.weight.data)
        self.decoder.bias.data.copy_(model.cls.predictions.decoder.bias.data)

    def freeze_pretrain(self):
        self.emb.freeze_pretrain()
        i = 0
        for layer in self.blocks:
            layer.freeze_pretrain()
            i += 1
        if self.mlm_head:
            for n, p in list(self.MLM.named_parameters()):
                p.requires_grad = False
            for n, p in list(self.decoder.named_parameters()):
                p.requires_grad = False

    def freeze_encoder(self):  
        for n, p in self.named_parameters(): 
            p.requires_grad = False  
        if self.class_head:
            for n, p in list(self.classifier.named_parameters()):
                p.requires_grad = True  
   
    def finetune_all(self):  
        for n, p in self.named_parameters(): 
            p.requires_grad = True  
        if self.mlm_head:
            for n, p in list(self.MLM.named_parameters()):
                p.requires_grad = False
            for n, p in list(self.decoder.named_parameters()):
                p.requires_grad = False

    def freeze_classifier(self):
        self.classifier.freeze_parameters()

    def get_config(self):
        return self.config
        
    #seqJoint_model(batch_sentence, batch_mask, batch_segment, batch_inter_mask, batch_prefix_mask, batch_target_mask, cls_no = cls_no)       
    def forward(self, x, mask, segment):
        #Embedding layer
        x, pos_emb, seg_emb = self.emb(x, segment) 
        neighbour_rep = None

        #Attention layer
        i = 0
        for layer in self.blocks:
            if i == self.neigh_layer_min:
                #Get the initial neighbors for non-cls token
                neighbour_rep = self.emb.nearest_neighbour(x[:, 1:], self.k, pos_emb[:, 1:], seg_emb[:, 1:]) #B, T-1, k, C
            x, neighbour_rep = layer(x, mask, neighbour_rep)
            i += 1
        return x

    def classify(self, x, local_class = None): #Classification performs after the decoder
        class_prob = self.classifier(x, local_class) #original classifier
        return class_prob         
            
    def class_loss(self, class_predict, target):
        class_prob = class_predict.gather(2, target.unsqueeze(2).repeat(1, class_predict.size(1), 1))#.squeeze(1)
        class_loss = -torch.sum(class_prob)
        class_loss = class_loss / class_predict.size(0)
        return class_loss
    
    def mlm(self, x):
        mask_rep = self.MLM(x)
        mask_head = self.drop(mask_rep)
        mask_predict = self.decoder(mask_head)
        return mask_predict

    