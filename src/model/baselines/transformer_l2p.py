import torch
import torch.nn as nn
from model.classifier import Classifier
import torch.nn.functional as F 
import logging 
import math
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

    def __init__(self, config):
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
    
    def self_att(self, x, mask):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x)
        q = self.query(x) 
        v = self.value(x)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))   
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, x.size(1), 1)
            mask = mask.unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9)   

        att = stable_softmax(att)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)    
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y # B, nh, T, T

    def forward(self, x, mask): 
        y = self.self_att(x, mask)
        y = self.layernorm(y + x) 
        return y

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
    def __init__(self, config):
        super(Block, self).__init__()
        self.attn = SelfAttention(config)
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
        
    def forward(self, x, mask):
        x  = self.attn(x, mask)
        tmp = x
        tmp = self.intermediate(tmp) 
        tmp = self.output(tmp)
        x = self.layernorm(x + tmp)
        return x
    
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
                
    def forward(self, x, token_type_ids):
        position_ids = self.position_ids[:, :x.size(1)].cuda()
        position = self.position_emb(position_ids)
        segment = self.segment_emb(token_type_ids)
        token = self.tok_emb(x)

        emb = token + segment + position
        emb = self.layernorm(emb)
        emb = self.drop(emb)
        return emb
         
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

class Prompt(nn.Module):
    def __init__(self, config, n_prompt, n_length):
        super(Prompt, self).__init__()
        self.prompt_pool = nn.Embedding(n_prompt*2, n_length * config.n_embd, padding_idx=0)
        self.prompt_key = nn.Embedding(n_prompt*2, config.n_embd, padding_idx=0)
        #self.pretrained_model = BertModel.from_pretrained(config.encoder_type)
        self.n_prompt = n_prompt
        self.n_length = n_length

    def key_query_loss(self, query, emb_key):         
        key_query_loss = (1 - F.cosine_similarity(query.unsqueeze(1).expand_as(emb_key), emb_key, dim = -1))
        key_query_loss = torch.sum(key_query_loss)/(emb_key.size(0) * emb_key.size(1))
        return key_query_loss
    
    def nearest_prompts(self, query):
        rep_matrix = self.prompt_key.weight.data
        #Cosine similarty
        norm_1 = torch.sum(query * query, dim = -1, keepdim = True)
        norm_2 = torch.sum(rep_matrix * rep_matrix, dim = -1, keepdim = True)
        norm_matrix = torch.sqrt(norm_1 @ norm_2.transpose(-1, -2))
        att = query @ rep_matrix.transpose(0,1) * (1.0 / norm_matrix)
        neighbour_index = att.topk(self.n_prompt, dim = -1)[1]    #B, n_prompt

        emb = self.prompt_pool(neighbour_index).view(query.size(0), self.n_prompt, self.n_length, query.size(-1))
        emb = emb.view(query.size(0), self.n_prompt * self.n_length, -1)
        #emb = self.drop(self.layernorm(emb))
        emb_key = self.prompt_key(neighbour_index)
        self.key_query_ls = torch.zeros((1,), requires_grad=True).cuda()
        self.key_query_ls = self.key_query_loss(query, emb_key)
        return emb

class Transformer(nn.Module):
    def __init__(self, config, class_head=True, mlm_head = False):
        super(Transformer, self).__init__()
        # input embedding stem
        self.emb = BertEmbedding(config)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.mlm_head = mlm_head
        self.class_head = class_head
        self.class_no = config.n_class
        # transformer
        self.layer = config.n_layer
        module = []
        for i in range(config.n_layer):
            module.append(Block(config))
        self.blocks = nn.ModuleList(module)
        if class_head:
            self.classifier = Classifier(config)
        if mlm_head:
            self.MLM = nn.Sequential(nn.Linear(config.n_embd, config.n_embd), GELU(), nn.LayerNorm(config.n_embd, eps=1e-12))
            self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=True)
    
        self.vocab = config.vocab_size 
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.n_prompt = 10
        self.n_length = 5
        self.prompt_pool = Prompt(config, self.n_prompt, self.n_length)
        self.method = 'l2p'
            
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

    def freeze_classifier(self):
        self.classifier.freeze_parameters()

    def get_config(self):
        return self.config
    
    def get_query(self, x, mask):
        for layer in self.blocks:
            x = layer(x, mask)
        query = x[:, 0, :]
        return query
    
    def forward(self, x, mask, segment):
        #Embedding layer
        emb = self.emb(x, segment) 

        #get query
        with torch.no_grad():
            query = self.get_query(emb, mask) #batch_size, hidden_dim
        #get nearest prompts
        emb_prompts = self.prompt_pool.nearest_prompts(query) #batch_size, self.n_prompts, hidden_dim
        emb = torch.cat((emb_prompts, emb), dim = 1)
        mask_prompts = torch.ones(emb_prompts.size(0), emb_prompts.size(1)).to(mask.device)
        mask = torch.cat((mask_prompts, mask), dim = 1)

        x = emb.clone()

        #Attention layer
        for layer in self.blocks:
            x = layer(x, mask)
        
        #cls prediction by average pooling of the prompts
        #x = torch.sum(x[:, :self.n_prompt*self.n_length], dim = 1, keepdim = True)/(self.n_prompt*self.n_length)
        x = x[:, self.n_prompt*self.n_length:]
        return x

    def classify(self, x, local_class = None): #Classification performs after the decoder
        class_prob = self.classifier(x, local_class) #original classifier
        return class_prob         
            
    def class_loss(self, class_predict, target):
        class_prob = class_predict.gather(2, target.unsqueeze(2).repeat(1, class_predict.size(1), 1))#.squeeze(1)
        class_loss = -torch.sum(class_prob)
        class_loss = class_loss / class_predict.size(0)
        loss = class_loss + self.prompt_pool.key_query_ls
        return loss
    
    def mlm(self, x):
        mask_rep = self.MLM(x)
        mask_head = self.drop(mask_rep)
        mask_predict = self.decoder(mask_head)
        return mask_predict

    