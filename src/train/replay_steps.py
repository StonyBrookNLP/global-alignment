from copy import deepcopy
from utils.helper import batchify
import torch
PRINT_FREQ = 50

def agem_step(model, steps_done, batch_size, replay_buffer, device = "cuda"):
    #Sample data from replay buffer
    replay_buffer.update_keys()
    #No A-GEM performed for the first task, i.e. len(replay_buffer) = 0
    if replay_buffer.keys_num > 0:
        #Store the current gradient
        cur_gradient = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                cur_gradient[n] = deepcopy(p.grad.data)
                p.grad.data = 0 * p.grad.data

        sentence_replay, segment_replay, mask_replay, label_replay, local_class_replay = replay_buffer.sample_batch(batch_size)
        sentence, segment, mask, label = sentence_replay.to(device), segment_replay.to(device), mask_replay.to(device), label_replay.to(device)
        local_class = local_class_replay.to(device)
        #Calculate the replay gradient
        x = model(sentence, mask, segment)
        class_predict = model.classify(x, local_class)
        cls_loss = model.class_loss(class_predict, label) 

        if steps_done % PRINT_FREQ == 0:
            loss_replay = cls_loss.data.cpu().numpy()
            correct_replay = class_predict.data.max(2)[1].long().eq(label.data.long()).sum().cpu().numpy()
            print("Sentence: " + str(steps_done) + "; The AGEM loss is:" + str(loss_replay) + "; The classification accuracy is:" + str(correct_replay/(batch_size)*100))

        # Backward pass
        cls_loss.backward()
        #Compare to the reference gradient
        dotcur = 0 #current gradient
        dotpast = 0 #ref gradient from past experience
        past_gradient = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                past_gradient[n] = deepcopy(p.grad.data) 
                dotcur += torch.sum(past_gradient[n].data * cur_gradient[n].data)
                dotpast += torch.sum(past_gradient[n].data * past_gradient[n].data)
        
        for n, p in model.named_parameters():
            if p.requires_grad and (p.grad is not None):
                if dotcur < 0:
                    p.grad.data = cur_gradient[n].data - (dotcur/(dotpast + 1e-9)) * past_gradient[n] 
                else:
                    p.grad.data = cur_gradient[n].data
    return 

def agem_push(model, data_current, batch_size, replay_buffer, device = "cuda"):
    sentence, segment, label = data_current['train']
    local_class = data_current['class']
    max_data_len = len(sentence)
    for i in range(0, max_data_len, batch_size):  
        batch_sentence, batch_segment, batch_mask, batch_label = batchify(sentence[i:i + batch_size], segment[i:i + batch_size], label[i:i + batch_size])
        batch_sentence, batch_segment, batch_mask, batch_label = batch_sentence.to(device), batch_segment.to(device), batch_mask.to(device), batch_label.to(device)
       
        x = model(batch_sentence, batch_mask, batch_segment)
        keys = x[:, :1]
        #Push all samples in the previous task to the replay buffer
        replay_buffer.push(keys.data.cpu().numpy(), (sentence[i:i + batch_size], segment[i:i + batch_size], label[i:i + batch_size], local_class.squeeze(0).data.numpy()), sample_rate = 1.0)
    return 

def er_step(model, er_frequency, steps_done, batch_size, replay_buffer, device = "cuda"):
    #If perform replay
    if (steps_done + 1) % er_frequency == 0:           
        #Store the current gradient
        cur_gradient = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                cur_gradient[n] = deepcopy(p.grad.data)
                p.grad.data = 0 * p.grad.data
        
        #Sample data from replay buffer
        replay_buffer.update_keys()
        if replay_buffer.keys_num > 0:
            sentence_replay, segment_replay, mask_replay, label_replay, local_class_replay = replay_buffer.sample_batch(batch_size)
            sentence, segment, mask, label = sentence_replay.to(device), segment_replay.to(device), mask_replay.to(device), label_replay.to(device)
            local_class = local_class_replay.to(device)
            #Calculate the replay gradient
            x = model(sentence, mask, segment)
            class_predict = model.classify(x, local_class)
            cls_loss = model.class_loss(class_predict, label) 

            loss_replay = cls_loss.data.cpu().numpy()
            correct_replay = class_predict.data.max(2)[1].long().eq(label.data.long()).sum().cpu().numpy()
            print("Sentence: " + str(steps_done) + "; The ER loss is:" + str(loss_replay) + "; The classification accuracy is:" + str(correct_replay/(batch_size)*100))

            # Backward pass
            cls_loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.data = (cur_gradient[n] + p.grad.data)/2   
    return
    
def erace_step(model, steps_done, all_class, batch_size, replay_buffer, device = "cuda"):     
    #Sample data from replay buffer
    replay_buffer.update_keys()
    if replay_buffer.keys_num > 0:
        #Store the current gradient
        cur_gradient = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                cur_gradient[n] = deepcopy(p.grad.data)
                p.grad.data = 0 * p.grad.data
                
        sentence_replay, segment_replay, mask_replay, label_replay, _ = replay_buffer.sample_batch(batch_size)
        sentence, segment, mask, label = sentence_replay.to(device), segment_replay.to(device), mask_replay.to(device), label_replay.to(device)
        all_class = all_class.to(device)
        #Calculate the replay gradient
        x = model(sentence, mask, segment)
        class_predict = model.classify(x, all_class) #class incremental replay, perform on all_seen class till the current task
        cls_loss = model.class_loss(class_predict, label) 

        if steps_done % PRINT_FREQ == 0:
            loss_replay = cls_loss.data.cpu().numpy()
            correct_replay = class_predict.data.max(2)[1].long().eq(label.data.long()).sum().cpu().numpy()
            print("Sentence: " + str(steps_done) + "; The ERACE loss is:" + str(loss_replay) + "; The classification accuracy is:" + str(correct_replay/(batch_size)*100))
            
        # Backward pass
        cls_loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad.data = (cur_gradient[n] + p.grad.data)/2
    return

