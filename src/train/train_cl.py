import os
import torch
import math
import numpy as np
from train.evaluate import evaluate_class
from utils.helper import visualize, set_optimizer, batchify
from utils.replay import ReplayMemory
from train.replay_steps import er_step, erace_step, agem_step, agem_push

def train_each_epoch(steps_done, local_class, all_class, optimizer, scheduler, model, sentence, segment, label, batch_size, erace, er, agem, replay_buffer, device = "cuda"):
    model.train()
    print_each = batch_size * 1000
    
    correct = 0
    loss_all = []

    #Shuffle the sentence
    index = np.random.permutation(len(sentence))
        
    sentence = np.array(sentence, dtype=object)[index].tolist()
    segment = np.array(segment, dtype=object)[index].tolist()
    label = np.array(label, dtype=object)[index].tolist()

    max_data_len = len(sentence)
    for i in range(0, max_data_len, batch_size):  
        batch_sentence, batch_segment, batch_mask, batch_label = batchify(sentence[i:i + batch_size], segment[i:i + batch_size], label[i:i + batch_size])
        batch_sentence, batch_segment, batch_mask, batch_label = batch_sentence.to(device), batch_segment.to(device), batch_mask.to(device), batch_label.to(device)
        local_class = local_class.to(device)
       
        x = model(batch_sentence, batch_mask, batch_segment)
        class_predict = model.classify(x, local_class)
        
        correct += class_predict.data.max(2)[1].long().eq(batch_label.data.long()).sum().cpu().numpy()

        loss = model.class_loss(class_predict, batch_label)
        loss_all.append(loss.data.cpu().numpy())

        if i % print_each == 0:
            print("Sentence: " + str(i) + "; The loss is:" + str(np.sum(loss_all)/len(loss_all)) + "; The classification accuracy is:" + str(correct/(i + len(batch_label))*100)) 
        
        #Optimization
        optimizer.zero_grad() 
        loss.backward()

        #Push samples to the replay_buffer
        if er or erace:
            keys = x[:, :1]
            replay_buffer.push(keys.data.cpu().numpy(), (sentence[i:i + batch_size], segment[i:i + batch_size], label[i:i + batch_size], local_class.squeeze(0).data.cpu().numpy()))
        
        #Update gradients based on different replay strategies
        if er:
            replay_freq = 100 #sparse replay
            er_step(model, replay_freq, steps_done, batch_size, replay_buffer, device)
        elif erace: #perform at every step
            erace_step(model, steps_done, all_class, batch_size, replay_buffer, device)
        elif agem:
            agem_step(model, steps_done, batch_size, replay_buffer, device)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()       
        if scheduler is not None:   
            scheduler.step()
        steps_done += 1
    return correct/(len(sentence)) * 100, steps_done

def train_task(steps_done, max_epoch, lr, model, data_current, batch_size, erace, er, agem, replay_buffer, device = "cuda", save_name = 'news_series', save_model = False, output_dir = './model_output'): 
    #Set optimizer   
    length_all = len(data_current['train'][0])
    training_steps = int(math.ceil(length_all/batch_size)) * max_epoch
    print('Training steps:'+ str(training_steps))
    optimizer, scheduler = set_optimizer(model, lr, training_steps)

    epoch = 1
    while epoch <= max_epoch:
        print("epoch:" + str(epoch))
        sentence, segment, label = data_current['train']
        local_class = data_current['class']
        all_class = data_current['all_class']
        
        acc_train, steps_done = train_each_epoch(steps_done, local_class, all_class, optimizer, scheduler, model, sentence, segment, label, batch_size, erace, er, agem, replay_buffer, device)
        if epoch == max_epoch: #if epoch == True:
            sentence_val, segment_val, label_val = data_current['dev']
            local_class = data_current['class']
            eval_acc_til, eval_acc_cil, _ = evaluate_class(local_class, model, sentence_val, segment_val, label_val, batch_size, device = device)
            print("The evaluation accuracy is:" + str(eval_acc_til)) 
 
        #Early Stop
        if save_model: 
            print("Saving the model at epoch " + str(epoch))
            model_to_save = model.module if hasattr(model, 'module') else model 
            output_model_file = os.path.join(output_dir, save_name + "_class.pth")
            torch.save(model_to_save.state_dict(), output_model_file)
        
        epoch += 1
    #seqJoint_model = best_model
    return eval_acc_til, eval_acc_cil, steps_done
    
def train(max_epoch, lr, model, data_all, batch_size, erace, er, agem, pf, freeze_pretrain, device = "cuda", save_name = 'news_series', plot_rep = False):
    output_dir = './model_output'

    before = []
    before_cil = []
    after = []
    after_cil = []
    
    print("==============================")
    print("Classification Only Training") 
    print("==============================")
    replay_buffer = ReplayMemory()
    steps_done = 0

    for task in range(len(data_all)):
        print("Task" + str(task))
        data_current = data_all[task]  
        save_model_train = False
        if pf and task > 0:
            print("Classifier Training: Probing")
            #Classifier only training
            model.freeze_encoder()
            class_lr = 1e-3
            class_max_epoch = 3
            eval_acc_til, eval_acc_cil, _ = train_task(0, class_max_epoch, class_lr, model, data_current, batch_size, False, False, False, None, device, save_name, save_model = save_model_train, output_dir = output_dir)
         
        #Fine-tuning
        print("All model Training")
        #Enable parameter training in both encoder and classifier
        model.finetune_all()
        if freeze_pretrain:
            #Freeze the pre-trained part in the encoder
            model.freeze_pretrain()
        if task > 0 and model.method == 'coda':
            model.coda_prompt.process_task_count()
        eval_acc_til, eval_acc_cil, steps_done = train_task(steps_done, max_epoch, lr, model, data_current, batch_size, erace, er, agem, replay_buffer, device, save_name, save_model = save_model_train, output_dir = output_dir)
        
         
        before.append(eval_acc_til)
        before_cil.append(eval_acc_cil)
         
        #Push data for AGEM
        if agem:
            agem_push(model, data_current, batch_size, replay_buffer, device)   
    
    print("CL Evaluation!")
    '''
    #Read model from saved file
    output_model_file = os.path.join(output_dir, save_name + "_class.pth")
    state_dict = torch.load(output_model_file)
    model.load_state_dict(state_dict)
    '''
    rep = []
    label = []
    for task in range(len(data_all)):
        sentence_val, segment_val, label_val = data_all[task]['dev']
        local_class = data_all[task]['class']
        save_name = data_all[task]['name']
        eval_acc_til, eval_acc_cil, rep_cur = evaluate_class(local_class, model, sentence_val, segment_val, label_val, batch_size, plot_rep, device)
        print("Task "+ str(task), ": The TIL accuracy is:" + str(eval_acc_til) + " The CIL acc is:" + str(eval_acc_cil))
        after.append(eval_acc_til)
        after_cil.append(eval_acc_cil)

        #Save some representations for plot
        num_plot = int(277 / 2 * (torch.sum(local_class).cpu().numpy())) 
        rep.extend(rep_cur[:num_plot])
        label.extend(np.array(label_val)[:num_plot, 0].tolist())
    
    if plot_rep:
        #randomly select 1000 samples for visualization
        index = np.random.RandomState(seed=1234).permutation(len(rep))
        vis_index = index[:1000]
        rep_plot = np.array(rep)[vis_index, :].tolist()
        label_plot = np.array(label)[vis_index].tolist()
        visualize(np.array(rep_plot), np.array(label_plot), filename = save_name + '.png')  

    avg_acc = np.mean(after)
    print("average accuracy is:" + str(avg_acc))
    avg_acc_cil = np.mean(after_cil)
    print("average class_incremental accuracy is:" + str(avg_acc_cil))
    if len(before) > 0:
        back_trans = np.sum(np.array(after) - np.array(before))/len(after)
        print("Backward transfer is:" + str(back_trans))