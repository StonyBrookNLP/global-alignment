import numpy as np
import torch
import random
from utils.helper import batchify

class ReplayMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, buffer=None):

        if buffer is None:
            self.memory = {}
        else:
            self.memory = buffer

        self.keys_num = 0
    
    def update_keys(self):
        self.keys_num = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(self.keys_num, 768)
        return
                
    def push(self, keys, examples, sample_rate = 0.1):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        sentences, segments, labels, local_class = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            random_num = np.random.random_sample()
            if random_num <= sample_rate:
                # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key
                self.memory.update(
                    {key.tobytes(): (sentences[i], segments[i], labels[i], local_class)})
        return

    #extract sentence, segments, etc from the stored experience
    def prepare_batch(self, sample):
        sentences = []
        segments = []
        labels = []
        local_classes = []
        # Iterate over experiences
        for sentence, segment, label, local_class in sample:
            sentences.append(sentence)
            segments.append(segment)
            labels.append(label)
            local_classes.append(local_class)
        batch_sentence, batch_segment, batch_mask, batch_label = batchify(sentences, segments, labels)
        local_classes = torch.LongTensor(local_classes)
        return (batch_sentence, batch_segment, batch_mask, batch_label, local_classes)
    
    def sample_batch(self, sample_size):
        if sample_size > self.keys_num:
            sample_size = self.keys_num
        keys = random.sample(list(self.memory),sample_size)
        samples = [self.memory[key] for key in keys]
        return self.prepare_batch(samples)
    
    #Find neighbors for MBPA
    def get_neighbors(self, keys, k=32):
        """
        Returns samples from buffer using nearest neighbour
        """
        #keys: B, 1, C
        #all_keys: num, C
        key_neigh = []
        batches = []
        # Iterate over all the input keys
        # to find neigbours for each of them
        all_keys = torch.Tensor(self.all_keys).float().to(keys.device)
        similarity_scores = keys @ all_keys.transpose(0,1) #B, 1, num
        similarity_scores = similarity_scores.data.cpu().numpy()
        
        for i in range(len(similarity_scores)):
            # compute similarity scores
            similarity_tmp = similarity_scores[i][0] #num
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_tmp, -k)[-k:]]
            key_neigh.append(torch.Tensor(K_neighbour_keys))
            
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]
            # converts experiences into batch
            batch = self.prepare_batch(neighbours)
            batches.append(batch)
        return key_neigh, batches