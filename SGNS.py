import logging
import pickle
import time
import numpy as np
       
from typing import Callable, List, Dict, Generator, Tuple
from scipy.special import expit as sigmoid
from os import listdir
from os.path import join

from vocab import BasicDataMaker

def data():
    dir_list = listdir('A0')
    paths = [join('A0', p) for p in dir_list]
    data = BasicDataMaker(paths=paths)
    return data.vocab, data.words_data, data.sent_gen

class SGNS:
    def __init__(self, data:Callable, w_size = 2, wv_size = 100, neg_smaple = 4, learning_rate = 0.15) -> None:
        self.vocal:List[str]
        self.word_info: Dict[str, Dict[str, int|float]]
        self.sent_gen: Callable[[None],Generator[List[List[str]], None, None]]  # 给出一篇文章的所有句子
        self.vocal, self.word_info, self.sent_gen = data()

        self.window_size = w_size
        self.wv_size = wv_size
        self.W = np.random.uniform(low=-0.5/(wv_size**(3/4)), high=0.5/(wv_size**(3/4)), size=(len(self.vocal), wv_size))
        self.C = np.random.uniform(low=-0.5/(wv_size**(3/4)), high=0.5/(wv_size**(3/4)), size=(len(self.vocal), wv_size))
        self.learning_rate = learning_rate
        self.neg_sample_num = neg_smaple
        
    def traindata_generator(self,) -> Generator[Tuple[int,int,List[int]], None, None]:
        word_index = np.array([self.word_info[word]['index'] for word in self.vocal])
        word_weighted_freq = np.array([self.word_info[word]['weighted_freq'] for word in self.vocal])
        assert len(word_index) == len(word_weighted_freq)
        for sentences in self.sent_gen():
            for n, sent in enumerate(sentences):
                # logging.info('train on %s th sentence, %s remaining', n+1, len(sentences) - n - 1)
                for index, target in enumerate(sent):
                    pos_words = sent[max(0, index-self.window_size):min(len(sent)-1, index+self.window_size+1)]
                    while target in pos_words:
                        pos_words.remove(target)  
                    for pos_word in pos_words:
                        target_index = self.word_info[target]['index']
                        pos_word_index = self.word_info[pos_word]['index']
                        neg_word_index = np.delete(word_index, [target_index, pos_word_index])
                        loss_prob_weight = word_weighted_freq[target_index] + word_weighted_freq[pos_word_index] # 损失的概率质量
                        neg_word_weighted_freq = np.delete(word_weighted_freq, [target_index, pos_word_index]) + loss_prob_weight / (len(self.vocal) - 2)
                        neg_words = np.random.choice(neg_word_index, 
                                                        size=self.neg_sample_num,
                                                        replace=False,
                                                        p=neg_word_weighted_freq)
                        logging.info('data: %s', (target_index, pos_word_index, neg_words))
                        yield target_index, pos_word_index, neg_words
                        
    def train(self,):
        data_generator = self.traindata_generator()
        epoch_loss = []
        for t_index, p_index, n_index in data_generator:
            t_vector = self.W[t_index].copy()
            p_vector = self.C[p_index].copy()
            n_matrix = self.C[n_index]  # 传入list时默认为副本
            
            epsilon = 1e-5
            sigmoid_values = sigmoid(np.vstack((p_vector, -n_matrix)).dot(t_vector))
            clamped_values = np.clip(sigmoid_values, epsilon, 1 - epsilon)  # 防止log接受到0
            loss = - np.sum(np.log(clamped_values))
    
            epoch_loss.append(loss)
            
            G_pos_part = (sigmoid(p_vector.dot(t_vector)) - 1)
            G_negs_part = (sigmoid(n_matrix.dot(t_vector))).reshape(-1, 1)
            G_target = (G_pos_part)*p_vector + np.sum(G_negs_part*n_matrix, axis=0)
            
            self.C[p_index] -= self.learning_rate*(G_pos_part*t_vector)
            for i, index in enumerate(n_index):
                self.C[index] -= self.learning_rate*(G_negs_part[i]*t_vector)
            self.W[t_index] -= self.learning_rate*G_target
        
        return sum(epoch_loss) / len(epoch_loss)
            
            
def main():
    model = SGNS(data)
    loss = []
    
    for epoch in range(60):
        st = time.time()
        logging.info("epoch—%s", epoch+1)
        l = model.train()
        loss.append(l)
        en = time.time()
        print(f'epoch {epoch + 1} cost {en - st}s, loss: {l}')
        save_data = {
        'W': model.W,
        'C':model.C,
    }
        with open(f'data{epoch+1}.pkl', 'wb') as file:
            pickle.dump(save_data, file)
        
    save_data = {
        'word_info':model.word_info,
        'loss': loss
    }
    
    with open('data.pkl', 'wb') as file:
        pickle.dump(save_data, file)
        
if __name__ == '__main__':
    main()