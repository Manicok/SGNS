import logging

from random import random
from math import sqrt
from os import listdir
from os.path import join
from xml import sax
from xml.sax import ContentHandler

from collections import Counter
from typing import List, Dict, Generator

logging.basicConfig(level=logging.INFO,
                    filename='voacl_log.txt',
                    encoding='utf8',
                    format='%(asctime)s - %(levelname)s : %(message)s')


class TextHandler(ContentHandler):
    '''
    用于从单个xml中提出文本并适当处理
    1.若未给出词表，则目标是提供词表
    2.若给了词表{word:freq}，则目标是根据词表给出训练数据（句子）
    '''
    def __init__(self, path: str, vocab: Dict[str, float]=None) -> None:
        super().__init__()
        self.path = path
        if vocab is None:
            self.vocab = Counter()
            self.sentences = None
        else:
            self.vocab = vocab
            self.sentences:List[List[str]] = []
        self.content = ''
        self.current = ''
        self.throw = False  # 用于处理一些需要丢弃的内容

    def startDocument(self) -> None:
        logging.info('Start processing Text: %s', self.path)
    
    def startElement(self, name: str, attrs) -> None:
        self.current = name
        if self.sentences is not None and name == 's':  # 一个新的句子
            self.new_sent = []
        
    def endElement(self, name: str) -> None:
        if self.sentences is None:
            if not self.throw and self.content:
                if name == "w":  # 更新词表
                    self.vocab[self.content] += 1

        else:
            if not self.throw: 
                if name == "w" and self.content in self.vocab.keys():  # 将词存入new_sent
                    prob = (sqrt(self.vocab[self.content] / 1e-3) + 1) * (1e-3 / self.vocab[self.content])  # 下采样，默认阈值1e-3
                    if random() < prob:
                        self.new_sent.append(self.content)
                elif name == 's':  # 存储句子序列
                    if len(self.new_sent) > 1:
                        self.sentences.append(self.new_sent)
                    del self.new_sent
        
    
    def characters(self, content: str) -> None:
        content = content.strip().lower()
        if content.isdigit():  # xml的数字属于 'w' 
            self.content = '<digit>'
        elif content == '(' and self.current == 'c':
            self.throw = True
            self.content = content
        elif content == ')' and self.current == 'c':
            self.throw = False
            self.content = content
        else:
            self.content = content
            
    def endDocument(self) -> None:
        pass
        
class BasicDataMaker:
    '''输入所有xml文件地址；给出词表，训练数据的生成器'''
    def __init__(self, paths: List[str], min_count: int=None, vocab_size: int=10000) -> None:
        self.min_count = int(.75*len(paths)) if not min_count else min_count
        self.vocab_size = vocab_size
        self.paths = paths
        
        self.vocab: List[str] 
        self.words_data: Dict[str, Dict[str, int]]
        self.get_WordsData(paths)
        
        self.sent_gen = self.sentence_generator
        
    def get_WordsData(self, paths: List[str]) -> None:
        v = Counter()
        parser = sax.make_parser()
        for path in paths:
            data = TextHandler(path)
            parser.setContentHandler(data)
            parser.parse(path)
            v += data.vocab
        v = v.most_common(self.vocab_size)
        v = {word: count for word, count in v if count >= self.min_count}
        
        count_sum = sum(v.values())
        weighted_sum = sum([count**.75 for count in v.values()])
        
        self.vocab = list(v.keys())
        self.words_data = {word: {'count': v[word], 
                                  'freq':v[word] / count_sum,
                                  'weighted_freq': v[word]**.75 / weighted_sum,
                                  'index':index}
                           for index, word in enumerate(self.vocab)}
    
    def sentence_generator(self) -> Generator[list[list[str]], None, None]:
        vocab_with_freq = {word: self.words_data[word]['freq']
                           for word in self.vocab}
        
        parser = sax.make_parser()
        for path in self.paths:
            data = TextHandler(path, vocab_with_freq)
            parser.setContentHandler(data)
            parser.parse(path)
            
            yield data.sentences
            
    
def test():
    dir_list = listdir('A0')
    paths = [join('A0', p) for p in dir_list]
    a = BasicDataMaker(paths=paths)
    v = a.vocab
    w = a.words_data
    s = a.sent_gen
    
    logging.info('++++> %s', v)
    logging.info('++++> %s', w)