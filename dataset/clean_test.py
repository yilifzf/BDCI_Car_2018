import codecs
import pandas as pd
import numpy as np
# import jieba
from pyhanlp import HanLP
# from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import os
# jieba.load_userdict("car_dict.txt")
# from pyltp import Segmentor
# LTP_ROOT = "/home/user_data/wangr/.data/ltp/ltp_data_v3.4.0/"
# LTP_CWS = os.path.join(LTP_ROOT, "cws.model")
# segmentor = Segmentor()
# segmentor.load_with_lexicon(LTP_CWS, 'car_dict.txt')

lines = codecs.open("../dataset/test_public_2.csv", encoding='utf-8').readlines()
fw = codecs.open("../data/test.txt", 'w', encoding='utf-8')
elements = len(lines[0].split(','))
print(elements)

instances = {}


for i in range(1, len(lines)):
    line = lines[i]
    splits = line.strip('\n').split(',')
    assert len(splits) == elements
    text = ' '.join([l.word for l in HanLP.segment(splits[1].strip())])
    # text = segmenter.segment(splits[1].strip())
    text = ' '.join(text.split())
    # attr = splits[2]
    # polarity = splits[3]
    # if polarity == '':
    #     polarity = '0'
    if text not in instances:
        instances[text] = []
    # if polarity != '':
    # instances[text].append(attr+'#'+polarity)

zero_sentences = 0
multi_sentences = 0
all_sentences = 0
max_attributes = 0
multi_set = []

from collections import Counter

for i in instances:
    # print(Counter(instances[i]).most_common(1)[0])
    fw.write(i + '\t' + '\t'.join(instances[i]) + '\n')
    # if len(instances[i]) == 0:
    #     zero_sentences += 1
    # elif len(instances[i]) > 1:
    #     # print(instances[i])
    #     multi_set.append('|'.join(instances[i]))
    #     multi_sentences += 1
    # max_attributes = len(instances[i]) if len(instances[i]) > max_attributes else max_attributes
    # all_sentences += 1


print(multi_set)
multi_counter = Counter(multi_set)
print(multi_counter)

print("number of sentences without attributes: %d" % zero_sentences)
print("number of sentences with multiple attributes: %d" % multi_sentences)
print("number of sentences: %d" % all_sentences)
print(max_attributes)