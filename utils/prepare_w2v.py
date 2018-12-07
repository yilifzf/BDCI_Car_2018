import codecs
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle

# w2v_model = KeyedVectors.load_word2vec_format('E:/embedding/vec_clear.txt')
# w2v_model = KeyedVectors.load_word2vec_format('E:/embedding/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5')
# w2v_model = KeyedVectors.load_word2vec_format('E:/embedding/sgns.merge.word')

from gensim.models.wrappers import FastText


def prepare_w2v(ds=None):
    w2v_model = KeyedVectors.load_word2vec_format('../embedding/cc.zh.300.bin')
    # w2v_model = FastText.load_fasttext_format('../embedding/cc.zh.300.bin')
    print("Finish Load")
    dim = len(w2v_model['好'])
    print(dim)
    # exit(-1)
    f1 = codecs.open("../data/train.txt", encoding='utf-8')
    f2 = codecs.open("../data/test.txt", encoding='utf-8')
    f_list = [f1, f2]
    fw1 = codecs.open("../embedding/embedding_all_fasttext_%d.txt" % (dim), 'w', encoding='utf-8')
    # fw2 = codecs.open("embedding_test_GN300.txt", 'w', encoding='utf-8')

    all_set = set()

    for f in f_list:
        for line in f:
            line = line.strip('\n').split('\t')[0]
            words = line.split(' ')
            for w in words:
                # if w == '' or w == "  " or w == ' ':
                #     w = 'UNK'
                # if re.findall(r"[A-Za-z0-9]", w):
                #     w = w.translate(str.maketrans('', '', string.punctuation))
                all_set.add(w)
    in_set = set()
    miss = 0
    word_list = list(all_set)
    vocab_dict = {x: i for i, x in enumerate(word_list)}
    print(len(word_list))
    # for w in all_set:
    #     if w in w2v_model:
    #         in_set.add(w)
    #     else:
    #         miss += 1
    #         print(w)
    #         # exit(-1)
    # print(len(all_set))
    # print(len(in_set))
    # print("miss:%d" % miss)
    embedding_matrix = np.zeros((len(all_set), dim))
    for index, w in enumerate(word_list):
        if index % 1000 == 0:
            print(index)
        try:
            # in_set.add(w)
            embeds = np.asarray(w2v_model[w])
        except:
            miss += 1
            print(w)
            embeds = np.random.uniform(-0.25, 0.25, dim)
        embedding_matrix[index] = embeds

    fw1.write(str(len(all_set)) + ' ' + str(dim)+'\n')
    for index, w in enumerate(word_list):
        fw1.write(w)
        for i in embedding_matrix[index]:
            fw1.write(' ' + str(i))
        fw1.write('\n')
    pickle.dump(vocab_dict, open('../data2/vocabulary.pkl', 'wb'))
    print(len(all_set))
    print("miss:%d" % miss)


def load_vocab():
    w2v_model = KeyedVectors.load_word2vec_format('../embedding/qiche_300d.txt', binary=False)
    print("Finish Load")
    dim = len(w2v_model['好'])
    fw1 = codecs.open("../embedding/embedding_all_qiche2_%d.txt" % (dim), 'w', encoding='utf-8')
    vocab_dict = pickle.load(open('../data/vocabulary.pkl', 'rb'))
    word_list = ['unk' for i in range(len(vocab_dict))]
    for k, v in vocab_dict.items():
        word_list[v] = k
    print(word_list)
    embedding_matrix = np.zeros((len(vocab_dict), dim))
    miss = 0
    for index, w in enumerate(word_list):
        if index % 1000 == 0:
            print(index)
        try:
            # in_set.add(w)
            embeds = np.asarray(w2v_model[w])
        except:
            miss += 1
            print(w)
            embeds = np.random.uniform(-0.25, 0.25, dim)
        embedding_matrix[index] = embeds

    fw1.write(str(len(word_list)) + ' ' + str(dim)+'\n')
    for index, w in enumerate(word_list):
        fw1.write(w)
        for i in embedding_matrix[index]:
            fw1.write(' ' + str(i))
        fw1.write('\n')
    pickle.dump(vocab_dict, open('../data/vocabulary2.pkl', 'wb'))
    print(len(word_list))
    print("miss:%d" % miss)


def load_ft():
    w2v_model = FastText.load_fasttext_format('../embedding/cc.zh.300.bin')
    print("Finish Load")
    dim = len(w2v_model['好'])
    fw1 = codecs.open("../embedding/embedding_all_ftoov_%d.txt" % (dim), 'w', encoding='utf-8')
    vocab_dict = pickle.load(open('../data/vocabulary.pkl', 'rb'))
    word_list = ['unk' for i in range(len(vocab_dict))]
    for k, v in vocab_dict.items():
        word_list[v] = k
    # print(word_list)
    embedding_matrix = np.zeros((len(vocab_dict), dim))
    miss = 0
    for index, w in enumerate(word_list):
        if index % 1000 == 0:
            print(index)
        try:
            # in_set.add(w)
            embeds = np.asarray(w2v_model[w])
        except:
            w2v_model.most_similar(w)
            miss += 1
            print(w)
            embeds = np.random.uniform(-0.25, 0.25, dim)
        embedding_matrix[index] = embeds

    fw1.write(str(len(word_list)) + ' ' + str(dim)+'\n')
    for index, w in enumerate(word_list):
        fw1.write(w)
        for i in embedding_matrix[index]:
            fw1.write(' ' + str(i))
        fw1.write('\n')
    pickle.dump(vocab_dict, open('../data/vocabulary2.pkl', 'wb'))
    print(len(word_list))
    print("miss:%d" % miss)


def test_miss():
    w2v_model = KeyedVectors.load_word2vec_format('../embedding/cc.zh.300.vec', binary=False)
    print("Finish Load")
    dim = len(w2v_model['好'])
    # fw1 = codecs.open("../embedding/embedding_all_fasttext2_%d.txt" % (dim), 'w', encoding='utf-8')
    vocab_dict = pickle.load(open('../data/vocabulary.pkl', 'rb'))
    word_list = ['unk' for i in range(len(vocab_dict))]
    for k, v in vocab_dict.items():
        word_list[v] = k
    print(word_list)
    embedding_matrix = np.zeros((len(vocab_dict), dim))
    miss = 0
    for index, w in enumerate(word_list):
        if index % 1000 == 0:
            print(index)
        try:
            # in_set.add(w)
            embeds = np.asarray(w2v_model[w])
        except:
            miss += 1
            print(w)
            embeds = np.random.uniform(-0.25, 0.25, dim)
        embedding_matrix[index] = embeds
    print(len(word_list))
    print("miss:%d" % miss)


def test():
    w2v_model = KeyedVectors.load_word2vec_format('data/%s/embedding_all_glove300.txt' % ds)
    print(w2v_model.most_similar("unhappy"))


if __name__ == '__main__':
    # main()
    # prepare_w2v('16res')
    prepare_w2v()
    # load_ft()
    # load_vocab()
    # test_miss()
    # test('16res')
