import codecs
import numpy as np
import pandas as pd


# def process_files():
#     from gensim.models.keyedvectors import KeyedVectors
#
#     my_model = KeyedVectors.load_word2vec_format('embedding_all_GN300.txt')
#     f_train1 = codecs.open("dataset/train_docs.txt", encoding='utf-8')
#     f_train2 = open("dataset/train_labels_a.txt")
#     f_train3 = open("dataset/train_labels_p.txt")
#     f_test1 = codecs.open("dataset/test_docs.txt", encoding='utf-8')
#     f_test2 = open("dataset/test_labels_a.txt")
#     f_test3 = open("dataset/test_labels_p.txt")
#
#     train_x_text = [line.strip().lower() for line in f_train1]
#     print('calm')
#     train_y_t = [[int(i) for i in line.strip('\r\n').split()] for line in f_train2]
#     train_y_p = [[int(i) for i in line.strip('\r\n').split()] for line in f_train3]
#     dev_x_text = [line.strip().lower() for line in f_test1]
#     dev_y_t = [[int(i) for i in line.strip('\r\n').split()] for line in f_test2]
#     dev_y_p = [[int(i) for i in line.strip('\r\n').split()] for line in f_test3]
#
#     train_feature_x = [line_to_tensor(s.split(' '), my_model) for s in train_x_text]  # [L*1*dim]
#     dev_feature_x = [line_to_tensor(s.split(' '), my_model) for s in dev_x_text]
#
#     for i in range(len(dev_y_t)):
#         if not len(dev_y_t[i]) == len(dev_feature_x[i]):
#             print(dev_y_t[i])
#             print(i)
#             print(dev_x_text[i])
#
#     pickle.dump(train_feature_x, open('data/train_x_f.pkl', 'wb'))
#     pickle.dump(dev_feature_x, open('data/dev_x_f.pkl', 'wb'))
#     pickle.dump(train_y_t, open('data/train_y_t.pkl', 'wb'))
#     pickle.dump(train_y_p, open('data/train_y_p.pkl', 'wb'))
#     pickle.dump(dev_y_t, open('data/dev_y_t.pkl', 'wb'))
#     pickle.dump(dev_y_p, open('data/dev_y_p.pkl', 'wb'))


def load_w2v(filename):
    f_w2v = codecs.open(filename, encoding="utf-8").readlines()
    vocab_size = int(f_w2v[0].split()[0])
    embed_size = int(f_w2v[0].split()[1])
    # print("Vocab size: %d" % vocab_size)
    # print("Embed size: %d" % embed_size)
    word2index = dict()
    W = np.zeros(shape=(vocab_size, embed_size), dtype='float32')
    for i in range(1, vocab_size+1):
        line = f_w2v[i].strip('\r\n').split(' ')
        w = ''.join(line[0:len(line)-embed_size])
        vec = line[len(line)-embed_size:]
        vec = [float(v) for v in vec]
        # print(embed_size)
        # print(len(vec))
        # print(w)
        assert len(vec) == embed_size
        word2index[w] = i - 1
        W[i-1] = vec
    # print(len(W))
    # print(W[0])
    return W, word2index


def load_attr_data(filename):
    fo = codecs.open(filename, encoding='utf-8')
    test_text = []
    labels = []
    for line in fo:
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0]
        text = text.strip().split(' ')
        test_text.append(text)
        label = []
        for pair in splits[1:]:
            aspect = pair.split('#')[0]
            label.append(aspect)
        labels.append(label)

    return test_text, labels


def load_abp_data(filename, dev=False, folds=5):  # aspect_based polarity
    fo = codecs.open(filename, encoding='utf-8').readlines()

    length = len(fo)
    print(length)
    np.random.seed(1024)
    index_list = np.random.permutation(length).tolist()

    train_index = index_list[0:length - length // folds]
    dev_index = index_list[length - length // folds:]

    train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects = [], [], [], [], [], []
    for i in train_index:
        line = fo[i]
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0].strip().split(' ')
        for pair in splits[1:]:
            aspect, p = pair.split('#')
            train_texts.append(text)
            train_labels.append(p)
            train_aspects.append(aspect)

    for i in dev_index:
        line = fo[i]
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0].strip().split(' ')
        for pair in splits[1:]:
            aspect, p = pair.split('#')
            test_texts.append(text)
            test_labels.append(p)
            test_aspects.append(aspect)
    return train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects


def load_abp_raw(filename):  # aspect_based polarity
    fo = codecs.open(filename, encoding='utf-8').readlines()

    return fo


def load_test_data(filename):
    fo = codecs.open(filename, encoding='utf-8')
    test_text = []
    labels = []
    for line in fo:
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0].strip().split(' ')
        test_text.append(text)

    return test_text


def load_ab_test(f1, f2):
    f_sentence = codecs.open(f1, encoding='utf-8').readlines()
    f_aspect = codecs.open(f2, encoding='utf-8').readlines()
    test_texts = []
    test_aspects = []
    labels = []
    assert len(f_sentence) == len(f_aspect)
    for line1, line2 in zip(f_sentence, f_aspect):
        splits = line1.strip('\n').split('\t')
        text = splits[0].strip().split(' ')
        # text = text.lower()
        aspects = line2.strip('\n').split('|')
        if len(aspects) == 0:
            print('error for aspect reading')
            aspects = [None]
        for a in aspects:
            if a == '':
                a = '动力'
            test_texts.append(text)
            test_aspects.append(a)
    assert len(test_texts) == len(test_aspects)

    return test_texts, test_aspects



def load_pos(ds):
    f_train = codecs.open("data/%s/train_pos.txt" % ds, encoding='utf-8')
    f_test = codecs.open("data/%s/test_pos.txt" % ds, encoding='utf-8')
    f_dict = codecs.open("data/%s/pos2id.txt" % ds, encoding='utf-8')

    pos2id = {}
    for line in f_dict:
        splits = line.strip('\r\n').split(' ')
        pos2id[splits[0]] = int(splits[1])
    train_pos = []
    test_pos = []
    for line in f_train:
        splits = line.strip('\r\n').split(' ')
        pos_label = [pos2id[s] for s in splits]
        train_pos.append(pos_label)
    for line in f_test:
        splits = line.strip('\r\n').split(' ')
        pos_label = [pos2id[s] for s in splits]
        test_pos.append(pos_label)
    return train_pos, test_pos, pos2id


def load_char2id(ds):
    f_dict = codecs.open("data/%s/char2id.txt" % ds, encoding='utf-8')

    char2id = dict()
    max_c_len = 0
    for line in f_dict:
        splits = line.strip('\r\n').split(' ')
        if len(splits) > 1:
            char2id[splits[0]] = int(splits[1])
        else:
            max_c_len = int(splits[0])

    return char2id, max_c_len


def generate_sentence_label(train_texts, train_ow):  # combine all ow labels for one sentence
    train_s_texts = []
    train_s_ow = []
    prev_text = ''
    train_s_t = []
    for i in range(len(train_texts)):
        if train_texts[i] != prev_text:
            prev_text = train_texts[i]
            train_s_texts.append(train_texts[i])
            train_s_ow.append([train_ow[i]])
        else:
            train_s_ow[-1].append(train_ow[i])
    print(len(train_s_texts))
    new_s_ow = []
    for t, o in zip(train_s_texts, train_s_ow):
        train_s_t.append([0 for i in range(len(t))])
        oarray = np.asarray(o)
        new_ow = oarray.max(axis=0).tolist()
        new_s_ow.append(new_ow)
        # print(str(t)+'\t'+str(o) + '\t' + str(new_ow))
    return train_s_texts, new_s_ow, new_s_ow


def parse_json(filename):
    entity_list = []
    df = pd.read_json(filename, encoding='utf-8')
    df = df['value']
    for d in df:
        entity_list.append(d['attribute2'])
    # entity_list.append('')
    entity_dict = {k: v for v, k in enumerate(entity_list)}
    return entity_list, entity_dict


if __name__ == '__main__':
    load_pos('14res')
