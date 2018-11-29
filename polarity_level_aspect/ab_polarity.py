import codecs
import sys

sys.path.append("..")
from utils.data_helper import load_attr_data, load_w2v, load_ab_test, load_abp_data, parse_json, load_abp_raw

import polarity_level_aspect.networks as networks
import utils.train_single as train_single
from utils.Data import Data, Data2, Data3
from utils.evaluate import score2

import argparse
import numpy as np
import torch
from collections import Counter
import os
import shutil
import time
import pickle

from sklearn.linear_model import LogisticRegression
# from mlxtend.classifier import StackingCVClassifier, StackingClassifier
# import xgboost as xgb
# from lightgbm import LGBMClassifier


parser = argparse.ArgumentParser()
parser.add_argument("--EPOCHS", type=int, default=5)
parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--optimizer", type=str,  default="Adam")

parser.add_argument("--model", type=str, default="HEAT")

parser.add_argument("--lr", type=float, default=0.2)
parser.add_argument("--freeze", type=bool, default=True)
parser.add_argument("--use_dev", type=bool, default=False)
parser.add_argument("--use_elmo", type=int, default=0)
parser.add_argument("--save", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--pretrain", type=int, default=1)
parser.add_argument("--check_dir", type=str, default="cp_New")

parser.add_argument("--mode", type=int, default=2)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--torch_seed", type=int, default=42)

# parser.add_argument("--test_model", type=str, default="TD_3LSTM_0.7626.pt")
parser.add_argument("--test_dir", type=str, default="cp_HEAT_0#cp_AT_LSTM_0#cp_HEAT_ft2#cp_AT_LSTM_ft2#cp_HEAT_2#cp_AT_LSTM_2#cp_HEAT_tc#cp_AT_LSTM_tc#cp_GCAE_0#cp_GCAE_2#cp_GCAE_ft2#cp_GCAE_tc#cp_Bert")
parser.add_argument("--saved", type=int, default=1)

parser.add_argument("--train_mode", type=int, default=1)
parser.add_argument("--w2v", type=str, default="merge")
args = parser.parse_args()

# torch.set_printoptions(profile="full")
print(args)

# seed = 314159
torch.manual_seed(args.torch_seed)
# seed = torch.initial_seed()
# print(seed)


class Classifier:  # Neural network method
    def __init__(self):
        self.classifier = None
        self.trained = False
        pass

    def train_from_data(self, train_raw_data, test_raw_data, W, word2index, polarity_dict, aspect_dict, args, Folds=0):

        word_embed_dim = W.shape[1]
        hidden_size = args.n_hidden
        vocab_size = len(W)
        output_size = len(polarity_dict)
        aspect_size = len(aspect_dict)

        if args.model == 'LSTM':
            self.classifier = networks.LSTM(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'Average_LSTM':
            self.classifier = networks.Average_LSTM(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'CNN':
            self.classifier = networks.CNN(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'AT_LSTM':
            self.classifier = networks.AT_LSTM(word_embed_dim, output_size, vocab_size, aspect_size, args)
        elif args.model == 'ATAE_LSTM':
            self.classifier = networks.ATAE_LSTM(word_embed_dim, output_size, vocab_size, aspect_size, args)
        elif args.model == 'GCAE':
            self.classifier = networks.GCAE(word_embed_dim, output_size, vocab_size, aspect_size, args)
        elif args.model == 'HEAT':
            self.classifier = networks.HEAT(word_embed_dim, output_size, vocab_size, aspect_size, args)
        train_elmo, test_elmo = [], []

        if args.use_elmo != 0:
            import h5py
            elmo_dict = h5py.File('../embedding/embeddings_elmo_ly-1.hdf5', 'r')
            for s in train_raw_data[0]:
                sentence = '\t'.join(s)
                sentence = sentence.replace('.', '$period$')
                sentence = sentence.replace('/', '$backslash$')
                # print(sentence)
                embeddings = torch.from_numpy(np.asarray(elmo_dict[sentence]))
                train_elmo.append(embeddings)
            for s in test_raw_data[0]:
                sentence = '\t'.join(s)
                sentence = sentence.replace('.', '$period$')
                sentence = sentence.replace('/', '$backslash$')
                embeddings = torch.from_numpy(np.asarray(elmo_dict[sentence]))
                test_elmo.append(embeddings)
            elmo_dict.close()
            print("finish elmo")
        if args.pretrain != 0:
            aspect_e_l = np.zeros((aspect_size, word_embed_dim))
            for a in aspect_dict:
                a_i = aspect_dict[a]
                # print(a)
                if a == '舒适性':
                    a = '舒适'
                a_e = W[word2index[a]]
                aspect_e_l[a_i] = a_e
            aspect_embeds = torch.from_numpy(aspect_e_l).float()
            # print(aspect_embeds)
            print("initial aspect")
            # print(attr_dict)
            self.classifier.AE.weight = torch.nn.Parameter(aspect_embeds)

        if args.train_mode == 1:
            train_data = Data3(train_raw_data, word2index, polarity_dict, args, target_dict=aspect_dict)
            # if args.use_dev:
            #     dev_data = Data(args, dev_input_s, dev_input_t, dev_y_tensor)
            # else:
            #     dev_data = None
            test_data = Data3(test_raw_data, word2index, polarity_dict, args, target_dict=aspect_dict)
            if args.use_elmo != 0:
                train_data.add_feature(train_elmo)
                test_data.add_feature(test_elmo)

            best_dict, max_acc = train_single.train(self.classifier, train_data, test_data, test_data, polarity_dict, W, args=args)
            best_model = "%s/checkpoint_%s_%.6f_%d.pt" % (args.check_dir, args.model, max_acc, Folds)
            if args.save != 0:
                torch.save(best_dict, best_model)
        pass

    def split_dev(self, train_texts, train_t, train_ow):
        instances_index = []
        curr_s = ""
        curr_i = -1
        for i, s in enumerate(train_texts):
            s = ' '.join(s)

            if s == curr_s:
                instances_index[curr_i].append(i)
            else:
                curr_s = s
                instances_index.append([i])
                curr_i += 1
        print(curr_i)
        print(len(instances_index))
        assert curr_i+1 == len(instances_index)
        length = len(instances_index)
        np.random.seed(1024)
        index_list = np.random.permutation(length).tolist()
        # np.random.shuffle(index_list)
        train_index = [instances_index[i] for i in index_list[0:length-length//5]]
        dev_index = [instances_index[i] for i in index_list[length-length//5:]]
        train_i_index = [i for l in train_index for i in l]
        dev_i_index = [i for l in dev_index for i in l]
        dev_texts, dev_t, dev_ow = ([train_texts[i] for i in dev_i_index], [train_t[i] for i in dev_i_index],
                                    [train_ow[i] for i in dev_i_index])
        train_texts, train_t, train_ow = ([train_texts[i] for i in train_i_index], [train_t[i] for i in train_i_index],
                                          [train_ow[i] for i in train_i_index])
        return train_texts, train_t, train_ow, dev_texts, dev_t, dev_ow

    def predict(self, rnn, test_raw_data, word2index, args):
        test_texts = test_raw_data[0]
        test_t = test_raw_data[1]
        test_ow = test_raw_data[2]

        test_input_s = [self.to_tensor(s, word2index) for s in test_texts]
        # print(train_input_s[0])
        test_elmo = []
        if args.use_elmo:
            import h5py
            elmo_dict = h5py.File('data/%s/elmo_layers.hdf5' % args.ds, 'r')
            for i, current_sentence in enumerate(test_texts):
                current_sentence = ' '.join(current_sentence)
                embeddings = torch.from_numpy(np.asarray(elmo_dict[current_sentence]))
                test_elmo.append(embeddings)
            elmo_dict.close()

        # print(train_input_s[0])
        test_input_t = [torch.LongTensor(t) for t in test_t]
        test_y_tensor = [torch.LongTensor(y) for y in test_ow]
        test_data = Data(args, test_input_s, test_input_t, test_y_tensor, features=test_elmo)
        with torch.no_grad():
            test_predict = predict(rnn, test_data, args)
        pred_acc_t = score(test_predict, test_data.labels)
        print("p:%.4f, r:%.4f, f:%.4f" % (pred_acc_t[0], pred_acc_t[1], pred_acc_t[2]))
        return test_predict


def kfold_split(length, k=5):
    np.random.seed(args.seed)
    index_list = np.random.permutation(length)

    l = length // k
    folds = []
    for i in range(k):
        test_idx = np.zeros(length, dtype=bool)
        test_idx[i*l:(i+1)*l] = True
        folds.append((index_list[~test_idx], index_list[test_idx]))
    return folds


def splits(fo, train_index, dev_index):
    train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects = [], [], [], [], [], []
    for i in train_index:
        line = fo[i]
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0].strip().split(' ')
        for pair in splits[1:]:
            aspect = pair.split('#')[0]
            p = pair.split('#')[1]
            train_texts.append(text)
            train_labels.append(p)
            train_aspects.append(aspect)

    for i in dev_index:
        line = fo[i]
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0].strip().split(' ')
        for pair in splits[1:]:
            aspect = pair.split('#')[0]
            p = pair.split('#')[1]
            test_texts.append(text)
            test_labels.append(p)
            test_aspects.append(aspect)
    return train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects


def count_instance(fo):
    count = 0
    index_list = []
    for line in fo:
        current_index = []
        splits = line.strip('\n').split('\t')
        for p in splits[1:]:
            assert '#' in p
            current_index.append(count)
            count += 1
        index_list.append(current_index)
    return count, index_list


def ensemble():
    f_train = "../data/train.txt"
    if args.w2v == "merge":
        f_w2v = "../embedding/embedding_all_merge_300.txt"
    elif args.w2v == "fasttext2":
        f_w2v = "../embedding/embedding_all_fasttext2_300.txt"
    elif args.w2v == "tencent":
        f_w2v = "../embedding/embedding_all_tencent_200.txt"
    else:
        print("error, no embedding")
        exit(-1)
    f_dict1 = "../dataset/polarity.json"
    f_dict2 = "../dataset/attribute.json"
    print(f_train)
    print(f_w2v)
    if not os.path.exists("%s" % args.check_dir):
        os.mkdir("%s" % args.check_dir)
    W, word2index2 = load_w2v(f_w2v)
    word2index = pickle.load(open("../data/vocabulary.pkl", 'rb'))
    assert word2index == word2index2
    polarity_list, polarity_dict = parse_json(f_dict1)
    attr_list, attr_dict = parse_json(f_dict2)
    kf = 0
    fo = load_abp_raw(f_train)
    for train_index, test_index in kfold_split(len(fo), args.folds):
        kf += 1
        print("FOLD:", kf)
        # print("TRAIN:", train_index, '\n', "TEST:", test_index, str(len(test_index)))
        train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects = splits(fo, train_index, test_index)
        print(len(train_texts))
        print(len(test_texts))
        # print(list(attr_dict.keys()))
        model = Classifier()
        print(attr_list)
        print(attr_dict)
        # exit(-1)
        # print(train_texts)
        model.train_from_data((train_texts, train_labels, train_aspects), (test_texts, test_labels, test_aspects),
                              W, word2index, polarity_dict, attr_dict, args, kf)


def main():
    f_train = "../data/train.txt"
    f_test = "data/test_p.txt"
    if args.w2v == "merge":
        f_w2v = "../embedding/embedding_all_merge_300.txt"
    elif args.w2v == "fasttext":
        f_w2v = "../embedding/embedding_all_fasttext_300.txt"
    elif args.w2v == "fasttext2":
        f_w2v = "../embedding/embedding_all_fasttext2_300.txt"
    elif args.w2v == "tencent":
        f_w2v = "../embedding/embedding_all_tencent_200.txt"
    else:
        print("error, no embedding")
        exit(-1)
    f_dict1 = "../dataset/polarity.json"
    f_dict2 = "../dataset/attribute.json"
    print(f_w2v)
    # train_texts, train_labels = load_attr_data(filename=f_train)
    # # test_text, test_labels = load_attr_data(filename=f_test)
    # train_texts, train_labels, test_texts, test_labels = split_dev(train_texts, train_labels)
    train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects = load_abp_data(f_train, folds=5)
    if not os.path.exists("%s" % args.check_dir):
        os.mkdir("%s" % args.check_dir)
    print(len(train_texts))
    print(len(test_texts))
    W, word2index2 = load_w2v(f_w2v)
    word2index = pickle.load(open("../data/vocabulary.pkl", 'rb'))
    assert word2index == word2index2
    polarity_list, polarity_dict = parse_json(f_dict1)
    attr_list, attr_dict = parse_json(f_dict2)
    # print(list(attr_dict.keys()))
    model = Classifier()
    print(polarity_list)
    print(polarity_dict)
    # exit(-1)
    # print(train_texts)
    model.train_from_data((train_texts, train_labels, train_aspects), (test_texts, test_labels, test_aspects),
                          W, word2index, polarity_dict, attr_dict, args)


def test():
    # model = Classifier()
    test_file1 = "../attribute_level/data/attribute_test.txt"
    test_file2 = "../attribute_level/test_predict.txt"
    test_texts, test_aspects = load_ab_test(test_file1, test_file2)
    f_w2v = "../embedding/embedding_all_merge_300.txt"
    W, word2index = load_w2v(f_w2v)

    f_dict1 = "../dataset/polarity.json"
    f_dict2 = "../dataset/attribute.json"
    polarity_list, polarity_dict = parse_json(f_dict1)
    attr_list, attr_dict = parse_json(f_dict2)

    assert len(test_texts) == len(test_aspects)

    files = [
        "checkpoint_HEAT_0.7189.pt",
        "checkpoint_HEAT_0.7062.pt"
    ]

    predicts = []
    for check_point in files:
        predict = []
        classifier = torch.load(check_point)
        for text, aspect in zip(test_texts, test_aspects):
            if aspect != '':
                if aspect is None:
                    print("error")
                test_data = Data3(([text], [None], [aspect]), word2index, polarity_dict, args, target_dict=attr_dict)
                test_predict = train_single.predict(classifier, test_data, args)
                assert len(test_predict) == 1
                polarity = str(test_predict[0].item()-1)
            else:
                print(aspect)
                print(text)
                polarity = '0'
            # fw.write(aspect+','+polarity+'\n')
            predict.append(aspect+','+polarity)
        predicts.append(predict)
    print(len(predicts))
    print(len(predicts[0]))
    fw = codecs.open("test_predict_polarity_ensemble.txt", 'w', encoding='utf-8')

    for j in range(len(predicts[0])):
        votes = [predicts[i][j] for i in range(len(predicts))]
        voted = Counter(votes).most_common(1)
        fw.write(voted+'\n')


def load_elmo(test_texts):
    test_elmo = []
    import h5py
    elmo_dict = h5py.File('../embedding/embeddings_elmo_ly-1.hdf5', 'r')
    for s in test_texts:
        sentence = '\t'.join(s)
        sentence = sentence.replace('.', '$period$')
        sentence = sentence.replace('/', '$backslash$')
        embeddings = torch.from_numpy(np.asarray(elmo_dict[sentence]))
        test_elmo.append(embeddings)
    elmo_dict.close()
    print("finish elmo")
    return test_elmo


def get_oof(clfs, fo, test_data, word2index, polarity_dict, attr_dict):
    NFOLDS = len(clfs)
    n_train, sentence2instance = count_instance(fo)
    print(n_train)
    n_test = len(test_data.sentences)
    class_num = 3
    oof_train = np.zeros((n_train, class_num))
    oof_train_y = np.zeros((n_train, 1))
    oof_test = np.zeros((n_test, class_num))
    oof_test_skf = np.zeros((NFOLDS, n_test, class_num))

    kf = 0
    for (train_index, test_index), checkpoint in zip(kfold_split(len(fo), NFOLDS), clfs):
        print(checkpoint)
        clf = torch.load(checkpoint)
        kf += 1
        print("FOLD:", kf)
        print("TRAIN:", str(len(train_index)), "TEST:", str(len(test_index)))
        # train_index, test_index = train_index.tolist(), test_index.tolist()
        train_texts, train_labels, train_aspects, dev_texts, dev_labels, dev_aspects = splits(fo, train_index,
                                                                                                 test_index)
        dev_data = Data3((dev_texts, dev_labels, dev_aspects), word2index, polarity_dict, args, target_dict=attr_dict)
        if args.use_elmo != 0:
            dev_elmo = load_elmo(dev_texts)
            dev_data.add_feature(dev_elmo)
        with torch.no_grad():
            dev_predict, oof_dev = train_single.predict_with_logit(clf, dev_data, args)
        pred_acc_p = score2(dev_predict, dev_data.labels)
        print("[p:%.4f, r:%.4f, f:%.4f] acc:%.4f" %
              (pred_acc_p[0], pred_acc_p[1], pred_acc_p[2], pred_acc_p[3]))
        # label_prf = label_analysis(dev_predict, dev_data.labels)
        # for i in range(len(label_prf)):
        #     print("%s : [%.4f, %.4f, %.4f] %.4f" %
        #           (list(attr_dict.keys())[i], label_prf[i][0], label_prf[i][1], label_prf[i][2], label_prf[i][3]))
        test_i_index = [i_index for sentence_index in test_index for i_index in sentence2instance[sentence_index]]
        assert len(test_i_index) == len(oof_dev)
        oof_train[test_i_index] = oof_dev
        dev_y = [l.detach().numpy() for l in dev_data.labels]
        oof_train_y[test_i_index] = dev_y
        _, oof_test_skf[kf - 1, :, :] = train_single.predict_with_logit(clf, test_data, args)
    oof_test[:] = oof_test_skf.mean(axis=0)
    dir = os.path.dirname(clfs[0])
    if not os.path.exists(os.path.join(dir, 'npy')):
        os.mkdir(os.path.join(dir, 'npy'))
    print(dir)
    np.save(os.path.join(dir, 'npy', "oof_train"), oof_train)
    np.save(os.path.join(dir, 'npy', "oof_train_y"), oof_train_y)
    np.save(os.path.join(dir, 'npy', "oof_test"), oof_test)
    return oof_train, oof_train_y, oof_test


def get_oof_test(clfs, test_data):
    NFOLDS = len(clfs)
    n_test = len(test_data.sentences)
    class_num = 3
    oof_test = np.zeros((n_test, class_num))
    oof_test_skf = np.zeros((NFOLDS, n_test, class_num))
    kf = 0
    for checkpoint in clfs:
        print(checkpoint)
        clf = torch.load(checkpoint)
        kf += 1
        print("FOLD:", kf)
        _, oof_test_skf[kf - 1, :, :] = train_single.predict_with_logit(clf, test_data, args)
    oof_test[:] = oof_test_skf.mean(axis=0)
    dir = os.path.dirname(clfs[0])
    if not os.path.exists(os.path.join(dir, 'npy')):
        os.mkdir(os.path.join(dir, 'npy'))
    print(dir)
    np.save(os.path.join(dir, 'npy', "oof_test"), oof_test)
    return oof_test


def load_oof_dir(dir):
    oof_train = np.load(os.path.join(dir, 'npy', "oof_train.npy"))
    oof_train_y = np.load(os.path.join(dir, 'npy', "oof_train_y.npy"))
    oof_test = np.load(os.path.join(dir, 'npy', "oof_test.npy"))
    print("loaded from: " + dir)
    return oof_train, oof_train_y, oof_test


def load_oof_test(dir):
    oof_test = np.load(os.path.join(dir, 'npy', "oof_test.npy"))
    print("loaded from: " + dir)
    return oof_test


def load_oof(clfs, fo, test_data, word2index, polarity_dict, attr_dict):
    dir = os.path.dirname(clfs[0])
    if os.path.isfile(os.path.join(dir, 'npy', "oof_train.npy")):
        oof_train = np.load(os.path.join(dir, 'npy', "oof_train.npy"))
        oof_train_y = np.load(os.path.join(dir, 'npy', "oof_train_y.npy"))
        oof_test = np.load(os.path.join(dir, 'npy', "oof_test.npy"))
        print("loaded from: " + dir)
    else:
        oof_train, oof_train_y, oof_test = get_oof(clfs, fo, test_data, word2index, polarity_dict=polarity_dict,
                                                   attr_dict=attr_dict)
    return oof_train, oof_train_y, oof_test


def load_oof3(clfs, fo, test_data, word2index, polarity_dict, attr_dict):  # re test
    dir = os.path.dirname(clfs[0])
    if os.path.isfile(os.path.join(dir, 'npy', "oof_train.npy")):
        oof_train = np.load(os.path.join(dir, 'npy', "oof_train.npy"))
        oof_train_y = np.load(os.path.join(dir, 'npy', "oof_train_y.npy"))
        oof_test = get_oof_test(clfs, test_data)
        print("loaded from: " + dir)
    else:
        oof_train, oof_train_y, oof_test = get_oof(clfs, fo, test_data, word2index, polarity_dict=polarity_dict,
                                                   attr_dict=attr_dict)
    return oof_train, oof_train_y, oof_test


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    print(e_x.shape)
    return e_x / e_x.sum(axis=-1, keepdims=True)


def stacking():
    # saved = True if args.saved != 0 else False
    saved = args.saved
    f_train = "../data/train.txt"
    test_file1 = "../data/test.txt"
    test_file2 = "../data/test_predict_aspect_ensemble.txt"
    test_texts, test_aspects = load_ab_test(test_file1, test_file2)
    # print(test_aspects)

    fo = load_abp_raw(f_train)
    word2index = pickle.load(open("../data/vocabulary.pkl", 'rb'))

    f_dict = "../dataset/polarity.json"
    polarity_list, polarity_dict = parse_json(f_dict)
    f_dict2 = "../dataset/attribute.json"
    attr_list, attr_dict = parse_json(f_dict2)

    paths = args.test_dir.split('#')
    models_files = []
    for path in paths:
        models_files.append([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    test_data = Data3((test_texts, None, test_aspects), word2index, polarity_dict, args, target_dict=attr_dict)
    if args.use_elmo != 0:
        test_elmo = load_elmo(test_texts)
        test_data.add_feature(test_elmo)

    x_train = []
    y_train = []
    x_test = []
    for dir, checkpoints_per_model in zip(paths, models_files):
        print(dir, checkpoints_per_model)
        if saved == 1:
            oof_train, oof_train_y, oof_test = load_oof_dir(dir)
        else:
            print(checkpoints_per_model)
            NFOLDS = len(checkpoints_per_model)
            print(NFOLDS)
            # assert NFOLDS == args.folds
            clfs = [None for i in range(NFOLDS)]
            for cp in checkpoints_per_model:
                fold = int(cp.replace('_', '.').split('.')[-2])
                clfs[fold-1] = cp
            if saved == 2:
                oof_train, oof_train_y, oof_test = load_oof(clfs, fo, test_data, word2index, polarity_dict=polarity_dict,
                                                            attr_dict=attr_dict)
            elif saved == 3:
                oof_train, oof_train_y, oof_test = load_oof3(clfs, fo, test_data, word2index, polarity_dict=polarity_dict,
                                                           attr_dict=attr_dict)
            elif saved == 0:
                oof_train, oof_train_y, oof_test = get_oof(clfs, fo, test_data, word2index, polarity_dict=polarity_dict,
                                                           attr_dict=attr_dict)
            else:
                print("saved error, [0:3]")
                exit(-1)
        x_train.append(oof_train)
        oof_train_y = oof_train_y.reshape(oof_train_y.shape[0], )
        if y_train == []:
            y_train = oof_train_y
        else:
            assert (y_train == oof_train_y).all()
        x_test.append(oof_test)
    x_train = np.concatenate(x_train, axis=1)
    x_test = np.concatenate(x_test, axis=1)

    y_train = np.asarray(y_train).reshape((len(y_train),))

    meta_clf = LogisticRegression()
    meta_clf.fit(x_train, y_train)
    test_predict = meta_clf.predict_proba(x_test)
    fw = codecs.open("../data/test_predict_polarity_ensemble.txt", 'w', encoding='utf-8')
    for j, prob in enumerate(test_predict):
        polarity = np.argmax(prob)-1
        fw.write(test_aspects[j] + ',' + str(polarity) + '\n')
    time_stamp = time.asctime().replace(':', '_').split()
    fw.close()
    shutil.copy2("../data/test_predict_polarity_ensemble.txt",
                 "../data/backup/test_predict_polarity_ensemble_%s.txt" % time_stamp)


def blending():
    # saved = True if args.saved != 0 else False
    saved = args.saved
    test_file1 = "../data/test.txt"
    test_file2 = "../data/test_predict_aspect_ensemble.txt"
    test_texts, test_aspects = load_ab_test(test_file1, test_file2)
    # print(test_aspects)

    word2index = pickle.load(open("../data/vocabulary.pkl", 'rb'))

    f_dict = "../dataset/polarity.json"
    polarity_list, polarity_dict = parse_json(f_dict)
    f_dict2 = "../dataset/attribute.json"
    attr_list, attr_dict = parse_json(f_dict2)

    paths = args.test_dir.split('#')
    models_files = []
    for path in paths:
        models_files.append([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    test_data = Data3((test_texts, None, test_aspects), word2index, polarity_dict, args, target_dict=attr_dict)
    if args.use_elmo != 0:
        test_elmo = load_elmo(test_texts)
        test_data.add_feature(test_elmo)

    x_test = []
    for dir, checkpoints_per_model in zip(paths, models_files):
        print(dir, checkpoints_per_model)
        if saved == 1:
            oof_test = load_oof_test(dir)
        else:
            clfs = checkpoints_per_model
            oof_test = get_oof_test(clfs, test_data)
        x_test.append(oof_test)
    x_test = np.stack(x_test, axis=1)
    print(x_test)
    print(x_test.shape)
    test_predict = np.mean(x_test, axis=1)
    fw = codecs.open("../data/test_predict_polarity_ensemble.txt", 'w', encoding='utf-8')
    for j, prob in enumerate(test_predict):
        polarity = np.argmax(prob)-1
        fw.write(test_aspects[j] + ',' + str(polarity) + '\n')
    time_stamp = time.asctime().replace(':', '_').split()
    fw.close()
    shutil.copy2("../data/test_predict_polarity_ensemble.txt",
                 "../data/backup/test_predict_polarity_ensemble_%s.txt" % time_stamp)


if __name__ == '__main__':
    if args.mode == 0:
        main()
    elif args.mode == 1:
        ensemble()
    elif args.mode == 2:
        stacking()

