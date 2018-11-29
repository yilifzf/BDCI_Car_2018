import codecs
# import torch
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset


sys.path.append("..")
from utils.data_helper import load_attr_data, load_w2v, load_pos, load_char2id, parse_json, load_test_data
import attribute_level.networks2 as networks
import utils.train_single as train_single
from utils.Data import Data, Data2
from utils.evaluate import score, label_analysis
import utils.train as train
import torch
import numpy as np
import argparse
import os
import shutil
import time
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--EPOCHS", type=int, default=5)
parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--optimizer", type=str,  default="Adam")
# parser.add_argument("--model", type=str, default="Average_LSTM2")
# parser.add_argument("--model", type=str, default="Binary_LSTM")
# parser.add_argument("--model", type=str, default="AttA3")
parser.add_argument("--model", type=str, default="CNN")
# parser.add_argument("--model", type=str, default="Attn_LSTM")
parser.add_argument("--lr", type=float, default=0.2)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--freeze", type=bool, default=True)
parser.add_argument("--use_dev", type=bool, default=False)

parser.add_argument("--mode", type=int, default=2)
parser.add_argument("--use_elmo", type=int, default=0)
parser.add_argument("--save", type=int, default=1)
parser.add_argument("--check_dir", type=str, default="checkpoints_00")
parser.add_argument("--saved", type=int, default=1)


parser.add_argument("--threshold_list", type=list, default=[0.45 for _ in range(10)])
parser.add_argument("--threshold", type=float, default=0.45)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--test_model", type=str, default="TD_3LSTM_0.7626.pt")
parser.add_argument("--train_mode", type=int, default=0)
parser.add_argument("--w2v", type=str, default="fasttext2")
# parser.add_argument("--test_dir", type=str, default="checkpoints_00")
parser.add_argument("--test_dir", type=str, default="cp_CNN_0#cp_CNN_ft2#cp_CNN_2#cp_CNN_tc#cp_AttA3_0#cp_AttA3_ft2#cp_AttA3_2#cp_AttA3_tc#cp_Bert")
parser.add_argument("--meta_dir", type=str, default="cp_1024")
args = parser.parse_args()
print(args)

# seed = 314159
# torch.manual_seed(seed)
# seed = torch.initial_seed()
# print(seed)


class AttributeClassifier:  # Neural network method
    def __init__(self):
        self.classifier = None
        self.trained = False
        pass

    def train_from_data(self, train_raw_data, test_raw_data, W, word2index, attr_dict, args, Fold=0):

        word_embed_dim = W.shape[1]
        hidden_size = args.n_hidden
        vocab_size = len(W)
        output_size = len(attr_dict)

        if args.model == 'LSTM':
            self.classifier = networks.LSTM(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'Fasttext':
            self.classifier = networks.Fasttext(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'Average_LSTM2':
            self.classifier = networks.Average_LSTM2(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'AttA3':
            self.classifier = networks.AttA3(word_embed_dim, output_size, vocab_size, args)
            aspect_e_l = []
            for a in attr_dict:
                # print(a)
                if a == '舒适性':
                    a = '舒适'
                a_e = torch.FloatTensor(W[word2index[a]])
                aspect_e_l.append(a_e)
            aspect_embeds = torch.cat(aspect_e_l, 0)
            # print(aspect_embeds)
            # print(attr_dict)
            self.classifier.AE.weight = torch.nn.Parameter(aspect_embeds)
        elif args.model == 'Binary_LSTM':
            self.classifier = networks.Binary_LSTM(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'CNN':
            self.classifier = networks.CNN(word_embed_dim, output_size, vocab_size, args)
        elif args.model == 'Attn_LSTM':
            self.classifier = networks.Attn_LSTM(word_embed_dim, output_size, vocab_size, args)

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

        train_data = Data(train_raw_data, word2index, attr_dict, args)
        # if args.use_dev:
        #     dev_data = Data(args, dev_input_s, dev_input_t, dev_y_tensor)
        # else:
        #     dev_data = None
        test_data = Data(test_raw_data, word2index, attr_dict, args)
        if args.use_elmo != 0:
            train_data.add_feature(train_elmo)
            test_data.add_feature(test_elmo)
        best_dict, max_acc = train.train(self.classifier, train_data, test_data, test_data, attr_dict, W, args=args)
        best_model = "%s/checkpoint_%s_%.6f_%d.pt" % (args.check_dir, args.model, max_acc, Fold)
        if args.save != 0:
            torch.save(best_dict, best_model)
        pass

    def load_model(self, check_point):
        self.classifier = torch.load(check_point)


def split_dev(train_texts, train_labels, folds=5):
    length = len(train_texts)
    np.random.seed(args.seed)
    index_list = np.random.permutation(length).tolist()
    # print(index_list)

    train_lines = [train_texts[i] for i in index_list[0:length - length // folds]]
    test_lines = [train_texts[i] for i in index_list[length - length // folds:]]
    train_y = [train_labels[i] for i in index_list[0:length-length//folds]]
    test_y = [train_labels[i] for i in index_list[length - length // folds:]]
    return train_lines, train_y, test_lines, test_y


def main():
    f_train = "../data/train.txt"
    # f_test = "data/test_attr2.txt"
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
    f_dict = "../dataset/attribute.json"
    print(f_w2v)
    train_texts, train_labels = load_attr_data(filename=f_train)
    train_texts, train_labels, test_texts, test_labels = split_dev(train_texts, train_labels)
    print(len(train_texts))
    print(len(test_labels))
    # train_texts2, train_labels2, test_texts, test_labels = split_dev(train_texts, train_labels)
    if not os.path.exists("%s" % args.check_dir):
        os.mkdir("%s" % args.check_dir)
    # test_texts, test_labels = load_attr_data(filename=f_test)
    W, word2index2 = load_w2v(f_w2v)
    word2index = pickle.load(open("../data/vocabulary.pkl", 'rb'))
    assert word2index == word2index2
    attr_list, attr_dict = parse_json(f_dict)
    print(list(attr_dict.keys()))
    model = AttributeClassifier()
    print(attr_list)
    print(attr_dict)
    # exit(-1)
    # print(train_texts)
    model.train_from_data((train_texts, train_labels), (test_texts, test_labels), W, word2index, attr_dict, args)


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


def ensemble():
    f_train = "../data/train.txt"
    # f_test = "data/test_attr2.txt"
    if args.w2v == "merge":
        f_w2v = "../embedding/embedding_all_merge_300.txt"
    elif args.w2v == "fasttext2":
        f_w2v = "../embedding/embedding_all_fasttext2_300.txt"
    elif args.w2v == "tencent":
        f_w2v = "../embedding/embedding_all_tencent_200.txt"
    else:
        print("error, no embedding")
        exit(-1)
    f_dict = "../dataset/attribute.json"
    print(f_train)
    print(f_w2v)
    if not os.path.exists("%s" % args.check_dir):
        os.mkdir("%s" % args.check_dir)
    raw_texts, raw_labels = load_attr_data(filename=f_train)
    W, word2index2 = load_w2v(f_w2v)
    word2index = pickle.load(open("../data/vocabulary.pkl", 'rb'))
    assert word2index == word2index2
    attr_list, attr_dict = parse_json(f_dict)
    kf = 0
    for train_index, test_index in kfold_split(len(raw_texts), args.folds):
        kf += 1
        print("FOLD:", kf)
        print("TRAIN:", str(len(train_index)), '\n', "TEST:", str(len(test_index)))
        # train_index, test_index = train_index.tolist(), test_index.tolist()
        test_texts, test_labels = [raw_texts[i] for i in test_index], [raw_labels[i] for i in test_index]
        train_texts, train_labels = [raw_texts[i] for i in train_index], [raw_labels[i] for i in train_index]
        print(len(train_texts))
        print(len(test_labels))
        model = AttributeClassifier()
        print(attr_list)
        print(attr_dict)
        # exit(-1)
        # print(train_texts)
        model.train_from_data((train_texts, train_labels), (test_texts, test_labels), W, word2index, attr_dict, args, kf)
    pass


def test():
    model = AttributeClassifier()
    check_point = "checkpoint_AttA3_0.8810.pt"
    model.load_model(check_point)

    test_file = "data/attribute_test.txt"
    test_texts = load_test_data(test_file)
    f_w2v = "../embedding/embedding_all_merge_300.txt"
    W, word2index = load_w2v(f_w2v)

    f_dict = "../dataset/attribute.json"
    attr_list, attr_dict = parse_json(f_dict)

    test_data = Data((test_texts, None), word2index)

    test_predict = train.predict(model.classifier, test_data, args)
    print(test_predict)

    fw = codecs.open("test_predict.txt", 'w', encoding='utf-8')
    for p in test_predict:
        attributes = []
        for i,l in enumerate(p):
            if l != 0:
                attributes.append(attr_list[i])
        fw.write('|'.join(attributes)+'\n')


def dev():
    model = AttributeClassifier()
    check_point = "checkpoints5/checkpoint_AttA3_0.8666.pt"
    model.load_model(check_point)

    f_train = "data/attribute_data.txt"
    # f_test = "data/test_attr2.txt"
    f_w2v = "../embedding/embedding_all_merge_300.txt"
    f_dict = "../dataset/attribute.json"
    print(f_w2v)
    raw_texts, raw_labels = load_attr_data(filename=f_train)
    W, word2index = load_w2v(f_w2v)
    attr_list, attr_dict = parse_json(f_dict)
    kf = 0

    _, test_index = kfold_split(len(raw_texts), args.folds)[2]
    test_texts, test_labels = [raw_texts[i] for i in test_index], [raw_labels[i] for i in test_index]
    test_data = Data((test_texts, test_labels), word2index, attr_dict, args)

    test_predict = train.predict(model.classifier, test_data, args)
    pred_acc_t = score(test_predict, test_data.labels)
    print(pred_acc_t)


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


def get_oof(clfs, raw_texts, raw_labels, test_data, word2index, attr_dict):
    NFOLDS = len(clfs)
    n_train = len(raw_texts)
    n_test = len(test_data.sentences)
    class_num = 10
    oof_train = np.zeros((n_train, class_num))
    oof_train_y = np.zeros((n_train, class_num))
    oof_test = np.zeros((n_test, class_num))
    oof_test_skf = np.zeros((NFOLDS, n_test, class_num))

    kf = 0
    for (train_index, test_index), checkpoint in zip(kfold_split(n_train, NFOLDS), clfs):
        print(checkpoint)
        clf = torch.load(checkpoint)
        kf += 1
        print("FOLD:", kf)
        print("TRAIN:", str(len(train_index)), "TEST:", str(len(test_index)))
        # train_index, test_index = train_index.tolist(), test_index.tolist()
        dev_texts, dev_labels = [raw_texts[i] for i in test_index], [raw_labels[i] for i in test_index]
        dev_data = Data((dev_texts, dev_labels), word2index, attr_dict, args)
        if args.use_elmo != 0:
            dev_elmo = load_elmo(dev_texts)
            dev_data.add_feature(dev_elmo)
        with torch.no_grad():
            dev_predict, oof_dev = train.predict_with_logit(clf, dev_data, args)
        pred_acc_p = score(dev_predict, dev_data.labels)
        print("[p:%.4f, r:%.4f, f:%.4f] acc:%.4f" %
              (pred_acc_p[0], pred_acc_p[1], pred_acc_p[2], pred_acc_p[3]))
        # label_prf = label_analysis(dev_predict, dev_data.labels)
        # for i in range(len(label_prf)):
        #     print("%s : [%.4f, %.4f, %.4f] %.4f" %
        #           (list(attr_dict.keys())[i], label_prf[i][0], label_prf[i][1], label_prf[i][2], label_prf[i][3]))
        oof_train[test_index] = oof_dev
        dev_y = [l[0].detach().numpy() for l in dev_data.labels]

        oof_train_y[test_index] = dev_y
        _, oof_test_skf[kf - 1, :, :] = train.predict_with_logit(clf, test_data, args)
    oof_test[:] = oof_test_skf.mean(axis=0)
    dir = os.path.dirname(clfs[0])
    if not os.path.exists(os.path.join(dir, 'npy')):
        os.mkdir(os.path.join(dir, 'npy'))
    print(dir)
    np.save(os.path.join(dir, 'npy', "oof_train"), oof_train)
    np.save(os.path.join(dir, 'npy', "oof_train_y"), oof_train_y)
    np.save(os.path.join(dir, 'npy', "oof_test"), oof_test)
    return oof_train, oof_train_y, oof_test


def load_oof(dir):
    oof_train = np.load(os.path.join(dir, 'npy', "oof_train.npy"))
    oof_train_y = np.load(os.path.join(dir, 'npy', "oof_train_y.npy"))
    oof_test = np.load(os.path.join(dir, 'npy', "oof_test.npy"))
    print("loaded from: " + dir)
    return oof_train, oof_train_y, oof_test


def stacking():
    saved = True if args.saved != 0 else False
    f_train = "../data/train.txt"
    test_file = "../data/test.txt"
    test_texts = load_test_data(test_file)
    raw_texts, raw_labels = load_attr_data(filename=f_train)
    word2index = pickle.load(open("../data/vocabulary.pkl", 'rb'))

    f_dict = "../dataset/attribute.json"
    attr_list, attr_dict = parse_json(f_dict)

    paths = args.test_dir.split('#')
    models_files = []
    for path in paths:
        models_files.append([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    test_data = Data((test_texts, None), word2index)
    if args.use_elmo != 0:
        test_elmo = load_elmo(test_texts)
        test_data.add_feature(test_elmo)

    x_train = []
    y_train = []  # TODO replace
    x_test = []
    for dir, checkpoints_per_model in zip(paths, models_files):
        print(dir, checkpoints_per_model)
        if saved == 1 and os.path.isfile(os.path.join(dir, 'npy', "oof_train.npy")):
            oof_train, oof_train_y, oof_test = load_oof(dir)
        else:
            NFOLDS = len(checkpoints_per_model)
            print(NFOLDS)
            assert NFOLDS == args.folds
            clfs = [None for i in range(NFOLDS)]
            for cp in checkpoints_per_model:
                fold = int(cp.replace('_', '.').split('.')[-2])
                print(fold)
                clfs[fold-1] = cp
            oof_train, oof_train_y, oof_test = get_oof(clfs, raw_texts, raw_labels, test_data, word2index, attr_dict)
        x_train.append(oof_train)
        if y_train == []:
            y_train = oof_train_y
        else:
            assert (y_train == oof_train_y).all()
        x_test.append(oof_test)
    x_train = np.stack(x_train, axis=2)
    x_test = np.stack(x_test, axis=2)

    print(x_train.shape)
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    test_predict = []
    for c in range(x_train.shape[1]):
        x_train_c = x_train[:, c, :].reshape(num_train, -1)
        x_test_c = x_test[:, c, :].reshape(num_test, -1)
        meta_clf_c = LogisticRegression()
        y_train_c = y_train[:, c]
        meta_clf_c.fit(x_train_c, y_train_c)
        test_predict_c = meta_clf_c.predict_proba(x_test_c)[:, 1]
        test_predict.append(test_predict_c)

    test_predict = np.stack(test_predict, axis=1)
    print(test_predict.shape)
    fw = codecs.open("../data/test_predict_aspect_ensemble.txt", 'w', encoding='utf-8')

    for prob in test_predict:
        attributes = []
        voted = [0 for a in range(len(attr_list))]

        for i in range(len(prob)):
            p = prob[i]
            # print(p)
            if p > args.threshold:
                voted[i] = 1
                # categories.append(attrC[i])
        if sum(voted) == 0:
            voted[prob.argmax()] = 1
        for i,l in enumerate(voted):
            if l != 0:
                attributes.append(attr_list[i])
        fw.write('|'.join(attributes) + '\n')
    time_stamp = time.asctime().replace(':', '_').split()
    fw.close()
    shutil.copy2("../data/test_predict_aspect_ensemble.txt",
                 "../data/backup/test_predict_aspect_ensemble_%s.txt" % time_stamp)


if __name__ == '__main__':
    if args.mode == 0:
        main()
    elif args.mode == 1:
        ensemble()
    elif args.mode == 2:
        stacking()

