import numpy as np
import csv
import os
seed =1024
import torch

def categories_from_output(output, t = 0.45):
    # categories = []
    predicted = [0 for i in range(output.size)]
    tensor = output
    # print(output)
    # print(tensor)
    for i in range(tensor.size):
        p = tensor[i]
        # print(p)
        if p > t:
            predicted[i] = 1
            # categories.append(attrC[i])
    if sum(predicted) == 0:
        predicted[np.argmax(tensor)] = 1
    return predicted


def kfold_split(length, k=5):
    np.random.seed(seed)
    index_list = np.random.permutation(length)

    l = length // k
    folds = []
    for i in range(k):
        test_idx = np.zeros(length, dtype=bool)
        test_idx[i*l:(i+1)*l] = True
        folds.append((index_list[~test_idx], index_list[test_idx]))
    return folds


def merge_oof(dir):
    NFOLDS = 5
    with open(os.path.join(dir, "1", "train.tsv"), "r", encoding='utf-8') as f:
        reader = f.readlines()
        train_length = len(reader) -1
    with open(os.path.join(dir, "1", "dev.tsv"), "r", encoding='utf-8') as f:
        reader = f.readlines()
        dev_length = len(reader) -1
    with open(os.path.join(dir, "test.tsv"), "r", encoding='utf-8') as f:
        reader = f.readlines()
        n_test = len(reader) -1
    print(train_length)
    print(dev_length)
    print(n_test)
    n_train = train_length + dev_length
    class_num = 3
    oof_train = np.zeros((n_train, class_num))
    oof_train_y = np.zeros((n_train,))
    oof_test = np.zeros((n_test, class_num))
    oof_test_skf = np.zeros((NFOLDS, n_test, class_num))
    for i in range(NFOLDS):
        fold = i + 1
        dev_index = np.loadtxt(os.path.join(dir, str(fold), 'dev.ind'), dtype=int)
        # print(dev_index2)
        # print(dev_index)
        # assert (dev_index2 == dev_index).all()
        dev_predict = np.load(os.path.join(dir, str(fold), 'oof_train.npy'))
        print(dev_predict.shape)
        print(dev_length)
        # assert dev_predict.shape == (dev_length, class_num)
        dev_labels = np.load(os.path.join(dir, str(fold), 'oof_train_y.npy'))
        # print(dev_labels.shape)
        # assert dev_labels.shape == (dev_length, class_num)
        test_predict = np.load(os.path.join(dir, str(fold), 'oof_test.npy'))
        print(test_predict.shape)
        print((n_test, class_num))
        assert test_predict.shape == (n_test, class_num)
        # dev_predict = torch.sigmoid(torch.Tensor(dev_predict)).numpy()
        # test_predict = torch.sigmoid(torch.Tensor(test_predict)).numpy()
        oof_train[dev_index] = dev_predict
        oof_train_y[dev_index] = dev_labels
        oof_test_skf[fold - 1, :, :] = test_predict
    oof_test[:] = oof_test_skf.mean(axis=0)
    # oof_train = torch.sigmoid(torch.Tensor(oof_train)).numpy()
    # oof_test = torch.sigmoid(torch.Tensor(oof_test)).numpy()
    if not os.path.exists(os.path.join(dir, 'npy')):
        os.mkdir(os.path.join(dir, 'npy'))
    print(dir)
    np.save(os.path.join(dir, 'npy', "oof_train"), oof_train)
    np.save(os.path.join(dir, 'npy', "oof_train_y"), oof_train_y)
    np.save(os.path.join(dir, 'npy', "oof_test"), oof_test)
    return oof_train, oof_train_y, oof_test


if __name__ == '__main__':
    import sys
    dir = sys.argv[1]
    merge_oof(dir)
