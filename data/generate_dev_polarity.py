import codecs
import numpy as np
seed = 1024


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


def load_abp_raw(filename):  # aspect_based polarity
    fo = codecs.open(filename, encoding='utf-8').readlines()
    return fo


def splits(fo, train_index, dev_index):
    train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects = [], [], [], [], [], []
    for i in train_index:
        line = fo[i]
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0].strip()
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
        text = splits[0].strip()
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


if __name__ == '__main__':
    f_train = 'train.txt'
    fo = load_abp_raw(filename=f_train)
    n_train, sentence2instance = count_instance(fo)
    for i, (train_index, test_index) in enumerate(kfold_split(len(fo), 5)):
        train_out = codecs.open("polarity_ensemble_online/%d/train.tsv" % (i+1), 'w', encoding='utf-8')
        dev_out = codecs.open("polarity_ensemble_online/%d/dev.tsv" % (i+1), 'w', encoding='utf-8')
        dev_index = codecs.open("polarity_ensemble_online/%d/dev.ind" % (i + 1), 'w', encoding='utf-8')
        train_out.write('text\taspect\tlabels\n')
        dev_out.write('text\taspect\tlabels\n')
        train_texts, train_labels, train_aspects, test_texts, test_labels, test_aspects = splits(fo, train_index,
                                                                                                 test_index)
        test_i_index = [i_index for sentence_index in test_index for i_index in sentence2instance[sentence_index]]
        for t, l, a in zip(train_texts, train_labels, train_aspects):
            # print(t, l, a)
            train_out.write(t + '\t' + a + '\t' + l + '\n')
        print(len(test_i_index))
        print(len(test_texts))
        for t, l, ind, a in zip(test_texts, test_labels, test_i_index, test_aspects):
            # print(t, l, a)
            dev_out.write(t + '\t' + a + '\t' + l + '\n')
            dev_index.write(str(ind) + '\n')
        train_out.close()
        dev_out.close()
        dev_index.close()
