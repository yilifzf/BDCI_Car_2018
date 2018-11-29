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

def load_attr_data(filename):
    fo = codecs.open(filename, encoding='utf-8')
    test_text = []
    labels = []
    for line in fo:
        splits = line.strip('\n').split('\t')
        # text = text.lower()
        text = splits[0]
        text = text.strip()
        test_text.append(text)
        label = []
        for pair in splits[1:]:
            aspect = pair.split('#')[0]
            label.append(aspect)
        labels.append('|'.join(label))

    return test_text, labels


if __name__ == '__main__':
    f_train = 'train.txt'
    raw_texts, raw_labels = load_attr_data(filename=f_train)
    for i, (train_index, test_index) in enumerate(kfold_split(len(raw_texts), 5)):
        train_out = codecs.open("aspect_ensemble/%d/train.tsv" % (i+1), 'w', encoding='utf-8')
        dev_out = codecs.open("aspect_ensemble/%d/dev.tsv" % (i+1), 'w', encoding='utf-8')
        dev_index = codecs.open("aspect_ensemble/%d/dev.ind" % (i + 1), 'w', encoding='utf-8')
        train_out.write('text    labels\n')
        dev_out.write('text    labels\n')
        test_texts, test_labels = [raw_texts[i] for i in test_index], [raw_labels[i] for i in test_index]
        train_texts, train_labels = [raw_texts[i] for i in train_index], [raw_labels[i] for i in train_index]
        for t, l in zip(train_texts, train_labels):
            train_out.write(t + '\t' + l + '\n')
        for t, l, ind in zip(test_texts, test_labels, test_index):
            dev_out.write(t + '\t' + l + '\n')
            dev_index.write(str(ind) + '\n')
        train_out.close()
        dev_out.close()
        dev_index.close()
