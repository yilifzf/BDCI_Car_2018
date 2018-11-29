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
    f_train = 'test.txt'
    raw_texts, raw_labels = load_attr_data(filename=f_train)
    fw = codecs.open("test.tsv", 'w', encoding='utf-8')
    fw.write("text    labels\n")
    for t, l in zip(raw_texts, raw_labels):
        fw.write(t + '\t' + l + '\n')
