import torch
import numpy as np


class Data:
    def __init__(self, train_raw_data, word2index, attr_dict=None, args=None):
        self.sentences = None
        self.targets = None
        self.labels = None
        self.chars = None
        self.pos = None
        self.char_lens = None
        self.char_recover = None
        self.features = None

        train_texts = train_raw_data[0]
        train_labels = train_raw_data[1]

        self.sentences = [self.to_tensor(s, word2index) for s in train_texts]  # [L*1*dim]
        if train_labels is not None:
            self.labels = [self.label2tensor(y, attr_dict) for y in train_labels]

    def get(self, index, cuda_flag):
        line_tensor = self.sentences[index]
        if self.labels is None:
            category_tensor = None
        else:
            category_tensor = self.labels[index]
            if cuda_flag:
                category_tensor = category_tensor.cuda()
        if cuda_flag:
            line_tensor = line_tensor.cuda()
        if self.features is None:
            elmo_tensor = None
        else:
            elmo_tensor = self.features[index]
            if cuda_flag:
                elmo_tensor = elmo_tensor.cuda()

        return (line_tensor, elmo_tensor), category_tensor

    def add_feature(self, features):
        self.features = features

    def get_input(self, index, cuda_flag):
        pass

    def to_tensor(self, text, word2id):
        index_list = []
        # print(text)
        for w in text:
            w = w
            if w in word2id:
                index_list.append(word2id[w])
            else:
                # print( w == '')
                # if ' ' not in w:
                print(w)
                print("2id error")
                w = 'UNK'
                # print(w)
                index_list.append(word2id[w])
        tensor = torch.from_numpy(np.asarray(index_list))
        tensor = tensor.view(tensor.size()[0], -1).long()
        return tensor

    def label2tensor(self, labels, attrDict):
        tensor = [0 for i in range(len(attrDict))]
        for l in labels:
            if l in attrDict:
                tensor[attrDict[l]] = 1
            # else:
            #     print(w)
            #     print("2id error")
            #     w = 'UNK'
            #     index_list.append(word2id[w])
        tensor = torch.from_numpy(np.asarray(tensor))
        tensor = tensor.view(1, tensor.size()[0]).long()
        return tensor

    def generate_char_tensor(self, text, char2id):
        char_len_list = list(map(len, text))
        words_length = len(text)
        max_seq_len = max(char_len_list)
        char_seq_tensor = torch.zeros((words_length, max_seq_len)).long()
        char_seq_lengths = torch.LongTensor(char_len_list)
        for idx, (word, charlen) in enumerate(zip(text, char_seq_lengths)):
            c_i_list = []
            for c in word:
                if c in char2id:
                    c_i_list.append(char2id[c])
                else:
                    print("char2id error")
            # print len(word), wordlen
            char_seq_tensor[idx, :charlen] = torch.LongTensor(c_i_list)
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        return char_seq_tensor, char_seq_lengths, char_seq_recover


class Data2:  # data for polarity:
    def __init__(self, train_raw_data, word2index, polarity_dict=None, args=None):
        self.sentences = None
        self.targets = None
        self.labels = None
        self.chars = None
        self.pos = None
        self.char_lens = None
        self.char_recover = None
        self.features = None

        train_texts = train_raw_data[0]
        train_labels = train_raw_data[1]

        self.sentences = [self.to_tensor(s, word2index) for s in train_texts]  # [L*1*dim]
        # print(train_input_s[0])

        # print(train_input_s[0])
        if train_labels is not None:
            self.labels = [self.label2tensor(y, polarity_dict) for y in train_labels]

    def get(self, index, cuda_flag):
        line_tensor = self.sentences[index]
        if self.labels is None:
            category_tensor = None
        else:
            category_tensor = self.labels[index]
            if cuda_flag:
                category_tensor = category_tensor.cuda()
        if cuda_flag:
            line_tensor = line_tensor.cuda()
        if self.features is None:
            elmo_tensor = None
        else:
            elmo_tensor = self.features[index]
            if cuda_flag:
                elmo_tensor = elmo_tensor.cuda()

        return (line_tensor, elmo_tensor), category_tensor

    def to_tensor(self, text, word2id):
        index_list = []
        for w in text:
            w = w
            if w in word2id:
                index_list.append(word2id[w])
            else:
                # print(w)
                print("2id error")
                w = 'UNK'
                index_list.append(word2id[w])
        tensor = torch.from_numpy(np.asarray(index_list))
        tensor = tensor.view(tensor.size()[0], -1).long()
        return tensor

    def label2tensor(self, labels, attrDict):
        # tensor = [0 for i in range(len(attrDict))]
        # for l in labels:
        #     if l in attrDict:
        #         tensor[attrDict[l]] = 1
        #     # else:
        #     #     print(w)
        #     #     print("2id error")
        #     #     w = 'UNK'
        #     #     index_list.append(word2id[w])
        # label = len(tensor)-1
        # for i, k in enumerate(tensor):
        #     if k != 0:
        #         label = i
        #         break
        label = attrDict[labels[0]]
        tensor = torch.from_numpy(np.asarray([label]))
        tensor = tensor.view(-1).long()
        return tensor


class Data3:  # data for aspect_polarity:
    def __init__(self, train_raw_data, word2index, polarity_dict=None, args=None, target_dict=None):
        self.sentences = None
        self.targets = None
        self.labels = None
        self.chars = None
        self.pos = None
        self.char_lens = None
        self.char_recover = None
        self.features = None

        train_texts = train_raw_data[0]
        train_labels = train_raw_data[1]

        targets = None
        if len(train_raw_data) > 2:
            if len(train_raw_data[2]) is not None:
                targets = train_raw_data[2]

        self.sentences = [self.to_tensor(s, word2index) for s in train_texts]  # [L*1*dim]
        # print(train_input_s[0])

        # print(train_input_s[0])
        if train_labels is not None:
            self.labels = [self.label2tensor(y, polarity_dict) for y in train_labels]
        if targets is not None:
            self.targets = [self.label2tensor(t, target_dict) for t in targets]

    def get(self, index, cuda_flag):
        line_tensor = self.sentences[index]
        if cuda_flag:
            line_tensor = line_tensor.cuda()
        if self.labels is None:
            category_tensor = None
        else:
            category_tensor = self.labels[index]
            if cuda_flag:
                category_tensor = category_tensor.cuda()
        if self.features is None:
            elmo_tensor = None
        else:
            elmo_tensor = self.features[index]
            if cuda_flag:
                elmo_tensor = elmo_tensor.cuda()
        if self.targets is None:
            target_tensor = None
        else:
            target_tensor = self.targets[index]
            if cuda_flag and target_tensor is not None:
                target_tensor = target_tensor.cuda()

        return (line_tensor, elmo_tensor, target_tensor), category_tensor

    def add_feature(self, features):
        self.features = features

    def to_tensor(self, text, word2id):
        index_list = []
        for w in text:
            w = w
            if w in word2id:
                index_list.append(word2id[w])
            else:
                # print(w)
                print("2id error")
                w = 'UNK'
                index_list.append(word2id[w])
        tensor = torch.from_numpy(np.asarray(index_list))
        tensor = tensor.view(tensor.size()[0], -1).long()
        return tensor

    def label2tensor(self, labels, attrDict):
        if labels in attrDict:
            label = attrDict[labels]
            tensor = torch.from_numpy(np.asarray([label]))
            tensor = tensor.view(-1).long()
        else:
            print(labels)
            print("label2tensor error")
        return tensor