import torch
from torch import nn
import torch.nn.functional as F


class WordRep(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, char_size, args):
        super(WordRep, self).__init__()
        # self.use_char = args.use_char
        self.use_elmo = args.use_elmo
        # self.elmo_mode = args.elmo_mode
        # self.elmo_mode2 = args.elmo_mode2
        # self.projected = args.projected
        # self.char_embed_dim = args.char_embed_dim
        self.word_embed = nn.Embedding(vocab_size, word_embed_dim)
        # if self.use_elmo:
        #     self.elmo_weights = nn.Linear(3, 1)
        #     self.elmo_proj = nn.Linear(1024, word_embed_dim)
        # if self.use_char:
        #     self.char_embed = nn.Embedding(char_size, self.char_embed_dim)
        #     self.char_lstm = nn.LSTM(self.char_embed_dim, self.char_embed_dim//2, num_layers=1, bidirectional=True)

    def forward(self, input_tensors):
        sentence = input_tensors[0]
        elmo_tensor = input_tensors[1]
        char_seq = None
        char_seq_len = None
        char_seq_recover = None
        words_embeds = self.word_embed(sentence)
        if self.use_elmo == 1:
            elmo_tensor = elmo_tensor.view(elmo_tensor.size()[0], 1, -1)
            words_embeds = torch.cat((words_embeds, elmo_tensor), dim=-1)
        elif self.use_elmo == 2:
            elmo_tensor = elmo_tensor.view(elmo_tensor.size()[0], 1, -1)
            words_embeds = elmo_tensor
        # if self.use_elmo:
        #     if self.elmo_mode == 2:
        #         elmo_tensor = elmo_tensor[-1]
        #     elif self.elmo_mode == 3:
        #         elmo_tensor = elmo_tensor[1]
        #     elif self.elmo_mode == 4:
        #         elmo_tensor = elmo_tensor[0]
        #     elif self.elmo_mode == 6:
        #         attn_weights = F.softmax(self.elmo_weights.weight, dim=-1)
        #         elmo_tensor = torch.matmul(attn_weights, elmo_tensor.t())
        #     else:
        #         elmo_tensor = elmo_tensor.mean(dim=0)
        #     if not self.projected:
        #         projected = elmo_tensor
        #     else:
        #         projected = self.elmo_proj(elmo_tensor)
        #     # print(words_embeds.size())
        #     # exit(-1)
        #     projected = projected.view(projected.size()[0], 1, -1)
        #     if self.elmo_mode2 == 1:
        #         words_embeds = words_embeds + projected
        #     elif self.elmo_mode2 == 2:
        #         words_embeds = words_embeds
        #     elif self.elmo_mode2 == 3:
        #         words_embeds = torch.cat((words_embeds, projected), dim=-1)
        #     else:
        #         words_embeds = projected
        # if self.use_char:
        #     char_embeds = self.char_embed(char_seq)
        #     pack_seq = pack_padded_sequence(char_embeds, char_seq_len, True)
        #     char_rnn_out, char_hidden = self.char_lstm(pack_seq)
        #     last_hidden = char_hidden[0].view(sentence.size()[0], 1, -1)
        #     # print(words_embeds)
        #     # print(last_hidden)
        #     words_embeds = torch.cat((words_embeds, last_hidden), -1)
        return words_embeds


class LSTM(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, args=None):
        super(LSTM, self).__init__()
        print("LSTM")

        self.input_size = word_embed_dim
        # if args.elmo_mode2 == 3 and args.projected and args.use_elmo:
        #     self.input_size += word_embed_dim
        # if args.elmo_mode2 == 0 and not args.projected and args.use_elmo:
        #     self.input_size = 1024
        # if args.elmo_mode2 == 3 and not args.projected and args.use_elmo:
        #     self.input_size += 1024
        self.hidden_size = args.n_hidden
        self.output_size = output_size
        self.max_length = 1

        self.word_rep = WordRep(vocab_size, word_embed_dim, None, args)
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=True)

        self.decoderP = nn.Linear(self.hidden_size*2, self.output_size)
        self.dropout = nn.Dropout(0.0)

    def forward(self, input_tensors):
        # print(sentence)
        sentence = self.word_rep(input_tensors)
        output, (hidden, _) = self.rnn(sentence)
        hidden = hidden.view(1, -1)
        decodedP = self.decoderP(hidden).view(1, -1)

        # outputP = F.softmax(decodedP, dim=-1)
        outputP = decodedP
        # print(outputP.size())
        return outputP


class CNN(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, args=None, max_length=20):
        super(CNN, self).__init__()
        print("CNN")
        self.input_size = word_embed_dim if (args.use_elmo == 0) else (
            word_embed_dim + 1024 if args.use_elmo == 1 else 1024)
        # if args.elmo_mode2 == 3 and args.projected and args.use_elmo:
        #     self.input_size += word_embed_dim
        # if args.elmo_mode2 == 0 and not args.projected and args.use_elmo:
        #     self.input_size = 1024
        # if args.elmo_mode2 == 3 and not args.projected and args.use_elmo:
        #     self.input_size += 1024
        self.hidden_size = args.n_hidden
        self.output_size = output_size
        self.max_length = max_length

        self.word_rep = WordRep(vocab_size, word_embed_dim, None, args)
        self.filter_size = [1, 2, 3, 4]
        self.map_size = 300

        self.convs1 = nn.ModuleList([nn.Conv1d(self.input_size, self.map_size, K) for K in self.filter_size])

        # self.pool1 = nn.MaxPool2d((max_length - self.filter_size[0] + 1, 1))
        # self.pool2 = nn.MaxPool2d((max_length - self.filter_size[1] + 1, 1))
        # self.pool3 = nn.MaxPool2d((max_length - self.filter_size[2] + 1, 1))

        # self.decoder = nn.Linear(self.map_size*3, output_size)
        self.decoder = nn.Sequential(
            nn.Linear(self.map_size*len(self.filter_size), self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.dropout = nn.Dropout(args.dropout)

        # self.pad = torch.nn.ConstantPad3d((0, 0, 0, 0, 0, ))
        # self.softmax = nn.Softmax()

    def forward(self, input_tensors):
        feature = self.word_rep(input_tensors)
        # input = F.pad(input, (0, 0, 0, 0, 0, self.max_length-input.size()[0])).view(1, self.max_length, -1)
        feature = feature.view(1, feature.size()[0], -1)
        feature = self.dropout(feature)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]

        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x0 = [i.view(i.size(0), -1) for i in x0]
        x0 = torch.cat(x0, 1)

        # c = self.dropout(c)
        decoded = self.decoder(x0)
        # decoded = self.dropout(decoded)
        # ouput = self.softmax(decoded)
        output = decoded
        output = F.sigmoid(output.view(1, -1))
        return output

    def optimize_step(self, input_tensors, category_tensor, optimizer):
        self.zero_grad()
        self.train()
        output = self.forward(input_tensors)
        # print(output)
        # print(category_tensor)

        # loss = F.multilabel_soft_margin_loss(output, category_tensor.float())
        loss = F.binary_cross_entropy(output, category_tensor.float())
        # loss = customized_loss2(output, category_tensor.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class AttA3(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, args=None):
        super(AttA3, self).__init__()

        self.input_size = word_embed_dim if (args.use_elmo == 0) else (
            word_embed_dim + 1024 if args.use_elmo == 1 else 1024)
        self.hidden_size = args.n_hidden
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005

        print(self.input_size)

        # self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.word_rep = WordRep(vocab_size, word_embed_dim, None, args)
        self.rnn_a = nn.LSTM(self.input_size, self.hidden_size//2, num_layers=1, bidirectional=True)

        self.AE = nn.Embedding(self.output_size, word_embed_dim)
        # if embeddings is not None:
        #     self.AE.weight = nn.Parameter(self.embeddings)
        self.W_h_a = nn.Linear(self.hidden_size, self.hidden_size)
        # self.W_v_a = nn.Linear(self.input_size, self.input_size)
        # self.w_a = nn.Linear(self.hidden_size + self.input_size, 1)
        self.W_v_a = nn.Linear(word_embed_dim, self.hidden_size)
        self.w_a = nn.Linear(self.hidden_size, 1)
        self.W_p_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_x_a = nn.Linear(self.hidden_size, self.hidden_size)

        # self.attn = nn.Linear(self.hidden_size, self.max_length)
        # self.attn_softmax = nn.Softmax()
        # self.W1 = nn.Linear(hidden_size, hidden_size)
        # self.W2 = nn.Linear(self.input_size, self.input_size)
        # self.combine = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        # self.combiner = nn.Linear(self.output_size, 1)

        self.decoders_a = nn.ModuleList([nn.Linear(self.hidden_size, 1) for i in range(output_size)])
        # self.decoders = [nn.Linear(self.hidden_size, 1) for i in range(output_size)]
        # self.decoder_a = nn.Linear(self.hidden_size, self.output_size)  # TODO
        self.dropout = nn.Dropout(args.dropout)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adadelta(self.parameters())

    def forward(self, input_tensors):
        # print(sentence)
        sentence = self.word_rep(input_tensors)

        length = sentence.size()[0]
        output2, (ht, ct) = self.rnn_a(sentence)
        ht = ht.view(1, -1)
        output = output2.view(1, length, -1)
        aspect_embedding = self.AE.weight

        aspect_embedding = aspect_embedding.view(self.output_size, 1, -1)
        # print(aspect)
        # print(aspect_embedding)

        aspect_embedding = aspect_embedding.expand(self.output_size, length, -1)
        output = output.expand(self.output_size, length, -1)
        # M = F.tanh(torch.cat((self.W_h_a(output), self.W_v_a(aspect_embedding)), dim=2))
        # print(aspect_embedding.size())
        # print(output.size())
        M = F.tanh(self.W_h_a(output) + self.W_v_a(aspect_embedding))
        # print(M)
        # print(self.w_a(M).view(12, -1))
        weights = F.softmax(self.w_a(M).view(self.output_size, -1), dim=1)
        r = torch.matmul(weights.view(self.output_size, -1), output2.view(length, -1))
        # print(r.t())
        # r = self.combiner(r.t()).view(1, -1)
        # print(r)
        # r = torch.sum(output2.view(length, -1), 0).view(1, -1)
        # print(self.W_x_a(ht))
        r = F.tanh(self.W_p_a(r) + self.W_x_a(ht))
        # r = ht.view(1, -1)
        r = self.dropout(r)
        decoded = []
        for i in range(r.size(0)):
            decoded.append(self.decoders_a[i](r[i]))
        decoded = torch.stack(decoded)
        # decoded = self.decoder_a(r).view(1, -1)
        # decoded = self.dropout(decoded)
        # print(decoded)
        # output = decoded.view(1,-1)
        output = F.sigmoid(decoded.view(1, -1))
        # output = F.softmax(decoded.view(1, -1), dim=-1)
        # print(output)
        return output

    def optimize_step(self, input_tensors, category_tensor, optimizer):
        self.zero_grad()
        self.train()
        output = self.forward(input_tensors)
        # print(output)
        # print(category_tensor)

        # loss = F.multilabel_soft_margin_loss(output, category_tensor.float())
        loss = F.binary_cross_entropy(output, category_tensor.float())
        # loss = customized_loss2(output, category_tensor.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()