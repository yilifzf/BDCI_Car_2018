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


class AT_LSTM(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, aspect_size, args=None):
        super(AT_LSTM, self).__init__()

        self.input_size = word_embed_dim if (args.use_elmo == 0) else (
            word_embed_dim + 1024 if args.use_elmo == 1 else 1024)
        self.hidden_size = args.n_hidden
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005

        self.word_rep = WordRep(vocab_size, word_embed_dim, None, args)
        # self.rnn = nn.LSTM(input_size, hidden_size)
        self.rnn_p = nn.LSTM(self.input_size, self.hidden_size // 2, bidirectional=True)

        self.AE = nn.Embedding(aspect_size, self.input_size)

        self.W_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(word_embed_dim, self.input_size)
        self.w = nn.Linear(self.hidden_size + self.input_size, 1)
        self.W_p = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_x = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_softmax = nn.Softmax(dim=0)
        # self.W1 = nn.Linear(hidden_size, hidden_size)
        # self.W2 = nn.Linear(hidden_size, hidden_size)
        self.decoder_p = nn.Linear(self.hidden_size, output_size)  # TODO
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.LogSoftmax()

    def forward(self, input_tensors):
        # print(sentence)
        assert len(input_tensors) == 3
        aspect_i = input_tensors[2]
        sentence = self.word_rep(input_tensors)

        length = sentence.size()[0]
        # for t in range(length):
        #     ht, ct = self.rnn(inputs[t], (ht, ct))
        output, hidden = self.rnn_p(sentence)

        hidden = hidden[0].view(1, -1)
        # print(output.size())
        # last_i = Variable(torch.LongTensor([output.size()[0] - 1]))
        # last = torch.index_select(output, 0, last_i).view(1, -1)
        # print(last)
        # print(hidden)
        output = output.view(output.size()[0], -1)

        # print(aspect_i)
        aspect_embedding = self.AE(aspect_i)
        aspect_embedding = aspect_embedding.view(1, -1)
        # print(aspect)
        aspect_embedding = aspect_embedding.expand(length, -1)
        M = F.tanh(torch.cat((self.W_h(output), self.W_v(aspect_embedding)), dim=1))
        weights = self.attn_softmax(self.w(M)).t()
        # print(weights)
        r = torch.matmul(weights, output)
        # print(r)

        # s = self.attn(output)
        # s = s.view(1,-1)
        # attn_weights = F.softmax(s)
        # # print(attn_weights)
        # output = torch.mm(attn_weights, output)
        r = F.tanh(torch.add(self.W_p(r), self.W_x(hidden)))

        # print(hidden)
        # r = ht.view(1,-1)
        # aspect = aspect.view(1, -1)
        # r = torch.cat((r, aspect), dim=1)
        # print(r)

        # r = self.dropout(r)
        decoded = self.decoder_p(r)
        # decoded = self.dropout(decoded)
        # ouput = F.softmax(decoded, dim=1)
        output = decoded
        return output


class GCAE(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, aspect_size, args=None):
        super(GCAE, self).__init__()
        self.args = args
        self.input_size = word_embed_dim if (args.use_elmo == 0) else (
            word_embed_dim + 1024 if args.use_elmo == 1 else 1024)
        V = vocab_size
        D = self.input_size
        C = output_size
        A = aspect_size

        Co = 100
        Ks = [2,3,4]

        self.word_rep = WordRep(V, word_embed_dim, None, args)
        # self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        self.AE = nn.Embedding(A, word_embed_dim)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(word_embed_dim, Co)

    def forward(self, input_tensors):
        feature = self.word_rep(input_tensors)
        aspect_i = input_tensors[2]
        aspect_v = self.AE(aspect_i)  # (N, L', D)
        # print(aspect_v.size())
        # aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        feature = feature.view(1, feature.size()[0], -1)
        # aspect_v = aspect_v.view(1, aspect_v.size()[0], -1)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        logit = self.fc1(x0)  # (N,C)
        return logit


class HEAT(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, aspect_size, args=None):
        super(HEAT, self).__init__()

        self.input_size = word_embed_dim if (args.use_elmo == 0) else (
            word_embed_dim + 1024 if args.use_elmo == 1 else 1024)
        self.hidden_size = args.n_hidden
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005

        self.word_rep = WordRep(vocab_size, word_embed_dim, None, args)

        self.rnn_a = nn.GRU(self.input_size, self.hidden_size // 2, bidirectional=True)
        self.AE = nn.Embedding(aspect_size, word_embed_dim)

        self.W_h_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v_a = nn.Linear(word_embed_dim, self.input_size)
        self.w_a = nn.Linear(self.hidden_size + word_embed_dim, 1)
        # self.w_as = nn.ModuleList([nn.Linear(self.hidden_size + self.input_size, 1) for i in range(self.output_size)])
        self.W_p_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_x_a = nn.Linear(self.hidden_size, self.hidden_size)
        # self.rnn = nn.LSTM(input_size, hidden_size)

        # self.rnn_p = nn.LSTM(input_size, hidden_size // 2, bidirectional=True)
        self.rnn_p = nn.GRU(self.input_size, self.hidden_size // 2, bidirectional=True)

        self.W_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(word_embed_dim+self.hidden_size, word_embed_dim+self.hidden_size)
        self.w = nn.Linear(2*self.hidden_size + word_embed_dim, 1)
        self.W_p = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_x = nn.Linear(self.hidden_size, self.hidden_size)

        # self.W1 = nn.Linear(hidden_size, hidden_size)
        # self.W2 = nn.Linear(hidden_size, hidden_size)
        self.decoder_p = nn.Linear(self.hidden_size+word_embed_dim, output_size)  # TODO
        # self.decoder_p = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(args.dropout)
        # self.softmax = nn.LogSoftmax()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.AE.weight.requires_grad = False

    def forward(self, input_tensors):
        # print(sentence)
        assert len(input_tensors) == 3
        aspect_i = input_tensors[2]
        sentence = self.word_rep(input_tensors)

        length = sentence.size()[0]
        # for t in range(length):
        #     ht, ct = self.rnn(inputs[t], (ht, ct))
        output_a, hidden = self.rnn_a(sentence)
        output_p, _ = self.rnn_p(sentence)
        output_a = output_a.view(output_a.size()[0], -1)
        output_p = output_p.view(length, -1)
        # print(aspect_i)
        aspect_e = self.AE(aspect_i)
        aspect_embedding = aspect_e.view(1, -1)
        aspect_embedding = aspect_embedding.expand(length, -1)
        # M_a = F.tanh(torch.cat((self.W_h_a(output_a), self.W_v_a(aspect_embedding)), dim=1))
        M_a = F.tanh(torch.cat((output_a, aspect_embedding), dim=1))
        weights_a = F.softmax(self.w_a(M_a), dim=0).t()
        # print(weights_a)
        r_a = torch.matmul(weights_a, output_a)

        r_a_expand = r_a.expand(length, -1)
        # print(r_a_expand)
        # print(aspect_embedding)
        query4PA = torch.cat((r_a_expand, aspect_embedding), dim=1)
        # print(self.W_h(output_p))
        # M_p = F.tanh(torch.cat((self.W_h(output_p), self.W_v(query4PA)), dim=1))
        M_p = F.tanh(torch.cat((output_p, query4PA), dim=1))
        g_p = self.w(M_p)
        # print(g_p)

        # M = torch.FloatTensor([[1 - abs(i-j)/length for j in range(length)] for i in range(length)])
        # if inputs.data.is_cuda:
        #     M = M.cuda()
        # m = torch.matmul(weights_a, Variable(M, volatile=False)).view(-1, 1)
        # weights_p = F.softmax(m*g_p, dim=0).t()

        weights_p = F.softmax(g_p, dim=0).t()
        # print(weights_p.data)
        r_p = torch.matmul(weights_p, output_p)
        # print(r_p)
        # r = F.tanh(torch.add(self.W_p(r), self.W_x(hidden)))

        r = torch.cat((r_p, aspect_e), dim=1)
        # r = r_p

        # r = self.dropout(r)
        decoded = self.decoder_p(r)
        # decoded = self.dropout(decoded)
        # ouput = F.softmax(decoded, dim=1)
        ouput = decoded
        return ouput