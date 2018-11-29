import torch
from torch import nn
import time
import numpy as np
import torch.nn.functional as F
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.evaluate import score, label_analysis

cuda_flag = True and torch.cuda.is_available()


def train(rnn, train_data, dev_data, test_data, attr_dict, W, args):
    # Train:
    # HyperParameter
    n_epochs = args.EPOCHS
    learning_rate = args.lr  # If you set this too high, it might explode. If too low, it might not learn
    if W is not None:
        W = torch.from_numpy(W)
        rnn.word_rep.word_embed.weight = nn.Parameter(W)
    # if W is not None:
    #     rnn.word_rep.word_embed.weight = nn.Parameter(W)
    print("CUDA: " +str(cuda_flag))
    if cuda_flag:
        rnn = rnn.cuda()
    # rnn = TextCNN(300, output_size, max_length)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(rnn.parameters())
    if args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(rnn.parameters())
    if args.freeze:
        for param in rnn.word_rep.word_embed.parameters():
            param.requires_grad = False
    print_every = 100
    plot_every = 30
    # Keep track of losses for plotting
    current_loss = []
    all_losses = []
    test_acc = []
    acc_index = 0
    es_len = 0

    start = time.time()
    max_acc = 0

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # np.random.seed([3, 1415])
    for epoch in range(1, n_epochs + 1):
        iterations = 0
        loss_sum = 0
        # scheduler.step()
        index_list = np.arange(len(train_data.sentences))
        np.random.shuffle(index_list)
        # print(index_list)
        for index in index_list:
            iterations += 1
            input_tensors, category_tensor = train_data.get(index, cuda_flag)
            loss = rnn.optimize_step(input_tensors, category_tensor, optimizer)
            # print(loss)
            current_loss.append(loss)
            loss_sum += loss

            # Add current loss avg to list of losses
            if (index+1) % plot_every == 0:
                # print(batch_epoch)
                all_losses.append(sum(current_loss) / len(current_loss))
                current_loss = []

            if iterations % (len(index_list) //1) == 0:
                with torch.no_grad():
                    dev_predict = predict(rnn, dev_data, args)
                pred_acc_p = score(dev_predict, dev_data.labels)
                print("Epoch:%d" % epoch)
                print("[p:%.4f, r:%.4f, f:%.4f] acc:%.4f" %
                      (pred_acc_p[0], pred_acc_p[1], pred_acc_p[2], pred_acc_p[3]))
                # with torch.no_grad():
                #     test_predict = predict(rnn, test_data, args)
                # pred_acc_t = score(test_predict, test_data.labels)
                # print("[p:%.4f, r:%.4f, f:%.4f] acc:%.4f" %
                #       (pred_acc_t[0], pred_acc_t[1], pred_acc_t[2], pred_acc_t[3]))

                if pred_acc_p[2] > max_acc:
                    best_predict = dev_predict
                    # label_prf = label_analysis(best_predict, test_data.labels)
                    # for i in range(len(label_prf)):
                    #     print("%s : [%.4f, %.4f, %.4f] %.4f" %
                    #           (list(attr_dict.keys())[i], label_prf[i][0], label_prf[i][1], label_prf[i][2], label_prf[i][3]))
                    max_acc = pred_acc_p[2]
                    max_print = ("Epoch%d\n" % epoch
                                 + "[p:%.4f, r:%.4f, f:%.4f] acc:%.4f\n"
                                 % (pred_acc_p[0], pred_acc_p[1], pred_acc_p[2], pred_acc_p[3]))
                    best_model = "%s/checkpoint_%s_%.6f.pt" % (args.check_dir, args.model, max_acc)
                    best_dict = copy.deepcopy(rnn)
        # test_acc.append(pred_acc)
        print("Epoch: %d, loss: %.4f" % (epoch, loss_sum))

    print(max_acc)
    print(max_print)
    label_prf = label_analysis(best_predict, test_data.labels)
    for i in range(len(label_prf)):
        print("%s : [%.4f, %.4f, %.4f] %.4f" %
              (list(attr_dict.keys())[i], label_prf[i][0], label_prf[i][1], label_prf[i][2], label_prf[i][3]))

    # plt.figure()
    plt.plot(all_losses)
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)
    plt.savefig("fig/foor_%s.png" % '_'.join(time_stamp))
    # plt.show()
    return best_dict, max_acc


def optimize_step(rnn, input_tensors, category_tensor, optimizer):
    rnn.zero_grad()
    rnn.train()
    output = rnn(input_tensors)
    # print(output)

    loss = F.multilabel_soft_margin_loss(output, category_tensor.float())
    # loss = F.binary_cross_entropy(output, category_tensor.float())
    # loss = customized_loss2(output, category_tensor.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def customized_loss2(input, target):
    length = target.size()[1]
    # print(length)
    ones = target.sum()
    zeros = target.size()[0] - ones
    # print(input)
    # print(target)
    cks = torch.mul(input, target)

    # print("fuck")
    # print(cks.size())
    # print(cks.expand(length, length))
    mask = torch.mm(target.view(-1,1), target.view(1,-1))
    mask = 1 - mask
    all_matrix = torch.mm(input.view(-1,1), target.view(1,-1))
    all_matrix = torch.mul(all_matrix, mask)
    ck_matrix = cks.expand(length, length)
    ck_matrix = torch.mul(ck_matrix, mask)
    sub = torch.exp(-(ck_matrix - all_matrix))
    sub = torch.mul(sub, target.expand(length, length))
    # print(all_matrix)
    # print(sub)
    sum = sub.sum() - ones**2
    loss = sum/(ones*(length-ones))
    # print(loss)

    return loss


def category_from_output(output):

    top_n, top_i = output.topk(1) # Tensor out of Variable with .data
    # print(top_i)
    category_i = top_i.view(output.size()[0]).detach()
    return category_i


def categories_from_output(output, threshold=[0.45 for _ in range(10)]):
    # categories = []
    predicted = [0 for i in range(output.size()[1])]
    tensor = output.detach()
    # print(output)
    # print(tensor)

    for i in range(tensor.size()[1]):
        p = tensor[0, i]
        # print(p)
        if type(threshold) == list:
            t = threshold[i]
        else:
            t = 0.45
        if p > t:
            predicted[i] = 1
            # categories.append(attrC[i])

    if sum(predicted) == 0:
        predicted[tensor.argmax()] = 1

    # if len(categories) < 1:
    #     categories = ["以上都不属于"]
    return predicted


def predict(rnn, dev_data, args):
    length = len(dev_data.sentences)
    rnn.eval()
    predicted_p = []
    conflict = 0
    for i in range(length):
        # category_tensor = Variable(torch.LongTensor([dev_y[i] - 1]))
        input_tensors, _ = dev_data.get(i, cuda_flag)
        output_p = rnn(input_tensors)
        # output_p = F.sigmoid(output_p)
        # if args.model == 'CNN':
        output_p = F.sigmoid(output_p)
        category_i_p = categories_from_output(output_p)
        # print(category_i_t)
        # for k in range(len(category_i_t)):
        #     if category_i_t[k] != 2 and category_i_p[k] != 2:
        #         conflict += 1
        #         if output_t.data[k][category_i_t[k]] > output_p.data[k][category_i_p[k]]:
        #             category_i_p[k] = 2
        #         else:
        #             category_i_t[k] = 2
        # exit(-1)
        predicted_p.append(category_i_p)

    return predicted_p


def predict_with_logit(rnn, dev_data, args):
    length = len(dev_data.sentences)
    rnn.eval()
    predicted_p = []
    conflict = 0
    class_num = 10
    oof_dev = np.zeros((length, class_num))
    for i in range(length):
        # category_tensor = Variable(torch.LongTensor([dev_y[i] - 1]))
        input_tensors, _ = dev_data.get(i, cuda_flag)
        output_p = rnn(input_tensors)
        oof_dev[i] = output_p[0].cpu().detach().numpy()
        # output_p = F.sigmoid(output_p)
        # if args.model == 'CNN':
        output_p = F.sigmoid(output_p)
        category_i_p = categories_from_output(output_p, args.threshold_list)

        predicted_p.append(category_i_p)

    return predicted_p, oof_dev

