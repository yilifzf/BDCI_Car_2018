import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support


def score_list(predicted, golden):
    assert len(predicted) == len(golden)
    correct = 0
    for p, g in zip(predicted, golden):
        # print(p)
        # print(g)
        # print(g[0].tolist())
        # exit(-1)
        if p == g:
            correct += 1
    acc = correct/len(golden)

    predicted_all = [l for p in predicted for l in p]
    golden_all = [l for g in golden for l in g]
    precision = precision_score(golden_all, predicted_all)
    recall = recall_score(golden_all, predicted_all)
    f1 = f1_score(golden_all, predicted_all)
    f12 = f1_score(predicted_all, golden_all)

    return precision, recall, f1, acc


def score(predicted, golden):
    assert len(predicted) == len(golden)
    correct = 0
    for p, g in zip(predicted, golden):
        # print(p)
        # print(g)
        # print(g[0].tolist())
        # exit(-1)
        if p == g[0].tolist():
            correct += 1
    acc = correct/len(golden)

    predicted_all = [l for p in predicted for l in p]
    golden_all = [l for g in golden for l in g[0].tolist()]
    precision = precision_score(golden_all, predicted_all)
    recall = recall_score(golden_all, predicted_all)
    f1 = f1_score(golden_all, predicted_all)
    f12 = f1_score(predicted_all, golden_all)

    return precision, recall, f1, acc


def label_analysis(predicted, golden):
    assert len(predicted) == len(golden)
    golden = np.asarray([g[0].tolist() for g in golden])
    predicted = np.asarray(predicted)
    rslt = []
    for i in range(golden.shape[1]):
        p = precision_score(golden[:, i], predicted[:, i])
        r = recall_score(golden[:, i], predicted[:, i])
        f = f1_score(golden[:, i], predicted[:, i])
        rate = sum(golden[:, i]) / golden.shape[0]
        rslt.append([p, r, f, rate])
    return rslt


def score2(predicted, golden):
    # print(len(predicted))
    # print(len(golden))
    assert len(predicted) == len(golden)
    correct = 0
    for p, g in zip(predicted, golden):
        if p == g[0].tolist():
            correct += 1
    acc = correct/len(golden)

    predicted_all = [p[0].tolist() for p in predicted]
    # print(predicted_all)
    golden_all = [g[0].tolist() for g in golden]
    # print(golden_all)
    p, r, f, _ = precision_recall_fscore_support(golden_all, predicted_all, average='micro')

    return p, r, f, acc


def label_analysis2(predicted, golden, label_num):
    assert len(predicted) == len(golden)

    rslt = []

    predicted_all = [p for p in predicted]
    golden_all = [g[0].tolist() for g in golden]
    P, R, F, Support = precision_recall_fscore_support(golden_all, predicted_all, labels=[i for i in range(label_num)])
    for p, r, f, s in zip(P, R, F, Support):
        rslt.append([p, r, f, s])
    # print(len(rslt))
    return rslt

def score_aspect(predict_list, true_list):
    correct = 0
    predicted = 0
    relevant = 0

    i = 0
    j = 0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]

        for num in range(len(true_seq)):
            if true_seq[num] == 0:
                if num < len(true_seq) - 1:
                    # if true_seq[num + 1] == '0' or true_seq[num + 1] == '1':
                    if true_seq[num + 1] != 1:
                        # if predict[num] == '1':
                        if predict[num] == 0 and predict[num + 1] != 1:
                            # if predict[num] == '1' and predict[num + 1] != '1':
                            correct += 1
                            # predicted += 1
                            relevant += 1
                        else:
                            relevant += 1

                    else:
                        if predict[num] == 0:
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == 1:
                                    if predict[j] == 1 and j < len(predict) - 1:
                                        # if predict[j] == '1' and j < len(predict) - 1:
                                        continue
                                    elif predict[j] == 1 and j == len(predict) - 1:
                                        # elif predict[j] == '1' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1

                                    else:
                                        relevant += 1
                                        break

                                else:
                                    if predict[j] != 1:
                                        # if predict[j] != '1':
                                        correct += 1
                                        # predicted += 1
                                        relevant += 1
                                        break


                        else:
                            relevant += 1

                else:
                    if predict[num] == 0:
                        correct += 1
                        # predicted += 1
                        relevant += 1
                    else:
                        relevant += 1

        for num in range(len(predict)):
            if predict[num] == 0:
                predicted += 1

        i += 1

    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1