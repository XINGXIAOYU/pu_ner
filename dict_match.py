# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 12:49
# @Author  : Xiaoyu Xing
# @File    : dict_match.py

import utils.data_utils, utils.dict_utils
import numpy as np


def compute_precision_recall_f1(labels, preds, flag, pflag):
    tp = 0
    np_ = 0
    pp = 0
    for i in range(len(labels)):
        sent_label = labels[i]
        sent_pred = preds[i]
        for j in range(len(sent_label)):
            item1 = np.array(sent_pred[j])
            item2 = np.array(sent_label[j])

            if (item1 == pflag).all() == True:
                pp += 1
            if (item2 == flag).all() == True:
                np_ += 1
                if (item1 == pflag).all() == True:
                    tp += 1
    if pp == 0:
        p = 0
    else:
        p = float(tp) / float(pp)
    if np_ == 0:
        r = 0
    else:
        r = float(tp) / float(np_)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = float(2 * p * r) / float((p + r))
    return p, r, f1, tp, np_, pp


def compute_precision_recall_f1_2(labels, preds, flag, pflag):
    tp = 0
    np_ = 0
    pp = 0
    for i in range(len(labels)):
        item1 = np.array(preds[i])
        item2 = np.array(labels[i])

        if item1 == pflag:
            pp += 1
        if item2 == flag:
            np_ += 1
            if item1 == pflag:
                tp += 1
    if pp == 0:
        p = 0
    else:
        p = float(tp) / float(pp)
    if np_ == 0:
        r = 0
    else:
        r = float(tp) / float(np_)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = float(2 * p * r) / float((p + r))
    return p, r, f1, tp, np_, pp


def getLabelsAndPreds(sentences):
    labels = []
    preds = []
    for sent in sentences:
        for word, label, pred in sent:
            if len(label.split('-')) > 1:
                label = label.split('-')[-1]
            else:
                label = label
            labels.append(label)
            preds.append(pred)
    return labels, preds


def dict_match_word(dp, dutils, fileName, dictName, flag, mode, dataset):
    sentences = dp.read_origin_file(fileName)
    length = [len(i) for i in sentences]
    maxLen = max(length)
    sentences,_ = dutils.lookup_in_Dic(dictName, sentences, flag, maxLen)
    if mode == "TRAIN":
        dp.writeFile("data/" + dataset + "/train." + flag + ".txt", mode, flag, sentences)
    ss = []
    for sentence in sentences:
        s = []
        for i, (word, label, pred) in enumerate(sentence):
            pred = np.array(pred)
            if pred[dutils.tag2Idx[flag]] == 1:
                s.append([word, label, 1])
            else:
                s.append([word, label, 0])
        ss.append(s)
    sentences = ss
    labels, preds = getLabelsAndPreds(sentences)
    p, r, f1, tp, np_, pp = compute_precision_recall_f1_2(labels, preds, flag, 1)
    return p, r, f1, tp, np_, pp


def dict_match_result(dp, dutils, fileName, dictName, flag, mode, dataset):
    sentences = dp.read_origin_file(fileName)
    length = [len(i) for i in sentences]
    maxLen = max(length)
    sentences, num = dutils.lookup_in_Dic(dictName, sentences, flag, maxLen)
    print(num)
    if mode == "TRAIN":
        dp.writeFile("data/" + dataset + "/valid." + flag + ".txt", mode, flag, sentences)
    ss = []
    for sentence in sentences:
        s = []
        for i, (word, label, pred) in enumerate(sentence):
            pred = np.array(pred)
            if pred[dutils.tag2Idx[flag]] == 1:
                s.append([word, label, 1])
            else:
                s.append([word, label, 0])
        ss.append(s)
    sentences = ss
    newSentences, newLabels, newPreds = dp.wordLevelGeneration(sentences)
    p, r, f1, tp, np_, pp = compute_precision_recall_f1(newLabels, newPreds, flag, 1)
    return p, r, f1, tp, np_, pp


def count_entity(dataset, type):
    filename = "dictionary/" + dataset + "/" + type + ".txt"
    s = set()
    with open(filename, "r") as fw:
        for line in fw:
            line = line.strip()
            s.add(line)
    print(type, len(s))


if __name__ == "__main__":
    dp = utils.data_utils.DataPrepare("twitter")
    dutils = utils.dict_utils.DictUtils()
    # num = 3

    # p1, r1, f11, tp, np_1, pp = dict_match_word(dp, dutils,
    #                                               "data/twitter/train.txt",
    #                                               "dictionary/twitter/person.txt",
    #                                               "PER",
    #                                               "TEST", "twitter")
    # print("%.4f" % p1, "%.4f" % r1, "%.4f" % f11, tp, np_1, pp)
    # # #
    # # count_entity("wikigold", "person")
    # #
    # p2, r2, f12, tp, np_2, pp = dict_match_word(dp, dutils, "data/twitter/train.txt",
    #                                               "dictionary/twitter/location.txt",
    #                                               "LOC",
    #                                               "TEST", "twitter")
    # print("%.4f" % p2, "%.4f" % r2, "%.4f" % f12, tp, np_2, pp)

    p3, r3, f13, tp, np_3, pp = dict_match_word(dp, dutils, "data/twitter/train.txt",
                                                  "dictionary/twitter/organization.txt",
                                                  "ORG", "TEST", "twitter")
    print("%.4f" % p3, "%.4f" % r3, "%.4f" % f13, tp, np_3, pp)

    # count_entity("conll2003", "organization")
    # p4, r4, f14, tp, np_4, pp = dict_match_result(dp, dutils, "data/twitter/test.txt",
    #                                               "dictionary/twitter/misc.txt",
    #                                               "MISC", "TEST",
    #                                               "twitter")
    # print("%.4f" % p4, "%.4f" % r4, "%.4f" % f14, tp, np_4, pp)

    # print(np_1 + np_2 + np_3 + np_4)





    # p = p1 + p2 + p3 + p4
    # r = r1 + r2 + r3 + r4
    # f = f11 + f12 + f13 + f14
    #
    # print(p / 4)
    # print(r / 4)
    # print(f / 4)

"""
PER
0.7391 0.0254 0.0491 17 669 23
0.8095 0.0258 0.0499 17 660 21
0.6522 0.0236 0.0455 15 636 23
0.7619 0.0252 0.0488 16 635 21
0.8261 0.0277 0.0537 19 685 23
0.5926 0.0246 0.0473 16 650 27
0.5143 0.0272 0.0516 18 662 35
0.7273 0.0247 0.0478 16 648 22
0.6207 0.0263 0.0505 18 684 29
0.5625 0.0134 0.0262 9 671 16

LOC
0.9630 0.5200 0.6753 364 700 378
0.9335 0.5098 0.6594 365 716 391
0.9665 0.4936 0.6534 346 701 358
0.9608 0.4973 0.6554 368 740 383
0.9446 0.4742 0.6314 358 755 379
0.9580 0.5321 0.6842 365 686 381
0.9634 0.4907 0.6502 368 750 382
0.9647 0.4955 0.6547 328 662 340
0.9669 0.4930 0.6530 351 712 363
0.9598 0.4986 0.6563 358 718 373

ORG
0.8682 0.3032 0.4494 191 630 220
0.8761 0.3139 0.4622 205 653 234
0.8333 0.2951 0.4358 180 610 216
0.8783 0.3112 0.4596 202 649 230
0.8540 0.3217 0.4673 193 600 226
0.8634 0.3111 0.4574 196 630 227
0.8912 0.3222 0.4733 213 661 239
0.8913 0.3218 0.4729 205 637 230
0.8462 0.3257 0.4703 198 608 234
0.8616 0.3002 0.4452 193 643 224

MISC
1.0000 0.3580 0.5272 121 338 121
1.0000 0.3672 0.5372 130 354 130
1.0000 0.3586 0.5279 123 343 123
1.0000 0.3583 0.5276 129 360 129
0.9907 0.3397 0.5059 107 315 108
1.0000 0.4064 0.5780 139 342 139
1.0000 0.3172 0.4816 118 372 118
1.0000 0.3409 0.5085 120 352 120
1.0000 0.3384 0.5056 112 331 112
1.0000 0.3595 0.5289 119 331 119

0.9849 0.3718 0.5398 261 702 265
0.9213 0.3059 0.4593 1405 4593 1525


twitter train
0.7926 0.2603 0.3919 577 2217 728
0.8589 0.3091 0.4545 925 2993 1077
0.8377 0.2058 0.3304 191 928 228

twitter test
0.8139 0.2770 0.4133 503 1816 618
0.8440 0.3044 0.4474 725 2382 859
0.8244 0.2014 0.3238 169 839 205


train+unlabel
per: 4278
loc: 8960
org: 5439
misc: 3668

train
per 482
loc 316
org 237
misc 113

train+unlabel
per 596
loc 352
org 249
misc 122

train
per: 2388
loc: 4493
org: 3198
misc: 1525
"""
