# -*- coding: utf-8 -*-
# @Time    : 2018/8/14 22:20
# @Author  : Xiaoyu Xing
# @File    : make_unlabeled_data.py

from utils.data_utils import DataPrepare
import os
import numpy as np


def make_unlabeled_data(dp, originFile, unlabeledFile, dataset, number):
    # label=-1 means unlabeled data
    trainSentences = dp.read_origin_file(originFile)
    unlabeledSentences = dp.read_unlabeled_data(unlabeledFile)#968779
    fdir = "data/" + dataset + "/unlabeled"
    if not os.path.isdir(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, "train.txt")
    allSentences = []
    for sent in trainSentences:
        allSentences.append(sent)
    for sent in unlabeledSentences:
        allSentences.append(sent)
    shuffle_indices = np.random.permutation(np.arange(len(allSentences)))
    allSentences = np.array(allSentences)[shuffle_indices]
    with open(fname, "w") as fw:
        for sentence in allSentences:
            for word, label, tagIdxList in sentence:
                labeled = 0
                fw.write(word + " " + str(label) + " " + str(labeled) + "\n")
            fw.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PU NER")
    # data

    parser.add_argument('--dataset', default="conll2003")
    parser.add_argument('--number', type=int, default=10000)
    args = parser.parse_args()

    dp = DataPrepare(args.dataset)
    make_unlabeled_data(dp, "data/" + args.dataset + "/train.txt", "data/eng.raw", args.dataset, args.number)
    print("Data Set Writing Successfully")
