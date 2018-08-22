# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 08:25
# @Author  : Xiaoyu Xing
# @File    : no_feature_pu_model_utils.py
from utils.plain_model_utils import ModelUtils
import numpy as np


class DetectionModelUtils(ModelUtils):
    def __init__(self, dp):
        super(DetectionModelUtils, self).__init__()
        self.dp = dp

    def make_PU_dataset(self, dataset):

        def _make_PU_dataset(x, y, flag):
            n_labeled = 0
            n_unlabeled = 0
            all_item = 0
            for item in flag:
                item = np.array(item)
                n_labeled += (item == 1).sum()
                item = np.array(item)
                n_unlabeled += (item == 0).sum()
                all_item += len(item)

            labeled = n_labeled
            unlabeled = n_unlabeled
            labels = np.array([0, 1])
            positive, negative = labels[1], labels[0]
            n_p = 0
            n_lp = labeled
            n_n = 0
            n_u = unlabeled
            for li in y:
                li = np.array(li)
                count = (li == positive).sum()
                n_p += count
                count2 = (li == negative).sum()
                n_n += count2

            if labeled + unlabeled == all_item:
                n_up = n_p - n_lp
            elif unlabeled == all_item:
                n_up = n_p
            else:
                raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
            prior = float(n_up) / float(n_u)
            print(prior)
            return x, y, flag, prior

        (_train_X, _train_Y, _labeledFlag), (_, _, _), (_, _, _) = dataset
        X, Y, FG, prior = _make_PU_dataset(_train_X, _train_Y, _labeledFlag)
        return list(zip(X, Y, FG)), prior

    def load_dataset(self, flag, datasetName):
        fname = "data/" + datasetName + "/train." + flag + ".txt"

        trainSentences = self.dp.read_processed_file(fname, flag)
        self.add_char_info(trainSentences)
        train_sentences_X, train_sentences_Y, train_sentences_LF = self.padding(
            self.createMatrices(trainSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        validSentences = self.dp.read_processed_file("data/" + datasetName + "/valid.txt", flag)
        self.add_char_info(validSentences)
        valid_sentences_X, valid_sentences_Y, valid_sentences_LF = self.padding(
            self.createMatrices(validSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        testSentences = self.dp.read_processed_file("data/" + datasetName + "/test.txt", flag)
        self.add_char_info(testSentences)
        test_sentences_X, test_sentences_Y, test_sentences_LF = self.padding(
            self.createMatrices(testSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        dataset = ((train_sentences_X, train_sentences_Y, train_sentences_LF),
                   (valid_sentences_X, valid_sentences_Y, valid_sentences_LF),
                   (test_sentences_X, test_sentences_Y, test_sentences_LF))

        trainSet, prior = self.make_PU_dataset(dataset)
        trainX, trainY, FG = zip(*trainSet)
        trainSet = list(zip(trainX, trainY, FG))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return trainSet, validSet, testSet, prior

    def load_new_dataset(self, flag, datasetName, iter, p):
        fname = "data/" + datasetName + "/train." + flag + str(iter) + ".txt"
        trainSentences = self.dp.read_processed_file(fname, flag)
        self.add_char_info(trainSentences)
        train_sentences_X, train_sentences_Y, train_sentences_LF = self.padding(
            self.createMatrices(trainSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        validSentences = self.dp.read_processed_file("data/" + datasetName + "/valid.txt", flag)
        self.add_char_info(validSentences)
        valid_sentences_X, valid_sentences_Y, valid_sentences_LF = self.padding(
            self.createMatrices(validSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        testSentences = self.dp.read_processed_file("data/" + datasetName + "/test.txt", flag)
        self.add_char_info(testSentences)
        test_sentences_X, test_sentences_Y, test_sentences_LF = self.padding(
            self.createMatrices(testSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        dataset = ((train_sentences_X, train_sentences_Y, train_sentences_LF),
                   (valid_sentences_X, valid_sentences_Y, valid_sentences_LF),
                   (test_sentences_X, test_sentences_Y, test_sentences_LF))

        trainSet, n_lp = self.make_PU_dataset(dataset)

        n = 0
        for i, sentence in enumerate(train_sentences_X):
            n += len(sentence[0])

        prior = float(n * p - n_lp) / float(n - n_lp)

        trainX, trainY, FG = zip(*trainSet)
        trainSet = list(zip(trainX, trainY, FG))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return trainSet, validSet, testSet, prior
