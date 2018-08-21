# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 21:27
# @Author  : Xiaoyu Xing
# @File    : supervised_model_utils.py

from utils.plain_model_utils import ModelUtils


class SupervisedModelUtils(ModelUtils):
    def __init__(self, dp):
        super(SupervisedModelUtils, self).__init__()
        self.dp = dp

    def load_dataset(self, flag, datasetName, percent):
        fname = "data/" + datasetName + "/train." + flag + ".txt"
        trainSentences = self.dp.read_processed_file(fname, flag)
        trainSize = int(len(trainSentences) * percent)
        trainSentences = trainSentences[:trainSize]
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

        trainSet = list(zip(train_sentences_X, train_sentences_Y, train_sentences_LF))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return trainSet, validSet, testSet
