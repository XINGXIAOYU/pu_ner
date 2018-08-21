# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 15:28
# @Author  : Xiaoyu Xing
# @File    : npu_model_utils.py

from utils.feature_pu_model_utils import FeaturedDetectionModelUtils
import numpy as np


class AddUnlabeledModelUtils(FeaturedDetectionModelUtils):
    def __init__(self, dp):
        super(AddUnlabeledModelUtils, self).__init__(dp)
        self.dp = dp

    def load_dataset_(self, flag, datasetName, iteration):
        fname = "data/" + datasetName + "/unlabeled/train." + flag + str(iteration) + ".txt"
        trainSentences = self.dp.read_processed_file(fname, flag)
        self.add_char_info(trainSentences)
        self.add_dict_info(trainSentences, 3, datasetName)
        train_sentences_X, train_sentences_Y, train_sentences_LF = self.padding(
            self.createMatrices(trainSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        validSentences = self.dp.read_processed_file("data/" + datasetName + "/valid.txt", flag)
        self.add_char_info(validSentences)
        self.add_dict_info(validSentences, 3, datasetName)
        valid_sentences_X, valid_sentences_Y, valid_sentences_LF = self.padding(
            self.createMatrices(validSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        testSentences = self.dp.read_processed_file("data/" + datasetName + "/test.txt", flag)
        self.add_char_info(testSentences)
        self.add_dict_info(testSentences, 3, datasetName)
        test_sentences_X, test_sentences_Y, test_sentences_LF = self.padding(
            self.createMatrices(testSentences, self.dp.word2Idx, self.dp.case2Idx, self.dp.char2Idx))

        dataset = ((train_sentences_X, train_sentences_Y, train_sentences_LF),
                   (valid_sentences_X, valid_sentences_Y, valid_sentences_LF),
                   (test_sentences_X, test_sentences_Y, test_sentences_LF))

        return dataset

    def load_origin_dataset(self, dataset, p):
        ((train_sentences_X, train_sentences_Y, train_sentences_LF),
         (valid_sentences_X, valid_sentences_Y, valid_sentences_LF),
         (test_sentences_X, test_sentences_Y, test_sentences_LF)) = dataset

        trainX = []
        trainUnlabeledX = []
        trainY = []
        trainUnlabeledY = []
        trainLF = []
        trainUnlabeledLF = []

        m = 0
        n = 0

        for i, sent in enumerate(train_sentences_X):
            # print(train_sentences_Y)
            if train_sentences_Y[i][0] == -1:
                for j in range(len(train_sentences_Y[i])):
                    assert train_sentences_Y[i][j] == -1
                trainUnlabeledX.append(sent)
                trainUnlabeledLF.append(train_sentences_LF[i])
                trainUnlabeledY.append(train_sentences_Y[i])
                n += len(train_sentences_Y[i])
            else:
                trainX.append(sent)
                trainY.append(train_sentences_LF[i])
                trainLF.append(train_sentences_Y[i])
                m += len(train_sentences_Y[i])

        # print(len(trainX), len(trainUnlabeledX))

        dataset2 = ((trainX, trainY, trainLF),
                    (valid_sentences_X, valid_sentences_Y, valid_sentences_LF),
                    (test_sentences_X, test_sentences_Y, test_sentences_LF))

        trainSet, prior, labeled = self.make_PU_dataset(dataset2)
        print(m, n, labeled)

        prior = float((m + n) * p - labeled) / float(m + n - labeled)
        print(prior)
        trainSet = list(zip(train_sentences_X, train_sentences_Y, train_sentences_LF))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))

        return trainSet, validSet, testSet, prior

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
            # print(n_up, n_u)
            prior = float(n_up) / float(n_u)
            # print(prior)
            return x, y, flag, prior, n_lp

        (_train_X, _train_Y, _labeledFlag), (_, _, _), (_, _, _) = dataset
        X, Y, FG, prior, n_lp = _make_PU_dataset(_train_X, _train_Y, _labeledFlag)
        return list(zip(X, Y, FG)), prior, n_lp

    def iterateSet(self, trainset, batchSize, mode, shuffle=True):
        if mode == "TRAIN":
            data_size = len(trainset)
            X, Y, FG = zip(*trainset)
            X = np.array(X)
            Y = np.array(Y)
            FG = np.array(FG)

            num_batches_per_epoch = int((len(trainset) - 1) / batchSize) + 1
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x = np.array(X)[shuffle_indices]
                y = np.array(Y)[shuffle_indices]
                flag = np.array(FG)[shuffle_indices]
            else:
                x = X
                y = Y
                flag = FG

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batchSize
                end_index = min((batch_num + 1) * batchSize, data_size)
                tokens = []
                caseing = []
                char = []
                features = []
                labels = []
                flags = []
                data_X = x[start_index:end_index]
                data_Y = y[start_index:end_index]
                data_FG = flag[start_index:end_index]
                for dt in data_X:
                    t, c, ch, f = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                    features.append(f)
                for dt in data_Y:
                    temp = []
                    for item in dt:
                        if item != -1:
                            item = np.array(item)
                            temp.append(np.eye(2)[item])
                        else:
                            temp.append(np.array([-1, -1]))
                    labels.append(temp)
                for dt in data_FG:
                    temp = []
                    for item in dt:
                        item = np.array(item)
                        temp.append(np.eye(2)[item])
                    flags.append(temp)

                yield np.asarray(tokens), np.asarray(caseing), np.asarray(char), np.asarray(features), np.asarray(
                    labels), np.asarray(
                    flags)
        else:
            data_size = len(trainset)
            X, Y, _ = zip(*trainset)
            X = np.array(X)
            Y = np.array(Y)

            num_batches_per_epoch = int((len(trainset) - 1) / batchSize) + 1
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x = np.array(X)[shuffle_indices]
                y = np.array(Y)[shuffle_indices]
            else:
                x = X
                y = Y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batchSize
                end_index = min((batch_num + 1) * batchSize, data_size)
                tokens = []
                caseing = []
                char = []
                features = []
                data_X = x[start_index:end_index]
                data_Y = y[start_index:end_index]
                for dt in data_X:
                    t, c, ch, f = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                    features.append(f)
                yield np.asarray(tokens), np.asarray(caseing), np.asarray(char), np.asarray(features), np.asarray(
                    data_Y)
