# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 13:08
# @Author  : Xiaoyu Xing
# @File    : supervised_model.py

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.data_utils import DataPrepare
from sub_model import CharCNN, CaseNet, WordNet, TimeDistributed
from utils.supervised_model_utils import SupervisedModelUtils
import torch.nn as nn
import torch
import numpy as np
from progressbar import *


class SupervisedModel(nn.Module):
    def __init__(self, dp, charModel, wordModel, caseModel, inputSize, hiddenSize, layerNum):
        super(SupervisedModel, self).__init__()
        self.dp = dp
        self.charModel = TimeDistributed(charModel, self.dp.char2Idx)
        self.wordModel = wordModel
        self.caseModel = caseModel
        self.lstm = nn.LSTM(inputSize, hiddenSize, layerNum, bias=0.5, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hiddenSize, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token, case, char):
        charOut, sortedLen1, reversedIndices1 = self.charModel(char)
        wordOut, sortedLen2, reversedIndices2 = self.wordModel(token)
        caseOut, sortedLen3, reversedIndices3 = self.caseModel(case)

        encoding = torch.cat([wordOut.float(), caseOut.float(), charOut.float()], dim=2)

        sortedLen = sortedLen1
        reverseIndices = reversedIndices1

        packed_embeds = pack_padded_sequence(encoding, sortedLen, batch_first=True)

        maxLen = sortedLen[0]
        mask = torch.zeros([len(sortedLen), maxLen, 2])
        for i, l in enumerate(sortedLen):
            mask[i][:l][:] = 1

        lstmOut, (h, _) = self.lstm(packed_embeds)

        paddedOut = pad_packed_sequence(lstmOut, sortedLen)

        # print(paddedOut)

        fcOut = self.fc(paddedOut[0])

        fcOut = fcOut * mask.cuda()
        fcOut = fcOut[reverseIndices]

        return fcOut

    def loss_func(self, input, target):
        return self.loss(input, target)


class Trainer(object):
    def __init__(self, model, lr):
        self.model = model
        self.learningRate = lr
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.learningRate,
                                          weight_decay=1e-5)
        self.bestResult = 0
        self.positive = np.eye(2)[1]
        self.negative = np.eye(2)[0]

    def train_mini_batch(self, batch, length):
        token, case, char, trueLabel = batch
        maxLen = max([x for x in length])

        mask = np.zeros([len(token), maxLen, 2])
        for i, l in enumerate(length):
            mask[i][:l][:] = 1

        lids = []
        for s in trueLabel:
            for j in s:
                lids.append(j)
        lids = np.array(lids)

        self.optimizer.zero_grad()
        result = self.model(token, case, char)
        result = result.masked_select(torch.from_numpy(mask).byte().cuda()).contiguous().view(-1, 2)

        # print(result)
        # print(lids)

        label = torch.from_numpy(lids).cuda()
        loss = self.model.loss_func(result, label)
        (loss).backward()
        self.optimizer.step()
        return loss.data

    def test(self, batch, length):
        token, case, char = batch
        maxLen = max([x for x in length])

        mask = np.zeros([len(token), maxLen, 2])
        for i, l in enumerate(length):
            mask[i][:l][:] = 1

        result = self.model(token, case, char)

        result = result.masked_select(torch.from_numpy(mask).byte().cuda()).contiguous().view(-1, 2)
        # print(result)
        pred = torch.argmax(result, dim=1)
        return pred.cpu().numpy()

    def save(self, dir):
        if dir is not None:
            torch.save(self.model.state_dict(), dir)


if __name__ == "__main__":
    import argparse

    torch.manual_seed(10)

    parser = argparse.ArgumentParser(description="PU NER")
    parser.add_argument('--flag', default="PER")
    parser.add_argument('--dataset', default="conll2003")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pert', type=float, default=1.0)
    args = parser.parse_args()

    dp = DataPrepare(args.dataset)
    mutils = SupervisedModelUtils(dp)
    trainSet, validSet, testSet = mutils.load_dataset(args.flag, args.dataset, args.pert)
    trainSize = len(trainSet)
    validSize = len(validSet)
    testSize = len(testSet)
    print(("train set size: {}, valid set size: {}, test set size: {}").format(trainSize, validSize, testSize))

    charcnn = CharCNN(dp.char2Idx)
    wordnet = WordNet(dp.wordEmbeddings, dp.word2Idx)
    casenet = CaseNet(dp.caseEmbeddings, dp.case2Idx)
    supModel = SupervisedModel(dp, charcnn, wordnet, casenet, 138, 200, 1)
    if torch.cuda.is_available:
        charcnn.cuda()
        wordnet.cuda()
        casenet.cuda()
        supModel.cuda()
    trainer = Trainer(supModel, args.lr)
    bar = ProgressBar(maxval=int((len(trainSet) - 1) / 400))
    time = 0

    train_sentences = dp.read_origin_file("data/" + args.dataset + "/train.txt")
    trainSize = int(len(train_sentences) * args.pert)
    train_sentences = train_sentences[:trainSize]
    train_words = []
    train_efs = []
    for s in train_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        train_words.append(temp)
        train_efs.append(temp2)

    valid_sentences = dp.read_origin_file("data/" + args.dataset + "/valid.txt")
    valid_words = []
    valid_efs = []
    for s in valid_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        valid_words.append(temp)
        valid_efs.append(temp2)

    test_sentences = dp.read_origin_file("data/" + args.dataset + "/test.txt")
    test_words = []
    test_efs = []
    for s in test_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        test_words.append(temp)
        test_efs.append(temp2)

    for e in range(1, 1000):
        print("Epoch: {}".format(e))
        bar.start()
        losses = []
        for step, (x_word_batch, x_case_batch, x_char_batch, y_true_batch) in enumerate(
                mutils.iterateSet(trainSet, batchSize=400, mode="TEST", shuffle=True)):
            bar.update(step)
            batch = [x_word_batch, x_case_batch, x_char_batch, y_true_batch]
            lengths = [len(x) for x in x_word_batch]
            loss = trainer.train_mini_batch(batch, lengths)
            losses.append(loss)
        meanLoss = np.mean(np.array(losses))
        print("loss: {}".format(meanLoss))

        if e % 10 == 0:
            # train set
            pred_train = []
            corr_train = []
            for step, (x_word_train_batch, x_case_train_batch, x_char_train_batch, y_true_train_batch) in enumerate(
                    mutils.iterateSet(trainSet, batchSize=100, shuffle=False, mode="TEST")):
                trainBatch = [x_word_train_batch, x_case_train_batch, x_char_train_batch]
                y_true = []

                for i, s in enumerate(y_true_train_batch):
                    for j, item in enumerate(s):
                        y_true.append(y_true_train_batch[i][j])

                y_true = np.array(y_true)
                lengths = [len(x) for x in x_word_train_batch]
                predLabels = trainer.test(trainBatch, lengths)
                assert len(predLabels) == len(y_true)
                start = 0
                for i, l in enumerate(lengths):
                    end = start + l
                    p = predLabels[start:end]
                    c = y_true[start:end]
                    pred_train.append(p)
                    corr_train.append(c)
                    start = end
            assert len(pred_train) == len(corr_train)

            newSentences = []
            for i, s in enumerate(train_words):
                sent = []
                for j, item in enumerate(s):
                    sent.append([item, train_efs[i][j], pred_train[i][j]])
                newSentences.append(sent)

            newSentences_, newLabels, newPreds = dp.wordLevelGeneration(newSentences)
            p, r, f1 = dp.compute_precision_recall_f1(newLabels, newPreds, args.flag, 1)
            print("Precision: {}, Recall: {}, F1: {}".format(p, r, f1))

            # validation set

            pred_valid = []
            corr_valid = []
            for step, (x_word_test_batch, x_case_test_batch, x_char_test_batch, y_true_test_batch) in enumerate(
                    mutils.iterateSet(validSet, batchSize=100, shuffle=False, mode="TEST")):
                validBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch]

                y_true_test = []

                for i, s in enumerate(y_true_test_batch):
                    for j, item in enumerate(s):
                        y_true_test.append(y_true_test_batch[i][j])
                y_true_test = np.array(y_true_test)

                lengths = [len(x) for x in x_word_test_batch]
                predLabels = trainer.test(validBatch, lengths)
                assert len(predLabels) == len(y_true_test)
                start = 0
                for i, l in enumerate(lengths):
                    end = start + l
                    p = predLabels[start:end]
                    c = y_true_test[start:end]
                    pred_valid.append(p)
                    corr_valid.append(c)
                    start = end

            assert len(pred_valid) == len(corr_valid)

            newSentences = []
            for i, s in enumerate(valid_words):
                sent = []
                for j, item in enumerate(s):
                    sent.append([item, valid_efs[i][j], pred_valid[i][j]])
                newSentences.append(sent)

            newSentences_, newLabels, newPreds = dp.wordLevelGeneration(newSentences)
            # print(newPreds)
            p_valid, r_valid, f1_valid = dp.compute_precision_recall_f1(newLabels, newPreds, args.flag, 1)
            print("Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid))

            if f1_valid <= trainer.bestResult:
                time += 1
            else:
                trainer.bestResult = f1_valid
                time = 0
                trainer.save(
                    ("saved_model/supervised_" + args.dataset + "_" + args.flag + "_" + str(args.pert) + ".model"))
            if time > 5:
                print(("BEST RESULT ON VALIDATE DATA:{}").format(trainer.bestResult))
                break

    supModel.load_state_dict(
        torch.load("saved_model/supervised_" + args.dataset + "_" + args.flag + "_" + str(args.pert) + ".model"))

    pred_test = []
    corr_test = []
    for step, (
            x_word_test_batch, x_case_test_batch, x_char_test_batch,
            y_test_batch) in enumerate(
        mutils.iterateSet(testSet, batchSize=100, mode="TEST", shuffle=False)):
        testBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch]
        correcLabels = []
        for x in y_test_batch:
            for xi in x:
                correcLabels.append(xi)
        lengths = [len(x) for x in x_word_test_batch]
        predLabels = trainer.test(testBatch, lengths)
        correcLabels = np.array(correcLabels)
        assert len(predLabels) == len(correcLabels)

        start = 0
        for i, l in enumerate(lengths):
            end = start + l
            p = predLabels[start:end]
            c = correcLabels[start:end]
            pred_test.append(p)
            corr_test.append(c)
            start = end

    newSentencesTest = []
    for i, s in enumerate(test_words):
        sent = []
        assert len(s) == len(test_efs[i]) == len(pred_test[i])
        for j, item in enumerate(s):
            sent.append([item, test_efs[i][j], pred_test[i][j]])
        newSentencesTest.append(sent)

    newSentencesValid_, newLabelsValid, newPredsValid = dp.wordLevelGeneration(newSentencesTest)
    p_test, r_test, f1_test = dp.compute_precision_recall_f1(newLabelsValid, newPredsValid, args.flag,
                                                             1)
    print("Test Result: Precision: {}, Recall: {}, F1: {}".format(p_test, r_test, f1_test))
