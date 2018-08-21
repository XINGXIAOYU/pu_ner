# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 22:27
# @Author  : Xiaoyu Xing
# @File    : supervised_model_evl.py

from utils.supervised_model_utils import SupervisedModelUtils
from utils.data_utils import DataPrepare
from supervised_model import SupervisedModel, Trainer
import torch
import argparse
from sub_model import CharCNN, CaseNet, WordNet
import numpy as np

torch.manual_seed(10)

parser = argparse.ArgumentParser(description="PU NER")
parser.add_argument('--model', default="")
parser.add_argument('--output', default=0)
parser.add_argument('--set', type=int, default=0)
parser.add_argument('--flag', default="PER")
parser.add_argument('--lr_rate', type=int, default=1e-4)
parser.add_argument('--dataset', default="conll2003")
parser.add_argument('--pert', type=float, default=1.0)
args = parser.parse_args()

dp = DataPrepare(args.dataset)
mutils = SupervisedModelUtils(dp)

trainSet, validSet, testSet = mutils.load_dataset(args.flag, args.dataset, args.pert)
setIter = [testSet, validSet]
detectionSet = setIter[args.set]
fileNameIter = ["data/" + args.dataset + "/test.txt", "data/" + args.dataset + "/valid.txt"]
fileName = fileNameIter[args.set]

charcnn = CharCNN(dp.char2Idx)
wordnet = WordNet(dp.wordEmbeddings, dp.word2Idx)
casenet = CaseNet(dp.caseEmbeddings, dp.case2Idx)

supModel = SupervisedModel(dp, charcnn, wordnet, casenet, 138, 200, 1)
supModel.load_state_dict(torch.load(args.model))
if torch.cuda.is_available:
    charcnn.cuda()
    wordnet.cuda()
    casenet.cuda()
    supModel.cuda()

pred_test = []
corr_test = []
trainer = Trainer(supModel, args.lr_rate)

for step, (x_word_test_batch, x_case_test_batch, x_char_test_batch, y_test_batch) in enumerate(
        mutils.iterateSet(detectionSet, batchSize=100, mode="TEST", shuffle=False)):
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

assert len(pred_test) == len(corr_test)

sentences = dp.read_origin_file(fileName)
words = []
efs = []
for s in sentences:
    temp = []
    temp2 = []
    for word, ef, lf in s:
        temp.append(word)
        temp2.append(ef)
    words.append(temp)
    efs.append(temp2)

newSentences = []
for i, s in enumerate(words):
    sent = []
    for j, item in enumerate(s):
        sent.append([item, efs[i][j], pred_test[i][j]])
    newSentences.append(sent)

newSentences_, newLabels, newPreds = dp.wordLevelGeneration(newSentences)
p, r, f1 = dp.compute_precision_recall_f1(newLabels, newPreds, args.flag, 1)
print("Precision: {}, Recall: {}, F1: {}".format(p, r, f1))

if args.output:
    outputFile = "result/supervised_" + args.dataset + "_" + args.flag + "_" + str(args.set) + ".txt"
    with open(outputFile, "w") as fw:
        for i, sent in enumerate(words):
            preds = pred_test[i]
            corrs = efs[i]
            for j, w in enumerate(sent):
                pred = preds[j]
                corr = corrs[j]
                fw.write(("{} {} {}\n").format(w, corr, pred))
            fw.write("\n")
