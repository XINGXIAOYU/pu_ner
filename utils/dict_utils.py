# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 12:45
# @Author  : Xiaoyu Xing
# @File    : dictUtils.py
import numpy as np
from collections import defaultdict


class DictUtils(object):
    def __init__(self):
        self.tag2Idx = {
            "PER": 0, "LOC": 1, "ORG": 2, "MISC": 3
        }
        self.idx2tag = {
            0: "PER", 1: "LOC", 2: "ORG", 3: "MISC"
        }

    def lookup_in_Dic(self, dicFile, sentences, tag, windowSize):
        tagIdx = self.tag2Idx[tag]
        dic = []
        labeled_word = set()
        with open(dicFile, "r") as fw:
            for line in fw:
                line = line.strip()
                if len(line) > 0:
                    dic.append(line)
        for i, sentence in enumerate(sentences):
            wordList = [word for word, label, dicFlags in sentence]
            trueLabelList = [label for word, label, dicFlags in sentence]
            isFlag = np.zeros(len(trueLabelList))
            j = 0
            while j < len(wordList):
                Len = min(windowSize, len(wordList) - j)
                k = Len
                while k >= 1:
                    words = wordList[j:j + k]
                    words_ = " ".join([w for w in words])

                    if words_ in dic:
                        # print(words_)
                        isFlag[j:j + k] = 1
                        j = j + k
                        break
                    k -= 1
                j += 1

            for m, flag in enumerate(isFlag):
                if flag == 1:
                    labeled_word.add(sentence[m][0])
                    # print(wordList[m] + " " + trueLabelList[m])
                    sentence[m][2][tagIdx] = 1

        return sentences, len(labeled_word)

        # def lookup_in_Dic(self, dicFile, sentences, tag, w):
        #     tagIdx = self.tag2Idx[tag]
        #     dic = []
        #
        #     with open(dicFile, "r") as fw:
        #         for line in fw:
        #             line = line.strip()
        #             if len(line) > 0:
        #                 dic.append(line)
        #     for i, sentence in enumerate(sentences):
        #         wordList = [word for word, label, dicFlags in sentence]
        #         trueLabelList = [label for word, label, dicFlags in sentence]
        #         isFlag = np.zeros(len(trueLabelList))
        #         for item in dic:
        #             item = item.split()
        #             s, e, mmax = self.match(wordList, item)
        #             if mmax == len(item):
        #                 isFlag[s:e + 1] = 1
        #
        #         for j, flag in enumerate(isFlag):
        #             if flag == 1:
        #                 print(wordList[j] + " " + trueLabelList[j])
        #                 sentence[j][2][tagIdx] = 1
        #     return sentences
        #
        # def match(self, s1, s2):
        #     m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        #     end = -1
        #     mmax = 0
        #     for i in range(len(s1)):
        #         for j in range(len(s2)):
        #             if s1[i] == s2[j]:
        #                 m[i + 1][j + 1] = m[i][j] + 1
        #                 if m[i + 1][j + 1] > mmax:
        #                     mmax = m[i + 1][j + 1]
        #                     end = i
        #     start = end + 1 - mmax
        #     return start, end, mmax
