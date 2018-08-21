# -*- coding: utf-8 -*-
# @Time    : 2018/8/12 18:36
# @Author  : Xiaoyu Xing
# @File    : lstm_crf.py

import torch
import torch.nn as nn
from sub_model import TimeDistributed
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
UNK_IDX = 3


class LSTM_CRF(nn.Module):
    def __init__(self, dp, charModel, wordModel, caseModel, inputSize, hiddenSize, layerNum, numTags):
        super(LSTM_CRF, self).__init__()
        self.dp = dp
        self.charModel = TimeDistributed(charModel, self.dp.char2Idx)
        self.wordModel = wordModel
        self.caseModel = caseModel
        self.lstm = nn.LSTM(inputSize, hiddenSize, layerNum, bias=0.5, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hiddenSize, numTags),



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
