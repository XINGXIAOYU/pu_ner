# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 12:01
# @Author  : Xiaoyu Xing
# @File    : test.py

from utils.data_utils import DataPrepare


def compute_all_prior(sentences):
    alltokens = 0
    labeledtokens = 0
    unlabeledtokens = 0
    for i, sentence in enumerate(sentences):
        for word, ef, lf in sentence:
            alltokens += 1
            if ef == 1:
                labeledtokens += 1
            else:
                unlabeledtokens += 1

    prior = float(labeledtokens / alltokens)
    assert alltokens == labeledtokens + unlabeledtokens
    print(prior)


if __name__ == "__main__":
    dataset = "muc"
    flag = "PER"
    dp = DataPrepare(dataset)
    filename = "data/"+dataset + "/train." + flag + ".txt"
    sentences = dp.read_processed_file(filename, flag)
    compute_all_prior(sentences)
    flag = "LOC"
    filename = "data/" + dataset + "/train." + flag + ".txt"
    sentences = dp.read_processed_file(filename, flag)
    compute_all_prior(sentences)
    flag = "ORG"
    filename = "data/" + dataset + "/train." + flag + ".txt"
    sentences = dp.read_processed_file(filename, flag)
    compute_all_prior(sentences)
    # flag = "MISC"
    # filename = "data/" + dataset + "/train." + flag + ".txt"
    # sentences = dp.read_processed_file(filename, flag)
    # compute_all_prior(sentences)

"""
conll2003
0.05465055176037835
0.040747270664617107
0.04923362521547384
0.02255661253014178

wikigold
0.04028527370855821
0.03700848111025443
0.050269853508095604
0.0358905165767155

webpages
0.08747044917257683
0.030929866036249015
0.09081954294720253
0.0884554767533491

muc
0.02234143426812319
0.025354930053125852
0.037616740488653926

"""