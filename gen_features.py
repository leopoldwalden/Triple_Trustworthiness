from gensim.models import Word2Vec
import numpy as np
from cal_relation_vec import *
from macro import ROOT_PATH


def sample2feature():
    # load model
    # ROOT_PATH = "/home/jilei/Desktop/PycharmProjects/Triple_Trustworthiness"
    model = Word2Vec.load(ROOT_PATH + "/model/embeddings.model")

    # load samples
    samples = []
    f = open(ROOT_PATH + "/data/samples.csv", "r")
    for line in f.readlines():
        line = line.strip("\n").split(",")
        samples.append(line)

    # load relation_vec
    relation_vec = cal_relation_embeding()
    # relation_vec = np.load("./data/relation_vec.npy")
    # print(relation_vec.shape)
    relation_dict = {}
    f = open(ROOT_PATH + "/data/rel2idx.csv", "r")
    for line in f.readlines():
        line = line.strip("\n").split(",")
        relation_dict[line[0]] = line[1]
    f = open(ROOT_PATH + "/data/relation_save_order.txt", "r")
    relation_list = []
    for line in f:
        # print(line)
        line = line.strip("\n")
        # print(line)
        relation_list.append(relation_dict[line])

    # conctate head tail and relation
    features = np.array([np.zeros(1+64*3)])
    # print(features.shape)
    for sample in samples:
        feature = np.zeros(1)

        label = np.array(sample[0])
        head_emb = model.wv[sample[1]]
        # print(head_emb)
        tail_emb = model.wv[sample[2]]
        relation_emb = relation_vec[relation_list.index(sample[3])]
        # print(relation_list.index(sample[3]))
        # print(relation_emb.shape)
        feature = np.append(feature, label)
        feature = np.append(feature, head_emb)
        feature = np.append(feature, tail_emb)
        feature = np.append(feature, relation_emb)
        feature = feature[1:]
        # print(feature.shape)
        features = np.append(features,np.array([feature]),axis=0)
        # print(features.shape)
    return features


sample2feature(samples)