from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import copy

ROOT_PATH = "/home/jilei/Desktop/PycharmProjects/Triple_Trustworthiness"
model = Word2Vec.load(ROOT_PATH+"/model/embeddings.model")

def cal_rel_norm(model, relation):
    # df = pd.read_csv
    path = ROOT_PATH+"/data/graph.csv"
    f = open(path,"r")
    triples = []
    for line in f.readlines():
        # print(line.strip("\n"))
        line = line.strip("\n").split(",")
        if line[2]==relation:
            triples.append([line[0],line[1]])# [head,tail]

    print(len(triples))

    relation_vec_sum = np.zeros(64)
    for triple in triples:
        # print(triple[0],triple[1])
        # print(model.wv[triple[0]])
        relation_vec_sum += model.wv[triple[0]] - model.wv[triple[1]]
    # vec_norm =
    norm_vec = relation_vec_sum/len(triples)
    return norm_vec

def cal_relation_embeding():
    relation_dict = {}
    f = open(ROOT_PATH+"/data/rel2idx.csv","r")
    for line in f.readlines():
        line = line.strip("\n").split(",")
        relation_dict[line[0]] = line[1]

    rel_emb_dict = {}
    for relation, index in relation_dict.items():
        norm_vec = cal_rel_norm(model,index)
        rel_emb_dict[relation] = np.array([norm_vec])

    relation_embeding = np.array([np.zeros(64)])
    # print(relation_embeding)
    with open(ROOT_PATH+"/data/relation_save_order.txt","w+") as f:
        for relation, embedding in rel_emb_dict.items():
            relation_embeding = np.append(relation_embeding,embedding,axis=0)
            print(relation,file = f)

    relation_embeding = relation_embeding[1:]
    # print(relation_embeding)
    print(relation_embeding.shape)
    # print(norm_vec)

    np.save(ROOT_PATH+"/relation_vec.npy",relation_embeding)

    return relation_embeding

