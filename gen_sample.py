import numpy as np
import random
from random import shuffle
from macro import ROOT_PATH

# ROOT_PATH = "/home/jilei/Desktop/PycharmProjects/Triple_Trustworthiness"

def gen_noise(triples:list):
    noise_triples = []
    length = len(triples)
    head_list, tail_list, relation_list = zip(*triples)

    # 1/2概率改变head,1/2概率改变tail
    for triple in triples:
        n = random.randint(0,length-1)
        p = random.random()
        # print(n, triple)
        if p > 0.5:
            noise_triple = [triple[0],triples[n][1],triple[2]]
        else:
            noise_triple = [triples[n][1],triple[1],triple[2]]
        noise_triples.append(noise_triple)
    return noise_triples

def gen_noise_sample():
    f = open(ROOT_PATH+"/data/graph.csv","r")
    triple_dict = {}
    for i, line in enumerate(f.readlines()):
        line = line.strip("\n").split(",")
        triple_dict[str(i)+","+line[0]+","+line[1]] = line[2]
    # print(len(triple_dict))

    relations = list(set([v for v in triple_dict.values()]))

    triple_clfed ={relation:[] for relation in relations}

    for head_tail,relation in triple_dict.items():
        # index = relations.index(rel)
        head_tail = head_tail.split(",")[1:]
        head_tail.append(relation)
        triple_clfed[relation].append(head_tail)

    noise_triple_clfed = {relation:gen_noise(triple_clfed[relation]) for relation in relations}

    noise_samples = []
    for relation in relations:
        for triple in noise_triple_clfed[relation]:
            noise_samples.append(triple)
    return noise_samples

def gen_labled_samples(triples,label):
    # print(triples)
    new_triples = []
    for triple in triples:
        triple.insert(0,label)
        new_triples.append(triple)
    return new_triples


def gen_samples():
    f = open(ROOT_PATH+"/data/graph.csv", "r")
    triple_dict = {}
    positive_samples = []
    for line in f.readlines():
        line = line.strip("\n").split(",")
        positive_samples.append(line)
    positive_samples = gen_labled_samples(positive_samples,"1")
    # print(positive_samples)
    # print(len(positive_samples))

    noise_samples = gen_noise_sample()
    # print(noise_samples)
    noise_samples = gen_labled_samples(noise_samples,"0")
    # print(len(noise_samples))
    # print(noise_samples)

    samples = positive_samples+noise_samples
    shuffle(samples)
    return samples

samples = gen_samples()
print(len(samples))
with open(ROOT_PATH+"/data/samples.csv","w+") as f:
    for sample in samples:
        print(",".join(sample),file = f)


