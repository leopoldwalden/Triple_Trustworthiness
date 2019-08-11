import pandas as pd
import json
from tqdm import tqdm
from tqdm._tqdm import trange
from macro import ROOT_PATH

# ROOT_PATH = "/home/jilei/Desktop/PycharmProjects/Triple_Trustworthiness"

df = pd.read_csv(ROOT_PATH+"/data/all_r_list.csv")
# print(df)
heads = df["head"].tolist()
tails = df["tail"].tolist()
relations = df["relation"].tolist()

entities = heads + tails
# print(len(entities))
entities = list(set(entities))
# print(len(entities))

# print(len(relation))
relations = list(set(relations))
# print(len(relations))
# print(relations)

ent2idx = {entity:i for i,entity in enumerate(entities)}
rel2idx = {relation:i for i,relation in enumerate(relations)}

with open(ROOT_PATH+"/data/ent2idx.csv","w+") as f:
    [print(str(k)+","+str(v),file=f) for k,v in ent2idx.items()]

with open(ROOT_PATH+"/data/rel2idx.csv","w+") as f:
    [print(str(k)+","+str(v),file=f) for k,v in rel2idx.items()]

# df = df.head(10)
heads = df["head"]
tails = df["tail"]
relations = df["relation"]

idx_df = pd.DataFrame(columns=["head","tail","relation"])

for i in trange(len(df)):
    # print(heads[i])
    heads[i] = ent2idx[heads[i]]
    # print(heads[i])
    tails[i] = ent2idx[tails[i]]
    relations[i] = rel2idx[relations[i]]

idx_df["head"] = heads
idx_df["tail"] = tails
idx_df["relation"] = relations
# idx_df = idx_df.join(heads)
# idx_df = idx_df.join(tails)
# idx_df = idx_df.join(relations)

idx_df.to_csv(ROOT_PATH+"/data/graph.csv",index=False)





