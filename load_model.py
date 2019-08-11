from gensim.models import Word2Vec
ROOT_PATH = "/home/jilei/Desktop/PycharmProjects/Triple_Trustworthiness"
model = Word2Vec.load(ROOT_PATH+"/model/embeddings.model")

# model.

result = model.wv.most_similar("2")
print(result)
print(model.wv.__class__)
print(model.wv["2829"])
print(model.wv["2"].shape)