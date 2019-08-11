import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
# g = nx.read_edgelist("./graph.csv",create_using = nx.DiGraph(),nodetype = str)
# print(g.__class__)
# print(nx.spring_layout(g))

# labels = dict((i,i) for i in g.nodes())
# nx.draw_networkx_labels(g, labels, pos=nx.spring_layout(g))
# plt.savefig(filename)
# plt.show

def createGraph(filename):
    G = nx.DiGraph()
    for line in open(filename) :
        strlist = line.split(",")
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        weight = int(strlist[2].strip("\n"))
        G.add_weighted_edges_from([(n1, n2, 1)]) #G.add_edges_from([(n1, n2)])
    return G

G = createGraph("./graph.csv")

nodes = G.nodes
edges = G.edges
# edge = G.edges['2829', '5154']
print(G.__class__)
print(nodes)
print(edges)
# print(edge)

# nx.draw_networkx(G)
# print(len(nodes))
# print(len(edges))



# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

graph = G
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

