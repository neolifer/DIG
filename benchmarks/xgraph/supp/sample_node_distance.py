import torch
from torch_geometric.datasets import CitationFull, Planetoid
import torch_geometric.transforms as T
import torch_geometric as pyg
import networkx
import argparse
import pickle as pk
from tqdm import tqdm
import random
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./datasets/')
parser.add_argument('--dataset_name', default='Cora')

parser = parser.parse_args()

dataset = Planetoid('./datasets', parser.dataset_name,split="public", transform = T.NormalizeFeatures())

G = pyg.utils.to_networkx(dataset[0])
distance_pair = {}
# for start in tqdm(G.nodes):
#     for end in G.nodes:
#         if start == end:
#             continue
#         try:
#             distance_pair[(start, end)] = networkx.algorithms.shortest_path_length(G, source = start, target= end)
#         except:
#             continue
# pk.dump(distance_pair, open(f'{parser.dataset_name}_distance_pair.pk','wb'))
distance_pair = pk.load(open(f'{parser.dataset_name}_distance_pair.pk','rb'))
start = random.randint(0, len(G.nodes) - 1)
nl = [start]
all = set(G.nodes)
all.remove(start)
while True:
    to_be_removed = set()
    for e in all:
        for n in nl:
            if (e,n) in distance_pair:
                if distance_pair[(e, n)] <= 2:
                    to_be_removed.add(e)
                    continue
            if (n, e) in distance_pair:
                if distance_pair[(n, e)] <= 2:
                    to_be_removed.add(e)
                    continue
    all = all - to_be_removed
    nl += random.sample(all, 1)
    all.remove(nl[-1])
    print(len(all))
    if len(all) == 0:
        break
pk.dump(nl, open(f'{parser.dataset_name}_exclude_nodes.pk','wb'))
# distances = []
# nl = pk.load(open(f'{parser.dataset_name}_exclude_nodes.pk','rb'))
# for start in tqdm(nl):
#     for end in nl:
#         if (start, end) in distance_pair:
#             distances.append(distance_pair[(start,end)])
# print(min(distances))



