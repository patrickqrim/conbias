import networkx as nx
import pandas as pd
from collections import Counter, OrderedDict
import json
import numpy as np
import itertools

def compute_clique_values(cliques, matrix):
    clique_values = {}
    # Loop through each clique
    for clique in cliques:
        sorted_clique = tuple(sorted(clique))  # Sort to avoid double counting
        # If we've already computed this clique, skip it
        if sorted_clique in clique_values:
            continue
        # Compute the value for the clique, here we'll take the minimum co-occurrence as an example
        value = min(matrix.loc[sorted_clique[i], sorted_clique[j]] for i in range(len(sorted_clique)) for j in range(i + 1, len(sorted_clique)))/11790
        clique_values[sorted_clique] = value
    return clique_values

def filter_cliques_by_percentile(clique_values, percentile_threshold):
    values = list(clique_values.values())
    percentile_value = np.percentile(values, percentile_threshold)
    # Filter cliques based on the computed percentile
    return [clique for clique, value in clique_values.items() if value > percentile_value], percentile_value



def remove_class_names(cliques_lb, cliques_wb):
    #For each item in clique, remove the class name if it is present
    for i in range(len(cliques_lb)):
        cliques_lb[i] = tuple([item for item in cliques_lb[i] if item != 'man'])

    for i in range(len(cliques_wb)):
        cliques_wb[i] = tuple([item for item in cliques_wb[i] if item != 'woman'])

    return cliques_lb, cliques_wb

G = nx.read_gexf('/home/rwiddhi/datadebias/concept_graphs/concept_graph_coco.gexf')
#Remove self loops
G.remove_edges_from(nx.selfloop_edges(G))
G = G.to_undirected()
print("Enumerating all cliques...")
#Only take sizes up to 4
all_cliques = list(itertools.takewhile(lambda x: len(x) <= 4, nx.enumerate_all_cliques(G)))

print("Done enumerating all cliques")
print("Number of cliques: ", len(all_cliques))
all_cliques_lb = [cl for cl in all_cliques if any(word == 'man' for word in cl)]
all_cliques_wb = [cl for cl in all_cliques if any(word == 'woman' for word in cl)]

#Remove landbird and waterbird from the cliques
all_cliques_lb, all_cliques_wb = remove_class_names(all_cliques_lb, all_cliques_wb)


all_cliques_lb_set = set(map(tuple, all_cliques_lb))
all_cliques_wb_set = set(map(tuple, all_cliques_wb))

#Ensure that the tuples are unique
assert len(all_cliques_lb) == len(all_cliques_lb_set)
assert len(all_cliques_wb) == len(all_cliques_wb_set)


common_cliques = all_cliques_lb_set.intersection(all_cliques_wb_set)

co_occurrence_matrix = pd.read_csv('/home/rwiddhi/datadebias/metadata/co_occurrence_matrix_coco.csv', index_col=0)
clique_number = max([len(clique) for clique in common_cliques])
print("Clique number: ", clique_number)

#We keep only cliques of size 2, 3 and 4
filtered_cliques = []
for k in range(2, clique_number + 1):
    # Filter for k-cliques
    print(f'Filtering for {k}-cliques')
    k_cliques = [clique for clique in common_cliques if len(clique) == k]
    clique_values = compute_clique_values(k_cliques, co_occurrence_matrix)

    #Get cliques above the 90th percentile
    cliques_above_percentile, percentile = filter_cliques_by_percentile(clique_values, 90)
    print(f'90th Percentile value: {percentile}')
    print(f'Cliques above 90th percentile: {len(cliques_above_percentile)}')
    filtered_cliques.extend(cliques_above_percentile)
    #filtered_cliques.extend(k_cliques)

print(len(common_cliques))
print(len(filtered_cliques))

#Save filtered cliques to a json file
with open('/home/rwiddhi/datadebias/metadata/common_cliques_filtered_coco.json', 'w') as f:
    json.dump(filtered_cliques, f)