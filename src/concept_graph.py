import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('/home/rwiddhi/datadebias/metadata/co_occurrence_matrix_coco.csv')


# Renaming the first column for clarity
df.rename(columns={df.columns[0]: "Item" }, inplace=True)

# Create a graph from the dataframe
G = nx.Graph()

# Adding nodes and weighted edges
for i in range(len(df)):
    for j in range(1, len(df.columns)):
        weight = df.iloc[i, j]
        if weight > 0:  # Only add edges with non-zero weight
            G.add_edge(df.iloc[i, 0], df.columns[j], weight=weight)

# Print the number of nodes and edges in the graph
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
#Print the nodes
print(G.nodes())

print('man' in G.nodes())
print('woman' in G.nodes())

with open('/media/exx/HDD/rwiddhic/coco/annotations/unique_categories.txt', 'r') as f:
    objects = f.readlines()
    objects = [x.strip() for x in objects]


#Check if objects list is the same as G.nodes()
print(set(objects) == set(list(G.nodes())))
#Check difference between object list and G.nodes()
print(set(objects) - set(list(G.nodes())))
#Save the graph to a gexf file
nx.write_gexf(G, '/home/rwiddhi/datadebias/concept_graphs/concept_graph_coco.gexf')