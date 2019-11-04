import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint


class Graph:
    def __init__(self, n, m):
        self.V = n  #vertex number
        self.E = m if m <= (n*(n-1)/2) else (n*(n-1)/2)  #edges number, masz number of edges is (n(n-1)/2)
        self.Vlist = []
        self.Elist = []
        self.Elist_dict_bothway = {} #needed for labeling
        self.Elist_dict_oneway = {} #need for prim-kruskal algorithm
        self.adjacency = []
        for i in range(0, self.V):
            self.Vlist.append(f"v{i}")
            row = []
            for j in range(0, self.V):
                row.append(0)
            self.adjacency.append(row)

    def add_vertex(self):
        number = len(self.Vlist)
        self.Vlist.append(f"v{number}")

    def add_edge(self, edges):
        for edge in edges:
            from_vertex, to_vertex, weight = edge
            if (from_vertex.lower() in self.Vlist) and (to_vertex.lower() in self.Vlist) and len(self.Elist) <= self.E:
                self.Elist.append((from_vertex, to_vertex, weight))
                print(f"From: {from_vertex}, To: {to_vertex}, Weight: {weight}")
            else:
                print(f"from_vertex: {from_vertex}, toVertes: {to_vertex} not in grah")

    def create_adjacency_matrix(self):
        for edge in self.Elist:
            from_vertex, to_vertex, weight = edge
            self.adjacency[int(from_vertex[-1])][int(to_vertex[-1])] = 1
            self.adjacency[int(to_vertex[-1])][int(from_vertex[-1])] = 1
        return self.adjacency

    def get_edges(self):
        return self.Elist

    def get_vertex(self):
        return self.Vlist

    def convert_edges_list_to_dict(self):
        for edge in self.Elist:
            from_vertex, to_vertex, weight = edge
            self.Elist_dict_bothway[(int(from_vertex[-1]), int(to_vertex[-1]))] = weight
            self.Elist_dict_bothway[(int(to_vertex[-1]), int(from_vertex[-1]))] = weight
        return self.Elist_dict_bothway

    def convert_edges_list_to_dict_one_way(self):
        pass



g = Graph(5,10)
g.add_edge([('v0', 'v1', 5), ('v0', 'v2', 5), ('v0', 'v3', 2), ('v0', 'v4', 1),
            ('v1', 'v3', 2), ('v1', 'v4', 1), ('v1', 'v2', 7),
            ('v2', 'v4', 10), ('v2', 'v3', 8),
            ('v3', 'v4', 6)
            ])

class SpanningTree:
    def __init__(self, graph: Graph):
        self.spanning_edges = []
        self.graph_edges = graph.Elist_dict_bothway

    def sort_graph_edges(self):
        pass

    def append_edges(self):
        pass



'''
Draw graph with labels
'''
adjacency_matrix = g.create_adjacency_matrix()
edge_list_dict = g.convert_edges_list_to_dict()
A = np.matrix(adjacency_matrix)
print(f"Adjancency matrix: \n{A}")
G = nx.from_numpy_matrix(A)
pos = nx.spring_layout(G)
print(f"Weight of edges: \n")
pprint(edge_list_dict)
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_list_dict)
plt.axis('off')
plt.show()





