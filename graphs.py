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
        for edge in self.Elist:
            from_vertex, to_vertex, weight = edge
            self.Elist_dict_oneway[(int(from_vertex[-1]), int(to_vertex[-1]))] = weight
        return self.Elist_dict_oneway


class SpanningTree:
    def __init__(self, graph: Graph):
        self.spanning_edges = []
        self.graph_edges = graph.Elist_dict_oneway
        self.edges_sorted = []

    def sort_graph_edges(self):
        self.edges_sorted = sorted(list((value, key) for key, value in self.graph_edges.items()))
        return self.edges_sorted


class Stack:
    def __init__(self):
        self.stack = []

    def pop(self):
        element = self.stack.pop(0)
        return element

    def push(self, element):
        self.stack.insert(0, element)

    def size(self):
        return len(self.stack)

    def empty(self):
        return True if self.size() == 0 else False

    def top(self):
        return self.stack[0] if self.size() > 0 else self.stack

    def all(self):
        return self.stack

    def remove_element(self, element):
        self.stack.remove(element)


class Cycle:

    def __init__(self, graph: Graph):
        self.edges = graph.Elist
        self.vertexes = graph.V
        self.vertexes_list = graph.Vlist
        self.adjacency_matrix = graph.adjacency
        self.all_vertex_nbr = {}
        self.stack_vertex = []
        self.cycle_vertex = []


    def vertex_nbr(self, vertex):
        nbr_list = []
        for index, value in enumerate(self.adjacency_matrix[int(vertex[-1])]):
            if value == 1:
                nbr_list.append(f"v{index}")
        self.all_vertex_nbr[vertex] = nbr_list

    def get_all_vertex_nbr(self):
        for vertex in self.vertexes_list:
            self.vertex_nbr(vertex)
        return self.all_vertex_nbr

    def stack_dfs(self):
        visited = [0 for _ in range(self.vertexes)]
        stack = Stack()
        v = 'v0'
        stack.push(v)
        visited[int(v[-1])] = 1
        while not stack.empty():
            v = stack.top()
            stack.pop()
            self.stack_vertex.append(v)
            for nbr in self.all_vertex_nbr[v]:
                if visited[int(nbr[-1])] == 1:
                    continue
                stack.push(nbr)
                # if v in self.all_vertex_nbr[nbr]:
                #     self.all_vertex_nbr[nbr].remove(v)
                visited[int(nbr[-1])] = 1

    def cycle_check(self):
        current_vertex = 'v0'
        visited = [False for _ in range(self.vertexes)]
        stack = Stack()
        stack.push(current_vertex)
        stack.push(-1)
        visited[int(current_vertex[-1])] = True
        while not stack.empty():
            from_vertex = stack.top()
            stack.pop()
            current_vertex = stack.top()
            stack.pop()
            for nbr in self.all_vertex_nbr[current_vertex]:
                if not visited[int(nbr[-1])]:
                    stack.push(nbr)
                    stack.push(current_vertex)
                    visited[int(nbr[-1])] = True
                else:
                    if nbr != from_vertex: #czy nasz sąsiad jest inny od wierzchołka z którego przyszliśmy ?
                        return True
        return False



'''
Draw graph with labels
'''
g = Graph(6,6)
# g.add_edge([('v0', 'v1', 5), ('v0', 'v2', 5), ('v0', 'v3', 2), ('v0', 'v4', 1),
#             ('v1', 'v3', 2), ('v1', 'v4', 1), ('v1', 'v2', 7),
#             ('v2', 'v4', 10), ('v2', 'v3', 20),
#             ('v3', 'v4', 6)
#             ])

g.add_edge([('v0', 'v1', 5),
            ('v1', 'v2', 5),
            ('v1', 'v5', 5),
            ('v1', 'v4', 5),
            ('v5', 'v4', 5),
            ('v2', 'v3', 5)])

# g.add_edge([('v0', 'v1', 5),
#             ('v0', 'v3', 5),
#             ('v1', 'v2', 5),
#             ('v2', 'v3', 5)
#             ])

# g.add_edge([('v0', 'v1', 5),
#             ('v0', 'v3', 5),
#             ('v0', 'v2', 5),
#             ('v2', 'v3', 5),
#             ('v3', 'v1', 5),
#             ('v2', 'v4', 5),
#             ('v4', 'v6', 5),
#             ('v3', 'v5', 5)])



adjacency_matrix = g.create_adjacency_matrix()
edge_list_dict_both_way = g.convert_edges_list_to_dict()
edge_list_dict = g.convert_edges_list_to_dict_one_way()

c = Cycle(g)
print(c.get_all_vertex_nbr())
print(c.cycle_check())
print(c.cycle_vertex)



# spanning_tree = SpanningTree(g)
# spanning_tree.sort_graph_edges()
#
A = np.matrix(adjacency_matrix)
# print(f"Adjancency matrix: \n{A}")
G = nx.from_numpy_matrix(A)
pos = nx.spring_layout(G)
# print(f"Weight of edges:")
# pprint(edge_list_dict)
'''green color for cycle vertex'''
color_map = ['blue' for _ in range(g.V)]
for node in c.cycle_vertex:
    color_map[int(node[-1])] = 'green'

nx.draw(G, pos, with_labels=True, node_color=color_map)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_list_dict_both_way)
plt.axis('off')
plt.show()





