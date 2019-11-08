import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import logging
plt.rcParams["figure.figsize"] = (15,15)
logging.basicConfig(format="%(message)s",level=logging.INFO)
logger = logging.getLogger('root')

class Graph:

    def __init__(self, edges):
        temp = []   #tablica która przyda siędo wyciągania wierzchołków z podanych krawędzi
        for edge in edges:
            from_vertex, to_vertex, weight = edge
            temp.append(from_vertex)
            temp.append(to_vertex)
        self.Vlist = list(set(temp))
        self.numbers = [int(vertex[-1]) for vertex in temp]
        self.Elist = edges
        self.Elist_dict_bothway = {} #needed for labeling
        self.Elist_dict_oneway = {} #need for prim-kruskal algorithm
        self.adjacency = []
        for i in range(max(self.numbers)+1):
            row = []
            for j in range(max(self.numbers)+1):
                row.append(0)
            self.adjacency.append(row)
        self.create_adjacency_matrix()
        self.convert_edges_list_to_dict_one_way()
        self.convert_edges_list_to_dict()


    def add_vertex(self):
        number = len(self.Vlist)
        self.Vlist.append(f"v{number}")


    def create_adjacency_matrix(self):
        for edge in self.Elist:
            from_vertex, to_vertex, weight = edge
            self.adjacency[int(from_vertex[-1])][int(to_vertex[-1])] = 1
            self.adjacency[int(to_vertex[-1])][int(from_vertex[-1])] = 1
        return self.adjacency

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


class Stack:
    def __init__(self, mylist=[]):
        self.stack = mylist

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

    def clear_stack(self):
        self.stack.clear()

    def __str__(self):
        return str(self.stack)


class Cycle:

    def __init__(self, graph: Graph):
        self.edges = graph.Elist
        self.vertexes_list = graph.Vlist
        self.adjacency_matrix = graph.adjacency
        self.all_vertex_nbr = {}

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

    def dfs(self):
        self.get_all_vertex_nbr()
        current_vertex = self.vertexes_list[0]
        visited = dict([key, False] for key in self.vertexes_list)
        stack = Stack()
        stack.clear_stack()
        stack.push(current_vertex)
        visited[current_vertex] = True
        while not stack.empty():
            current_vertex = stack.top()
            stack.pop()
            for nbr in self.all_vertex_nbr[current_vertex]:
                if visited[nbr]:
                    continue
                stack.push(nbr)
                visited[nbr] = True

        return visited

    def cycle_check(self):

        self.get_all_vertex_nbr()
        current_vertex = self.vertexes_list[0]
        visited = dict([key, False] for key in self.vertexes_list)
        stack = Stack()
        stack.clear_stack()
        stack.push(current_vertex)
        stack.push(-1)
        visited[current_vertex] = True
        while not stack.empty():
            from_vertex = stack.top()
            stack.pop()
            current_vertex = stack.top()
            stack.pop()
            for nbr in self.all_vertex_nbr[current_vertex]:
                if not visited[nbr]:
                    stack.push(nbr)
                    stack.push(current_vertex)
                    visited[nbr] = True
                else:
                    if nbr != from_vertex:
                        '''czy nasz sąsiad jest inny od wierzchołka z którego przyszliśmy ?
                        #jeśli sąsiad jest już odwiedzony i ten sąsiad nie jest wirzchołkiem z którego przyszliśmy to
                        oznacza, że graf posiada cykl, ponieważ znowu odwiedzamy wierzchołek w którym już byliśmy
                        '''
                        return True
        return False


class MinimumSpanningTree():

    def __init__(self, graph: Graph):
        self.edges = graph.Elist
        self.edges_dict = graph.convert_edges_list_to_dict_one_way()
        self.vertex_list = graph.Vlist
        self.edges_sorted = []
        self.minimum_tree_edges = []

    def sort_edges_in_weight_order(self):
        helper = sorted([(value, key) for key, value in self.edges_dict.items()])
        self.edges_sorted = Stack([(f"v{value[0]}", f"v{value[1]}", key) for key, value in helper])
        return self.edges_sorted

    def generate_min_tree(self):
        self.sort_edges_in_weight_order()
        checked_vertex = set()
        condition = False
        while not condition:
            edge = self.edges_sorted.pop()
            from_vertex, to_vertex, weight = edge
            self.minimum_tree_edges.append(edge)
            g = Graph(self.minimum_tree_edges)
            cycle = Cycle(g)
            if cycle.cycle_check():
                self.minimum_tree_edges.remove(edge)
            checked_vertex.add(from_vertex)
            checked_vertex.add(to_vertex)
            if len(list(checked_vertex)) == len(self.vertex_list):
                my_graph = Graph(self.minimum_tree_edges)
                my_cycle = Cycle(my_graph)
                condition = all(my_cycle.dfs().values())   #jeśli wszystkie wierzchołi są już w drzewie to sprawdzamy spójność algorytmem dfs
        print(self.minimum_tree_edges)
        w_sum = sum([weight for *args, weight in self.minimum_tree_edges])
        return Graph(self.minimum_tree_edges), w_sum, self.minimum_tree_edges

'''
Draw graph with labels
'''

# edges = ([('v0', 'v1', 5),
#         ('v1', 'v2', 9),
#         ('v2', 'v7', 3),
#         ('v6', 'v7', 9),
#         ('v6', 'v5', 6),
#         ('v0', 'v3', 9),
#         ('v1', 'v4', 8),
#           ('v6', 'v4', 1),
#           ('v2', 'v6', 5),
#           ('v2', 'v4', 4),
#           ('v2', 'v3', 9),
#           ('v7', 'v1', 7),
#           ('v0', 'v6', 3),
#           ('v1', 'v5', 6),
#           ('v3', 'v6', 1),
#           ('v5', 'v4', 1)])
edges = ([('v0', 'v1', 4),
          ('v1', 'v2', 2),
          ('v2', 'v3', 8),
          ('v3', 'v4', 6),
          ('v4', 'v0', 2),
          ('v0', 'v5', 1),
          ('v1', 'v5', 2),
          ('v4', 'v5', 7),
          ('v3', 'v5', 3)])

# edges = ([('v0', 'v1', 4),
#           ('v1', 'v2', 2),
#           ('v2', 'v3', 20),
#           ('v1', 'v4', 3),
#           ('v1', 'v5', 10),
#           ('v4', 'v5', 6)])

g = Graph(edges)


def generate_plot(g: Graph, min_span_tree=False):
    weight_sum = None
    A = np.matrix(g.adjacency)
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='orange', edge_color='green', font_size=20, node_size=400)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=g.convert_edges_list_to_dict(), font_size=20)
    if min_span_tree:
        tree = MinimumSpanningTree(g)
        z, weight_sum, tree_edges = tree.generate_min_tree()
        tree_edges = [(int(f[-1]), int(t[-1])) for f, t, *args in tree_edges]
        nx.draw_networkx_edges(G, pos, edgelist=tree_edges, edge_color='r', width=2)
    plt.axis('off')
    os.remove("static/images/graph") if "graph" in os.listdir("static/images") else None
    os.remove("static/images/graph.jpg") if 'graph.jpg' in os.listdir("static/images") else None
    plt.savefig("static/images/graph", format="png")
    os.chdir("static/images")
    os.rename('graph', 'graph.jpg')
    return weight_sum
    # plt.show()


generate_plot(g, min_span_tree=True)







