import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import logging

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
        self.number = len(list(temp))
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


    def add_vertex(self):
        number = len(self.Vlist)
        self.Vlist.append(f"v{number}")


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
        wag_sum = 0
        checked_vertex = set()
        condition = False
        while not condition:
            edge = self.edges_sorted.pop()
            from_vertex, to_vertex, weight = edge
            self.minimum_tree_edges.append(edge)
            g = Graph(self.minimum_tree_edges)
            g.create_adjacency_matrix()
            g.convert_edges_list_to_dict()
            g.convert_edges_list_to_dict_one_way()
            cycle = Cycle(g)
            if cycle.cycle_check():
                self.minimum_tree_edges.remove(edge)
            checked_vertex.add(from_vertex)
            checked_vertex.add(to_vertex)
            if len(list(checked_vertex)) == len(self.vertex_list):
                my_graph = Graph(self.minimum_tree_edges)
                my_graph.create_adjacency_matrix()
                my_graph.convert_edges_list_to_dict()
                my_graph.convert_edges_list_to_dict_one_way()
                my_cycle = Cycle(my_graph)
                condition = all(my_cycle.dfs().values())   #jeśli wszystkie wierzchołi są już w drzewie to sprawdzamy spójność
        print(self.minimum_tree_edges)
        w_sum = sum([weight for *args, weight in self.minimum_tree_edges])
        return Graph(self.minimum_tree_edges), w_sum












'''
Draw graph with labels
'''

edges = ([('v0', 'v1', 5),
        ('v1', 'v2', 9),
        ('v2', 'v7', 3),
        ('v6', 'v7', 9),
        ('v6', 'v5', 6),
        ('v0', 'v3', 9),
        ('v1', 'v4', 8),
          ('v6', 'v4', 1),
          ('v2', 'v6', 5),
          ('v2', 'v4', 4),
          ('v2', 'v3', 9),
          ('v7', 'v1', 7),
          ('v0', 'v6', 3),
          ('v1', 'v5', 6),
          ('v3', 'v6', 1),
          ('v5', 'v4', 1)])
g = Graph(edges)


tree = MinimumSpanningTree(g)
g, weight_sum = tree.generate_min_tree()
print(weight_sum)


adjacency_matrix = g.create_adjacency_matrix()
edge_list_dict_both_way = g.convert_edges_list_to_dict()
edge_list_dict = g.convert_edges_list_to_dict_one_way()

A = np.matrix(adjacency_matrix)
G = nx.from_numpy_matrix(A)
pos = nx.spring_layout(G)

'''green color for cycle vertex'''

nx.draw(G, pos, with_labels=True, node_color='orange', edge_color='blue')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_list_dict_both_way)
plt.axis('off')
plt.savefig("graph", format="png")
plt.figtext(.5, .9, f"Spannig Tree weight sum = {weight_sum}")
plt.show()

if 'graph.jpg' in os.listdir():
    os.remove('graph.jpg')
os.rename('graph', 'graph.jpg')







