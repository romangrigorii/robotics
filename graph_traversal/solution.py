###
# This file holds the two solutions asked for the in the problem statement.
# The goal is to make it clear how k>0 is dealth with.
# This a file that can be expanded upon to compute the times of traversal and other metrics. 

from path_algs import Dijkstra, StackBasedApproach

class Solutions():
    def solution1(self, graph, pos_start, pos_end, k):
        if k == 0 : return -1 # no traversals
        traversal, optimal_path = Dijkstra(graph, pos_start, pos_end)
        traversals = [list(traversal)] # this represents the graph traversals we will take at every k
        while k > 1:
            traversal.append(optimal_path)
            k-=1

    def solution2(self, graph, pos_start, pos_end, k):
        explored_graph = {}
        stack = []
        traversals = []
        for k_i in range(k):
            visited, optimal_path, explored_graph, stack = StackBasedApproach(graph = graph, pos_start = pos_start, pos_end = pos_end, graph_traversed=explored_graph, pos_stack=stack, first = k_i == 0)
            traversals.append(optimal_path)
            print(optimal_path)
        return traversals
    