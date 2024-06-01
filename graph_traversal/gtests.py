import unittest
from typing import Optional, List
from path_algs import *
from maze import *

class Test(unittest.TestCase, SearchAlgs):
    '''
    graph formats here are {start node : [[ connecting_node, cost], [connecting_node, cost], ...]}
    '''
    graph1 = {(0,0): set([((1,0), 1), ((0,1), 2)]), (1,0) : set([((0,0), 1), ((1,1), 3)]), (0,1): set([((0,0), 2), ((1,1), 1)]), (1,1): set([((0,1), 1), ((1,0), 3), ((2,1),1)]), (2,1): set([((1,1), 1)])}
    # deptiction of graph1
    #
    #  0,0 ---(1)--- 1,0
    #   |             |
    #   |             |
    #  (2)           (3)
    #   |             |
    #   |             |
    #  0,1 ---(1)--- 1,1 ---(1)--- (2,1)
    #
    #  The optimal path should be 0,0 -> 0,1 -> 1,1 -> (2,1)
    #
    graph2 = {(0,0): set([((1,0), 1)]), (1,0): set([((0,0), 1)]), (0,1):set([((1,1), 1)]), (1,1) : set([((0,1), 1)])}
    # deptiction of graph2
    #  0,0 ---(1)--- 1,0
    #
    #   
    #  
    #   
    #   
    #  0,1 ---(1)--- 1,1
    #  There is no path from 0,0 to 1,1 so the algorithm will return -1

    def test_(self):    
        self.assertEqual(SearchAlgs.Dijkstra(self.graph1, (0,0),(1,1))[1], [(1,1), (0,1), (0,0)])
        self.assertEqual(SearchAlgs.Dijkstra(self.graph2, (0,0),(1,1))[1], -1)
        graph_traversed = {}
        stack = []
        vis = set()
        for i in range(2):
            visited, optimal_path, graph_traversed, stack, vis = SearchAlgs.StackBasedApproach(self.graph1, (0,0), (2,1), graph_traversed=graph_traversed, pos_stack=stack, first = i==0, visited=vis)
            if i == 0:                
                self.assertEqual(optimal_path, [(2,1), (1,1), (1,0), (0,0)]) # first solution is suboptimal 
            if i == 1:
                self.assertEqual(optimal_path, [(2,1), (1,1), (0,1), (0,0)]) # optimal solution is found on second traversal
        self.assertEqual(SearchAlgs.StackBasedApproach(self.graph2, (0,0),(1,1))[1], -1)
if __name__ == "__main__":
    unittest.main()
