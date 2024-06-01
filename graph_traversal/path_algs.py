import heapq
import time

class SearchAlgs:        
    @staticmethod
    def extract_optimal_path(pos_start, pos_end, path_cost, graph):
        '''
        Given the graph and the cost traversal to each of the visited nodes computed the optimal path
        '''
        path = []
        path_set = set([pos_end])
        path.append(pos_end)
        pos_cur = pos_end
        while pos_cur != pos_start:
            pos_cur = min(graph[pos_cur], key=lambda x: path_cost.get(x[0], float('inf')) + x[1] if x[0] not in path_set else float('inf'))
            pos_cur = pos_cur[0]
            path_set.add(pos_cur)
            path.append(pos_cur)
        return path
   
    @staticmethod
    def Dijkstra(graph, pos_start = (0,0), pos_end = (0,0), visited_all = set()):
        '''
        @input : 
        graph is the dictionary of positions (tuples) and a list of corresponding nodes that can be visited from them.
        The format connected nodes is in the [weight, (xcoord, ycoord)] format, where weight is the vertes weight. 
        pos_start, pos_end, visited_all (used for Stack Version)
        '''
        possible_paths = {pos_start: 0}
        visited = set()
        pos_cur = (pos_start, 0)
        visited.add(pos_start)
        while pos_end != pos_cur[0]:
            pos_st = pos_cur[0]
            for q in graph.get(pos_cur[0], []):
                if q[0] not in visited and q[0] in visited_all if visited_all else True:
                    if q[1] in possible_paths:
                        possible_paths[q[0]] = min(possible_paths[q[1]], q[1] + pos_cur[1])
                    else:
                        possible_paths[q[0]] =  q[1] + pos_cur[1]
            pos_cur = min(possible_paths.items(), key = lambda x: x[1] if x[0] not in visited else float('inf')) # take the min cost route
            visited.add(pos_cur[0])
            if pos_cur[0] == pos_st: return -1, -1 # no path possible
        v = SearchAlgs.extract_optimal_path(pos_start, pos_end, possible_paths, graph)
        return visited | visited_all, v
    
    @staticmethod
    def StackBasedApproach(graph, pos_start = (0,0), pos_end = (0,0), graph_traversed = {}, pos_stack = None, visited = set(), first = True):
        '''
        @input : 
        graph is the dictionary of positions (tuples) and a list of corresponding nodes that can be visited from them.
        The format connected nodes is in the [weight, (xcoord, ycoord)] format, where weight is the vertes weight. 
        graph_traversed holdds the graph that is traversed thus far
        pos_stack is the stack of nodes
        visited are nodes already visited
        '''
        if not pos_stack: pos_stack = [(pos_start, 0)]
        pos_cur = (pos_start, 0)
        stack_popped = len(pos_stack)//2
        while ((pos_end != pos_cur[0] and first )or (not first and stack_popped<len(pos_stack))) and pos_stack:
            pos_cur = pos_stack.pop()
            visited.add(pos_cur[0])
            current_choices = []
            for q in graph.get(pos_cur[0], []):
                graph_traversed.setdefault(pos_cur[0], set()).add(q) # add information to the explored graph
                graph_traversed.setdefault(q[0], set()).add((pos_cur[0],q[1]))  
                if q[0] not in visited: current_choices.append(q)
            current_choices.sort(reverse=True, key = lambda x: x[1] if x[0]!= pos_end else 0)      
            pos_stack.extend(current_choices) 
        if pos_end not in graph_traversed: return -1, -1, {}, [], set()
        res = SearchAlgs.Dijkstra(graph_traversed, pos_start, pos_end, visited)
        return res[0], res[1], graph_traversed, pos_stack, visited
    
    @staticmethod
    def Astar(graph, pos_start = (0,0), pos_end = (0,0)):
        '''
        @input : 
        graph is the dictionary of positions (tuples) and a list of corresponding nodes that can be visited from them.
        The format connected nodes is in the [weight, (xcoord, ycoord)] format, where weight is the vertes weight. 
        '''
        def Astar_heuristic(p1,p2,type = 1):
            if type == 0: # manhattan
                return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
            elif type == 1: # euclidean
                return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            else:
                raise Exception("heuristic type can either be manhattan (0) or euclidean (1)")
        possible_paths = {pos_start: 0}
        visited = set()
        pos_cur = [0, 0, pos_start] # heuristic + g, g, node locations
        visited.add(pos_start)
        while pos_end != pos_cur[1]:
            pos_st = pos_cur
            for q in graph.get(pos_cur[1], []):
                if q[0] not in visited:
                    possible_paths[q[0]] =  q[1] + Astar_heuristic(q[0], pos_end)
            pos_cur = min(possible_paths.items(), key = lambda x: x[1] if x[0] not in visited else float('inf')) # take the min cost route
            pos_cur = [pos_cur[1], pos_cur[0]]
            visited.add(pos_cur[1])
            if pos_cur[1] == pos_st[1]: return -1, -1 # no path possible
        v = SearchAlgs.extract_optimal_path_recursive(pos_start, pos_end, possible_paths, graph)        
        return visited, v
    
    @staticmethod
    def extract_optimal_path_recursive(pos_start, pos_end, path, graph):
        '''
        This is a method for finding the most direct path through the traversed nodes. 
        '''
        visited = set(pos_start)
        path_so_far = [pos_start]
        def helper(path_so_far):
            if path_so_far[-1] == pos_end: return 1
            posibilities = [p[0] for p in graph[path_so_far[-1]] if p[0] in path and p[0] not in visited]
            posibilities.sort(key = lambda x: path[x])
            if not posibilities: return 0
            else:
                for p in posibilities:
                    visited.add(p)
                    path_so_far.append(p)
                    a = helper(path_so_far)
                    if a == 0: path_so_far.pop()
                    else: return 1 # terminate early
                return 0
        helper(path_so_far)
        return path_so_far