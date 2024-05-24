import heapq
import time

class SearchAlgs:
    @staticmethod
    def extract_optimal_path_Dijkstra(pos_start, pos_end, path_cost, graph):
        path = set()
        path.add(pos_end)
        pos_cur = pos_end
        while pos_cur != pos_start:
            #print(path_cost[pos_cur])
            #time.sleep(1)
            pos_cur = min(graph[pos_cur], key=lambda x: path_cost.get(x[1], float('inf')))[1]
            path.add(pos_cur)
        return list(path)
    
    @staticmethod
    def extract_optimal_path_Astar(pos_start, pos_end, path, graph):
        visited = set(pos_start)
        path_so_far = [pos_start]
        def helper(path_so_far):
            if path_so_far[-1] == pos_end: return 1
            posibilities = [p[1] for p in graph[path_so_far[-1]] if p[1] in path and p[1] not in visited]
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
    
    @staticmethod
    def Astar(graph, pos_start = (0,0), pos_end = (0,0)):
        '''
        @input : 
        graph is the dictionary of positions (tuples) and a list of corresponding nodes that can be visited from them.
        The format connected nodes is in the [weight, (xcoord, ycoord)] format, where weight is the vertes weight. 
        '''
        def Astar_heuristic(p1,p2,type):
            if type == 0: # manhattan
                return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
            elif type == 1: # euclidean
                return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            else:
                raise Exception("heuristic type can either be manhattan (0) or euclidean (1)")
        openlist = []
        pos_cur = [0, 0, pos_start] # the first entry of pos cur repseresnts heuristic + g cost, second entry is just g cost
        heapq.heappush(openlist, pos_cur) # these are all the conurrent positions we may want to visit
        visited = set()
        q = len(visited)
        while pos_end != pos_cur[2] and openlist:
            #openlist.sort(key = (lambda x: x[2] + x[1]), reverse=True) # we sort according to heuristic + distance traveled so far
            pos_cur = heapq.heappop(openlist) # we take the node with shortest estimated value of travel cost 
            visited.add(pos_cur[2])
            for q in graph.get(pos_cur[2], []):
                q = q[1]
                if q not in visited:
                    heapq.heappush(openlist, [Astar_heuristic(q,pos_end,1)+pos_cur[1], pos_cur[1], q]) # we add new nodes to the heap
            if q == len(visited): break
        v = SearchAlgs.extract_optimal_path_Astar(pos_start, pos_end, list(visited), graph)        
        return visited, v
    
   
    @staticmethod
    def Dijkstra(height, graph, pos_start = (0,0), pos_end = (0,0), heightmap = []):
        '''
        @input : 
        graph is the dictionary of positions (tuples) and a list of corresponding nodes that can be visited from them.
        The format connected nodes is in the [weight, (xcoord, ycoord)] format, where weight is the vertes weight. 
        '''
        possible_paths = {pos_start: 0}
        visited = set()
        pos_cur = [0, pos_start]
        visited.add(pos_start)
        while pos_end != pos_cur[1]:
            for q in graph.get(pos_cur[1], []):
                if q[1] not in visited:
                    if q[1] in possible_paths:
                        possible_paths[q[1]] = min(possible_paths[q[1]], q[0] + possible_paths[pos_cur[1]])
                    else:
                        possible_paths[q[1]] =  q[0] + possible_paths[pos_cur[1]]
            pos_cur = min(possible_paths.items(), key = lambda x: x[1] if x[0] not in visited else float('inf')) # take the min cost route
            pos_cur = [pos_cur[1], pos_cur[0]]
            visited.add(pos_cur[1])
        v = SearchAlgs.extract_optimal_path_Dijkstra(pos_start, pos_end, possible_paths, graph)        
        return visited, v