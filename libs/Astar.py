class RoboticsAlgorithms:
    @staticmethod
    def clean_up_path(pos_start, pos_end, path, graph):
        visited = set(pos_start)
        path_so_far = [pos_start]
        def helper(path_so_far):
            if path_so_far[-1] == pos_end: return 1
            posibilities = [p for p in graph[path_so_far[-1]] if p in path and p not in visited]
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
        Graph is the dictionary of positions and the corresponding positions which can be visited from them.
        '''
        def Astar_heuristic(p1,p2,type):
            if type == 0: # manhattan
                return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
            elif type == 1: # euclidean
                return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            else:
                raise Exception("heuristic type can either be manhattan (0) or euclidean (1)")
        pos_cur = [pos_start,0,0] # node, value to get to this node, heuristic to get to final node
        openlist = [pos_cur] # these are all the conurrent positions we may want to visit
        visited = []
        visited_path = []
        q = len(visited)
        while pos_end != pos_cur[0] and openlist:
            openlist.sort(key = (lambda x: x[2] + x[1]), reverse=True) # we sort according to heuristic + distance traveled so far
            pos_cur = openlist.pop() # we take the node with shortest estimated value of travel cost 
            visited_path.append(pos_cur[0])
            for q in graph.get(pos_cur[0], []):
                if q not in visited:
                    openlist += [[q,pos_cur[1]+1, Astar_heuristic(pos_cur[0],q,1)]] # we add new nodes to the list
            visited.append(pos_cur[0])
            if q == len(visited): break
        print('here')
        v = RoboticsAlgorithms.clean_up_path(pos_start, pos_end, visited_path, graph)        
        return visited_path, v
    
   
