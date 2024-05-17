import pygame, sys, time ,random, numpy as np, pickle
from maze import *


class search (MAZE):

    def Astar_heuristic(self,p1,p2,type):
        if type == 0: # manhattan
            return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
        else: # euclidean
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def Astar(self, pos_start = (0,0), pos_end = (0,0), visualize = True):
        pos_cur = [pos_start,0,0] # node, value to get to this node, heuristic to get to final node
        openlist = [pos_cur] # these are all the conurrent positions we may want to visit
        visited = []
        while pos_end != pos_cur[0] and openlist:
            time.sleep(1)
            openlist.sort(key = (lambda x: x[2] + x[1]), reverse=True) # we sort according to heuristic + distance traveled so far
            pos_cur = openlist.pop() # we take the node with shortest estimated value of travel cost 
            if self.graph.get(pos_cur[0]):
                for q in self.graph.get(pos_cur[0]):
                    if q not in visited:
                        openlist += [[q,pos_cur[1]+1,self.heuristic(pos_cur[0],q,1)]] # we add new nodes to the list
            visited.append(pos_cur[0])
            self.draw_rect(self.GREEN,4,pos_cur[0],self.wx/2,self.wy/2)


m = SEARCH_ALGS(visualize = True)
m.state_machine()
m.Astar(pos_end = (20,20))