##############################
# Python maze generator code #
##############################

USE_SAVED_MAZE = False

import pygame, numpy as np, pickle
import matplotlib.pyplot as plt
from collections import deque
import heapq
import time 

class MazeBuilder:

    WHITE = (255,255,255)
    GREEN = (0,255,0)
    BLUE = (0,0,255)
    YELLOW = (255,255,0)
    RED = (255,0,0)
    BLACK = (0,0,0)

    def __init__(self, maze_type = 0, heightmap = False, width = 100, height = 100, ncellsx = 20, ncellsy = 20):
        '''
        For all mazes the start and end node are positioned at (0,0) and (ncellsx-1,ncellsy-1), respectively 
        maze types:
        0 - this is a maze in which hthe start node position can be reached via multiple paths and each traversal has a cost.
        1 - this is a maze in which the start and end node are connedted by exctly ONE global path. 
        '''
        # set whether or not we want to visualize the maze building in real time

        self.maze_type = maze_type
        self.FPS = 1
        if width < ncellsx*3: width = ncellsx*3 # make sure each path is at least 1 pixel wide
        if height < ncellsy*3: height = ncellsy*3
        self.width = width
        self.height = height
        self.size = (self.width,self.height) # this is the size of the map in pixels
        self.screen = None
        self.clock = None

        self.ncellsx = ncellsx
        self.ncellsy = ncellsy        
        self.wx = self.width//self.ncellsx
        self.wy = self.height//self.ncellsy
        self.width = self.ncellsx*self.wx   # we rescale to make sure the maze looks cohesive
        self.height = self.ncellsy*self.wy
        self.pos_start = (0,0)
        self.pos_end = (ncellsx-1,ncellsy-1)
        self.pos = self.pos_start

        self.visited = []
        self.visited_set = set()
        self.stack = []
        self.graph = {}
        self.paths = []
        self.heightmap = [[0]*ncellsy for i in range(ncellsx)]
        if heightmap:
            self.heightmap = [[np.exp(-((x - ncellsx/4)**2 )/ncellsx**2*10 - ((y - ncellsy/2)**2)/ncellsy**2*60) for y in range(ncellsy)] for x in range(ncellsx)]
            self.heightmap = [[self.heightmap[x][y] + np.exp(-((x - ncellsx*3/4)**2 )/ncellsx**2*30 - ((y - ncellsy*3/4)**2)/ncellsy**2*30) for y in range(ncellsy)] for x in range(ncellsx)]
            self.heightmap = [[self.heightmap[x][y] + np.exp(-((x - ncellsx*1/4)**2 )/ncellsx**2*30 - ((y - ncellsy*1/4)**2)/ncellsy**2*30) for y in range(ncellsy)] for x in range(ncellsx)]
            self.heightmap = [[self.heightmap[x][y] + np.exp(-((x)**2 )/ncellsx**2*10 - ((y - ncellsy)**2)/ncellsy**2*10) for y in range(ncellsy)] for x in range(ncellsx)]
            self.heightmap = [[self.heightmap[x][y] + np.exp(-((x - ncellsx)**2 )/ncellsx**2*10 - ((y)**2)/ncellsy**2*10) for y in range(ncellsy)] for x in range(ncellsx)]
            ma = max(sum(self.heightmap,[]))
            self.heightmap = [[self.heightmap[x][y]/ma for y in range(ncellsy)] for x in range(ncellsx)]

    def init_maze_environment(self, background_color = BLACK):
        '''
        pygame is a a template environment which will help us visualize the 
        robot navigating the space and performing tasks. 
        '''
        pygame.init()
        pygame.mixer.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(background_color)
        pygame.display.set_caption('a maze')
        window = pygame.display.set_mode(self.size)
        return window
    
    
    def draw_grid_from_graph(self, color = WHITE, width = 1):
        '''
        Fills out the grid with walls from the graph
        '''
        for pos in self.graph:
            possible_walls = [(pos[0]+1, pos[1]),(pos[0]-1, pos[1]),(pos[0], pos[1]-1),(pos[0], pos[1]+1)]
            walls = []
            for i, q in enumerate(possible_walls):
                if q not in [a[0] for a in self.graph[pos]]:
                    walls.append(i)
            for i in walls:
                if i < 2:
                    pygame.draw.line(self.screen, color, [self.wx*(pos[0] + possible_walls[i][0] + 1)/2, self.wy*possible_walls[i][1]], [self.wx*(pos[0] + possible_walls[i][0] + 1)/2, self.wy*(1+possible_walls[i][1])], width)
                else:
                    pygame.draw.line(self.screen, color, [self.wx*possible_walls[i][0], self.wy*(pos[1] + possible_walls[i][1] + 1)/2], [self.wx*(1+possible_walls[i][0]), self.wy*(pos[1] + possible_walls[i][1] + 1)/2], width)
            pygame.display.update()

    def build_maze(self):
        '''
        build_maze will construct a graph representing the allowed traversals within a virtual maze environment. 
        The graph is represented as a dictionary - a position wihtin a maze is a key and corresponding values represent
        allowed moves from that position in the format [weight, (posx, posy)] where weight is the cost of transition
        or a weight of the vertex connecting the two nodes. that is applicable when we use heightmap. 

        maze_type 0: is a maze in which there are multiple paths to the target node by 
        maze_type 1: is a maze that has one path toward the target node

        '''        
        if self.maze_type == 0:
            self.visited # this keeps track of all positions we have visited
            self.stack.append(self.pos)   # this keeps track of which next positions we should explore
            while self.stack:            
                locs = np.random.permutation([(1,0),(-1,0),(0,1),(0,-1)]) # pick a random direction
                flag = 1
                for loc in [a for a in locs]: # we vist a new location
                    newpos = (loc[0]+self.pos[0],loc[1]+self.pos[1])
                    if newpos[0]>=0 and newpos[0]<self.ncellsx and newpos[1]>=0 and newpos[1]<self.ncellsy and newpos not in self.visited:
                        q = newpos
                        q_cost = self.heightmap[q[0]][q[1]]
                        p = self.pos
                        p_cost = self.heightmap[p[0]][p[1]]
                        self.graph.setdefault(self.pos,set()).add((newpos, q_cost)) # we append the positions we can visit from a given node
                        self.graph.setdefault(newpos,set()).add((self.pos, p_cost))
                        flag = 0
                        self.visited.append(newpos)
                        self.stack.append(self.pos)
                        self.pos = newpos # we move to the new location
                        #break
                if flag and self.stack: # if we enter a dead end we select the last position visited from the stack
                    if self.pos_end not in self.visited:
                        ii = np.random.randint(len(self.visited))
                        self.pos = self.visited[ii]
                        self.visited = self.visited[:ii] + self.visited[ii+1:]
                    else:
                        p = np.random.randint(len(self.stack))
                        self.pos = self.stack[p]
                        self.stack = self.stack[:p] + self.stack[p+1:]

        if self.maze_type == 1:
            self.visited_set.add(self.pos) # this keeps track of all positions we have visited
            d_to_fin = (self.pos[0] - self.pos_end[0])**2 +  (self.pos[1] - self.pos_end[1])**2
            heapq.heappush(self.visited, [d_to_fin, self.pos])
            self.stack.append(self.pos)  # this keeps track of which next positions we should explore
            while self.stack:            
                locs = np.random.permutation([(1,0),(-1,0),(0,1),(0,-1)]) # pick a random direction
                flag = 1
                for loc in [a for a in locs]: # we vist a new location
                    newpos = (loc[0]+self.pos[0],loc[1]+self.pos[1])
                    d_to_fin = (newpos[0] - self.pos_end[0])**2 +  (newpos[1] - self.pos_end[1])**2
                    if newpos[0]>=0 and newpos[0]<self.ncellsx and newpos[1]>=0 and newpos[1]<self.ncellsy and newpos not in self.visited_set:
                        self.graph.setdefault(self.pos,set()).add((newpos, 0)) # we append the positions we can visit from a given node
                        self.graph.setdefault(newpos,set()).add((self.pos, 0))
                        flag = 0
                        heapq.heappush(self.visited,[d_to_fin,newpos])
                        self.visited_set.add(newpos)
                        self.stack.append(self.pos)
                        self.pos = newpos # we move to the new location
                        break
                if flag and self.stack: # if we enter a dead end we select the last position visited from the stack
                    if self.pos_end not in self.visited_set:
                        self.pos = heapq.heappop(self.visited)[1]
                        self.visited_set.remove(self.pos)
                    else:
                        p = np.random.randint(len(self.stack))
                        self.pos = self.stack[p]
                        self.stack = self.stack[:p] + self.stack[p+1:]
        return self.graph
    
    def run(self):
        flag = 1
        pygame.display.flip() 
        while flag:
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                flag = event.type != pygame.QUIT
    
    def fill_height_map(self):
        for x in range(self.ncellsx):
            for y in range(self.ncellsy):
                pygame.draw.rect(self.screen, (self.heightmap[x][y]*255,0,0), (x*self.wx, y*self.wy, self.wx, self.wy))
        pygame.display.update()

    def draw_pos(self, pos, color = WHITE, scale=.5):
        pygame.draw.circle(self.screen, color, ((pos[0]+.5)*self.wx, (pos[1]+.5)*self.wy), self.wx*scale/2)
        pygame.display.update()

    def draw_all_pos(self, path, color = WHITE, scale = .5, linked = True, st_end = (0,0)):
        for pos in path:
            pygame.draw.circle(self.screen, color, ((pos[0]+.5)*self.wx, (pos[1]+.5)*self.wy), self.wx*scale/2)
        if linked:
            for ii in range(len(path)-1):
                start_pos=((path[ii][0]+.5)*self.wx, (path[ii][1]+.5)*self.wy)
                end_pos=((path[ii+1][0]+.5)*self.wx, (path[ii+1][1]+.5)*self.wy)
                print(start_pos)
                print(end_pos)
                pygame.draw.line(surface=self.screen, color=color, start_pos=start_pos, end_pos=end_pos, width = int(np.ceil(self.wx*scale/2)))
        pygame.display.update()

    def save_graph(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.graph, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_graph(self,filename):
        with open(filename,'rb') as file:
            self.graph = pickle.load(file)

if __name__ == "__main__":
    # A sample routine to verify maze functionality
    def fill_maze():
        if type == 0:
            m.fill_height_map()
        m.draw_grid_from_graph(width = 3)

    while(1):
        type = int(input('Provide maze type. 0 for 1 solution maze, 1 for multuple solution maze:'))
        if type not in [0,1]:
            print('Invalid option.')
            break
        m = MazeBuilder(maze_type=type, heightmap = False, width = 1000, height = 1000, ncellsx = 10, ncellsy = 10)
        m.init_maze_environment() # set up pygame 
        m.build_maze()            # randomly generate maze
        fill_maze()               # fill the maze onto the pygame instance
        m.run()                   # run pygame instance 
