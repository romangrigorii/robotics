##############################
# Python maze generator code #
##############################

import pygame, sys, time ,random, numpy as np, pickle
from Astar import RoboticsAlgorithms

class Environment:

    WHITE = (255,255,255)
    GREEN = (0,255,0)
    BLUE = (0,0,255)
    YELLOW = (255,255,0)
    RED = (255,0,0)
    BLACK = (0,0,0)

    def __init__(self, width = 1000, height = 500, ncellsx = 100, ncellsy = 50, visualize_maze_building = True):
        
        # set whther or not we want to visualize the maze building in real time
        self.visualize_maze_building = visualize_maze_building 
        if width < ncellsx*3: width = ncellsx*3
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

        self.grid = []
        self.visited = []
        self.stack = []
        self.solution = {}
        self.graph = {}
        self.paths = []

        self.FPS = 1


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
        return pygame
    
    def draw_full_grid(self, color = WHITE):
        '''
        Fills out the grid with walls
        '''
        for x in range(self.ncellsx+1):
            pygame.draw.line(self.screen, color, [x*self.wx,0],[x*self.wx,self.height])
        for y in range(self.ncellsy+1):
            pygame.draw.line(self.screen, color, [0,y*self.wy],[self.width,y*self.wy], width = 30)

    def draw_grid_from_graph(self, color = WHITE):
        '''
        Fills out the grid with walls from the graph
        '''
        for pos in self.graph:
            possible_walls = [(pos[0]+1, pos[1]),(pos[0]-1, pos[1]),(pos[0], pos[1]-1),(pos[0], pos[1]+1)]
            walls = []
            for i, q in enumerate(possible_walls):
                if q not in self.graph[pos]:
                    walls.append(i)
            for i in walls:
                if i < 2:
                    pygame.draw.line(self.screen, color, [self.wx*(pos[0] + possible_walls[i][0] + 1)/2, self.wy*possible_walls[i][1]], [self.wx*(pos[0] + possible_walls[i][0] + 1)/2, self.wy*(1+possible_walls[i][1])], width = 1)
                else:
                    pygame.draw.line(self.screen, color, [self.wx*possible_walls[i][0], self.wy*(pos[1] + possible_walls[i][1] + 1)/2], [self.wx*(1+possible_walls[i][0]), self.wy*(pos[1] + possible_walls[i][1] + 1)/2], width = 1)
            pygame.display.update()

    def build_maze(self):
        self.visited.append(self.pos) # this keeps track of all positions we have visited
        self.stack.append(self.pos)   # this keeps track of which next positions we should explore
        self.stack.append(self.pos_end)   # this keeps track of which next positions we should explore
        while self.stack:            
            locs = np.random.permutation([(1,0),(-1,0),(0,1),(0,-1)]) # pick a random direction
            flag = 1
            for loc in [a for a in locs]: # we vist a new location
                newpos = (loc[0]+self.pos[0],loc[1]+self.pos[1])
                if newpos[0]>=0 and newpos[0]<self.ncellsx and newpos[1]>=0 and newpos[1]<self.ncellsy and newpos not in self.visited:
                    self.graph.setdefault(self.pos,[]).append(newpos) # we append the positions we can visit from a given node
                    self.graph.setdefault(newpos,[]).append(self.pos)
                    flag = 0
                    self.visited.append(newpos)
                    self.stack.append(self.pos)
                    self.pos = newpos # we move to the new location
                    break
            if flag and self.stack: # if we enter a dead end we select the last position visited from the stack
                if self.pos_end not in self.visited:
                    self.visited = sorted(self.visited, key = lambda x : (x[0] - self.pos_end[0])**2 +  (x[1] - self.pos_end[1])**2)
                    self.pos = self.visited[0]
                    self.visited = self.visited[1:]
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

    def draw_pos(self, pos, color = WHITE):
        pygame.draw.circle(self.screen, color, ((pos[0]+.5)*self.wx, (pos[1]+.5)*self.wy), self.wx/3)
        pygame.display.update()

    def draw_rect(self,color,direction,pos,dx,dy):
        if direction == 'left' or direction == 3 :
            pygame.draw.rect(self.screen,color, (pos[0]*self.wx-self.wx/2-dx/2+1,pos[1]*self.wy+1,dx + self.wx - 1, self.wy-1))
        if direction == 'right' or direction == 1 :
            pygame.draw.rect(self.screen,color, (pos[0]*dx+(self.wx-dx)/2+1,pos[1]*dy+1,dx + self.wx-1, self.wy-1))
        if direction == 'up' or direction == 0:
            pygame.draw.rect(self.screen,color, (pos[0]*dx+1,pos[1]*dy-self.wx/2-dy/2+1,self.wx-1, dx + self.wy-1))
        if direction == 'down' or direction == 2:
            pygame.draw.rect(self.screen,color, (pos[0]*dx+1,pos[1]*dy+1+(self.wy-dy)/2,self.wx - 1, dx + self.wy-1))
        else:
            pygame.draw.rect(self.screen,color, (pos[0]*self.wx+1+(self.wx-dx)/2,pos[1]*self.wy+(self.wy-dy)/2+1,dx-1, dy-1))
        pygame.display.update()

    def draw_rect_2(self,color,pos1,pos2):
        loc = (pos2[0]-pos1[0], pos2[1]-pos1[1])
        direction = int(loc==(1,0)) + 2*int(loc==(0,1)) + 3*int(loc == (-1,0))
        self.draw_rect(color,direction,pos1,self.wx,self.wy)

    def save_graph(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.graph, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_graph(self,filename):
        with open(filename,'rb') as file:
            self.graph = pickle.load(file)

if __name__ == "__main__":
    while(1):
        m = Environment(width = 500, height = 500, ncellsx = 70, ncellsy = 70, visualize_maze_building = True)
        rb = RoboticsAlgorithms()
        m.init_maze_environment()
        m.build_maze()
        m.draw_grid_from_graph()   
        m.save_graph("amaze.pickle")    
        path, shortest = rb.Astar(graph = m.graph, pos_start = m.pos_start, pos_end = m.pos_end)
        for p in path:
            m.draw_pos(p)
        for p in shortest:
            m.draw_pos(p, color=m.BLUE)
        m.run()
