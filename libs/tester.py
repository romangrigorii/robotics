from maze import Environment
from Astar import RoboticsAlgorithms

e = Environment()
rb = RoboticsAlgorithms()
p = e.init_maze_environment()
e.build_maze()
e.draw_grid_from_graph()
path = rb.Astar(e.graph, pos_start = e.pos_start, pos_end = e.pos_end)
print(path)
for p in path:
    e.draw_pos(p)
# e.draw_pos((0,0), color=e.WHITE)
# e.draw_pos((0,0), color=e.BLACK)
e.run()