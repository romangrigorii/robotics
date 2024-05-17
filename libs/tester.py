from maze import Environment

e = Environment()
p = e.init_maze_environment()
e.build_maze()
e.draw_grid_from_graph()
e.run()