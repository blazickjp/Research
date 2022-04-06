import numpy as np
import collections
from gym.envs.toy_text import discrete
from gym.envs.classic_control import rendering

import time
import pickle 
import os

CELL_SIZE = 100
MARGIN = 10

def get_coords(row, col, loc = "center"):
    xc = (col + 1.5) * CELL_SIZE
    yc = (row + 1.5) * CELL_SIZE
    
    if loc == "center": 
        return xc, yc
    elif loc == "interior_corners":
        half_size = CELL_SIZE // 2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = xc - half_size, xc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == "interior_triangle":
        x1, y1 = xc, yc + CELL_SIZE//3
        x2, y2 = xc + CELL_SIZE // 3, yc - CELL_SIZE // 3
        x3, y3 = xc - CELL_SIZE // 3, yc - CELL_SIZE // 3
        return [(x1, y1), (x2, y2), (x3, y3)]
    
def draw_object(coords_list):
    if len(coords_list) == 1: # draw circle
        obj = rendering.make_circle(int(0.25*CELL_SIZE))
        obj_transform = rendering.Transform()
        obj.add_attr(obj_transform)
        obj_transform.set_translation(*coords_list[0])
        obj.set_color(.2,.2,.2) # black
    elif len(coords_list) == 3: # Draw triangle
        obj = rendering.FilledPolygon(coords_list)
        obj.set_color(.9,.6,.2) # yellow
    elif len(coords_list) > 3: # Draw something else?
        obj = rendering.FilledPolygon(coords_list)
        obj.set_color(.4,.4,.8) # blue
    print(coords_list)
    return obj


class GridWorldEnv(discrete.DiscreteEnv):
    def __init__(self, num_rows=4, num_cols=6, delay=.05):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay

        move_up = lambda row, col: (max(row - 1, 0), col)
        move_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_left = lambda row, col: (row, max(col - 1, 0))
        move_right = lambda row, col: (row, min(col + 1, num_cols - 1))

        self.action_defs = {
            0: move_up,
            1: move_right,
            2: move_down,
            3: move_left
        }
    
        nS = num_cols * num_rows
        nA = len(self.action_defs)
        self.grid2state_dict = {(s//num_cols, s%num_cols): s for s in range(nS)}
        self.state2grid_dict = {s: (s//num_cols, s%num_cols) for s in range(nS)}

        # Gold State
        gold_cell = (num_rows // 2, num_cols - 2)
        gold_state = self.grid2state_dict[gold_cell]

        # Trap States
        trap_cells = [
            ((gold_cell[0] + 1), gold_cell[1]),
            (gold_cell[0] - 1, gold_cell[1]-1),
            ((gold_cell[0] - 1), gold_cell[1])
        ]
        trap_states = [self.grid2state_dict[(r,c)] for (r,c) in trap_cells]

        self.terminal_states = [gold_state] + trap_states
        print(self.terminal_states)

        # Build the transition probability
        P = collections.defaultdict()
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            P[s] = collections.defaultdict(list)
            for a in range(nA):
                action = self.action_defs[a]
                next_s = self.grid2state_dict[action(row, col)]

                # Terminal State
                if self.is_terminal(next_s):
                    r = 1.0 if next_s == self.terminal_states[0] else 0.0
                else:
                    r = 0.0
                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False
                P[s][a] = [(1., next_s, r, done)]
        
        # Initial State Distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)
        self.viewer = None
        self._build_display(gold_cell, trap_cells)
    
    def is_terminal(self, state):
        return state in self.terminal_states

    def _build_display(self, gold_cell, trap_cells):
        screen_width = (self.num_cols + 2) * CELL_SIZE
        screen_height = (self.num_rows + 2) * CELL_SIZE
        self.viewer = rendering.Viewer(screen_width, screen_height)
        all_objects = []
        
        # List of border points
        bp_list = [
            (CELL_SIZE - MARGIN, CELL_SIZE - MARGIN),
            (screen_width - CELL_SIZE + MARGIN, CELL_SIZE - MARGIN),
            (screen_width - CELL_SIZE + MARGIN, screen_height - CELL_SIZE + MARGIN),
            (CELL_SIZE - MARGIN, screen_height - CELL_SIZE + MARGIN)
        ]
        border = rendering.PolyLine(bp_list, True)
        border.set_linewidth(5)
        all_objects.append(border)

        # Vertical Lines
        for col in range(self.num_cols+1):
            x1, y1 = (col+1) * CELL_SIZE, CELL_SIZE
            x2, y2 = (col+1) * CELL_SIZE, (self.num_rows+1) * CELL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)

        # Horizontal Lines
        for row in range(self.num_rows+1):
            x1, y1 = CELL_SIZE, (row+1) * CELL_SIZE
            x2, y2 = (self.num_cols+1) * CELL_SIZE, (row+1) * CELL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)

        # Traps
        for cell in trap_cells:
            trap_coords = get_coords(*cell, loc = "center")
            all_objects.append(draw_object([trap_coords]))

        # Gold
        gold_coords = get_coords(*gold_cell, loc="interior_triangle")
        all_objects.append(draw_object(gold_coords))

        # Agent
        agent_coords = get_coords(0,0, loc = "interior_corners")
        agent = draw_object(agent_coords)
        self.agent_trans = rendering.Transform()
        agent.add_attr(self.agent_trans)
        all_objects.append(agent)

        for obj in all_objects:
            self.viewer.add_geom(obj)
    
    def render(self, mode='human', done=False):
        if done:
            sleep_time = 1.0
        else:
            sleep_time = self.delay
        
        x_coord = self.s % self.num_cols
        y_coord = self.s // self.num_cols
        x_coord = (x_coord+0) * CELL_SIZE
        y_coord = (y_coord+0) * CELL_SIZE
        self.agent_trans.set_translation(x_coord, y_coord)
        rend = self.viewer.render(return_rgb_array = (mode == 'rgb_array'))

        time.sleep(sleep_time)
        return rend

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = GridWorldEnv(5,6)
    for i in range(1):
        s = env.reset()
        env.render(mode = 'human', done = False)

        while True:
            action = np.random.choice(env.nA)
            res = env.step(action)
            print('Action ', env.s, action, ' -> ', res)
            env.render(mode = 'human', done = res[2])
            if res[2]:
                break