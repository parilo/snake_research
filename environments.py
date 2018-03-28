import numpy as np
import matplotlib.pyplot as plt
import random

class Snake:

    def __init__(self, grid_size=(8, 8)):
        """
        Classic Snake game implemented as Gym environment.

        Parameters
        ----------
        grid_size: tuple
            tuple of two parameters: (height, width)
        """

        self.height, self.width = grid_size
        self.state = np.zeros(grid_size)
        self.x, self.y = [], []
<<<<<<< HEAD

=======
        self.dir = None
        self.food = None
        self.opt_tab = self.opt_table(grid_size)
        
>>>>>>> master
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns
        -------
        observation: numpy.array of size (width, height, 1)
            the initial observation of the space.
        """

        self.state = np.zeros((self.height, self.width))

<<<<<<< HEAD


        while 1:

            x_tail = np.random.randint(self.height)
            y_tail = np.random.randint(self.width)

            xs = [x_tail, ]
            ys = [y_tail, ]

            try_count = 0

            # for i in range(random.randint(2, 8)):
            for i in range(2):
                nbrs = self.get_neighbors(xs[-1], ys[-1])
                while 1:
                    idx = np.random.randint(0, len(nbrs))
                    x0 = nbrs[idx][0]
                    y0 = nbrs[idx][1]
                    occupied = [list(pt) for pt in zip(xs, ys)]
                    try_count += 1
                    if not [x0, y0] in occupied:
                        xs.append(x0)
                        ys.append(y0)
                        break
                    if try_count > 100:
                        break
                if try_count > 100:
=======
        x_tail = np.random.randint(self.height)
        y_tail = np.random.randint(self.width)
        
        xs = [x_tail, ]
        ys = [y_tail, ]
        
        for i in range(2):
            nbrs = self.get_neighbors(xs[-1], ys[-1])
            while 1:
                idx = np.random.randint(0, len(nbrs))
                x0 = nbrs[idx][0]
                y0 = nbrs[idx][1]
                occupied = [list(pt) for pt in zip(xs, ys)]
                if not [x0, y0] in occupied:
                    xs.append(x0)
                    ys.append(y0)
>>>>>>> master
                    break

            if try_count < 100:
                break


        for x_t, y_t in list(zip(xs, ys)):
            self.state[x_t, y_t] = 1

        self.generate_food()
        self.x = xs
        self.y = ys
        self.update_dir()

        return self.get_state()

    def step(self, a):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Args
        ----
        action: int from {0, 1, 2, 3}
            an action provided by the environment

        Returns
        -------
        observation: numpy.array of size (width, height, 1)
            agent's observation of the current environment
        reward: int from {-1, 0, 1}
            amount of reward returned after previous action
        done: boolean
            whether the episode has ended, in which case further step()
            calls will return undefined results
        """
<<<<<<< HEAD

        x_, y_ = self.next_cell(self.x[-1], self.y[-1], a)

        # snake continues moving in current direction if the
        # action forces it to turn head by 180 degrees
        if (x_, y_) == (self.x[-2], self.y[-2]):
            x_ = 2 * self.x[-1] - self.x[-2]
            y_ = 2 * self.y[-1] - self.y[-2]
=======
        
        self.update_dir()
        x_, y_ = self.next_cell(self.x[-1], self.y[-1], a)
>>>>>>> master

        # snake dies if hitting the walls
        if x_ < 0 or x_ == self.height or y_ < 0 or y_ == self.width:
            return self.get_state(), -1, True

        # snake dies if hitting its tail with head
        if self.state[x_, y_] == 1:
<<<<<<< HEAD
            return self.get_state(), -1, True

=======
            if (x_ == self.x[0] and y_ == self.y[0]):
                pass
            else:
                return self.get_state(), -1, True
        
>>>>>>> master
        self.x.append(x_)
        self.y.append(y_)

        # snake elongates after eating a food
        if self.state[x_, y_] == 3:
            self.state[x_, y_] = 1
<<<<<<< HEAD
            self.generate_food()
            return self.get_state(), 1, False

        # snake moves forward if cell ahead is empty
        if self.state[x_, y_] == 0:
            self.state[x_, y_] = 1
            self.state[self.x[0], self.y[0]] = 0
            self.x = self.x[1:]
            self.y = self.y[1:]
            return self.get_state(), 0, False

    def get_head_coords(self):
        return self.x[-1], self.y[-1]

    def get_state(self):
        state = self.state.copy()
        state[self.x[-1], self.y[-1]] = 2

        x_head = self.x[-1]
        y_head = self.y[-1]

        x_prev = self.x[-2]
        y_prev = self.y[-2]

        if x_head > x_prev:
            self.turn_state = 2
        if x_head < x_prev:
            self.turn_state = 0
        if y_head > y_prev:
            self.turn_state = 1
        if y_head < y_prev:
            self.turn_state = 3

        return state.reshape((self.height, self.width, 1))

    def generate_food(self):
        free = np.where(self.state == 0)
        idx = np.random.randint(free[0].size)
        self.state[free[0][idx], free[1][idx]] = 3

    def next_cell(self, i, j, a):
        if a == 0: return i-1, j
        if a == 1: return i, j-1
        if a == 2: return i+1, j
        if a == 3: return i, j+1

=======
            return self.get_state(), 1, self.generate_food()

        # snake moves forward if cell ahead is empty
        # or currently occupied by its tail
        self.state[x_, y_] = 1
        self.state[self.x[0], self.y[0]] = 0
        self.x = self.x[1:]
        self.y = self.y[1:]
        return self.get_state(), 0, False  
        
    def get_state(self):
        state = np.zeros((self.height, self.width, 4))
        state[self.x[1:-1], self.y[1:-1], 0] = 1
        state[self.x[-1], self.y[-1], 1] = 1
        state[self.x[0], self.y[0], 2] = 1
        state[self.food[0], self.food[1], 3] = 1
        return state
        
    def generate_food(self):
        free = np.where(self.state == 0)
        if free[0].size == 0:
            return True
        else:
            idx = np.random.randint(free[0].size)
            self.food = free[0][idx], free[1][idx]
            self.state[self.food] = 3
            return False
        
    def next_cell(self, i, j, a):
        if a == 0: 
            return i+self.dir[0], j+self.dir[1]
        if a == 1: 
            return i-self.dir[1], j+self.dir[0]
        if a == 2: 
            return i+self.dir[1], j-self.dir[0]
        
>>>>>>> master
    def plot_state(self):
        state = self.get_state()
        img = sum([state[:,:,i]*(i+1) for i in range(3)])
        plt.imshow(img, vmin=0, vmax=4, interpolation='nearest')

    def get_neighbors(self, i, j):
        """
        Get all the neighbors of the point (i, j)
        (excluding (i, j))
        """
        h = self.height
        w = self.width
        nbrs = [[i + k, j + m] for k in [-1, 0, 1] for m in [-1, 0, 1]
                if i + k >=0 and i + k < h and j + m >= 0 and j + m < w
                and not (k == m) and not (k == -m)]
        return nbrs
    
    def update_dir(self):
        x_dir = self.x[-1] - self.x[-2]
        y_dir = self.y[-1] - self.y[-2]
        self.dir = (x_dir, y_dir)
        
    ########################## Optimal action selection ##########################
    
    def opt_table(self, grid_size):
        n = grid_size[0]
        t = np.zeros(grid_size, dtype=np.int)
        t[0] = np.arange(n)
        for i in range(n//2):
            t[1:,(n-1)-2*i] = np.arange(n-1) + n+2*i*(n-1)
            t[1:,(n-2)-2*i][::-1] = np.arange(n-1) + 2*n-1+2*i*(n-1)
        return t
    
    def opt_action(self):
        x, y = self.x[-1], self.y[-1]
        self.update_dir()
        n = self.height
        mod = n ** 2
        tab_xy = self.opt_tab[x, y]
        pos_a = -1
        for a in range(3):
            x_, y_ = self.next_cell(x, y, a)
            if (x_<n and y_<n and x_>=0 and y_>=0):
                tab_xy_ = self.opt_tab[x_, y_]
                if ((tab_xy+1) % mod == tab_xy_):
                    return a
                if ((tab_xy-1) % mod == tab_xy_):
                    pos_a = a
        return pos_a