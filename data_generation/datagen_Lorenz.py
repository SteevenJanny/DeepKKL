from scipy.integrate import solve_ivp
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Simulate multiple trajectories of Lorenz's system")
parser.add_argument('--nb_ex', default=200, type=int, help='number of trajectories to simulate')
parser.add_argument('--span', default=5, type=int, help='length of a trajectory (in seconds)')
parser.add_argument('--fps', default=1500, type=str, help='number of points per second')
parser.add_argument('--dirout', default='../Dataset/lorenz/test.npy', type=str, help='where to store created files')
args = parser.parse_args()

# Simulation Parameters
SIGMA = 10
BETA = 8 / 3
RHO = 24


def dynamics(t, y):
    """Lorenz model"""
    out = np.zeros_like(y)
    out[0] = SIGMA * (y[1] - y[0])
    out[1] = RHO * y[0] - y[1] - y[0] * y[2]
    out[2] = y[0] * y[1] - BETA * y[2]
    return out


def generate(nb_ex):
    dataset = np.zeros((nb_ex, 3, 100))  # That will store each simulation
    t_span = [0, args.span]
    t_eval = np.linspace(0, args.span, args.span * args.fps)  # Time vector

    # Change this line to configure how much you downsample the data, and the final time range
    idx = np.arange(2 * args.fps, 4 * args.fps, 30)

    for k in tqdm(range(nb_ex)):
        y0 = np.random.random(3) * 2 - 1
        y0 = y0 * 20
        sol = solve_ivp(dynamics, y0=y0, t_span=t_span, t_eval=t_eval)
        dataset[k] = sol.y[:, idx]

    np.save(args.dirout, dataset)


if __name__ == '__main__':
    generate(args.nb_ex)
