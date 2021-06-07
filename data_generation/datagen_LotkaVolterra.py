from scipy.integrate import solve_ivp
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Simulate multiple trajectories of LotkaVolterra's system")
parser.add_argument('--nb_ex', default=200, type=int, help='number of trajectories to simulate')
parser.add_argument('--span', default=25, type=int, help='length of a trajectory (in seconds)')
parser.add_argument('--fps', default=40, type=str, help='number of points per second')
parser.add_argument('--dirout', default='../Dataset/lotkavolterra/val.npy', type=str,
                    help='where to store created files')
args = parser.parse_args()

# Simulation Parameters
ALPHA = 2 / 3
BETA = 4 / 3
GAMMA = 1
DELTA = 1


def dynamics(t, y):
    """Lotka Volterra model"""
    out = np.zeros_like(y)
    out[0] = y[0] * (ALPHA - BETA * y[1])
    out[1] = y[1] * (DELTA * y[0] - GAMMA)
    return out


def generate(nb_ex):
    dataset = np.zeros((nb_ex, 2, 100))  # That will store each simulation
    t_span = [0, args.span]
    t_eval = np.linspace(0, args.span, args.span * args.fps)  # Time vector

    # Change this line to configure how much you downsample the data, and the final time range
    idx = np.arange(0, args.span * args.fps, 10)

    for k in tqdm(range(nb_ex)):
        y0 = np.random.random(2) * 2
        sol = solve_ivp(dynamics, y0=y0, t_span=t_span, t_eval=t_eval)
        dataset[k] = sol.y[:, idx]

    np.save(args.dirout, dataset)


if __name__ == '__main__':
    generate(args.nb_ex)
