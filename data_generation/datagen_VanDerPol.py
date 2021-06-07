from scipy.integrate import solve_ivp
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Simulate multiple trajectories of Van Der Pol's system")
parser.add_argument('--nb_ex', default=200, type=int, help='number of trajectories to simulate')
parser.add_argument('--span', default=25, type=int, help='length of a trajectory (in seconds)')
parser.add_argument('--fps', default=40, type=str, help='number of points per second')
parser.add_argument('--dirout', default='../Dataset/vanderpol/test.npy', type=str, help='where to store created files')
args = parser.parse_args()

# Simulation Parameters
OMEGA = 1
EPSILON = 1


def dynamics(t, y):
    out = np.zeros_like(y)
    out[0] = y[1]
    out[1] = EPSILON * OMEGA * (1 - y[0] ** 2) * y[1] - OMEGA ** 2 * y[0]
    return out


def generate(nb_ex):
    dataset = np.zeros((nb_ex, 2, 100))
    t_span = [0, args.span]
    t_eval = np.linspace(0, args.span, args.span * args.fps)
    idx = np.arange(0, args.span * args.fps, 10)

    for k in range(nb_ex):
        y0 = np.random.random(2) * 2 - 1
        y0 = y0 * 5
        sol = solve_ivp(dynamics, y0=y0, t_span=t_span, t_eval=t_eval)
        dataset[k] = sol.y[:, idx]

    np.save(args.dirout, dataset)


if __name__ == '__main__':
    generate(args.nb_ex)