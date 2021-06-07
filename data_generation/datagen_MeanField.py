from scipy.integrate import solve_ivp
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Simulate multiple trajectories of Mean Field's system")
parser.add_argument('--nb_ex', default=200, type=int, help='number of trajectories to simulate')
parser.add_argument('--span', default=6, type=int, help='length of a trajectory (in seconds)')
parser.add_argument('--fps', default=20, type=str, help='number of points per second')
parser.add_argument('--dirout', default='../Dataset/meanfield/test.npy', type=str, help='where to store created files')
args = parser.parse_args()

# Simulation Parameters
MU = 0.1
OMEGA = 1
A = -0.1
LAMBDA = 10


def dynamics(t, y):
    """Mean Field model"""
    out = np.zeros_like(y)
    out[0] = MU * y[0] - OMEGA * y[1] + A * y[0] * y[2]
    out[1] = OMEGA * y[0] + MU * y[1] + A * y[1] * y[2]
    out[2] = -LAMBDA * (y[2] - y[0] ** 2 - y[1] ** 2)
    return out


def generate(nb_ex):
    dataset = np.zeros((nb_ex, 3, 120))  # That will store each simulation
    t_span = [0, args.span]
    t_eval = np.linspace(0, args.span, args.span * args.fps)

    for k in tqdm(range(nb_ex)):
        # Complicated initial condition...
        y0 = np.zeros(3)
        r = np.random.random(1) * 1.1
        theta = np.random.random(1) * 2 * np.pi
        y0[0] = r * np.cos(theta)
        y0[1] = r * np.sin(theta)
        y0[2] = y0[0] ** 2 + y0[1] ** 2

        sol = solve_ivp(dynamics, y0=y0, t_span=t_span, t_eval=t_eval)
        dataset[k] = sol.y

    np.save(args.dirout, dataset)


if __name__ == '__main__':
    generate(args.nb_ex)
    # demo()
