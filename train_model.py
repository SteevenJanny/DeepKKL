import torch
import numpy as np
from Dataset.dataloader import DynamicalSystem
from torch.utils.data import DataLoader
from models.kkl import KKL
from models.baselines import RNN, GRU
import argparse

parser = argparse.ArgumentParser(description="Train/Evaluate model")
parser.add_argument('--epoch', default=0, type=int, help="Num. of Epochs, set to 0 to evaluate trained model")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--timespan', default=5, type=int, help="Number of known timesteps")
parser.add_argument('--dataset', default='vanderpol', type=str,
                    help="Dataset selection {vanderpol, lorenz, lotkavolterra, meanfield}")
parser.add_argument('--model', default='kkl', type=str, help="Model to train")
args = parser.parse_args()

TIMESPAN = args.timespan

models = {"kkl": KKL, "rnn": RNN, "gru": GRU}


def evaluate():
    dataloader = DataLoader(DynamicalSystem("test", args.dataset, horizon=100), batch_size=1, shuffle=False)
    latent_space_dim = dataloader.dataset.dim * 2 + 1
    model = models[args.model](observation_dim=1,
                               latent_space_dim=latent_space_dim,
                               prediction_horizon=100,
                               mlp_hidden_dim=128)
    try:
        model.load_state_dict(torch.load("trained_models/" + args.model + "_" + args.dataset + ".nn"))
    except FileNotFoundError:
        model.load_state_dict(torch.load("../trained_models/" + args.model + "_" + args.dataset + ".nn"))

    # Set viz to True to visualize the prediction
    print("TEST error : ", validate(model, dataloader, viz=False))


def visualise(gt, pred):
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    plt.title(args.model + " Prediction")
    plt.plot(gt[0], label="Ground Truth")
    plt.plot(pred[0], '--', label="Prediction")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Observation")
    plt.show()


def validate(model, dataloader, viz=False):
    model.eval()
    loss = []
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            obs = x[:, 0, :].unsqueeze(-1)
            prediction = model(obs[:, :TIMESPAN])
            error = ((obs[:, TIMESPAN:] - prediction[:, TIMESPAN:]) ** 2).mean()
            if viz is True:
                visualise(obs, prediction)
            loss.append(error)
        return np.mean(loss)


def train():
    print(args)

    ## CREATE DATALOADER
    train_dataloader = DataLoader(DynamicalSystem("train", args.dataset, horizon=50), shuffle=True,
                                  batch_size=64)
    val_dataloader = DataLoader(DynamicalSystem("val", args.dataset, horizon=50), shuffle=False, batch_size=64)

    ## DEFINE MODEL
    latent_space_dim = train_dataloader.dataset.dim * 2 + 1
    model = models[args.model](observation_dim=1,
                               latent_space_dim=latent_space_dim,
                               prediction_horizon=50,
                               mlp_hidden_dim=128)

    ## OPTIMIZER
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(train_dataloader):
            obs = x[:, 0, :].unsqueeze(-1)  # Extract the observation
            prediction = model(obs[:, :TIMESPAN])  # Predict
            cost = ((obs - prediction) ** 2).mean()  # Compute error

            ## BACKPROP
            optim.zero_grad()
            cost.backward()
            optim.step()

        cost = validate(model, val_dataloader)

        # Save model weights
        try:
            torch.save(model.state_dict(), "trained_models/" + args.model + "_" + args.dataset + ".nn")
        except FileNotFoundError:
            torch.save(model.state_dict(), "../trained_models/" + args.model + "_" + args.dataset + ".nn")


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        train()
        evaluate()
