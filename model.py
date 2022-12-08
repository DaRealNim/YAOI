import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import math
import random
from environment import Board
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(65, 128, device=device).double()
        self.hidden_layer_1 = nn.Linear(128, 128, device=device).double()
        self.hidden_layer_2 = nn.Linear(128, 128, device=device).double()
        self.output_layer = nn.Linear(128, 64, device=device).double()

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.tanh(x)

        x = self.hidden_layer_1(x)
        x = torch.tanh(x)

        x = self.hidden_layer_2(x)
        x = torch.tanh(x)

        out = self.output_layer(x)
        return out

policy_net = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

EPS_START       = 0.9
EPS_END         = 0.05
EPS_DECAY       = 200
TARGET_UPDATE   = 10
GAMMA           = 0.99
LOSSDISPLAY     = 50
optimizer = optim.RMSprop(policy_net.parameters())

def _make_state(board, color):
    state = np.append(board.arr.reshape(64), color)
    return torch.from_numpy(state).to(device)

def usepolicy(board, color):
    with torch.no_grad():
        out = policy_net(_make_state(board, color)).cpu().numpy()
        illegal_map = board.get_impossibles_moves_map(color)
        out[illegal_map] = np.nan
        return np.unravel_index(np.nanargmax(out), (8,8))


def select_action(board, color, steps_done):
    v = random.random()
    eps_treshold = EPS_END + (EPS_START-EPS_END) * math.exp(-1*steps_done / EPS_DECAY)
    if v > eps_treshold:
        return usepolicy(board, color)
    else:
        move = random.choice(board.get_possible_moves(color))
        return move

def optimize(state, nextstate, reward):
    state_action_val = policy_net(state)
    next_state_val = target_net(nextstate).detach()
    expected_state_action_val = (next_state_val * GAMMA) + reward
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_val, expected_state_action_val)
    lossval = loss.item()
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return lossval

def save():
    torch.save(policy_net, "policy.model")
    torch.save(target_net, "target.model")

def load():
    policy_net = torch.load("policy.model")
    target_net = torch.load("target.model")

def train(epochs):
    agentcolor = 1
    losshistory = []
    rewardhistory = []
    for e in tqdm(range(epochs)):
        gamereward = []
        if e % LOSSDISPLAY == 0 and len(losshistory)>LOSSDISPLAY:
            avgl = np.array(losshistory[len(losshistory)-LOSSDISPLAY : len(losshistory)]).mean()
            avgr = np.array(rewardhistory[len(rewardhistory)-LOSSDISPLAY : len(rewardhistory)]).mean()
            print("Average last 50 losses : ", avgl)
            print("Average last 50 games reward : ", avgr)
        b = Board()
        currentturn = -1
        agentcolor *= -1
        steps = 0
        for i in itertools.count():
            if b.is_over():
                winner = 1 if b.score(1) > b.score(-1) else -1
                if winner == agentcolor:
                    reward = 100
                else:
                    reward = -100
                break
            if not b.get_possible_moves(currentturn):
                currentturn *= -1
                continue
            state = _make_state(b, currentturn)

            putrow, putcol = select_action(b, currentturn, steps)
            b.put(putrow, putcol, currentturn)
            score = b.score(currentturn)
            done = b.is_over()

            currentturn *= -1
            nextstate = _make_state(b, currentturn)

            if -currentturn == agentcolor:
                steps += 1
                if done:
                    winner = 1 if b.score(1) > b.score(-1) else -1
                    if winner == agentcolor:
                        reward = 100
                    else:
                        reward = -100
                else:
                    reward = score - b.score(currentturn)

                gamereward.append(reward)
                lossval = optimize(state, nextstate, reward)
                losshistory.append(lossval)

                if i % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        rewardhistory.append(np.array(gamereward).sum())

            
    plt.plot(losshistory)
    plt.show()
    plt.plot(rewardhistory)
    plt.show()

if __name__ == "__main__":
    train(1000)