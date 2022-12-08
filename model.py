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

class Agent():
    def __init__(self):
        self.policy_net = Net().to(device)
        self.target_net = Net().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.EPS_START       = 0.9
        self.EPS_END         = 0.05
        self.EPS_DECAY       = 200
        self.TARGET_UPDATE   = 10
        self.GAMMA           = 0.99
        self.LOSSDISPLAY     = 50
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def _make_state(self, board, color):
        state = np.append(board.arr.reshape(64), color)
        return torch.from_numpy(state).to(device)

    def usepolicy(self, board, color):
        with torch.no_grad():
            out = self.policy_net(self._make_state(board, color)).cpu().numpy()
            illegal_map = board.get_impossibles_moves_map(color)
            out[illegal_map] = np.nan
            return np.unravel_index(np.nanargmax(out), (8,8))


    def select_action(self, board, color, steps_done):
        v = random.random()
        eps_treshold = self.EPS_END + (self.EPS_START-self.EPS_END) * math.exp(-1*steps_done / self.EPS_DECAY)
        if v > eps_treshold:
            return self.usepolicy(board, color)
        else:
            move = random.choice(board.get_possible_moves(color))
            return move

    def optimize(self, state, nextstate, reward):
        state_action_val = self.policy_net(state)
        next_state_val = self.target_net(nextstate).detach()
        expected_state_action_val = (next_state_val * self.GAMMA) + reward
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_val, expected_state_action_val)
        lossval = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return lossval

    def save(self, name="model", folder="checkpoints"):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, folder + "/" + name+".chkpt")

    def load(self, name="model", folder="checkpoints"):
        checkpoint = torch.load(folder + "/" + name+".chkpt")
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    def evaluate(self, against, n=100):
        agentcolor = -1
        bar = tqdm(total=n)
        wins = 0
        draws = 0
        for e in range(n):
            bar.update()
            b = Board()
            currentturn = -1
            agentcolor *= -1
            
            while True:
                bar.write(str(b.arr))
                if b.is_over():
                    wscore = b.score(1)
                    bscore = b.score(-1)
                    winner = 1 if wscore > bscore else (-1 if wscore < bscore else 0)
                    if winner == 0:
                        draws += 1
                    if winner == agentcolor:
                        wins += 1
                    break
                if not b.get_possible_moves(currentturn):
                    currentturn *= -1
                    continue
                if currentturn == agentcolor:
                    row, col = self.usepolicy(b, currentturn)
                else:
                    row, col = against.usepolicy(b, currentturn)
                b.put(row, col, currentturn)
                currentturn *= -1
                
        return wins, draws



    def train(self, epochs, fixed_opponent=None):
        agentcolor = 1
        losshistory = []
        rewardhistory = []
        bar = tqdm(total=epochs)
        for e in range(epochs):
            bar.update()
            gamereward = []
            if e % self.LOSSDISPLAY == 0 and len(losshistory)>self.LOSSDISPLAY:
                avgl = np.array(losshistory[len(losshistory)-self.LOSSDISPLAY : len(losshistory)]).mean()
                avgr = np.array(rewardhistory[len(rewardhistory)-self.LOSSDISPLAY : len(rewardhistory)]).mean()
                bar.write("Average last 50 losses : " + str(avgl))
                bar.write("Average last 50 games reward : " + str(avgr))
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

                state = self._make_state(b, currentturn)

                if fixed_opponent and currentturn != agentcolor:
                    putrow, putcol = fixed_opponent.usepolicy(b, currentturn)
                else:
                    putrow, putcol = self.select_action(b, currentturn, steps)
                b.put(putrow, putcol, currentturn)
                score = b.score(currentturn)
                done = b.is_over()

                currentturn *= -1
                nextstate = self._make_state(b, currentturn)

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
                    lossval = self.optimize(state, nextstate, reward)
                    losshistory.append(lossval)

                    if i % self.TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    break
            rewardhistory.append(np.array(gamereward).sum())

                
        plt.plot(losshistory)
        plt.show()
        plt.plot(rewardhistory)
        plt.show()


if __name__ == "__main__":
    agent1 = Agent()
    agent1.load()
    agent2 = Agent()
    agent2.load("_fixed")
    agent1.train(5000, fixed_opponent=agent2)