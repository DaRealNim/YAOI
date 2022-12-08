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
import os


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

        self.totalepochs = 0

        self.EPS_START       = 0.9
        self.EPS_END         = 0.05
        self.EPS_DECAY       = 200
        self.TARGET_UPDATE   = 10
        self.GAMMA           = 0.99
        self.LOSSDISPLAY     = 100

        self.epsilon = self.EPS_START
        self.optimizer = optim.Adam(self.policy_net.parameters())

    def _make_state(self, board, color):
        state = np.append(board.arr.reshape(64), color)
        return torch.from_numpy(state).to(device)

    def usepolicy(self, board, color):
        with torch.no_grad():
            out = self.policy_net(self._make_state(board, color)).cpu().numpy()
            illegal_map = board.get_impossibles_moves_map(color)
            out[illegal_map] = np.nan
            return np.unravel_index(np.nanargmax(out), (8,8))


    def select_action(self, board, color):
        v = random.random()
        if v > self.epsilon:
            return self.usepolicy(board, color)
        else:
            move = random.choice(board.get_possible_moves(color))
            return move

    def optimize(self, memory):
        # state_action_val = self.policy_net(state)
        # next_state_val = self.target_net(nextstate).detach()
        # expected_state_action_val = (next_state_val * self.GAMMA) + reward

        nextstates = torch.vstack(memory["nextstates"]).to(device)
        # print(nextstates, nextstates.shape)
        # print("\n")
        rewards = torch.tensor(memory["rewards"]).unsqueeze(1).to(device)
        # print(rewards)
        # print("\n")
        states = torch.vstack(memory["states"]).to(device)
        # print(states, states.shape)
        # print("\n")
        actions = torch.tensor(memory["actions"]).unsqueeze(1).to(device)
        # print(actions)
        # print("\n")

        target_next_val = self.target_net(nextstates).detach().max(1)[0].unsqueeze(1)
        # print(target_next_val, target_next_val.shape)
        # print("\n")
# 
        target_val = (self.GAMMA*target_next_val) + rewards
        # print(target_val, target_val.shape)
        # print("\n")

        expected_val = self.policy_net(states).gather(1, actions)
        # print(expected_val, expected_val.shape)
        # print("\n")
        # exit()


        criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_val, expected_state_action_val)
        loss = criterion(target_val, expected_val)
        lossval = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return lossval

    def save(self, name="model", folder="checkpoints"):
        torch.save({
            'epochs': self.totalepochs,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon' : self.epsilon,
        }, folder + "/" + name+".chkpt")

    def load(self, name="model", folder="checkpoints"):
        checkpoint = torch.load(folder + "/" + name+".chkpt")
        self.totalepochs = int(checkpoint["epochs"])
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]


    def evaluate(self, against, n=1, agentcolor=-1):
        agentcolor *= -1
        bar = tqdm(total=n)
        wins = 0
        draws = 0
        cumreward = 0
        for e in range(n):
            bar.update()
            b = Board()
            currentturn = -1
            agentcolor *= -1
            
            while True:
                # bar.write(str(b.arr))
                if b.is_over():
                    # if e%10 == 0:
                    #     bar.write(str(b.arr) + "\n\n")
                    wscore = b.score(1)
                    bscore = b.score(-1)
                    winner = 1 if wscore > bscore else (-1 if wscore < bscore else 0)
                    if winner == 0:
                        cumreward -= 30
                        draws += 1
                    elif winner == agentcolor:
                        cumreward += 100
                        wins += 1
                    else:
                        cumreward -= 100
                    break
                if not b.get_possible_moves(currentturn):
                    currentturn *= -1
                    continue
                if currentturn == agentcolor:
                    row, col = self.usepolicy(b, currentturn)
                else:
                    row, col = against.usepolicy(b, currentturn)
                b.put(row, col, currentturn)

                if currentturn == agentcolor:
                    cumreward += b.score(currentturn) - b.score(-currentturn)

                currentturn *= -1
                
        return wins, draws, cumreward



    def train(self, epochs, lastmodel=None):
        agentcolor = 1
        losshistory = []
        cumreward = []
        lastchk=""
        if lastmodel:
            lastchk = lastmodel
            self.load(lastmodel)
        bar = tqdm(initial=self.totalepochs, total=self.totalepochs + epochs)
        for e in range(self.totalepochs, self.totalepochs + epochs):
            bar.update()
            self.epsilon = self.EPS_END + (self.EPS_START-self.EPS_END) * math.exp(-1*self.totalepochs / self.EPS_DECAY)
            if e % self.LOSSDISPLAY == 0 and len(losshistory)>self.LOSSDISPLAY:
                self.save("checkpoint_e"+str(e))
                if not lastchk == "":
                    x = Agent()
                    x.load(lastchk)
                    _, _, cumr1 = self.evaluate(x)
                    _, _, cumr2 = self.evaluate(x, agentcolor=1)
                    cumr = cumr1+cumr2
                    cumreward.append(cumr)
                    os.remove("checkpoints/"+lastchk+".chkpt")
                    
                    bar.write("Evaluation to previous version cumulative reward : " + str(cumr1+cumr2))
                lastchk = "checkpoint_e"+str(e)
                avgl = np.array(losshistory[len(losshistory)-self.LOSSDISPLAY : len(losshistory)]).mean()
                bar.write("Average last 100 losses : " + str(avgl))
                bar.write("Epsilon : " + str(self.epsilon))
                bar.write("+----------------------------------------+")
            b = Board()
            currentturn = -1
            agentcolor *= -1
            memory = {
                "states"     : [],
                "nextstates" : [],
                "rewards"    : [],
                "actions"    : []
            }
            for i in itertools.count():
                if b.is_over():
                    wscore = b.score(1)
                    bscore = b.score(-1)
                    winner = 1 if wscore > bscore else (-1 if wscore < bscore else 0)
                    if winner == 0:
                        reward = -30
                    elif winner == agentcolor:
                        reward = 100
                    else:
                        reward = -100
                    break
                if not b.get_possible_moves(currentturn):
                    currentturn *= -1
                    continue

                state = self._make_state(b, currentturn)

                # if fixed_opponent and currentturn != agentcolor:
                #     putrow, putcol = fixed_opponent.usepolicy(b, currentturn)
                # else:
                putrow, putcol = self.select_action(b, currentturn)
                b.put(putrow, putcol, currentturn)
                score = b.score(currentturn)
                done = b.is_over()

                currentturn *= -1
                nextstate = self._make_state(b, currentturn)

                if -currentturn == agentcolor:
                    if done:
                        winner = 1 if b.score(1) > b.score(-1) else -1
                        if winner == agentcolor:
                            reward = 100
                        else:
                            reward = -100
                    else:
                        reward = score - b.score(currentturn)

                    memory["states"].append(state)
                    memory["nextstates"].append(nextstate)
                    memory["rewards"].append(reward)
                    memory["actions"].append(np.ravel_multi_index((putrow, putcol), (8,8)))

                    if i % self.TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    break
            lossval = self.optimize(memory)
            losshistory.append(lossval)
            self.totalepochs += 1
                
        plt.plot(losshistory)
        plt.show()
        plt.plot(cumreward)
        plt.show()


if __name__ == "__main__":
    a = Agent()
    a.train(1000)