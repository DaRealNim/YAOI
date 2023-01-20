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
from gymenv import OthelloEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, device=device).double()
        self.conv2 = nn.Conv2d(8, 16, 3, device=device).double()

        self.fc1 = nn.Linear(16*8*8, 512, device=device).double()
        self.fc2 = nn.Linear(512, 256, device=device).double()
        self.fc3 = nn.Linear(256, 64, device=device).double()
        


        # self.input_layer = nn.Linear(64, 128, device=device).double()
        # self.hidden_layer_1 = nn.Linear(128, 128, device=device).double()
        # self.hidden_layer_2 = nn.Linear(128, 128, device=device).double()
        # self.output_layer = nn.Linear(128, 64, device=device).double()

    def forward(self, x):
        # x = self.input_layer(x)
        # x = torch.tanh(x)

        # x = self.hidden_layer_1(x)
        # x = torch.tanh(x)

        # x = self.hidden_layer_2(x)
        # x = torch.tanh(x)

        # out = self.output_layer(x)
        x = self.conv1(x.reshape((-1, 1, 8, 8)))
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(-1, 16*8*8)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        return x

class Agent():
    def __init__(self):
        self.policy_net = Net().to(device)
        self.target_net = Net().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.totalepochs = 0

        self.EPS_START       = 0.99
        self.EPS_END         = 0.05
        self.EPS_DECAY       = 2000
        self.TARGET_UPDATE   = 10
        self.GAMMA           = 0.99
        self.DISPLAY         = 200
        self.EVALUATE        = 20

        self.epsilon = self.EPS_START
        self.optimizer = optim.Adam(self.policy_net.parameters())

    def _make_state(self, board, color):
        return torch.from_numpy(board.arr.reshape(64)*color).to(device)

    def usepolicy(self, board, color):
        with torch.no_grad():
            out = self.policy_net(self._make_state(board, color)).cpu().numpy()
            illegal_map = board.get_impossibles_moves_map(color)
            out[illegal_map] = np.nan
            return np.unravel_index(np.nanargmax(out), (8,8)), out[np.nanargmax(out)]


    def select_action(self, board, color):
        v = random.random()
        if v > self.epsilon:
            action, _ = self.usepolicy(board, color)
            return action
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
        print(target_next_val)
        print(rewards)
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


    def evaluate(self, against, agentcolor, returnscore=False):
        env = OthelloEnv()
        observation = env.reset()
        reward = 0
        while True:
            currentturn = observation["turn"]
            if currentturn == agentcolor:
                action, _ = self.usepolicy(observation["board"], currentturn)
            else:
                action, _ = against.usepolicy(observation["board"], currentturn)

            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                reward = 1 if observation["board"].has_won(agentcolor) else 0 if observation["board"].is_draw() else -1
                if returnscore:
                    return observation["board"].score(agentcolor), observation["board"].score(-agentcolor)
                return reward

    def _run_evaluation(self, lastchk, bar, cumreward, eval_against):
        if not lastchk == "" or eval_against:
            x = Agent()
            if eval_against:
                x.load(eval_against)
            else:
                x.load(lastchk)
            rewardBlack = self.evaluate(x, -1)
            rewardWhite = self.evaluate(x, 1)
            cumreward.append(rewardBlack)
            cumreward.append(rewardWhite)
            # bar.write("Evaluation to previous version cumulative reward : " + str(cumr1+cumr2))
            # bar.write("Mean cumulative reward since training started : " +str(np.array(cumreward).mean()))

    def _display_progress(self, epoch, lastchk, bar, losshistory, cumreward):
        if lastchk != "":
            os.remove("checkpoints/"+lastchk+".chkpt")
        lastchk = "checkpoint_e"+str(epoch)
        self.save(lastchk)
        
        avgl = np.array(losshistory[len(losshistory)-self.DISPLAY : len(losshistory)]).mean()
        bar.write("Average last "+str(self.DISPLAY)+" losses       : " + str(avgl))
        if cumreward:
            bar.write("Cumulative reward against previous version : " + str(np.sum(cumreward)))
        bar.write("Epsilon : " + str(self.epsilon))
        bar.write("+----------------------------------------+")
        cumreward.clear()
        return lastchk



    def train(self, epochs, lastmodel=None, eval_against=None):
        # initialize Gym environment
        env = OthelloEnv()

        losshistory = []
        cumreward = []
        lastchk=""

        # Load a potential previous model for evaluation
        if lastmodel:
            lastchk = lastmodel
            self.load(lastmodel)

        bar = tqdm(initial=self.totalepochs, total=self.totalepochs + epochs)
        for e in range(self.totalepochs, self.totalepochs + epochs):
            bar.update()
            self.epsilon = self.EPS_END + (self.EPS_START-self.EPS_END) * math.exp(-1*self.totalepochs / self.EPS_DECAY)
            if e % self.EVALUATE == 0:
                self._run_evaluation(lastchk, bar, cumreward, eval_against)
            if e % self.DISPLAY == 0 and len(losshistory)>=self.DISPLAY:
                lastchk = self._display_progress(e, lastchk, bar, losshistory, cumreward)

            memory = {
                "states"     : [],
                "nextstates" : [],
                "rewards"    : [],
                "actions"    : []
            }

            observation = env.reset()
            while True:
                currentturn = observation["turn"]
                
                memory["states"].append(self._make_state(observation["board"], observation["turn"]))

                putrow, putcol = self.select_action(observation["board"], observation["turn"])
                memory["actions"].append(np.ravel_multi_index((putrow, putcol), (8,8)))

                observation, reward, terminated, truncated, info = env.step((putrow, putcol))
                memory["rewards"].append(reward)
                memory["nextstates"].append(self._make_state(observation["board"], observation["turn"]))
                
                if terminated or truncated:
                    break

            observation = env.reset()
            lossval = self.optimize(memory)
            losshistory.append(lossval)
            self.totalepochs += 1
            if e % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            
                
        plt.plot(losshistory)
        plt.show()
        plt.plot(cumreward)
        plt.show()


if __name__ == "__main__":
    a = Agent()
    a.train(1000)