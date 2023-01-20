import gym
from gym import spaces
import numpy as np

from environment import Board

class OthelloEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([8, 8])
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,8), dtype=np.int8)
        self.reset()

    def reset(self):
        self.board = Board()
        self.player = -1
        self.finished = False
        return self._get_obs()

    def step(self, action):
        row, col = action
        legal = self.board.put(row, col, self.player)
        reward = 0

        if legal:
            currentscore = self.board.score(self.player)
            oppositescore = self.board.score(-1*self.player)
            reward = currentscore-oppositescore
            if self.board.get_possible_moves(-1 * self.player): #if opponent has possible moves, switch turn
                self.player *= -1
            elif not self.board.get_possible_moves(self.player): #else, if we don't have any possible moves, the game is over
                self.finished = True
                if self.board.has_won(self.player):
                    reward = 50
                elif self.board.has_won(-self.player):
                    reward = -50
            # else just play again
        
        return self._get_obs(), reward, self.finished, False, None



    def render(self, mode="human"):
        pass

    def _get_obs(self):
        return {
            "board" : self.board,
            "turn" : self.player,
        }