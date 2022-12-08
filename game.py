from tkinter import *
import numpy as np
from environment import Board
import model
import time
import sys

PLAYERCOLOR = 1

class Reversi():
    def __init__(self, agent):
        self.agent = agent
        self.window = Tk()
        self.window.title = "YARE interface"
        self.canvas = Canvas(self.window, width=800, height=800)
        self.canvas.configure(bg="darkgreen")
        self.canvas.pack()

        self.window.bind('<Button-1>', self.click)
        self.initialize_board()
        self.draw_pieces()

        self.turn = -1

        self.canvas.update_idletasks()
        time.sleep(2)
        if self.turn != PLAYERCOLOR:
            self.model_play()

    def model_play(self):
        row, col = self.agent.usepolicy(self.board, self.turn)
        self.board.put(row, col, self.turn)
        self.draw_pieces()
        if self.board.is_over():
            print(self.board.score(-1), self.board.score(1))
            return
        if not self.board.get_possible_moves(-self.turn):
            self.model_play()
        self.turn *= -1

    def initialize_board(self):
        self.board = Board()
        self.turn = -1
        for i in range(8):
            self.canvas.create_line(i*100, 0, i*100, 800)
            self.canvas.create_line(0, i*100, 800, i*100)

    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                if self.board.arr[row, col] != 0:
                    fillc = "white" if self.board.arr[row, col] == 1 else "black"
                    self.canvas.create_oval(col*100 + 20, row*100 + 20, col*100 + 80, row*100 + 80, fill=fillc)

    def mainloop(self):
        self.window.mainloop()

    
    def click(self, event):
        col = event.x//100
        row = event.y//100
        if self.turn == PLAYERCOLOR:
            res = self.board.put(row, col, self.turn)
            if not res:
                return
            self.draw_pieces()
            if self.board.is_over():
                print(self.board.score(-1), self.board.score(1))
                return
            
            if not self.board.get_possible_moves(-self.turn):
                return
            self.turn *= -1
            self.canvas.update_idletasks()
            time.sleep(1)
            self.model_play()
        pass

    def quit(self):
        self.window.quit()

# assumes we have a model named "model" in checkpoints for now
agent = model.Agent()
agent.load(sys.argv[1])
instance = Reversi(agent)
instance.mainloop()