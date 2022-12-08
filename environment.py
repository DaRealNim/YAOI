import numpy as np
import itertools

class Board():
    def __init__(self):
        self.arr = np.zeros((8,8))
        self.arr[3][3] = 1
        self.arr[4][3] = -1
        self.arr[3][4] = -1
        self.arr[4][4] = 1

    def is_legal_position(self, row, col):
        return 0 <= col < 8 and 0 <= row < 8 and self.arr[row][col] == 0

    def check_line(self, row, col, drow, dcol, color, to_flip):
        tmp = []
        while True:
            col += dcol
            row += drow
            if (not (0<=col<8 and 0<=row<8)) or self.arr[row][col] == 0:
                break
            if self.arr[row][col] != color:
                tmp.append((row, col))
                continue
            to_flip += tmp
            break

    def would_flip(self, row, col, color):
        to_flip = []
        if not self.is_legal_position(row, col):
            return to_flip
        for drow, dcol in itertools.product([-1, 0, 1], repeat=2):
            if (drow, dcol) == (0, 0):
                continue
            self.check_line(row, col, drow, dcol, color, to_flip)
        return to_flip

    def put(self, row, col, color):
        w_flip = self.would_flip(row, col, color)
        if not w_flip:
            return False
        self.arr[row][col] = color
        w_flip = np.array(w_flip)
        self.arr[w_flip[:, 0], w_flip[:, 1]] *= -1
        return True

    def get_possible_moves(self, color):
        moves = []
        for row, col in itertools.product(range(8), repeat=2):
            if self.would_flip(row, col, color):
                moves.append((row, col))
        return moves

    def is_over(self):
        return not (self.get_possible_moves(-1) or self.get_possible_moves(1))

    def score(self, color):
        return np.count_nonzero(self.arr == color)

    def get_impossibles_moves_map(self, color):
        a = np.ones((8,8))
        possmoves = np.array(self.get_possible_moves(color))
        a[possmoves[:,0], possmoves[:,1]] = 0
        a = a.reshape(64).astype(bool)
        return a



if __name__ == "__main__":
    board = Board()
    print(board.arr)