# -*- coding: utf-8 -*-
# blockgo.py
# Katsuki Ohto

from enum import IntEnum

import numpy as np

import go.go as go

# rule
ILLEGAL_BLOCK = 5
DOUBLE_BLOCK = 6
DISTINCT_BLOCK = 7
IGNORE_STAR = 8
NO_BLOCK = 9

# direction
def check_dir(dir):
    return 0 <= dir and dir < 4

# block
class BlockType(IntEnum):
    NONE = -8 # block -> color の変換で色が EMPTY になるように
    B = 0
    I = 1
    J = 2
    L = 3
    O = 4
    S = 5
    T = 6
    Z = 7
    NUM = 8
BlockTypeChar = "BIJLOSTZ"

def check_bt(bt):
    return 0 <= bt and bt < BlockType.NUM
def char_to_bt(c):
    return BlockTypeChar.find(c)

BLOCK_COMPONENTS = [
    [[0, 0]], # B
    [[0, 0], [0, 1], [0, 2], [0, 3]], # I
    [[0, 0], [0, 1], [0, 2], [1, 2]], # J
    [[0, 0], [0, 1], [0, 2], [1, 0]], # L
    [[0, 0], [0, 1], [1, 0], [1, 1]], # O
    [[0, 0], [1, 0], [1, 1], [2, 1]], # S
    [[0, 0], [0, 1], [0, 2], [1, 1]], # T
    [[0, 0], [0, 1], [1, 1], [1, 2]]  # Z
]

BLOCK_COMPONENTS_Z = [[go.IMAGE_LENGTH * cmp[0] + cmp[1] for cmp in bc] for bc in BLOCK_COMPONENTS]
BLOCK_NUM = [2, 1, 1, 1, 1, 1, 1, 1]
BLOCK_STONES = [len(cmp) for cmp in BLOCK_COMPONENTS]
BLOCK_LENGTH_X = [1, 1, 2, 2, 2, 3, 2, 2]
BLOCK_LENGTH_Y = [1, 4, 3, 3, 2, 2, 3, 3]

BLOCK_COMPONENTS_ZD = [[[go.IMAGE_LENGTH * cmp[0] + cmp[1] for cmp in bc],
                        [go.IMAGE_LENGTH * cmp[1] - cmp[0] for cmp in bc],
                        [go.IMAGE_LENGTH * -cmp[0] - cmp[1] for cmp in bc],
                        [go.IMAGE_LENGTH * -cmp[1] + cmp[0] for cmp in bc]]
                       for bc in BLOCK_COMPONENTS]

ROT4 = [[[1, 0], [0, 1]],
        [[0, 1], [-1, 0]],
        [[-1, 0], [0, -1]],
        [[0, -1], [1, 0]]]

def rotate(tuple, dir):
    return ROT4[dir][0][0] * tuple[0] + ROT4[dir][0][1] * tuple[1], ROT4[dir][1][0] * tuple[0] + ROT4[dir][1][1] * tuple[1]

BLOCK_COMPONENTS_D = [[[rotate(cmp, dir) for cmp in bc] for dir in range(4)]for bc in BLOCK_COMPONENTS]

def bt_dir_range(bt, dir, ix, iy):
    rx, ry = rotate((BLOCK_LENGTH_X[bt], BLOCK_LENGTH_Y[bt]), dir)
    if rx > 0:
        ixmin, ixmax = ix, ix + rx - 1
    else:
        ixmin, ixmax = ix + rx + 1, ix
    if ry > 0:
        iymin, iymax = iy, iy + ry - 1
    else:
        iymin, iymax = iy + ry + 1, iy
    return ixmin, ixmax, iymin, iymax

def is_on_board_block_ixy(bt, dir, ix, iy, l):
    ixmin, ixmax, iymin, iymax = bt_dir_range(bt, dir, ix, iy)
    return 0 <= ixmin and ixmax < l and 0 <= iymin and iymax < l

def is_on(bt, dir, z, tz):
    for dz in BLOCK_COMPONENTS_ZD[bt][dir]:
        if tz == z + dz:
            return True
    return False

def to_block(c, bt):
    return (c - go.BLACK) * BlockType.NUM + bt

def block_string(bt, dir = 0):
    dir = (dir + 3) % 4
    rx, ry = rotate((BLOCK_LENGTH_X[bt], BLOCK_LENGTH_Y[bt]), dir)
    ix, iy = 0, 0
    if rx > 0:
        ixmin, ixmax = ix, ix + rx - 1
    else:
        ixmin, ixmax = ix + rx + 1, ix
    if ry > 0:
        iymin, iymax = iy, iy + ry - 1
    else:
        iymin, iymax = iy + ry + 1, iy
    print(ixmin, ixmax, iymin, iymax)
    s = ""
    for dx in range(ixmin - 1, ixmax + 1):
        for dy in range(iymin - 1, iymax + 1):
            if (dx, dy) in BLOCK_COMPONENTS_D[bt][dir]:
                if (dx, dy) == (0, 0):
                    s += "■"
                else:
                    s += "●"
            else:
                s += " "
        s += "\n"
    return s

def block2color(b):
    return (b // BlockType.NUM) + go.BLACK
def block2type(b):
    return b % BlockType.NUM

class Board(go.Board):
    def __init__(self, l):
        super().__init__(l)
        self.blockturn = 0
        self.blockcell = np.zeros((go.IMAGE_SIZE), dtype=np.int32)
        self.blocks = np.zeros((2, BlockType.NUM), dtype=np.int32)
        self.z_star = [go.XYtoZ(self.padding + 3, self.padding + 3),
                  go.XYtoZ(self.padding + 3, self.padding + self.length - 4),
                  go.XYtoZ(self.padding + self.length - 4, self.padding + 3),
                  go.XYtoZ(self.padding + self.length - 4, self.padding + self.length - 4)]
        self.clear()

    def clear(self):
        super().clear()
        self.blockturn = 0
        for z in range(go.IMAGE_SIZE):
            self.blockcell[z] = BlockType.NONE
        for ic in range(2):
            for bt in range(BlockType.NUM):
                self.blocks[ic][bt] = BLOCK_NUM[bt]

    def check_double(self, bt, dir, z):
        for dz in BLOCK_COMPONENTS_ZD[bt][dir]:
            if self.blockcell[z + dz] != BlockType.NONE:
                return False
        return True
    def check_adjacent(self, c, bt, dir, z):
        for dz in BLOCK_COMPONENTS_ZD[bt][dir]:
            for dz4 in go.DIR4:
                b = self.blockcell[z + dz + dz4]
                if block2color(self.blockcell[z + dz + dz4]) != BlockType.NONE and block2color(self.blockcell[z + dz + dz4]) == c:
                    return True
        return False
    def check_star(self, bt, dir, z):
        for tz in self.z_star:
            if is_on(bt, dir, z, tz):
                return True
        return False

    def check_block(self, c, bt, dir, z):
        ix = go.ZtoX(z) - self.padding
        iy = go.ZtoY(z) - self.padding
        if not check_bt(bt) or not check_dir(dir) or z == go.PASS: # パス
            return go.VALID
        if self.blocks[go.to_color_index(c)][bt] <= 0:
            return NO_BLOCK
        if not is_on_board_block_ixy(bt, dir, ix, iy, self.length): # 盤外
            return go.OUT
        if not self.check_double(bt, dir, z): # 二重置き
            return DOUBLE_BLOCK
        if self.blockturn < 4:
            if not self.check_star(bt, dir, z):
                return IGNORE_STAR
        else:
            if not self.check_adjacent(c, bt, dir, z):
                return DISTINCT_BLOCK
        return go.VALID

    def move_block(self, c, bt, dir, z, force = False):
        if not force:
            validity = self.check_block(c, bt, dir, z)
            if validity != go.VALID:
                return validity
        if not check_bt(bt) or not check_dir(dir) or z == go.PASS: # パス
            super().move(c, go.PASS)
        else:
            for dz in BLOCK_COMPONENTS_ZD[bt][dir]:
                self.blockcell[z + dz] = to_block(c, bt)
                super().move(c, z + dz)
            self.blocks[go.to_color_index(c)][bt] -= 1
        self.blockturn += 1
        return go.VALID

    def to_string(self):
        s = ""
        for x in range(self.length):
            for y in range(self.length):
                s += str(self.blockcell[go.XYtoZ(x + self.padding, self.padding + y)]) + " "
            s += "\n"
        return s + super().to_string() + "Block Turn : " + str(self.blockturn) + "\n"

"""def print_line(lst):
    length_x = 0, length_y = 0
    max_size = 0
    splitted_lst = []
    for l in lst:
        splitted = l.split("\n")
        splitted_lst.append(splitted)
        length_y = max(length_y, len(splitted))
        for s in splitted:
            max_size = max(max_size, len(s))
    length_x = len(lst)
    print """


if __name__ == '__main__':
    for bt in range(BlockType.NUM):
        print(bt, BlockType(bt))
        for dir in range(4):
            print(block_string(bt, dir))

    ok = True
    """for i in range(100):
        bd = Board(13)
        print(bd.to_string())
        bt, dir, z = np.random.randint(BlockType.NUM), np.random.randint(4), bd.z_star[np.random.randint(4)]
        print(bt, dir, z)
        v = bd.move_block(go.BLACK, bt, dir, z)
        if v != go.VALID:
            print(v)
            of = False
            break
        print(bd.to_string())
    if ok:
        print("ok")
    else:
        print("failed")"""