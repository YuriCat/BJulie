# -*- coding: utf-8 -*-
# go.py
# Katsuki Ohto

import copy
import math
import time

import numpy as np

# color
EMPTY = 0
BLACK = 1
WHITE = 2
WALL = 3

colorString = ["EMPTY", "BLACK", "WHITE", "WALL"]
colorChar = " BW/"
xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

def to_turn_color(t):
    return (t % 2) + BLACK;

def flip_color(c):
    return BLACK + WHITE - c

def to_color_index(c):
    return c - BLACK

def char_to_ix(c):
    return xChar.find(c)

# rule
KOMIX2 = 7 # for block-go

VALID = 0
LEGAL = 0
DOUBLE = 1
SUICIDE = 2
KO = 3
OUT = 4
EYE = 16

ruleString = {VALID: "VALID",
              DOUBLE: "DOUBLE",
              SUICIDE: "SUICIDE",
              KO: "KO",
              OUT: "OUT",
              EYE: "EYE"}

MAX_LENGTH = 19
MAX_SIZE = MAX_LENGTH * MAX_LENGTH

# board image
PADDING = 2

IMAGE_LENGTH = MAX_LENGTH + PADDING * 2
IMAGE_SIZE = IMAGE_LENGTH * IMAGE_LENGTH

IMAGE_PLAINS = 40

DIR4 = [-IMAGE_LENGTH, -1, +1, +IMAGE_LENGTH]

def XYtoZ(x, y):
    return x * IMAGE_LENGTH + y
def ZtoX(z):
    return z // IMAGE_LENGTH
def ZtoY(z):
    return z % IMAGE_LENGTH

def is_on_board_z(z, l):
    padding = PADDING + (MAX_LENGTH - l) // 2
    ix = ZtoX(z) - padding
    iy = ZtoY(z) - padding
    return 0 <= ix and ix < l and 0 <= iy and iy < l

# move
PASS = 0
RESIGN = -1

# hash key
stoneHashTable = np.zeros((IMAGE_SIZE, 4), dtype = np.uint64)
koHashTable = np.zeros((IMAGE_SIZE), dtype = np.uint64)
for i in range(IMAGE_SIZE):
    for j in range(1, 4):
        stoneHashTable[i][j] = (np.uint64(np.random.randint(2 ** 32)) * (2 ** 32)) + np.uint64(np.random.randint(2 ** 32))
    koHashTable[i] = (np.uint64(np.random.randint(2 ** 32)) * (2 ** 32)) + np.uint64(np.random.randint(2 ** 32))

class Board:
    def __init__(self, l):
        self.length = l
        self.captured = [0, 0]
        #self.lastColor = WHITE
        self.lastZ = RESIGN
        self.turn = 0
        self.koZ = PASS
        self.boardKeySet = set()
        self.boardKey = np.uint64(0)
        self.cell = np.zeros((IMAGE_SIZE), dtype = int)
        self.padding = PADDING + (MAX_LENGTH - l) // 2
        for z in range(IMAGE_SIZE):
            self.cell[z] = WALL
        for x in range(self.length):
            for y in range(self.length):
                z = XYtoZ(self.padding + x, self.padding + y)
                self.cell[z] = EMPTY

    def clear(self):
        self.captured[0] = 0
        self.captured[1] = 0
        self.lastZ = RESIGN
        self.turn = 0
        self.koZ = PASS
        self.boardKey = np.uint64(0)
        self.boardKeySet = set()
        for z in range(IMAGE_SIZE):
            self.cell[z] = WALL
        for x in range(self.length):
            for y in range(self.length):
                z = XYtoZ(self.padding + x, self.padding + y)
                self.cell[z] = EMPTY

    def reverse(self):
        # 盤面を黒白反転
        self.captured.reverse()
        for z in range(IMAGE_SIZE):
            if self.cell[z] == BLACK:
                self.cell[z] = WHITE
            elif self.cell[z] == WHITE:
                self.cell[z] = BLACK


    def countLibertyAndStringSub(self, flag, c, z):
        lib = 0
        str = 0
        flag[z] = 1
        str += 1
        for i in range(4):
            tz = z + DIR4[i]
            if flag[tz]:
                continue # チェック済
            if self.cell[tz] == EMPTY:
                flag[tz] = 1;
                lib += 1 # 隣が空なので呼吸点
            elif self.cell[tz] == c:
                # 隣が同じ色の石ならば連になるので、さらに探索
                tlib, tstr = self.countLibertyAndStringSub(flag, c, tz)
                lib += tlib
                str += tstr
        return (lib, str)

    def countLibertyAndString(self, z):
        flag = np.zeros(IMAGE_SIZE, dtype = int)
        return self.countLibertyAndStringSub(flag, self.cell[z], z)

    def remove(self, c, z):
        # 連となっている石を全て取り除く
        if self.cell[z] != c:
            return -1
        self.cell[z] = EMPTY
        # ハッシュ値更新
        self.boardKey ^= stoneHashTable[z][c]
        for i in range(4):
            tz = z + DIR4[i]
            if self.cell[tz] == c:
                self.remove(c, tz)
                        
    def check(self, c, z):
        #if z == XYtoZ(16 - 1 + PADDING, 19 - 1 + PADDING): # 2017/1/15 KGS bot tournament vs LeelaBot
        #    return KO
        if z == PASS: # パス
            return VALID
        if not is_on_board_z(z, self.length): # 盤外
            return OUT
        if self.cell[z] != EMPTY: # 二重置き
            return DOUBLE
        if z == self.koZ:
            return KO # コウ
        
        ac = np.empty(4, dtype = int)
        alib = np.empty(4, dtype = int)
        astr = np.empty(4, dtype = int)
            
        oc = flip_color(c)
            
        # 4方向のダメの数と石数
        empty = 0 # 空白
        wall = 0 # 盤外
        aliveBuddies = 0
        capture = 0 # 取る石の数
        koCandidate = 0 # コウかもしれない場所
            
        for i in range(4):
            tz = z + DIR4[i]
            tc = self.cell[tz]
            #print("%d : %d" % (i, tc))
            if tc == EMPTY:
                empty += 1
            elif tc == WALL:
                wall += 1
    
            ac[i] = tc
            if tc == EMPTY or tc == WALL:
                # 盤外や空点には呼吸点は無いし連でもない
                alib[i] = 0
                astr[i] = 0
                continue
            # 呼吸点と連のサイズを数える
            tlib, tstr = self.countLibertyAndString(tz)
            alib[i] = tlib
            astr[i] = tstr
                
            if tc == oc and tlib == 1:
                # 元々呼吸点が1つなので、取ることが出来る
                # 取った場合には、コウになる可能性がある
                capture += tstr
                koCandidate = tz
            if tc == c and tlib >= 2:
                # 呼吸点が2つ以上あるので、同じ色の生きた石である
                aliveBuddies += 1

        if capture == 0 and empty == 0 and aliveBuddies == 0:
            # 石を取るわけでも、周りに空点もなく、生きた味方とつながるわけでもないので自殺手
            return SUICIDE # 自殺手
        if wall + aliveBuddies == 4:
            # 周囲が味方で囲われているので眼
            return EYE # 眼
        return VALID

    def checkRoot(self, c, z):
        if z == PASS:
            return VALID
        ret = self.check(c, z)
        if ret != VALID:
            return ret
        # スーパーコウを判定
        after = copy.deepcopy(self)
        after.move(c, z)
        if after.boardKey in after.boardKeySet:
            return SUPER_KO
        return VALID
            
    def move(self, c, z, force = False):
        self.boardKeySet.add(self.boardKey) # ハッシュ値保存
        if z == PASS: # パス
            if self.koZ != PASS:
                # コウのハッシュ値を戻しておく
                self.boardKey ^= koHashTable[self.koZ]
            self.koZ = PASS
            self.lastZ = z
            self.turn += 1
            return VALID
        if not is_on_board_z(z, self.length): # 盤外
            return OUT
        if self.cell[z] != EMPTY: # 二重置き
            return DOUBLE
        if z == self.koZ:
            return KO # コウ
            
        ac = np.empty(4, dtype = int)
        alib = np.empty(4, dtype = int)
        astr = np.empty(4, dtype = int)
                            
        oc = flip_color(c)
                                
        # 4方向のダメの数と石数
        empty = 0 # 空白
        wall = 0 # 盤外
        aliveBuddies = 0
        capture = 0 # 取る石の数
        koCandidate = 0 # コウかもしれない場所
                                                    
        for i in range(4):
            tz = z + DIR4[i]
            tc = self.cell[tz]
            if tc == EMPTY:
                empty += 1
            elif tc == WALL:
                wall += 1
                        
            ac[i] = tc
            if tc == EMPTY or tc == WALL:
                # 盤外や空点には呼吸点は無いし連でもない
                alib[i] = 0
                astr[i] = 0
                continue
                                                                                                    
            # 呼吸点と連のサイズを数える
            tlib, tstr = self.countLibertyAndString(tz)
            alib[i] = tlib
            astr[i] = tstr
            if tc == oc and tlib == 1:
                # 元々呼吸点が1つなので、取ることが出来る
                # 取った場合には、コウになる可能性がある
                capture += tstr
                koCandidate = tz
            if tc == c and tlib >= 2:
                # 呼吸点が2つ以上あるので、同じ色の生きた石である
                aliveBuddies += 1
            
        if capture == 0 and empty == 0 and aliveBuddies == 0:
            # 石を取るわけでも、周りに空点もなく、生きた味方とつながるわけでもないので自殺手
            return SUICIDE # 自殺手
        if force == False and wall + aliveBuddies == 4:
            # 周囲が味方で囲われているので眼
            return EYE # 眼
            
        # 石を取る操作
        for i in range(4):
            tz = z + DIR4[i]
            if self.cell[tz] == oc: # 既に消した石を再び消さないように毎回チェック
                if alib[i] == 1: # 呼吸点が1つなので、今置く石で取られる
                    self.remove(oc, tz)
                    self.captured[to_color_index(c)] += astr[i]

        # 石を置く
        self.cell[z] = c
        self.lastZ = z

        # ハッシュ値更新
        if self.koZ != PASS:
            # コウのハッシュ値を戻しておく
            self.boardKey ^= koHashTable[self.koZ]
        self.boardKey ^= stoneHashTable[z][c]
        
        # コウかどうかを判断
        lib, str = self.countLibertyAndString(z)
            
        if capture == 1 and lib == 1 and str == 1:
            # 石を1つ取って、サイズ1、呼吸点1の連が出来るならば、コウになる
            self.koZ = koCandidate
            if koCandidate != PASS:
                # コウのハッシュ値
                self.boardKey ^= koHashTable[self.koZ]
        else:
            self.koZ = PASS
        self.turn += 1
        return VALID
        
    def countLiberty(self, z):
        lib, str = self.countLibertyAndString(z)
        return lib
    
    def count_chinese_score2(self):
        score = 0
        kind = [0, 0, 0] # empty, black, white
        kind[BLACK] = kind[WHITE] = 0
        for ix in range(self.length):
            for iy in range(self.length):
                 z = XYtoZ(ix + self.padding, iy + self.padding)
                 c = self.cell[z]
                 kind[c] += 1
                 if c != EMPTY:
                     continue
                 mk = [0, 0, 0, 0]
                 for i in range(4):
                     tz = z + DIR4[i]
                     mk[self.cell[tz]] += 1
                 if mk[BLACK] and mk[WHITE] == 0:
                     score += 1
                 elif mk[WHITE] and mk[BLACK] == 0:
                     score -= 1
        score += kind[BLACK] - kind[WHITE]
        return score * 2 - KOMIX2
    """def count_settled_score2_sub(self, z, flag):
        flag[z] = 1
        for dz in go.DIR4:
            if
    def count_settled_score2(self):
        score = 0
        for ix in range(self.length):
            for iy in range(self.length):
                flag = np.zeros((IMAGE_SIZE), dtype=np.int32)"""

    def to_string(self):
        ret = ""
        stoneChar = ".●○.■□"
        for iy in range(self.length - 1, -1, -1):
            if iy < 9:
                ret += str(iy + 1) + ' '
            else:
                ret += str(iy + 1)
            for ix in range(0, self.length):
                z = XYtoZ(ix + self.padding, iy + self.padding)
                if z == self.lastZ:
                    ret += ' ' + stoneChar[self.cell[z] + 3]
                else:
                    ret += ' ' + stoneChar[self.cell[z]]
            ret += '\n'
        ret += '  '
        for ix in range(self.length):
            ret += ' ' + xChar[ix]
        ret += '\n'
        ret +=  "Turn : " + str(self.turn) + "  Turn-Color : " + colorString[to_turn_color(self.turn)] + "  Ko : "
        if self.koZ == PASS:
            ret += "none"
        else:
            ret += xChar[ZtoX(self.koZ) - self.padding] + str(ZtoY(self.koZ) - self.padding + 1)
        ret += '\n'
        return ret

IMAGE_OUT = 0
IMAGE_MINE = 1
IMAGE_OPP = 2
IMAGE_SENSIBLENESS = 3
IMAGE_LAST1 = 4
IMAGE_LAST2 = 5
IMAGE_KO = 6
IMAGE_AFTERKO = 7
IMAGE_LIBERTY = 8
IMAGE_AFTERLIBERTY = 16
IMAGE_CAPTURE = 24
IMAGE_SELFATARI = 32

def to_image(bd, c):
    ci = to_color_index(c)
    mcap = bd.captured[ci]
    img = np.zeros((IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = np.float32)
    for x in range(IMAGE_LENGTH):
        for y in range(IMAGE_LENGTH):
            img[x][y][IMAGE_OUT] = 1 # wall
    for ix in range(bd.length):
        for iy in range(bd.length):
            x = ix + bd.padding
            y = iy + bd.padding
            z = XYtoZ(x, y)
            tc = bd.cell[z]
            
            img[x][y][IMAGE_OUT] = 0 # on board
            if tc == c:
                img[x][y][IMAGE_MINE] = 1 # my color
            elif tc == flip_color(c):
                img[x][y][IMAGE_OPP] = 1 # opp color
            elif bd.check(c, z) == VALID:
                img[x][y][IMAGE_SENSIBLENESS] = 1 # sensibleness

            if IMAGE_PLAINS <= 4:
                continue
            
            if bd.cell[z] != EMPTY:
                lib = bd.countLiberty(z)
                if lib > 0: # liberty
                    img[x][y][IMAGE_LIBERTY + min(7, lib - 1)] = 1
            
            if bd.lastZ == z: # last move
                img[x][y][IMAGE_LAST1] = 1
            
            if bd.koZ == z: # current ko
                img[x][y][IMAGE_KO] = 1
            
            tbd = copy.deepcopy(bd)
            validity = tbd.move(c, z)
            
            if validity == VALID or validity == EYE:
                if tbd.koZ != PASS: # after ko
                    img[x][y][IMAGE_AFTERKO] = 1
                after_mcap = tbd.captured[ci]
                mcap_dist = after_mcap - mcap
                if mcap_dist > 0: # capture
                    img[x][y][IMAGE_CAPTURE + min(7, mcap_dist - 1)] = 1

                lib, str = tbd.countLibertyAndString(z)
                if lib > 0: # after liberty
                    img[x][y][IMAGE_AFTERLIBERTY + min(7, lib - 1)] = 1
                if lib == 1 and str > 0: # self atari
                    img[x][y][IMAGE_SELFATARI + min(7, str - 1)] = 1
    return img

def amplify_image2(img):
    # 画像の対称形 x2 を計算
    amp_img = np.zeros((2, IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = np.float32)
    for x in range(IMAGE_LENGTH):
        for y in range(IMAGE_LENGTH):
            for p in range(IMAGE_PLAINS):
                val = img[x][y][p]
                amp_img[0][x][y][p] = val
                amp_img[1][IMAGE_LENGTH - 1 - x][IMAGE_LENGTH - 1 - y][p] = val
    return amp_img

def amplify_image3(img):
    # 画像の対称形 x3 を計算
    amp_img = np.zeros((3, IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = np.float32)
    for x in range(IMAGE_LENGTH):
        for y in range(IMAGE_LENGTH):
            for p in range(IMAGE_PLAINS):
                val = img[x][y][p]
                amp_img[0][x][y][p] = val
                amp_img[1][IMAGE_LENGTH - 1 - x][IMAGE_LENGTH - 1 - y][p] = val
                amp_img[2][y][x][p] = val
    return amp_img

def amplify_image4(img):
    # 画像の対称形 x4 を計算
    amp_img = np.zeros((4, IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = np.float32)
    for x in range(IMAGE_LENGTH):
        for y in range(IMAGE_LENGTH):
            for p in range(IMAGE_PLAINS):
                val = img[x][y][p]
                amp_img[0][x][y][p] = val
                amp_img[1][IMAGE_LENGTH - 1 - x][y][p] = val
                amp_img[2][x][IMAGE_LENGTH - 1 - y][p] = val
                amp_img[3][IMAGE_LENGTH - 1 - x][IMAGE_LENGTH - 1 - y][p] = val
    return amp_img

def amplify_image8(img):
    # 画像の対称形 x8 を計算
    amp_img = np.zeros((8, IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = np.float32)
    for x in range(IMAGE_LENGTH):
        for y in range(IMAGE_LENGTH):
            for p in range(IMAGE_PLAINS):
                val = img[x][y][p]
                amp_img[0][x][y][p] = val
                amp_img[1][IMAGE_LENGTH - 1 - x][y][p] = val
                amp_img[2][x][IMAGE_LENGTH - 1 - y][p] = val
                amp_img[3][IMAGE_LENGTH - 1 - x][IMAGE_LENGTH - 1 - y][p] = val
                amp_img[4][y][x][p] = val
                amp_img[5][IMAGE_LENGTH - 1 - y][x][p] = val
                amp_img[6][y][IMAGE_LENGTH - 1 - x][p] = val
                amp_img[7][IMAGE_LENGTH - 1 - y][IMAGE_LENGTH - 1 - x][p] = val
    return amp_img

"""
def string_to_board_log(str):
    v = str.split(' ')
    sl = {}
    sl['mv'] = v[0]
    sl['color'] = v[1]
    sl['ko'] = v[2]
    st = np.empty(SIZE, dtype = int)
    for i in range(SIZE):
        st[i] = int(v[3 + i])
    sl['stone'] = st
    return sl
def load_board_logs(file_path):
    f = open(file_path)
    logs = []
    for line in f:
        line = line.rstrip()
        ml = string_to_board_log(line)
        logs.append(ml)
    return logs"""

def read_bin_uint64(s):
    ret = 0
    for i in range(8):
        ret += ord(str(s[8 - 1 - i]))
        ret = ret << 8
    return ret

def string_to_middle_image_log(str):
    v = str.split(' ')
    il = {}
    il['mv'] = int(v[0])
    img = np.empty((IMAGE_LENGTH, IMAGE_LENGTH), dtype = int)
    for x in range(IMAGE_LENGTH):
        for y in range(IMAGE_LENGTH):
            val = np.int64(v[1 + x * IMAGE_LENGTH + y])
            img[x][y] = val
    il['img'] = img
    return il

def load_middle_image_logs(file_path):
    f = open(file_path)
    logs = []
    for line in f:
        line = line.rstrip()
        il = string_to_middle_image_log(line)
        logs.append(il)
    f.close()
    return logs

"""def load_image_moves(file_path, size):
    # sizeの数が入った画像データを読む
    f = open(file_path)
    images = np.empty((size, IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = np.float32)
    moves = np.zeros((size, IMAGE_SIZE), dtype = np.float32)
    for i, line in enumerate(f):
        v = line.split(' ')
        moves[i][int(v[0])] = 1
        for x in range(IMAGE_LENGTH):
            for y in range(IMAGE_LENGTH):
                val = np.int64(v[1 + x * IMAGE_LENGTH + y])
                for p in range(IMAGE_PLAINS):
                    images[i][x][y][p] = (val >> p) & 1
    f.close()
    return images, moves"""
def load_image_moves(file_path, size):
    # sizeの数が入った画像データを読む
    f = open(file_path)
    images = np.empty((size, IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = np.float32)
    moves = np.zeros((size, IMAGE_SIZE), dtype = np.float32)
    
    #tmp = np.empty((IMAGE_LENGTH, IMAGE_LENGTH), dtype = np.int64)
    #one = np.ones((IMAGE_LENGTH, IMAGE_LENGTH), dtype = np.int64)
    #t = 0
    for i, line in enumerate(f):
        #print(i)
        v = line.split(' ')
        moves[i][int(v[0])] = 1
        #t = time.time()
        for x in xrange(IMAGE_LENGTH):
            for y in xrange(IMAGE_LENGTH):
                val = np.int64(v[1 + x * IMAGE_LENGTH + y])
                for p in xrange(IMAGE_PLAINS):
                    images[i][x][y][p] = (val >> p) & 1
        """for x in xrange(IMAGE_LENGTH):
            for y in xrange(IMAGE_LENGTH):
                tmp[x][y] = np.int64(v[1 + x * IMAGE_LENGTH + y])
        print(time.time() - t)
        t = time.time()
        for p in xrange(IMAGE_PLAINS):
            tmp = np.left_shift(tmp, p)
            t1 = np.bitwise_and(tmp, 1)
            #images[i] = t1
            for x in xrange(IMAGE_LENGTH):
                for y in xrange(IMAGE_LENGTH):
                    images[i][x][y] = np.float32(t1[x][y])"""
        #print(time.time() - t)
    f.close()
    return images, moves

def load_image_moves_np(path, n):
    # numpy.saveで保存した形式の画像データをロード
    # image_name = path + "_np_img" + str(n) + ".npy"
    # label_name = path + "_np_lbl" + str(n) + ".npy"
    image_name = path + "_np_img" + str(n) + ".npz"
    label_name = path + "_np_lbl" + str(n) + ".npz"
    imagez = np.load(image_name)
    labelz = np.load(label_name)
    ret = (imagez['arr_0'], labelz['arr_0'])
    imagez.close()
    labelz.close()
    return ret


"""def load_image_logs(file_path, n):
    f = open(file_path, 'rb')
    logs = []
    
    for i in range(n):
        buf = f.read(8)
        mv = read_bin_uint64(buf) # 選ばれた着手(I系列)
        img = np.empty((IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_PLAINS), dtype = int)
        for x in range(IMAGE_LENGTH):
            for y in range(IMAGE_LENGTH):
                buf = f.read(8)
                val = read_bin_uint64(buf) # 1画素の画像データ
                print(val)
                # 各ビットの値を読む
                for p in range(IMAGE_PLAINS):
                    img[x][y][p] = (val >> p) & 1
    logs.append({'mv' : mv, 'img' : img})
    f.close()
    return logs
"""


