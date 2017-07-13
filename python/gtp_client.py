# -*- coding: utf-8 -*-
# gtp_client.py
# Katsuki Ohto

import os, sys, time, copy
import numpy as np
import go.go as go
import blockgo
import go.policy_client as pc

def print_error(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def send_msg(msg):
    print_error("Client << %s" % msg)
    sys.stdout.write(msg)
    sys.stdout.flush()
    #print(msg)

def recv_msg():
    msg = sys.stdin.readline()
    print_error(msg)
    return msg

#def search(board):
#    for

def to_GTP_move_string(l, z):
    if z == go.RESIGN:
        return "resign"
    elif z == go.PASS:
        return "pass"
    else:
        x = go.ZtoX(z)
        y = go.ZtoY(z)
        padding = go.PADDING + (go.MAX_LENGTH - l) // 2
        return go.xChar[x - padding] + str(y - padding + 1)

class Engine:
    def __init__(self, time_limit):
        self.org_time_limit = time_limit

    def set_length(self, l):
        print_error('Engine.set_length()')
        self.length = l
        self.board = blockgo.Board(l)
        self.init_game([])
    
    def init_all(self, arg_list):
        print_error('Engine.init_all()')
        self.policy = pc.PolicyClient(arg_list[0]) # モデル読み込み
        return 0
    
    def init_game(self, arg_list):
        print_error('Engine.init_game()')
        self.board.clear()
        self.time_limit = self.org_time_limit
        print_error(self.board.to_string())
        return 0
    
    def close_game(self, arg_list):
        print_error('Engine.close_game()')
        return 0
    
    def close_all(self, arg_list):
        print_error('Engine.close_all()')
        return -1
    
    def play(self, my_color):
        print_error('Engine.play()')
        
        print_error("remained time = %s" % self.time_limit)
        tstart = time.time()
        
        sc2 = self.board.count_chinese_score2()
        print_error("current chinese score x 2 = %d" % sc2)
        if self.board.lastZ == go.PASS:
            if self.board.turn >= 200:
                return blockgo.BlockType.NONE, -1, go.PASS
            if my_color == go.BLACK and sc2 > 0:
                self.time_limit -= time.time() - tstart
                return blockgo.BlockType.NONE, -1, go.PASS
            elif my_color == go.WHITE and sc2 < 0:
                self.time_limit -= time.time() - tstart
                return blockgo.BlockType.NONE, -1, go.PASS
        
        img = go.to_image(self.board, my_color)
        
        amp_img = go.amplify_image8(img)
        prob = self.policy.calc_amp8(amp_img)
        
        print_error(go.colorString[my_color])
        for ix in range(self.board.length):
            print_error(str([int(prob[go.XYtoZ(ix + self.board.padding, iy + self.board.padding)] * 100) for iy in range(self.board.length)]))
        
        vlist = []
        for bt in range(blockgo.BlockType.NUM):
            for dir in range(4):
                for z in range(1, go.IMAGE_SIZE):
                    validity = self.board.check_block(my_color, bt, dir, z)
                    if validity == go.VALID: # 合法である
                        # このブロックのスコアを計算
                        psum, pprod = 0, 1
                        for dz in blockgo.BLOCK_COMPONENTS_ZD[bt][dir]:
                            p = prob[z + dz]
                            psum += p
                            pprod *= p
                        pmean, ppmean = psum / blockgo.BLOCK_NUM[bt], pprod ** (1 / blockgo.BLOCK_NUM[bt])
                        score = (0.3 * pmean + 0.7 * ppmean) * blockgo.BLOCK_STONES[bt]
                        # 初期の手に対するヒューリスティクス
                        ix = go.ZtoX(z) - self.board.padding
                        iy = go.ZtoY(z) - self.board.padding
                        ixmin, ixmax, iymin, iymax = blockgo.bt_dir_range(bt, dir, ix, iy)
                        if self.board.blockturn < 6:
                            if ixmax < 3 or self.board.length - 4 < ixmin or iymax < 3 or self.board.length - 4 < iymin:
                                score = 0
                        elif self.board.blockturn < 8:
                            if ixmax < 2 or self.board.length - 3 < ixmin or iymax < 2 or self.board.length - 3 < iymin:
                                score = 0
                        # 地を確定させる手
                        """tbd = copy.deepcopy(self.board)
                        tbd.move_block(my_color, bt, dir, z + dz)
                        tsc2 = tbd.count_chinese_score2()
                        if my_color == go.BLACK:
                            score += max(tsc2 - sc2, 0)
                        else:
                            score += max(sc2 - tsc2, 0)"""
                        vlist.append((bt, dir, z, score))
        if len(vlist) == 0:
            print_error("no valid move.")
            self.time_limit -= time.time() - tstart
            return blockgo.BlockType.NONE, -1, go.PASS
        
        np.random.shuffle(vlist) # 同一局面を避けるために先にランダムに並べ替え
        sorted_vlist = sorted(vlist, key = lambda x : -x[3])
        self.time_limit -= time.time() - tstart
        return sorted_vlist[0][0], sorted_vlist[0][1], sorted_vlist[0][2]
    
    def recv_play(self, c, bt, dir, z):
        print_error('Engine.recv_play()')
        print(go.colorChar[c], blockgo.BlockTypeChar[bt], dir, to_GTP_move_string(self.board.length, z))
        self.board.move_block(c, bt, dir, z, force = True)
        print_error(self.board.to_string())
        return 0

def gtp_main(my_id, my_version, config_directory, model_path):
    
    engine = Engine(1800)
    ingame = False
    end = False

    engine.init_all([model_path])

    while not end:
        rmsg_str = recv_msg()
        if len(rmsg_str) == 0:
            print_error("disconnected...")
            end = True
        rmsg_list = str(rmsg_str).split('\n')
        for command in rmsg_list:
            if len(command) == 0:
                continue
            print_error("Server >> %s" % command)
            com_list = command.split(' ')
            com = com_list[0].lower()
            # メッセージが複雑なのでこちらで分岐する
            if com == "boardsize":
                l = int(com_list[1])
                engine.set_length(l)
                ingame = False
                send_msg("= \n\n")
            elif com == "clear_board":
                # ログアウト判定
                #if ingame and os.path.isfile(config_directory + "stop_signal"):
                #    end = True
                engine.init_game([])
                send_msg("= \n\n")
            elif com == "name":
                send_msg("= " + my_id + "\n\n")
            elif com == "version":
                send_msg("= " + my_version + "\n\n")
            elif com == "genmove":
                bt, dir, move_z = blockgo.BlockType.NONE, -1, go.RESIGN
                if com_list[1].lower() == "b":
                    bt, dir, move_z = engine.play(go.BLACK)
                    engine.recv_play(go.BLACK, bt, dir, move_z)
                elif com_list[1].lower() == "w":
                    bt, dir, move_z = engine.play(go.WHITE)
                    engine.recv_play(go.WHITE, bt, dir, move_z)
                send_msg("= " + to_GTP_move_string(engine.length, move_z) + "\n\n")
            elif com == "play":
                move_z = go.RESIGN
                if com_list[2].lower().find("resign") >= 0:
                    end = True
                else:
                    if com_list[2].lower().find("pass") >= 0:
                        bt = blockgo.BlockType.NONE
                        dir = -1
                        move_z = go.PASS
                    else:
                        bt = blockgo.char_to_bt(com_list[2])
                        dir = int(com_list[3])
                        padding = go.PADDING + (go.MAX_LENGTH - engine.length) // 2
                        x = go.char_to_ix(com_list[4][0].upper()) + padding
                        y = int(com_list[4][1 : ]) - 1 + padding
                        move_z = go.XYtoZ(x, y)
                    
                    if com_list[1].lower() == "b":
                        engine.recv_play(go.BLACK, bt, dir, move_z)
                    elif com_list[1].lower() == "w":
                        engine.recv_play(go.WHITE, bt, dir, move_z)
                send_msg("= \n\n")
            elif com == "final_status_list":
                send_msg("= \n\n")
            elif com == "list_commands":
                lst = ["boardsize", "clear_board", "name", "version", "genmove", "play", "list_commands"]
                send_msg("= " + '\n'.join(lst) + "\n\n")
            else:
                send_msg("= \n\n")
    return

if __name__ == '__main__':

    for bt in range(blockgo.BlockType.NUM):
        print(bt, blockgo.BlockType(bt))
        for dir in range(4):
            print(blockgo.block_string(bt, dir))
    
    my_id = "BJulie"
    my_version = "170702"
    config_directory = "/Users/ohto/dropbox/go/kgs/"
    model_path = "/Users/ohto/documents/data/go/tfmodel/rn170405/tf_policy_cnn.ckpt"
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    for c, s in enumerate(sys.argv):
        if s == '-i' and c < len(sys.argv) - 1:
            model_path = sys.argv[c + 1]
    
    gtp_main(my_id, my_version, config_directory, model_path)