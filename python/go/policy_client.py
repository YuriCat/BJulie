# -*- coding: utf-8 -*-
# policy_client.py
# Katsuki Ohto
# http://kivantium.hateblo.jp/entry/2015/11/18/233834

import sys

import numpy as np
import tensorflow as tf

import go.go as go

#import model.cnn_policy as mdl
import go.model.rn_policy as mdl
#import model.rn_policy2 as mdl

#definition

N_PHASES = 1

class PolicyClient:
    
    def __init__(self, mdl_file = None):
        self.x_placeholder = tf.placeholder("float", shape = (None, go.IMAGE_LENGTH, go.IMAGE_LENGTH, go.IMAGE_PLAINS))
        self.forward_op = [mdl.make(self.x_placeholder) for ph in range(N_PHASES)]
        self.prob_op = [mdl.normalize(self.forward_op[ph]) for ph in range(N_PHASES)]
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        if mdl_file is not None:
            self.restore(mdl_file)

    def restore(self, mdl_file):
        print(mdl_file)
        self.saver.restore(self.sess, mdl_file)

    def calc(self, img, phase = 0):
        return self.sess.run(self.prob_op[phase], feed_dict = {self.x_placeholder : [img]})[0]

    def calc_amp2(self, img, phase = 0):
        # 対称形 x2 を考慮して確率計算
        amp_prob = self.sess.run(self.prob_op[phase], feed_dict = {self.x_placeholder : img})
        prob = np.zeros(go.IMAGE_SIZE, dtype = np.float32)
        for x in range(go.IMAGE_LENGTH):
            for y in range(go.IMAGE_LENGTH):
                pb = 0
                pb += amp_prob[0][go.XYtoZ(x, y)]
                pb += amp_prob[1][go.XYtoZ(go.IMAGE_LENGTH - 1 - x, go.IMAGE_LENGTH - 1 - y)]
                prob[go.XYtoZ(x, y)] = pb / 2
        return prob

    def calc_amp3(self, img, phase = 0):
        # 対称形 x3 を考慮して確率計算
        amp_prob = self.sess.run(self.prob_op[phase], feed_dict = {self.x_placeholder : img})
        prob = np.zeros(go.IMAGE_SIZE, dtype = np.float32)
        for x in range(go.IMAGE_LENGTH):
            for y in range(go.IMAGE_LENGTH):
                pb = 0
                pb += amp_prob[0][go.XYtoZ(x, y)]
                pb += amp_prob[1][go.XYtoZ(go.IMAGE_LENGTH - 1 - x, go.IMAGE_LENGTH - 1 - y)]
                pb += amp_prob[2][go.XYtoZ(y, x)]
                prob[go.XYtoZ(x, y)] = pb / 3
        return prob

    def calc_amp4(self, img, phase = 0):
        # 対称形 x4 を考慮して確率計算
        amp_prob = self.sess.run(self.prob_op[phase], feed_dict = {self.x_placeholder : img})
        prob = np.zeros(go.IMAGE_SIZE, dtype = np.float32)
        for x in range(go.IMAGE_LENGTH):
            for y in range(go.IMAGE_LENGTH):
                pb = 0
                pb += amp_prob[0][go.XYtoZ(x, y)]
                pb += amp_prob[1][go.XYtoZ(go.IMAGE_LENGTH - 1 - x, y)]
                pb += amp_prob[2][go.XYtoZ(x, go.IMAGE_LENGTH - 1 - y)]
                pb += amp_prob[3][go.XYtoZ(go.IMAGE_LENGTH - 1 - x, go.IMAGE_LENGTH - 1 - y)]
                prob[go.XYtoZ(x, y)] = pb / 4
        return prob

    def calc_amp8(self, img, phase = 0):
        # 対称形 x8 を考慮して確率計算
        amp_prob = self.sess.run(self.prob_op[phase], feed_dict = {self.x_placeholder : img})
        prob = np.zeros(go.IMAGE_SIZE, dtype = np.float32)
        for x in range(go.IMAGE_LENGTH):
            for y in range(go.IMAGE_LENGTH):
                pb = 0
                pb += amp_prob[0][go.XYtoZ(x, y)]
                pb += amp_prob[1][go.XYtoZ(go.IMAGE_LENGTH - 1 - x, y)]
                pb += amp_prob[2][go.XYtoZ(x, go.IMAGE_LENGTH - 1 - y)]
                pb += amp_prob[3][go.XYtoZ(go.IMAGE_LENGTH - 1 - x, go.IMAGE_LENGTH - 1 - y)]
                pb += amp_prob[4][go.XYtoZ(y, x)]
                pb += amp_prob[5][go.XYtoZ(go.IMAGE_LENGTH - 1 - y, x)]
                pb += amp_prob[6][go.XYtoZ(y, go.IMAGE_LENGTH - 1 - x)]
                pb += amp_prob[7][go.XYtoZ(go.IMAGE_LENGTH - 1 - y, go.IMAGE_LENGTH - 1 - x)]
                prob[go.XYtoZ(x, y)] = pb / 8
        return prob

    def wait(self, host, port, mdl_file):
        # 通信待ち状態に入る
        self.restore(mdl_file)
        backlog = 10
    
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        end = False
        with closing(sock):
            sock.bind((host, port))
            sock.listen(backlog)
            conn, address = sock.accept()
            while not end:
                rmsg_str = recv_msg(sock, buf_size)
                if len(rmsg_str) == 0:
                    print("disconnected...")
                    end = True
                # rmsg_str を画像データとして読む
                img = np.zeros((go.IMAGE_LENGTH, go.IMAGE_LENGTH, go.IMAGE_PLAINS), dtype = np.float32)
                for i, s in enumerate(rmsg_str):
                    val = np.int64(s)
                    x = go.ZtoX(i)
                    y = go.ZtoY(i)
                    for p in go.IMAGE_PLAINS:
                        img[x][y][p] = (val >> p) & 1
                prob = self.calc(img) # 着手確率計算
                ret = ' '.join([p for p in prob])
                send_msg(sock, ret)

def test(img_file, mdl_file):
    
    # 精度テスト
    policy = PolicyClient(mdl_file)
    
    # 画像ログの読み込み
    middle_image_logs = go.load_middle_image_logs(img_file)
    
    phase_index_vector = [[] for _ in range(N_PHASES)]
    for i, il in enumerate(middle_image_logs):
        phase_index_vector[0].append(i)
    phase_data_num = [len(v) for v in phase_index_vector]
    for ph in range(N_PHASES):
        np.random.shuffle(phase_index_vector[ph])

    image_vector = [np.empty((phase_data_num[ph], go.IMAGE_LENGTH, go.IMAGE_LENGTH, go.IMAGE_PLAINS)
                         , dtype = np.float32) for ph in range(N_PHASES)]
    move_vector = [np.zeros((phase_data_num[ph], go.IMAGE_SIZE), dtype = np.float32)
                   for ph in range(N_PHASES)]

    for ph in range(N_PHASES):
        for i in range(phase_data_num[ph]):
            il = middle_image_logs[phase_index_vector[ph][i]]
            for x in range(go.IMAGE_LENGTH):
                for y in range(go.IMAGE_LENGTH):
                    val = il['img'][x][y]
                    for p in range(go.IMAGE_PLAINS):
                        b = (val >> p) & 1
                        image_vector[ph][i][x][y][p] = b
            move_vector[ph][i][il['mv']] = 1
    
    for ph in range(N_PHASES):
        for i, img in enumerate(image_vector[ph]) :
            prob = policy.calc(img)
            print(i)
            for x in range(go.IMAGE_LENGTH):
                print([int(prob[go.XYtoZ(x, y)] * 100) for y in range(go.IMAGE_LENGTH)])

def test_reverse(mdl_file):
    
    import copy
    
    # 黒白入替テスト
    policy = PolicyClient(mdl_file)
    
    board0 = go.Board(19)
    test_go.s2board(board0,"""
   A B C D E F G H J K L M N O P Q R S T
19 . X . . . . . . . . . . . . . . . . . 19
18 . O X . O . O . O . . . . . . . . . . 18
17 . O . X X O . O X . . . . X . . . . . 17
16 X X X X O O . O X + . . . . . X X X . 16
15 . O O O O . . O X . . . . . . O X O . 15
14 . . . X X X . . X . . . . . . O O . . 14
13 . O . X O O O O . . . . . . . . . . . 13
12 . X X X X X . . . X . . . . O O X . . 12
11 . X O O X . O O O O . . . . . X O . . 11
10 X O O X X . X X . + . . . . . X O . . 10
 9 . X O O X . X . . . . . . . X O O O . 9
 8 . . . . O O . . X . . X . . X O O X . 8
 7 . . X O . O . O . . . . . . O X O X . 7
 6 . O X X X X O O O O O O O . O X X X . 6
 5 . X . X O O X X O X . X . . O . . . . 5
 4 . O X O O X . X O + . . . . . O O X . 4
 3 . X O O O O O X X O O X . . . O X X . 3
 2 . X X O X . O X . X O O . . . . O X . 2
 1 . . . X . O X X X . X . . . . . . . . 1
   A B C D E F G H J K L M N O P Q R S T
White to play
"""[1:-1])
    
    board0.koZ = 100
    
    board1 = copy.deepcopy(board0)
    board1.reverse()
    
    img0 = go.to_image(board0, go.BLACK)
    img1 = go.to_image(board1, go.WHITE)
    
    print(board0.to_string())
    print(board1.to_string())
    
    p0 = policy.calc(img0)
    p1 = policy.calc(img1)
    
    for i, p in enumerate(p0):
        print(p0[i], p1[i])
    
    return 0

if __name__ == '__main__':

    mdl_file = sys.argv[1]
    
    if sys.argv[2] == "-t":
        img_file = sys.argv[3]
        test(img_file, mdl_file) # 評価テストモード
    elif sys.argv[2] == "-c":
        wait(host, port, mdl_file) # クライアントモード
    elif sys.argv[2] == "-r":
        test_reverse(mdl_file) # 黒白入れ替えテスト


