import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet :
    def __init__(self, input_size, hidden_size, output_size) :
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = 0.01 * np.random.randn(I, H)   
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(0)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치화 기울기를 리스트에 모은다
        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads
