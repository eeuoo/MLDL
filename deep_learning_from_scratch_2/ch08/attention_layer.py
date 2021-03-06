# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
import numpy as np
from common.layers import  Softmax

class WeightSum :
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def bacward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis = 2)

        return dhs, da


class AttentionWeight :
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hs
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a

    def backward(self, da) :
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, aixs=1)

        return dhs, dh

# attention weight + weight sum = Attention
# Encoder 가 건네주는 정보 hs에서 중요한 원소에 주목하여, 그것을 바탕으로 맥락 벡터를 구해 위쪽 계측으로 전파
# LSTM 계층과 Affine 계층 사이에 추가하면 됨 
class Attention :
    def __init__(self) :
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()  # Encoder 가 출력하는 각 단어 벡터 hs에 주목하여 해당 단어의 가중치 a를 구함
        self.weight_sum_layer = WeightSum() # a와 hs의 가중합을 구하고 결과를 맥락 벡터 c로 출력
        self.attention_weight = None

    def forward(self, hs, h) :
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return  out

    def backward(self, dout) :
        dhs0, da = self.weight_sum_layer.bacward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

class TimeAttention :
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.emtpy_like(hs_dec)
        self.layers = []
        self.attention_weights = []   # 각 계층이 각 단어의 가중치를 보관

        ## Attention을 필요한 수만큼 만듦. 여기서는 T개 생성.
        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T) :
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec



### 총정리
# 번역이나 음성 인식 등, 한 시계열 데이터를 다른 시계열 데이터로 변환하는 작업에서는 시계열 데이터 사이의 대응 관계가 존재하는 경우가 많다.
# 어텐션은 두 시계열 데이터 사이의 상응 관계를 데이터로부터 학습한다.
# 어텐션에서는 (하나의 방법으로서) 벡터의 내적을 사용해 벡터 사이의 유사도를 구하고, 그 유사도를 이용한 가중합 벡터가 어텐션의 출력이 된다.
# 어텐션에 사용하는 연산은 미분 가능하기 때문에 오차역전파법으로 학습할 수 있다.
# 어텐션이 산출하는 가중치(확률)를 시각화하면 입출력의 대응 관계를 볼 수 있다.
# 외부 메모리를 활용한 신경망 확장 연구 예에서는 메모리를 읽고 쓰는 데 어텐션을 사용한다.