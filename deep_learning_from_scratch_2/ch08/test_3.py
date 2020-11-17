# coding: utf-8
import sys
sys.path.append('..')
from common.layers import Softmax
import numpy as np

N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
h = np.random.randn(N, H)
hr = h.reshape(N, 1, H).repeat(T, axis=1)

t = hs * hr
print(t.shape)
# (10, 5, 4)

s = np.sum(t, axis=2)
print(s.shape)
# (10, 5)

softmax = Softmax()
a = softmax.forward(s)
print(a.shape)
# (10, 5)


# 각 단어의 중요도를 나타내는 가중치 a
# 이 가중치 a의 합(가중합)을 이용해 맥박 벡터를 얻을 수 있다
# Decoder의 LSTM 계층의 은닉 상태 벡터를 h라고 했을 때,
# 지금 목표는 h가 hs의 각 단어 벡터와 얼마나 비슷한가를 수치로 나타내는 것
# 여기서는 가장 단순한 방법인 벡터의 내적을 이용한다 - 두 벡터가 얼마나 같은 방향을 향하고 있는가, 두 벡터의 유사도

