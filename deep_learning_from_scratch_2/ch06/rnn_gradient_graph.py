import numpy as np
import matplotlib.pyplot as plt

N = 2  # 미니배치 크키
H = 3  # 은닉 상태 벡터의 차원 수
T = 20 # 시계열 데이터의 길이

dh = np.ones((N, H))
np.random.seed(3)  # 재현할 수 있도록 난수의 시드 고정
Wh = np.random.randn(H, H)

norm_list = []
for t in range(T) :
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dg**2)) / N
    norm_list.append(norm)


    