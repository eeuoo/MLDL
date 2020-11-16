import numpy as np

### 가중합의 계산 ###

N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
a = np.random.randn(N, T)
ar = a.reshape(N, T, 1).repeat(H, axis=2)
# ar = a.reshape(N, T, 1)    # 브로드캐스트를 사용하는 경우

t = hs * ar
print(t.shape)
# (10, 5, 4)


c = np.sum(t, axis=1)
print(c.shape)
# (10, 4)


# a (N, T) -> ar (N, T, H) X hs (N, T, H) -> t (N, T, H) -> c (N, H)
# repeat 노드를 사용해 a 복제, 원소별 곱을 계산한 다음 sum 노드로 합을 구함.
# sum의 역전하는 repeat, repeat의 역전파는 sum
