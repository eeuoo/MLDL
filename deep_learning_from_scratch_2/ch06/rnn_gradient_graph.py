import numpy as np
import matplotlib.pyplot as plt

## 기울기 폭발과 기울기 소실에 대한 예시 그래프 

N = 2  # 미니배치 크키
H = 3  # 은닉 상태 벡터의 차원 수
T = 20 # 시계열 데이터의 길이

dh = np.ones((N, H))
np.random.seed(3)  # 재현할 수 있도록 난수의 시드 고정
Wh = np.random.randn(H, H)  # 변경 전 (기울기 폭발) 
Wh = np.random.randn(H, H) * 0.5  # 변경 후 (기울기 소실)

norm_list = []
for t in range(T) :
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)



print(norm_list)

# 그래프 그리기
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('시간 크기(time step)')
plt.ylabel('노름(norm)')
plt.show()


# Wh를 T번 곱했기 때문에 기울기 폭발이나 소실이 나타남
# Wh가 스칼라면 1보다 크면 지수적으로 증가, 1보다 작으면 지수적으로 감소 
# Wh가 행렬이면 '특잇값'이 척도가 됨. 특잇값의 최대값이 1보다 크면 지수적으로 증가, 1보다 작으면 지수적으로 감소. 
# 행렬의 특잇값 : 데이터가 얼마나 퍼져 있는지 
