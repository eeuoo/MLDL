import numpy as np

class MatMul :   # Matrix Multiply
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW           # 생략기호를 사용하면 '깊은 복사', 메모리 위치 이동이 아닌 실제 값 덮어 쓰기
        
        return dx


class Sigmoid :
    def __init__(self) :
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x) :
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout) :
        dx = dout * (1.0 - self.out) * self.out

        return dx
