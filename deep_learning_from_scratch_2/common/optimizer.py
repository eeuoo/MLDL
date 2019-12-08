
# 확률적경사하강법 (Stochastic Gradient Descent)
class SGD :    
    def __init__(self, lr = 0.01) :
        self.lr = lr

    def update(self, params, grads) :
        for i in range( len(params) ) :
            params[i] -= self.lr * grads[i]  # 신경망의 매개변수 갱신