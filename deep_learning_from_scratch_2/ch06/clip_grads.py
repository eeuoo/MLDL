import numpy as np

dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)

    if rate < 1:
        for grad in grads:
            grad *= rate


print('before:', dW1.flatten())
clip_grads(grads, max_norm)
print('after:', dW1.flatten())


# 결과
# before: [2.1761174  7.384534   4.28052605 1.84073141 0.50402541 4.35053292 1.56027608 6.06908582 7.41833507]
# after: [0.48272062 1.63808572 0.94953434 0.4083231  0.11180622 0.96506372 0.34611066 1.34628438 1.64558369]