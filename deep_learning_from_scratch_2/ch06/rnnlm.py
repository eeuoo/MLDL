import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel

class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        rn = np.random.rnadn

        # 가중치 초기화 
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftWithLoss()
        self.lstm_layer = self.layers[1]

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads


    def predict(self, xs) :
        # Softmax 계층 직전까지를 처리하는 predict() 메서드 추가 
        # 문장생서에 사용됨 
        for layer in self.layers :
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts) :
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, xs, ts) :
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self) :
        self.lstm_layer.reset_state()

    def save_params(self, file_name="Rnnlm.pkl") :
        # 매개변수 쓰기
        with open(file_name, 'wb') as f :
            pickle.dump(self.params, f)

    def load_params(self, file_name='Rnnlm.pkl') :
        # 매개변수 읽기
        with open(file_name, 'rb') as f :
            self.params = pickle.load(f)
