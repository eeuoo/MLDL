# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder

# seq2seq 의 두 번째 개선
# peeky = 엿보기
# 중요한 정보가 담긴 Encoder의 출력 h를 Decoder의 다른 계층에게도 전해주는 것.
# 모든 시각의 Affine 계층과 LSTM 계층에 h를 전해주며 기존에는 하나의 LSTM만이 소유하던 중요 정보 h를 여러 계층에서 공유함.
# 집단지성과 같음. 중요 정보를 한 사람이 독점하는 게 아니라 많은 사람과 공유하면 올바른 결정 내릴 가능성 높아질 것.
class PeekyDecoder :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)

        for layer in (self.embed, self.lstm, self.affine) :
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None


    def forward(self, xs, h) :
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)   # h를 시계열만큼 복제해서 hs에 저장.
        out = np.concatenate((hs, out), axis=2)  # hs와 Embedding 계층의 출력을 연결.

        out = self.lstm.forward(out)  # 연결한 것을 LSTM 계층에 입력.
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)   # 연결한 것을 Affine 계층에 입력.
        self.cache = H

        return score


    def backward(self, dscore) :
        H =  self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:]. dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)

        return dh


    def generate(self, h, start_id, sample_size) :
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)

        for _ in range(sample_size) :
            x =  np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id =  np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled


class PeekySeq2seq(Seq2seq) :  # Seq2seq 계승
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(v, D, H)
        self.encoder = PeekyDecoder(V, D, H)  # seq2seq 과 다르게 PeekyDecoder를 사용.
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.encoder.grads


# Encoder의 정보를 널리 퍼지게 하는 Peeky
# Peeky 를 이용하게 되면 신경망은 가중치 매개변수가 커져서 계산량도 늘어난다.
# 커진 매개변수만큼의 핸디캡을 잘 감안해야 할 것.
# 또한 seq2seq의 정확도는 하이퍼파라미터 영향을 크게 받는다.