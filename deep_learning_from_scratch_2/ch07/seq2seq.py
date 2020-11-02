import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel

# seq2seq 는 2개의 RNN을 연결한 신경망 (Encoder + Decoder)


class Encoder:
    # Encoder 는 문자열(정확하게는 문자 ID)을 받아 벡터 h(문자 벡터)를 반환한다.
    # Embedding 계층과 LSTM 계층으로 구성됨.
    # LSTM 계층은 오른쪽으로는 은닉 상태와 셀을 출력하고 위쪽으로는 은닉 상태만 출력한다.
    # 위에 다른 계층이 없으니 위쪽 출력은 폐기되고 Encoder에서는 마지막 문자를 처리한 후 LSTM 계층의 은닉 상태 h를 출력.
    # 은닉 상태 h가 Decoder로 전달된다.

    def __init__(self, vocab_size, wordvec_size, hidden_size) :

        V = vocab_size   # 어휘 수 문자의 종류, 0~9 숫자와 '+', '', '_' 합쳐 총 13가지 문자 사용
        D = wordvec_size   # 문자 벡터의 차원의 수
        H = hidden_size  # LSTM 계층의 은닉 차원의 수
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)   # Embedding 계층
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, sateful=False)  # LSTM 계층, 상태 유지를 하지 않기 때문에 sateful=False

        self.params = self.embed.params + self.lstm.params  # 가중치 매개변수
        self.grads = self.embed.grads + self.lstm.grads  # 기울기
        self.hs = None

    def forward(self, xs) :
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        
        return hs[:, -1, :]

    def backward(self, dh) :
        # dh =  LSTM 계층의 마지막 은닉 상태에 대한 기울기, Decoder가 전해주는 기울기
        dhs = np.zeros_like(self.hs)  # 원소가 모두 0인 텐서
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        
        return dout




class Decoder :
    # Encoder 가 출력한 h를 받아 목적으로 하는 다른 문자열을 출력.
    # RNN 계층으로 구성됨.
    # RNN 계층으로 문장을 생성할 때 추론 시 최초 시작을 알리는 구분 문자(ex, '_') 하나만 준다.
    # ex) 입력 데이터 : ['-', '6', '2', ''] 출력 데이터 : ['6', '2', '', ''] 되도록 학습시킴.
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn*H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f')
        affing_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []

        for layer in (self.embed, self.lstm, self.affime) :
            self.params += layer.params
            self.grads += layer.grads

        def forward(self, xs, h) :
            self.lstm.set_state(h)

            out = self.embed.forward(xs)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            return score

        def backward(self, dscore) :
            dout = self.affine.backward(dscore)
            dout = self.lstm.backward(dout)
            dout = self.embed.backward(dout)
            dh = self.lstm.dh

            return dh

        def generate(self, h, start_id, sample_size) :
            sampled = []
            sample_id = start_id
            self.lstm.set_state(h)

            for _ in range(sample_size) :
                x = np.array(sample_id).reshape((1,1))
                out = self.embed.forward(x)
                out = self.lstm.forward(out)
                score = self.affine.forward(out)

                sample_id = np.argmax(score.flatten())
                sampled.append(int(sample_id))

            return sampled


class Seq2sq(BaseModel) :
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout =  self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled