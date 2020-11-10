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

        # Time Embedding, Time LSTM, TimeAffine 의 3가지 계층으로 구성됨.
        # Encdoer의 출력 h를 Decoder의 Time LSTM 계층의 상태로 설정함. 즉, 상태를 갖도록(stateful) 한 것.
        # 단, 한번 설정된 이 은닉 상태는 재설정되지 않고, Endocer의 h를 유지하면서 순전파가 이뤄짐.
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []

        for layer in (self.embed, self.lstm, self.affine) :
            self.params += layer.params
            self.grads += layer.grads


        def forward(self, xs, h) :
            # forward : 학습할 때 사용되는 메서드.
            self.lstm.set_state(h)

            out = self.embed.forward(xs)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            return score

        def backward(self, dscore) :
            # 위쪽의 Softmax with Loss 계층으로부터 기울기 dscore를 받아 아래 3계층으로 전파.
            dout = self.affine.backward(dscore)
            dout = self.lstm.backward(dout)
            dout = self.embed.backward(dout)
            dh = self.lstm.dh  # LSTM 의 시간 방향으로의 기울기는 인스턴스 변수 dh에 저장되어 있음.

            return dh

        def generate(self, h, start_id, sample_size) :
            # generate : Decoder 클래스에 문장 생성을 담당하는 generate() 메서드.
            # h : Encoder로부터 받은 은닉 상태.
            # start_id : 최초로 주어지는 문자 ID.
            # sample_size : 생성하는 문자 수.
            # 문자를 주고 Affine 계층이 출력하는 점수가 가장 큰 문자 ID를 선택하는 작업을 반복한다.
            sampled = []
            sample_id = start_id
            self.lstm.set_state(h)

            for _ in range(sample_size) :
                x = np.array(sample_id).reshape((1,1))
                out = self.embed.forward(x)
                out = self.lstm.forward(out)
                score = self.affine.forward(out)

                sample_id = np.argmax(score.flatten())  # argmax : 최댓값을 가진 원소의 인덱스(문자 ID)를 선택하는 노드
                sampled.append(int(sample_id))

            return sampled


class Seq2sq(BaseModel) :
    # Encoder 클래스와 Decoder 클래스를 연결하고, Time Softmax with Loss 계층을 이용해 손실을 계산한다.
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



# seq2seq은 한 시계열 데이터를 다른 시계열 데이터로 변환한다.
# 예시)
# 기계 번역 : 한 언어의 문장을 다른 언어의 문자으로 변환
# 자동 요약 : 긴 문장을 짧게 요약된 문장으로 변환
# 질의응답 : 질문을 응답으로 변환
# 메일 자동 응답 : 받은 메일의 문장을 답변 글로 변환


# 총정리
# RNN 을 이용한 언어 모델은 새로운 문장을 생성할 수 있다.
# 문장을 생성할 때는 하나의 단어(혹은 문자)를 주고 모델의 출력(확률분포)에서 샘플링하는 과정을 반복한다.
# RNN을 2개 조합함으로써 시계열 데이터를 다른 시계열 데이터로 변환할 수 있다.
# seq2seq는 Encoder가 출발어 입력문을 인코딩하고, 인코딩된 정볼르 Decoder가 받아 디코딩하여 도착어 출력문을 얻는다.
# 입력문을 반전시키는 기법 Revers, 인코딩된 정보를 Decoder의 여러 계층에 전달하는 기법은 seq2seq 정확도 향상에 효과적이다.
# 기계번역, 챗봇, 이미지 캡셔닝 등 seq2seq는 다양한 애플리케이션에 이용할 수 있다.