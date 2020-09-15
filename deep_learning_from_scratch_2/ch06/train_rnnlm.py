import sys
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_preplexiy
from dataset import ptb
from rnnlm import Rnnlm

# 하이퍼파라미터 설정
batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
tx = corpus[1:]

### 모델 생성
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)  # RnnlmTrainer 클래스를 사용해 모델을 학습 

### 기울기 클리핑을 적용하여 학습
# fit() 메서드는 모델의 기울기를 구해 모델의 매개변수를 갱신 
# 이때 인수로 max_grad를 지정해 기울기 클리핑을 적용 (기울기 폭발 대책)
# eval_interval=20 은 20번째 반복마다 퍼플렉서티를 평가하라는 뜻, 데이터가 크므로 모든 에폭에서 평가하지 않고 20번에서 평가
trainer.fit(xs, tx, max_epoch, batch_size, time_size, max_grad, eval_interval=20)  
trainer.plot(ylim=(0, 500))

### 테스트 데이터로 평가
# 학습이 끝난 후 테스트 데이터를 사용해 퍼플렉서티를 평가 
# 모델 상태(LSTM의 은닉 상태와 기억 셀)를 재설정하여 평가를 수행햇
model.reset_state()
ppl_test = eval_preplexity(model, corpus_test)
print('테스트 퍼블렉서티: ', ppl_test)

### 매개변수 저장
model.save_params()