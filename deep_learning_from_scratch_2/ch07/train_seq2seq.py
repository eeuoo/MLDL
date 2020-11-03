import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq


# 데이터셋 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vecab()

# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# 모델 / 옵티마이저 / 트레이너 생성
model = Seq2seq(vocab_size, wordvec_size, hideen_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []

for epoch in range(max_epoch) :
    trainer.fit(x_train, t_train, max_epoch = 1,
                batch_size = batch_size, max_grad = max_grad)

    correct_num = 0

    for i in range(len(x_test)) :
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        # eval_seq2seq : 문제(question)를 모델에 주고, 문자열을 생성하게 하여 그것이 답과 같은지를 판정한다, 모델이 맞으면 1, 아니면 0 리턴.
        # 인자를 6개정도 받음. model, question(문제 문장 ID의 배열), correct(정답 문장 ID의 배열),
        # id_to_char(문자 ID와 문자의 변환을 수행하는 딕셔너리), verbose(결과를 출력할지 여부), is_reverse(입력문을 반전했는지 여부)
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose)
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)

    print('검증 정확도 %.3f%%' % (acc * 100))