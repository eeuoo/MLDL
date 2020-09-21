import sys
sys.path.append('..')
import numpy as np 
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm) :
    def generate(self, start_id, skip_ids=None, sample_size=100) :
        # 문장 생성을 수행하는 메서드
        # start_id : 최초로 주는 ID  
        # sample_size : 샘플링하는 단어의 수 
        # skip_ids : 단어 ID의 리스트, 이 리스트에 속하는 건 샘플링 되지 않도록 함. PTB 데이터셋에 있는 <unk>나 N 등, 전처리된 단어를 샘플링하지 않게 하는 용도로 사용
        
        word_ids = [start_id]

        x = start_id

        while len(word_ids) < sample_size :
            x = np.array(x).reshape(1, 1) # 미니배치 처리를 하므로 입력 x는 2차원 배열이 되어야 함, 단어 ID를 하나만 입력하더라도 미니배치 크기는 1로 간주해 1X1 넘파이 배열로 성형함
            score = self.predict(x)  # 각 단어의 점수를 출력(점수는 정규화 되기 전 값)
            p = softmax(score.flatten())  # 소프트맥스로 정규화, 이걸 목표로 하는 확률분포 p를 얻을 수 있음

            smapled =  np.random.choice(len(p), size=1, p=p) # 확률분포 p로부터 다음 단어를 샘플링, 네거티브 샘플링에서 사용됨 

            if (skip_ids is None) or (sampled not in skip_ids) :
                x = sampled 
                word_ids.append(int(x))

        return word_ids 

    def get_state(self) :
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self) :
        self.lstm_layer.set_state(*state)