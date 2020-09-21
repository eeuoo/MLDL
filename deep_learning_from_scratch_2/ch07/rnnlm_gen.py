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
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            smapled =  np.random.choice(len(p), size=1, p=p)

            if (skip_ids is None) or (sampled not in skip_ids) :
                x = sampled 
                word_ids.append(int(x))

        return word_ids 