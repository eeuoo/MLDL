import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
#model.load_params('../ch06/Rnnlm.pkl') # 앞 장에서 학습을 끝낸 가중치 매개변수

# 시작(start) 문자와 건너뜀(skip) 문자 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$'] # 샘플링하지 않을 단어 지정 
skip_ids = [word_to_id[w] for w in skip_words] 

# 문자 생성
word_ids = model.generate(start_id, skip_ids) # 문장을 생성하는 generate() 는 단어 ID들을 배열 형태로 반환함 
txt = ' '.join([id_to_word[i] for  i in word_ids]) 
txt = replace(' <eos>', '.\n')
print(txt)