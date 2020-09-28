import sys
sys.path.append('..')
from dataset import sequence

# 문자 ID가 저장되어 있는 x_train, t_train
(x_train, t_train) , (x_test, t_test) = sequnce.load_data('addition.txt', seed=1984)

# 문자 ID와 문자의 대응 관계는 char_to_id, id_to_char 로 상호변환
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

print(x_train[0])
print(t_train[0])

print(''.join([id_to_char[c] for c in x_train[0]]))
print(''.join([id_to_char[c] for c in t_train[0]]))


# 정석대로라면 데이터 셋을 훈련용, 검증용, 테스트용으로 나눠 사용해야 한다. 
# 훈련용 : 학습
# 검증용 : 하이퍼파라미터 튜닝
# 테스트 : 모델 성능 평가 
