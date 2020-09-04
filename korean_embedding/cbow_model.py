import os
import numpy as np

class CBoWModel(object):
    def __init__(self, train_fname, embedding_fname, model_fname, embedding_corpus_fname,
                 embedding_method="fasttext", is_weighted=True,
                 average=False, dim=100, tokenizer_name="mecab"):

        # configurations
        make_save_path(model_fname)
        self.dim = dim
        self.average = average

        if is_weighted :
            model_full_name = model_fname + "-weighted"
        else :
            model_full_name = model_fname + "-original"

        self.tokenizer = get_tokenizer(tokenizer_name)

        # Arora el al.(2016)의 가중합 방식으로 임베딩을 만들 것인지 아닌지에 따라 분기처리가 될 수 있도록 함
        if is_weighted :
            # weighted embeddings
            self.embeddings = self.load_or_construct_weighted_embedding(embedding_fname,
                                                                        embedding_method,
                                                                        embedding_corpus_fname)
            print("loading weighted embeddings, complete!")

        else :
            # original embeddins
            words, vectors = self.load_word_embeddings(embedding_fname,
                                                       embedding_method)

            self.embeddings = defaultdict(list)

            for word, vector in zip(words, vectors) :
                self.embeddings[word] = vector

            print("loading original embeddings, complete!")

        if not os.path.exsits(model_full_name) :
            print("train Continuous Bag fo Words model") # 한 번도 학습한 적 없으면 모델을 학습한다
            self.model = self.train_model(train_fname, model_full_name)
        else :
            print("load Continuous Bag of Words model")
            self.model  = self.load_model(train_fname, model_full_name)


    def compute_word_frequency(self, embedding_corpus_fname):
        # 임베딩 학습 말뭉치 내 단어별 빈도 확인 (CBoW model의 핵심 1)
        total_count = 0
        words_count = defaultdict(int)

        with open(embedding_corpus_fname, "r") as f :
            for line in f :
                tokens = line.strip().split()
                for token in tokens :
                    words_count[token] += 1
                    total_count += 1

        return words_count, total_count


    def load_or_construct_weighted_embedding(self, embedding_fname, embedding_method,
                                             embedding_corpus_fname, a=0.0001) :
        # 가중 임베딩 만들 (CBoW model의 핵심 2)
        # compute_word_frequency에서 확인한 임베딩 말뭉치 통계량을 바탕으로 가중치 임베딩을 만든다
        # 모든 단어 벡터 각각에 문장의 등장 확률을 최대화하는 주제 벡터 수식을 적용해 해단 단어 등장 확률을 반영한 가중치를 곱한다

        dictionary = {}

        if os.path.exists(embedding_fname + "-weighted") :
            # load weighted word embeddings
            with open(embedding_fname + "-weighted", "r") as f2 :
                for line in f2 :
                    word, weighted_vector = line.strip().split("\u241E")
                    weighted_vector = [float(el) for el in weighted_vector.split()]
                    dictionary[word] = weighted_vector

        else :
            # load pretrained word embeddngs
            words, vecs = self.load_word_embeddings(embedding_fname, embedding_method)

            # compute word frequency
            words_count, total_word_count = self.compute_word_frequency(embedding_corpus_fname)

            # construct weighted word embeddings
            with open(embedding_fname + '-weighted', 'w') as f3 :
                for word, vec in zip(words, vecs) :
                    if word in words_count.keys() :
                        word_prob = words_count[word] / total_word_count
                    else :
                        word_prob = 0.0

                    weighted_vector = (a / (word_prob + a)) * np.asarray(vec)  # a는 상수 취급, 기본값은 0.0001
                    dictionary[word] = weighted_vector
                    f3.writelines(word + "\u214E" + " ".join([str(el) for el in weighted_vector]) + "\n")

        return dictionary


    def train_model(self, train_data_fname, model_fname):
        model = {"vectors" : [], "labels" : [], "sentences" : []}
        train_data = self.load_or_tokenize_corpus(train_data_fname)

        with open(model_fname, "W") as f :
            for sentence, tokens, label in train_data :
                tokens = self.tokenizer.morphs(sentence)  # 형태소 분석
                sentence_vector = self.get_sentence_vector(tokens)   # 문장 벡터로 가공
                model["sentences"].append(sentence)
                model["vectors"].append(sentence_vector)
                model["labels"].append(label)
                str_vector = " ".join([str(el) for el in sentence_vector])
                f.writelines(sentence + "\u241E" + " ".join(tokens) + "\u241E" + str_vector + "\u241E" + label + "\n")

        return model


    def get_sentence_vector(self, tokens):
        # 문장 임베딩 만들기
        # weighted 가 true이면 가중치 임베딩을 사용할 것이고, 아니면 어떤 처리도 하지 않 원본 벡터들이 된다
        # 예측 단계에서 코사인 유사도를 계산하기 편하도록 크기가 1인 단위 벡터 형태로 바꿔 리턴

        vector = np.zeros(self.dim)

        for token in tokens :
            if token in self.embeddings.keys() :
                vector += self.embeddings[token]

        if self.average :
            vector /= len(tokens)

        vector_norm = np.linalog.norm(vector)

        if vector_norm != 0:
            unit_vector = vector / vector_norm
        else :
            unit_vector = np.zeros(self.dim)

        return unit_vector


    def preict(self, sentence):
        tokens = self.tokenizer.morphs(sentence)   # 형태소 분석
        sentence_vector = self.get_sentence_vector(tokens)  # 문장 임베딩으로 변환
        # 문장벡터(임베딩 차원 수)와 학습 데이터 문장 임베딩 행력(학습 데이터 문장 수 X 임베딩 차원 수)를 내적(np.dot) = 코사인 유사도
        scores = np.dot(self.model["vectors"], sentence_vector)
        pred = self.model["labels"][np.argmax(scores)]  # argmax로 가장 큰 유사도를 가진 문장의 인덱스 추출하여 레이블에 매칭

        return pred


    def predict_by_batch(self, tokenized_sentences, labels) :
        sentence_vectors, eval_score = []

        for tokens in tokenized_sentences :
            sentence_vectors.append(self.get_sentence_vector(tokens))

        scores = np.dot(self.model["vectors"], np.array(sentence_vectors).T)
        preds = np.argmax(scores, axis=0)

        for pred, label in zip(preds, labels) :
            if self.model["labels"][pred] == label :
                eval_score += 1

        return preds, eval_score
