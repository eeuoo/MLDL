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
            print("train Continuous Bag fo Words model")
            self.model = self.train_model(train_fname, model_full_name)
        else :
            print("load Continuous Bag of Words model")
            self.model  = self.load_model(train_fname, model_full_name)


    def compute_word_frequency(self, embedding_corpus_fname):
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

                    weighted_vector = (a / (word_prob + a)) * np.asarray(vec)
                    dictionary[word] = weighted_vector
                    f3.writelines(word + "\u214E" + " ".join([str(el) for el in weighted_vector]) + "\n")

        return dictionary


    def train_model(self, train_data_fname, model_fname):
        model = {"vectors" : [], "labels" : [], "sentences" : []}
        train_data = self.load_or_tokenize_corpus(train_data_fname)

        with open(model_fname, "W") as f :
            for sentence, tokens, label in train_data :
                tokens = self.tokenizer.morphs(sentence)
                sentence_vector = self.get_sentence_vector(tokens)
                model["sentences"].append(sentence)
                model["vectors"].append(sentence_vector)
                model["labels"].append(label)
                str_vector = " ".join([str(el) for el in sentence_vector])
                f.writelines(sentence + "\u241E" + " ".join(tokens) + "\u241E" + str_vector + "\u241E" + label + "\n")

        return model


    def get_sentence_vector(self, tokens):
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
