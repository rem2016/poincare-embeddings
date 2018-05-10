import numpy as np
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import TfidfVectorizer


stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer()


class Td:
    def __init__(self, l, r, s1, s2, score):
        self.l_word = l
        self.r_word = r
        self.s1 = word_tokenize(s1.replace('<b>', '').replace('</b>', ''))
        self.s2 = word_tokenize(s2.replace('<b>', '').replace('</b>', ''))
        self.l_index = self._index(self.s1, l)
        self.r_index = self._index(self.s2, r)
        self.score = score

    @staticmethod
    def _index(words, word):
        for i in range(1, len(words)):
            if words[i].lower() == word.lower():
                return i
        raise ValueError(f'{words} [{word}]')

    def __repr__(self):
        return f"Td({self.l_word},{self.r_word},...,{self.score})"

    def show(self):
        return f"{self.l_word}, {self.r_word}\n\n{self.s1}\n\n{self.s2}\n\n{self.score}"

    def get_l_context(self, size=6):
        return self._get_context(self.s1, self.l_index, size)

    def get_r_context(self, size=6):
        return self._get_context(self.s2, self.r_index, size)

    @staticmethod
    def _get_context(s, index, size):
        l_index = max(0, index - size)
        r_index = min(index + size, len(s))
        tokens = [x for x in s[l_index: r_index] if x not in stop_words]
        return tokens


class Tfidf:
    __voc = None
    __idf = None
    path = './model/idf.pkl'

    def __init__(self):
        self.__mean = 1
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
                self.__voc = data[0]
                self.__idf = data[1]
                self.__mean = sum(self.__idf) / len(self.__idf)

    def fit(self):
        if self.__voc is not None:
            return
        corpus = (' '.join([stemmer.stem(x.lower()) for x in s]) for s in gutenberg.sents())
        tfidf = TfidfVectorizer(use_idf=True, stop_words='english')
        tfidf.fit(corpus)
        self.__voc = tfidf.vocabulary_
        self.__idf = tfidf.idf_
        self.__mean = sum(self.__idf) / len(self.__idf)
        with open(self.path, 'wb') as f:
            pickle.dump([self.__voc, self.__idf], f)

    def idf(self, word):
        word = stemmer.stem(word)
        if self.__voc is None:
            return 1
        if word not in self.__voc:
            return self.__mean
        return self.__idf[self.__voc[word]]

    def idf_words(self, words):
        return [self.idf(word) for word in words]

