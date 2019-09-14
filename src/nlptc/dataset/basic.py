import codecs
import os

from mxnet import nd


class DataSet(object):
    __data_sets = []

    @classmethod
    def data_sets(cls):
        return cls.__data_sets


class Vocab(object):
    def __init__(self, data, min_freq=0, unk=None):
        self.count_dir = self.__count(data)

        rm_words = []
        for w in self.count_dir.keys():
            if self.count_dir[w] < min_freq:
                rm_words.append(w)
        for w in rm_words:
            self.count_dir.pop(w)

        self.idx_to_word = list(self.count_dir.keys())
        self.unk = unk
        if self.unk is not None:
            self.idx_to_word.append(unk)

        self.word_to_idx = {}
        for i, w in enumerate(self.idx_to_word):
            self.word_to_idx[w] = i

    def __count(self, data):
        count_dir = {}
        for d in data:
            for w in d:
                if w not in count_dir:
                    count_dir[w] = 0
                count_dir[w] += 1
        return count_dir

    def to_indices(self, seq):
        if self.unk is not None:
            return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx[self.unk] for w in seq]
        else:
            return [self.word_to_idx[w] for w in seq]

    def to_words(self, idx):
        return [self.idx_to_word[i] for i in idx]



class Embedding(object):
    __embedding_set = ['news_tensite.msr.words.50d', 'news_tensite.pku.words.50d', 'news_tensite.200d']

    def __init__(self, name, vocab):
        if name not in self.__embedding_set:
            raise RuntimeError('embedding not exist!')
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        path = os.path.join(root_path, 'data', 'embeddings', name)
        dim = int(name.split(".")[-1][:-1])

        self.idx_to_vec = self.__load(path, vocab, dim)
        self.idx_to_word = vocab.idx_to_word
        self.word_to_idx = vocab.word_to_idx

    def __load(self, path, vocab, dim):
        token_to_vec = {}
        with codecs.open(path, 'r', 'utf8') as f:
            for line in f.readlines():
                tokens = line.strip().split()
                token_to_vec[tokens[0]] = tokens[1:]
        matrix = []
        for w in vocab.idx_to_word:
            if w in token_to_vec:
                matrix.append(token_to_vec[w])
            else:
                matrix.append([0] * dim)
        return nd.array(matrix)
