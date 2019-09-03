class DataSet(object):
    __data_sets = []

    @classmethod
    def data_sets(cls):
        return cls.__data_sets


class Vocab(object):
    def __init__(self, data, min_freq=0, unk='<unk>'):
        self.count_dir = self.__count(data)

        rm_words = []
        for w in self.count_dir.keys():
            if self.count_dir[w] < min_freq:
                rm_words.append(w)
        for w in rm_words:
            self.count_dir.pop(w)

        self.idx_to_word = list(self.count_dir.keys())
        self.unk = unk
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
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx[self.unk] for w in seq]
