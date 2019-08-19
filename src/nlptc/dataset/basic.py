
class DataSet(object):
    __data_sets = []

    @classmethod
    def data_sets(cls):
        return cls.__data_sets


class Vocab(object):
    def __init__(self, data, min_freq=0):
        self.count_dir = self.__count(data)
        for w in self.count_dir.keys():
            if self.count_dir[w] < min_freq:
                self.count_dir.pop(w)

        self.idx_to_word = list(self.count_dir.keys())
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

