import codecs
import os

from . import DataSet
from ..nlp_tools import tagger as nlp_tagger


class SegmentData(DataSet):
    __data_sets = ["sighan2005-msr", "sighan2005-pku"]

    def __init__(self, data_name, tagger=None, **kwargs):
        super().__init__(**kwargs)
        if data_name not in SegmentData.__data_sets:
            raise RuntimeError("Data set dose not exist!")
        self.file_path = os.path.join("/Users/xh0543/workspace/nlp-practice/data/segment", data_name)
        self.train_data = None
        self.test_data = None
        self.tagger = None
        if tagger == 'bmes':
            self.tagger = nlp_tagger.bmes_tagger
        self.__load_all()

    def __load_all(self):
        train_path = os.path.join(self.file_path, "train.txt")
        self.train_data = self.__load(train_path)

        test_path = os.path.join(self.file_path, "test.txt")
        self.test_data = self.__load(test_path)

    def __load(self, path):
        data = []
        with codecs.open(path, 'r', 'utf8') as f:
            for line in f.readlines():
                tokens = line.strip().split()
                data.append(tokens)
        if self.tagger is None:
            return data

        tag_data = []
        for seq in data:
            tag_data.append(self.tagger(seq))

        return list(zip(*tag_data))


