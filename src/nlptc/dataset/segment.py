import codecs
import os

from . import DataSet
from ..nlp_tools import tagger as nlp_tagger


class SegmentData(DataSet):
    __data_sets = ['sighan2005-msr', 'sighan2005-pku']

    def __init__(self, data_name, tagger=None, **kwargs):
        super().__init__(**kwargs)
        if data_name not in SegmentData.__data_sets:
            raise RuntimeError('Data set dose not exist!')
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        self.file_path = os.path.join(root_path, 'data', 'segment', data_name)
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.tagger = None
        if tagger == 'bmes':
            self.tagger = nlp_tagger.BMESTagger()
        self.__load_all()

    def __load_all(self):
        train_path = os.path.join(self.file_path, 'train.txt')
        if os.path.exists(train_path):
            self.train_data = self.__load(train_path)

        test_path = os.path.join(self.file_path, 'test.txt')
        if os.path.exists(test_path):
            self.test_data = self.__load(test_path)

        val_path = os.path.join(self.file_path, 'val.txt')
        if os.path.exists(val_path):
            self.val_data = self.__load(val_path)

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
            tag_data.append(self.tagger.tag(seq))

        return list(zip(*tag_data))
