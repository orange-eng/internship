import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, file, min_count):
        self.inputFileName = file
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        # 词频表
        f = open(self.inputFileName, 'r')
        word_frequency = dict()
        for line in f.readlines():
            line = line.split(',')
            self.sentences_count += 1
            for word in line:
                self.token_count += 1
                word_frequency[word] = word_frequency.get(word, 0) + 1
        f.close()
        # 建立词汇表
        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:  # 删除出现频率比较小的词汇
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # 为负采样建立frequency表
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


class Metapath2vecDataset(Dataset):
    def __init__(self, data, window_size):
        # read in data, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, 'r')

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split(',')

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            pair_catch.append((u, v, self.data.getNegatives(v, 5)))
                    return pair_catch

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)