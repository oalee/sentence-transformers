from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from torch.utils.data import Dataset
from typing import List
from ..readers.InputExample import InputExample
import numpy as np
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import vaex
import tqdm


class DenoisingAutoEncoderDataset(Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    :param sentences: A list of sentences
    :param noise_fn: A noise function: Given a string, it returns a string with noise, e.g. deleted words
    """

    def __init__(self, sentences: List[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s)):
        self.sentences = sentences
        self.noise_fn = noise_fn

    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[self.noise_fn(sent), sent])

    def __len__(self):
        return len(self.sentences)

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            # guarantee that at least one word remains
            keep_or_not[np.random.choice(n)] = True
        words_processed = TreebankWordDetokenizer(
        ).detokenize(np.array(words)[keep_or_not])
        return words_processed


class ListedDenoisingAutoEncoderDataset(Dataset):
    def __init__(self, file_paths: List[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s), chunk_size=10_000_000, max_chunks=1_000_000, num_workers=32):
        self.file_paths = file_paths
        self.noise_fn = noise_fn
        self.num_workers = num_workers
        self.file_lengths = self.compute_file_lengths()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.cache = OrderedDict()

    def compute_length(self, file_path):
        df = vaex.open(file_path)
        return len(df)

    def compute_file_lengths(self):
        lengths = []
        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = {executor.submit(self.compute_length, file_path)
                       for file_path in self.file_paths}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                lengths.append(future.result())
        return lengths

    def find_file_and_row(self, idx):
        for file_idx, file_length in enumerate(self.file_lengths):
            if idx < file_length:
                return file_idx, idx
            idx -= file_length
        raise IndexError("Index out of range")

    def get_sentence(self, idx):
        file_idx, row_idx = self.find_file_and_row(idx)
        file_path = self.file_paths[file_idx]

        chunk_idx = row_idx // self.chunk_size
        cache_key = (file_idx, chunk_idx)

        if cache_key not in self.cache:
            if len(self.cache) >= self.max_chunks:
                # remove the least recently used (LRU) chunk
                self.cache.popitem(last=False)

            start = chunk_idx * self.chunk_size
            end = min((chunk_idx + 1) * self.chunk_size,
                      self.file_lengths[file_idx])
            # self.cache[cache_key] = pd.read_parquet(file_path, skiprows=range(
            #     1, start + 1), nrows=(end - start), encoding='utf8')['text'].tolist()
        #             start = chunk_idx * self.chunk_size
        # end = min((chunk_idx + 1) * self.chunk_size, self.file_lengths[file_idx])
            df = vaex.open(file_path)
            self.cache[cache_key] = df['text'][start:end].tolist()
        else:
            self.cache.move_to_end(cache_key)  # update the access order

        return self.cache[cache_key][row_idx % self.chunk_size]

    def __getitem__(self, item):
        sent = self.get_sentence(item)
        return InputExample(texts=[self.noise_fn(sent), sent])

    def __len__(self):
        return sum(self.file_lengths)

    @staticmethod
    def delete(text, del_ratio=0.6):
        words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True
        words_processed = TreebankWordDetokenizer(
        ).detokenize(np.array(words)[keep_or_not])
        return words_processed
