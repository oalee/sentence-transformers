from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
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
    def __init__(self, file_paths: List[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s), max_chunks=1_000_000, num_workers=32):
        self.file_paths = file_paths
        self.noise_fn = noise_fn
        self.num_workers = num_workers
        self.file_lengths = self.compute_file_lengths()
        # self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.cache = OrderedDict()
        self.locks = {}

    def compute_length(self, file_path):
        df = vaex.open(file_path)
        return len(df)

    def compute_file_lengths(self):
        lengths = []

        print("Computing file lengths...")
        print("Total number of files:", len(self.file_paths))

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = {file_path: executor.submit(self.compute_length, file_path)
                       for file_path in self.file_paths}
            for file_path in tqdm.tqdm(self.file_paths):
                future = futures[file_path]
                lengths.append(future.result())
        return lengths

    def find_file_and_row(self, idx):
        for file_idx, file_length in enumerate(self.file_lengths):
            if idx < file_length:  # idx is within current file
                return file_idx, idx
            idx -= file_length
        # If idx is still non-zero after going through all files, then it's out of range
        raise IndexError("Index out of range")

    def get_sentence(self, idx):
        file_idx, row_idx = self.find_file_and_row(idx)
        file_path = self.file_paths[file_idx]

        cache_key = file_idx  # Now cache_key is only dependent on file_idx

        if cache_key not in self.cache:
            self.cache[cache_key] = vaex.open(file_path)['text'].tolist()
        else:
            self.cache.move_to_end(cache_key)  # update the access order

        if len(self.cache) >= self.max_chunks:
            # Remove the least recently used (LRU) file
            self.cache.popitem(last=False)

        for i in range(1, 100):  # prefetch next 5 files
            next_file_idx = file_idx + i
            if next_file_idx < len(self.file_paths) and next_file_idx not in self.cache:
                if next_file_idx not in self.locks:
                    self.locks[next_file_idx] = threading.Lock()
                threading.Thread(target=self.load_file_into_cache,
                                 args=(next_file_idx,)).start()

        return self.cache[cache_key][row_idx]

    def load_file_into_cache(self, file_idx):
        with self.locks[file_idx]:
            if file_idx not in self.cache:
                # Check if cache is full

                file_path = self.file_paths[file_idx]
                df = vaex.open(file_path)
                self.cache[file_idx] = df['text'].tolist()

        # if cache_key not in self.cache:
        #     if len(self.cache) >= self.max_chunks:
        #         # remove the least recently used (LRU) file
        #         self.cache.popitem(last=False)

        #     # Now read the whole file into cache, no chunking
        #     df = vaex.open(file_path)
        #     self.cache[cache_key] = df['text'].tolist()
        # else:
        #     self.cache.move_to_end(cache_key)  # update the access order
        # # Start prefetching next file(s) if not already in cache

        # next_file_idx = file_idx + 1
        # if next_file_idx < len(self.file_paths) and next_file_idx not in self.cache:
        #     threading.Thread(target=self.load_file_into_cache,
        #                      args=(next_file_idx,)).start()

        # # row_idx now directly indexes into the file, no modulo operation
        # return self.cache[cache_key][row_idx]

    def __getitem__(self, item):
        sent = self.get_sentence(item)
        try:
            return InputExample(texts=[self.noise_fn(sent), sent])
        except:
            import ipdb
            ipdb.set_trace()
            sent = self.get_sentence(item)
            sent = self.get_sentence(item-1)
            return InputExample(texts=[self.noise_fn(sent), sent])

    def __len__(self):
        return sum(self.file_lengths)

    @ staticmethod
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
