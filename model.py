from typing import List, Set, Union

import os

import torch
import torch.nn as nn
from torch.nn import functional as func

import numpy as np

import utils


class Vectorizer():
    '''


    Parameters
    ----------
    embedding_dim : int, optional
        Size of text embedding vectors.
        Default
    corpus : str or list of str, optional
        Space-delimited string or list of strings representing the entire corpus.
    corpus_filepath : str, optional
        Path to text file containing the entire corpus.
    corpus_dir : str, optional
        Path to directory containing text files that represent the corpus.

    Attributes
    ----------


    Methods
    -------


    '''

    def __init__(self,
                 embedding_dim: int = 100,
                 corpus: Union[str, List[str]] = None,
                 corpus_filepath: str = None, corpus_dir: str = None):
        self._vocab: List[str] = self._init_vocab(corpus, corpus_filepath, corpus_dir)
        self.module = nn.Embedding(embedding_dim, len(self._vocab))

    def _init_vocab(self,
                    corpus: Union[str, List[str]] = None,
                    corpus_filepath: str = None,
                    corpus_dir: str = None) \
                    -> List[str]:
        vocab: Set[str] = set()

        if sum([corpus_param is not None for corpus_param in [corpus, corpus_filepath, corpus_dir]]) > 1:
            raise ValueError("Cannot have multiple corpora")
        if corpus is not None:
            if isinstance(corpus, str):
                if not len(corpus):
                    raise ValueError("`corpus` cannot be empty")
                vocab.update(utils.clean_word(word)
                             for word
                             in corpus.split())
            elif isinstance(corpus, list) and all(isinstance(word, str) for word in corpus):
                if not len(corpus):
                    raise ValueError("`corpus` cannot be empty")
                vocab.update(utils.clean_word(word)
                             for word
                             in corpus)
            else:
                raise TypeError("`corpus` must be str or list of str")
        elif corpus_filepath is not None:
            if isinstance(corpus_filepath, str):
                with open(corpus_filepath) as corpus_file:
                    vocab.update(utils.clean_word(word)
                                 for word
                                 in corpus_file.read()
                                               .strip()
                                               .split())
            else:
                raise ValueError("`corpus_filepath` must be str")
        elif corpus_dir is not None:
            if isinstance(corpus_dir, str):
                for filename in os.listdir(corpus_dir):
                    with open(os.path.join(corpus_dir, filename)) as corpus_file:
                        vocab.update(utils.clean_word(word)
                                     for word
                                     in corpus_file.read()
                                                   .strip()
                                                   .split())
            else:
                raise ValueError("`corpus_dir` must be str")
        else:
            raise ValueError("A corpus must be provided")
        vocab.discard('')

        return ["<start>", "<end>"] + list(vocab)

    def vectorize(self, words: List[str]) -> torch.Tensor:
        '''


        '''
        if not (isinstance(words, list) and all(isinstance(word, str) for word in words)):
            raise ValueError("`words` must be list of str")
        try:
            indices = torch.tensor([self._vocab.index(word) for word in words])
        except ValueError as e:
            raise ValueError("'{}' is not in vocabulary".format(str(e).split('\'')[1]))
        return self.module(indices)

    def lookup(self, indices: torch.Tensor) -> List[str]:
        '''


        '''
        raise NotImplementedError()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        raise NotImplementedError()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        raise NotImplementedError()

def main():
    corpus = "This is the entire corpus. Seriously, it's contained in here. All of it."
    print("corpus: \n\t{}".format(corpus))
    vectorizer = Vectorizer(corpus=corpus)
    print("Vectorizer initialized successfully.")
    print("vocabulary:")
    for word in vectorizer._vocab:
        print("\t{}".format(word))
    print("[\"all\", \"this\"] vectorized: \n\t{}".format(vectorizer.vectorize(["all", "this"])))

if __name__ == "__main__":
    main()
