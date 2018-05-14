from typing import List, Set, Union

import os

import torch
import torch.nn as nn
from torch.nn import functional as func

import numpy as np

import utils


class Vectorizer():
    def __init__(self, embedding_dim: int = 100, corpus: Union[str, List[str]] = None, corpus_filename: str = None, corpus_dir: str = None):
        '''

        '''
        self._vocab: List[str] = self._init_vocab(corpus=corpus, corpus_filename=corpus_filename, corpur_dir=corpur_dir)
        self.embedding: nn.Embedding = self._init_embedding(embedding_dim)

    def _init_vocab(self, corpus: Union[str, List[str]] = None, corpus_filename: str = None, corpus_dir: str = None) -> List[str]:
        '''


        Parameters
        ----------
        embedding_dim : int, optional

        corpus : str or list of str, optional

        corpus_filename : str, optional

        corpus_dir : str, optional


        Returns
        -------
        vocab : set


        '''
        vocab: Set[str] = set()

        if sum([corpus_param is None for corpus_param in [corpus, corpus_filename, corpus_dir]]) > 1:
            raise ValueError("Cannot have multiple corpora")
        if corpus is not None:
            if isinstance(corpus, str):
                if not len(corpus):
                    raise ValueError("`corpus` cannot be empty")
                raise NotImplementedError()
            elif isinstance(corpus, list) and all(isinstance(elem, str) for elem in corpus):
                if not len(corpus):
                    raise ValueError("`corpus` cannot be empty")
                vocab.update(
                    utils.clean_word(word)
                    for word
                    in corpus
                )
            else:
                raise TypeError("`corpus` must be str or list of str")
        elif corpus_filename is not None:
            if isinstance(corpus_filename, str):
                with open(corpus_filename) as corpus_file:
                    vocab.update(
                        utils.clean_word(word)
                        for word
                        in corpus_file.read()
                                      .lower()
                                      .strip()
                                      .split()
                    )
            else:
                raise ValueError("`corpus_filename` must be str")
        elif corpus_dir is not None:
            if isinstance(corpus_dir, str):
                for filename in os.listdir(corpus_dir):
                    with open(os.path.join(corpus_dir, filename)) as corpus_file:
                        vocab.update(
                            utils.clean_word(word)
                            for word
                            in corpus_file.read()
                                          .lower()
                                          .strip()
                                          .split()
                        )
            else:
                raise ValueError("`corpus_dir` must be str")
        else:
            raise ValueError("A corpus must be provided")
        vocab.discard('')

        return ["<SOS>", "<EOS>"] + list(vocab)

    def _init_embedding(self, embedding_dim: int) -> nn.Embedding:
        '''


        Parameters
        ----------


        Returns
        -------

        '''
        return nn.Embedding(embedding_dim, len(self._vocab))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
