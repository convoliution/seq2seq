from typing import List, Set, Union

import os

import torch


class Vocabulary():
    '''


    Parameters
    ----------
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
                 corpus: Union[str, List[str]] = None,
                 corpus_filepath: str = None,
                 corpus_dir: str = None):
        self._vocab: List[str] = self._init_vocab(corpus, corpus_filepath, corpus_dir)

    def __len__(self):
        return len(self._vocab)

    @staticmethod
    def clean_word(word: str) -> str:
        return word.strip()                    \
                   .lower()                    \
                   .replace('’', '\'')         \
                   .lstrip('<(\'\"“')          \
                   .rstrip('>)\'\"”.,!?:;—-…')

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
                vocab.update(Vocabulary.clean_word(word)
                             for word
                             in corpus.split())
            elif isinstance(corpus, list) and all(isinstance(word, str) for word in corpus):
                if not len(corpus):
                    raise ValueError("`corpus` cannot be empty")
                vocab.update(Vocabulary.clean_word(word)
                             for word
                             in corpus)
            else:
                raise TypeError("`corpus` must be str or list of str")
        elif corpus_filepath is not None:
            if not isinstance(corpus_filepath, str):
                raise TypeError("`corpus_filepath` must be str")
            with open(corpus_filepath) as corpus_file:
                vocab.update(Vocabulary.clean_word(word)
                             for word
                             in corpus_file.read()
                                           .strip()
                                           .split())
        elif corpus_dir is not None:
            if not isinstance(corpus_dir, str):
                raise TypeError("`corpus_dir` must be str")
            for filename in os.listdir(corpus_dir):
                with open(os.path.join(corpus_dir, filename)) as corpus_file:
                    vocab.update(Vocabulary.clean_word(word)
                                 for word
                                 in corpus_file.read()
                                               .strip()
                                               .split())
        else:
            raise ValueError("A corpus must be provided")
        vocab.discard('')

        return ["<start>", "<end>"] + list(vocab)

    def indexify(self, words: List[str]) -> torch.Tensor:
        '''


        '''
        if not (isinstance(words, list) and all(isinstance(word, str) for word in words)):
            raise TypeError("`words` must be list of str")
        try:
            return torch.tensor([self._vocab.index(word) for word in words])
        except ValueError as e:
            raise ValueError("'{}' is not in vocabulary".format(str(e).split('\'')[1]))

    def lookup(self, indices: torch.Tensor) -> List[str]:
        '''


        '''
        if not isinstance(indices, torch.Tensor):
            raise TypeError("`indices` must be a Tensor")
        return [self._vocab[index] for index in indices.tolist()]
