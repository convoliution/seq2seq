from typing import List, Set, Union

import os


class Vocabulary():
    '''
    Object for creating and facilitating a mapping of unique words to unique integers.

    Very very very very loosely based on the `Lang` class from
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

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
    size : int
        Number of words stored in the vocabulary.

    Methods
    -------
    clean_word(word)
        Static. Returns `word` converted to lowercase with whitespace and certain punctionation stripped off the ends.
    indexify(words)
        Returns indices corresponding to mappings from `words`.
    wordify(indices)
        Returns words corresponding to mappings from `indices`.

    '''

    def __init__(self,
                 corpus: Union[str, List[str]] = None,
                 corpus_filepath: str = None,
                 corpus_dir: str = None):
        self._vocab: List[str] = self._init_vocab(corpus, corpus_filepath, corpus_dir)

    @property
    def size(self):
        return len(self._vocab)

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
        '''
        Collects unique words from the corpus, cleaning them with `clean_word()` in the process.

        Parameters
        ----------
        corpus : str or list of str, optional
            Space-delimited string or list of strings representing the entire corpus.
        corpus_filepath : str, optional
            Path to text file containing the entire corpus.
        corpus_dir : str, optional
            Path to directory containing text files that represent the corpus.

        Returns
        -------
        vocab : list of str
            Every unique word from the corpus in an arbitrary order. However, a
            start-of-sequence token "<start>" and an end-of-sequence token "<end>"
            are at indices 0 and 1, respectively.

        '''
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

    def indexify(self, words: List[str]) -> List[int]:
        '''
        Translates words into indices based on this instance's internal vocabulary mapping.

        Parameters
        ----------
        words : list of str
            Words present in this instance's vocabulary.

        Returns
        -------
        indices : list of int
            Indices corresponding to mappings from `words`.

        '''
        if not (isinstance(words, list) and all(isinstance(word, str) for word in words)):
            raise TypeError("`words` must be list of str")
        try:
            return [self._vocab.index(word) for word in words]
        except ValueError as e:
            raise KeyError("'{}' is not in vocabulary".format(str(e).split('\'')[1]))

    def wordify(self, indices: List[int]) -> List[str]:
        '''
        Translates indices into words based on this instance's internal vocabulary mapping.

        Parameters
        ----------
        indices : list of int
            Indices that map to words in this instance's vocabulary.

        Returns
        -------
        words : list of str
            Words corresponding to mappings from `indices`.

        '''
        if not (isinstance(indices, list) and all(isinstance(index, int) for index in indices)):
            raise TypeError("`indices` must be list of int")
        for index in indices:
            if index < 0:
                raise KeyError("`indices` cannot contain negative values")

        try:
            return [self._vocab[index] for index in indices]
        except IndexError:
            raise KeyError("`indices` cannot contain values greater than size of vocabulary")
