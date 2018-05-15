import torch

import utils


corpus = "This is the entire corpus. Seriously, it's contained in here. All of it."
print("Corpus: \n\t{}".format(corpus))

vocabulary = utils.Vocabulary(corpus=corpus)
print("`vocabulary` initialized successfully.")
print("Size of `vocabulary`: {}".format(vocabulary.size))
print("Contents of `vocabulary`:")
for word in vocabulary.words:
    print("\t{}".format(word))

indices = vocabulary.indexify(["all", "this"])
print("[\"all\", \"this\"] indexified: \n\t{}".format(indices))

words = vocabulary.wordify(indices)
print("and converted back into words: \n\t{}".format(words))
