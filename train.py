import utils

corpus = "This is the entire corpus. Seriously, it's contained in here. All of it."
print("Corpus: \n\t{}".format(corpus))

vocabulary = utils.Vocabulary(corpus=corpus)
print("`vocabulary` initialized successfully.")
print("Length of `vocabulary`: {}".format(len(vocabulary)))
print("Contents of `vocabulary`:")
for word in vocabulary._vocab:
    print("\t{}".format(word))

indices = vocabulary.indexify(["all", "this"])
print("[\"all\", \"this\"] indexified: \n\t{}".format(indices))

words = vocabulary.wordify(indices)
print("and converted back into words: \n\t{}".format(words))
