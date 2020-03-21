from nltk.corpus import treebank
import itertools

datapaths = ["../parsed/wsj_{0:04}.prd".format(num) for num in range(1, 200)]
sentences = list(itertools.chain.from_iterable(list(map(treebank.parsed_sents, datapaths))))
