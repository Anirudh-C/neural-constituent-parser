from nltk.corpus import treebank
from nltk.tree import Tree
import re
import torch
from torch.utils.data import Dataset, DataLoader
import itertools

from utils import cnf, collapse

datapaths = ["../parsed/wsj_{0:04}.prd".format(num) for num in range(1, 200)]

class TreebankDataset(Dataset):
    """
    Penn Treebank Dataset that returns (s,r) for each sentence
    """
    def __init__(self, pathString):
        """
        Initiates ground truth for PTB after removing function labels, null values
        and converting to CNF
        :param: pathString - string that defines the treebank files to load from
        """
        self._datapaths = [pathString.format(num) for num in range(1,200)]
        sentences = itertools.chain(*map(treebank.parsed_sents, self._datapaths))
        self.filterSentences(list(sentences))

    def __len__(self):
        """
        Returns length of dataset, that is, number of sentences
        """
        return len(self._sentences)

    def filterSentences(self, sentences):
        """
        Filters the dataset removing function labels, null values, converting to CNF
        and forming the spans.
        :param: sentences - list of sentences to filter
        """
        self._sentences = [collapse(self.removeNulls(cnf(self.removeFunctionLabels(sent))))
                           for sent in sentences]
        # print(type(self._sentences[0].productions()[0].rhs()[2]))

    def encodeRule(rule):
        return rule

    def removeFunctionLabels(self, sentence):
        """
        Removes function labels
        :param: sentence
        """
        if isinstance(sentence, Tree):
            sentence.set_label(sentence.label().split("-")[0])
            for subtree in sentence:
                subtree = self.removeFunctionLabels(subtree)
        return sentence

    def nullMatch(self, word):
        """
        Returns true if the word is a null element or a pseudo attachment
        :param: word
        """
        # Match all null elements and pseudo attachments
        if re.search("\*-\d", word) or \
           re.search("\*T\*-\d", word) or \
           re.search("\*U\*", word) or \
           re.search("\*ICH\*|\*PPA\*|\*RNR\*|\*EXP\*", word):
            return True
        return False

    def removeNulls(self, sentence):
        """
        Removes the null values in a sentence
        :param: sentence
        """
        if isinstance(sentence, Tree):
            for i, child in enumerate(sentence):
                if isinstance(child, str) and self.nullMatch(child):
                    sentence.remove(child)
                else:
                    sentence[i] = self.removeNulls(child)
        return sentence

    def __getitem__(self, idx):
        """
        Returns an element from the ground truth, that is, a sentence in (s,r) form.
        """
        return self._sentences[idx]

    def getLabels(sentence):
        labels = []
        if isinstance(sentence, Tree):
            labels.append(sentence.label())
            for subtree in sentence:
                labels = labels + getLabels(subtree)
        return labels

if __name__=="__main__":
    treebank = TreebankDataset("../parsed/wsj_{0:04}.prd")
    treebankLoader = DataLoader(treebank, batch_size=1, shuffle=True)
    for (i,batch) in enumerate(treebankLoader):
        for sent in batch:
            print(sent)
        if i == 1:
            break
