from nltk.corpus import treebank
from nltk.tree import Tree

import re
import itertools

import numpy as np
import torch
from torch.utils.data import Dataset
from allennlp.commands.elmo import ElmoEmbedder

from utils import cnf, collapse, spanRepresentation

datapaths = ["../parsed/wsj_{0:04}.prd".format(num) for num in range(1, 200)]

class TreebankDataset(Dataset):
    """
    Penn Treebank Dataset that returns (s,r) for each sentence
    """
    def __init__(self, train=True, samples=5, pathString="../parsed/wsj_{0:04}.prd", computeVecs=False):
        """
        Initiates ground truth for PTB after removing function labels, null values
        and converting to CNF
        :param: pathString - string that defines the treebank files to load from
        """
        self._train = train
        self._computeVecs = computeVecs
        self._datapaths = [pathString.format(num) for num in range(1,200)]
        sentences = itertools.chain(*map(treebank.parsed_sents, self._datapaths))
        self.filterSentences(list(sentences))

        # Number of train samples
        self._samples = samples

        print("Filtered Dataset")

        self._labels = []
        for sent in self._sentences:
            self._labels = self._labels + self.getLabel(sent)
        self._labels = sorted(list(set(self._labels)))

        self._rules = np.array([[self.encodeRule(rule) for rule in sent.productions()]
                                for sent in self._sentences])
        print("Encoded Rules")

        self._spans = [self.getSpanIndices(self.getSpans(sent), sent.leaves()) for sent in self._sentences]
        print("Computed spans")

        print("Loading ELMo Embeddings...")
        self.elmo = ElmoEmbedder()
        if self._computeVecs:
            print("Computing word vectors...")
            self._wordVectors = [self.elmo.embed_sentence(sent.leaves())
                                 for sent in self._sentences]

    def getLabel(self, sentence):
        """
        Get the labels in a given sentence
        :param: sentence
        """
        labels = []
        if isinstance(sentence, Tree):
            labels.append(sentence.label())
            for subtree in sentence:
                labels = labels + self.getLabel(subtree)
        return labels

    def __len__(self):
        """
        Returns length of dataset, that is, number of sentences
        """
        if self._train:
            return self._samples
        else:
            return len(self._sentences) - self._samples

    def __getitem__(self, idx):
        """
        Returns an element from the ground truth, that is, a sentence in (s,r) form.
        """
        if not self._train:
            idx = idx + self._samples

        numSpans = len(self._spans[idx])
        sentence = np.zeros((numSpans, 2, 2048))
        for i in range(numSpans):
            span = np.zeros(2048)
            span[:4] = self._spans[idx][i]
            span[4:30] = self._rules[idx][i]

            if self._computeVecs:
                wordVecs = spanRepresentation(self._wordVectors[idx], self._spans[idx][i])
            else:
                wordVecs = spanRepresentation(self.elmo.embed_sentence(self._sentences[idx].leaves())[2],
                                              self._spans[idx][i])

            sentence[i] = np.stack((wordVecs, span))

        return sentence

    # Functions to encode the input
    def oneHotEncode(self, label):
        """
        One Hot encode a label if it is in self._labels
        :param: label
        """
        if label in self._labels:
            encodedLabel = np.zeros(len(self._labels))
            encodedLabel[self._labels.index(label)] = 1
            return encodedLabel
        else:
            raise ValueError

    def encodeRule(self, rule):
        """
        Produce the encoding for a given production
        :param: rule
        """
        encodedRule = np.zeros(len(self._labels))
        encodedRule += self.oneHotEncode(rule.lhs().symbol())
        phraseCount = 0
        for term in rule.rhs():
            if not isinstance(term, str):
                phraseCount += 1
                encodedRule += (3**phraseCount) * self.oneHotEncode(term.symbol())
        return encodedRule

    def getSpans(self, tree):
        """
        Produce the spans of words for the given sentence for the
        production rules
        :param: tree
        """
        spans = []
        if isinstance(tree, Tree):
            phrases = []
            for child in tree:
                if isinstance(child, Tree):
                    phrases.append(child.leaves())
                else:
                    phrases.append(child)
            spans.append(phrases)
            for child in tree:
                if isinstance(child, Tree):
                    spans += self.getSpans(child)
        return spans

    def spanIndices(self, span, sentence, givenFirst=None):
        """
        Given an individual span return the span indices
        in the sentence as (i,j)
        :param: span
        :param: sentence
        """
        if isinstance(span, str):
            span = [span]

        subLen = len(span)
        for idx in [i for i,x in enumerate(sentence) if x == span[0]]:
            if sentence[idx:idx+subLen] == span:
                i = idx
                j = idx + subLen - 1
                if givenFirst and i <= givenFirst[0] and j <= givenFirst[0]:
                    continue
                else:
                    break

        return i,j

    def getSpanIndices(self, spans, sentence):
        """
        Given a list of full spans for a sentence return
        the indices as (i,j,k,l)
        :param: spans
        :param: sentence
        """
        indices = []
        for span in spans:
            if len(span) == 1:
                i,j = self.spanIndices(span[0], sentence)
                k,l = j,j
            else:
                i,j = self.spanIndices(span[0], sentence)
                k,l = self.spanIndices(span[1], sentence, (i,j))
            indices.append((i,j,k,l))
        return indices

    # Functions to filter a given sentence
    def filterSentences(self, sentences):
        """
        Filters the dataset removing function labels, null values, converting to CNF
        and forming the spans.
        :param: sentences - list of sentences to filter
        """
        self._sentences = [collapse(cnf(self.removeNulls(self.removeFunctionLabels(sent))))
                 for sent in sentences]

    def removeFunctionLabels(self, sentence):
        """
        Removes function labels
        :param: sentence
        """
        if isinstance(sentence, Tree):
            sentence.set_label(sentence.label().split("-")[0])
            sentence.set_label(sentence.label().split("=")[0])
            sentence.set_label(sentence.label().split("|")[0])
            for subtree in sentence:
                subtree = self.removeFunctionLabels(subtree)
        return sentence

    def nullMatch(self, word):
        """
        Returns true if the word is a null element or a pseudo attachment
        :param: word
        """
        # Match all null elements and pseudo attachments
        if re.search("\*|\*T\*", word ) or \
           re.search("\*-\d", word) or \
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

def tree_collate(batch):
    """
    Custom collate function for DataLoader
    """
    return torch.as_tensor(np.concatenate(batch, axis=0))

if __name__=="__main__":
    treebank = TreebankDataset(train=False, samples=6)
    print("Loaded treebank with {} sentences.".format(len(treebank._sentences)))

