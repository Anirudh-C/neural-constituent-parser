import torch
from torch import nn

class Score(nn.Module):
    """
    Score model to score each rule and span in treebank for parsing
    """
    def __init__(self, device):
        super(Score, self).__init__()
        self._device = device
        self._ruleWeights = nn.Linear(26, 16).to(self._device)
        self._wordWeights = nn.Sequential(nn.Linear(2048, 512),
                                          nn.ReLU(),
                                          nn.Linear(512, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 16),
                                          nn.ReLU()).to(self._device)

    def forward(self, x):
        """
        In a forward step compute the score for a batch
        """
        words = x[:,0,:]
        rules = x[:,1,4:30]
        return torch.sum(torch.addcmul(torch.zeros(16, dtype=torch.float).to(self._device),
                                       self._wordWeights(words),
                                       self._ruleWeights(rules)), 1)


def treeLoss(scores):
    """
    The training loss is the negative sum of the scores for all the spans
    in a given batch.
    """
    return - torch.sum(torch.abs(scores))

if __name__=="__main__":
    from data import TreebankDataset
    from torch.utils.data import DataLoader

    treebank = TreebankDataset()
    treebankLoader = DataLoader(treebank, batch_size=4, shuffle=False)

    scoreFunc = Score()
    for i, batch in enumerate(treebankLoader):
        if i == 1:
            break
        scores = scoreFunc(batch.to(dtype=torch.float))
        loss = treeLoss(scores)
        print(scores, loss)
