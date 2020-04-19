import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from data import TreebankDataset, tree_collate
from model import treeLoss, Score

def log(string):
    """
    Log the string into logs/run.log
    :param: string
    """
    with open('logs/run.log', 'a') as logFile:
        print(string, file=logFile)

def train(batches=50, epochs=5, split=0.8, samples=2000, cuda=True):
    """
    Train the score function over treebank
    :param: batches (default = 50)
    :param: epochs (default = 5)
    :param: split (default = 0.8) - train test split ratio
    :param: samples (default = 2000) - number of samples to train on in treebank
    :param: cuda (default = True)
    """
    # Create train test split
    treebank = TreebankDataset(train=True, samples=samples)
    trainSize = int(split * len(treebank))
    testSize = len(treebank) - trainSize
    print("Generating train test split")
    treebankTrain, treebankTest = random_split(treebank, [trainSize, testSize])

    trainLoader = DataLoader(treebankTrain, batch_size=batches, collate_fn=tree_collate)
    testLoader = DataLoader(treebankTest, batch_size=batches, collate_fn=tree_collate)

    if cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Score(device).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)

    for epoch in range(epochs):
        print("Epoch step {} of {}".format(epoch+1, epochs))

        # Train
        print("Training..")
        epochLoss = 0.0
        loss = torch.zeros(1)
        for i, batch in enumerate(tqdm(trainLoader)):
            scores = model(batch.to(device, dtype=torch.float))
            loss = treeLoss(scores)
            log("[%d, %d] loss: %.3f" %
                (epoch+1, i+1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        print("[%d, %d] loss: %.3f" %
              (epoch+1, epochs, epochLoss / trainSize))

        #Test
        print("Validation..")
        testLoss = 0.0
        for batch in tqdm(testLoader):
            with torch.no_grad():
                scores = model(batch.to(device, dtype=torch.float))
                loss = treeLoss(scores)

                testLoss += loss.item()

        log("[%d %d] val loss: %.3f" %
            (epoch+1, epochs, testLoss / testSize))
        print("[%d %d] val loss: %.3f" %
              (epoch+1, epochs, testLoss / testSize))

        torch.save(model.state_dict(), 'models/run.pt')

if __name__ == "__main__":
    train()
