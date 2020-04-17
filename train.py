import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from data import TreebankDataset
from model import treeLoss, Score

def train(batch=64, epochs=2, split=0.8, cuda=True):
    # Create train test split
    treebank = TreebankDataset(train=True)
    trainSize = int(split * len(treebank))
    testSize = len(treebank) - trainSize
    print("Generating train test split")
    treebankTrain, treebankTest = random_split(treebank, [trainSize, testSize])

    trainLoader = DataLoader(treebankTrain, batch_size=batch, shuffle=True)
    testLoader = DataLoader(treebankTest, batch_size=batch, shuffle=True)

    if cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Score(device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print("Epoch step {} of {}".format(epoch+1, epochs))

        # Train
        print("Training..")
        runningLoss = 0.0
        for i, batch in enumerate(tqdm(trainLoader)):
            scores = model(batch.to(device, dtype=torch.float))
            loss = treeLoss(scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

        print("[%d, %d] loss: %.3f" %
              (epoch+1, epochs, runningLoss / trainSize))

        #Test
        print("Validation..")
        testLoss = 0.0
        for batch in tqdm(testLoader):
            with torch.no_grad():
                scores = model(batch.to(device, dtype=torch.float))
                loss = treeLoss(scores)

                testLoss += loss.item()

        print("[%d %d] val loss: %.3f" %
              (epoch+1, epochs, testLoss / testSize))

    torch.save(model.state_dict(), 'models/run-1.pth')

if __name__ == "__main__":
    train()
