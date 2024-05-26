import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics
import torch.nn.functional as F
device = ('cuda' if torch.cuda.is_available() else 'cpu')
train_data = datasets.MNIST(root='data',
                                  train=True,
                                  download=True,
                                  transform=ToTensor())
test_data = datasets.MNIST(root='data',
                                  train=False,
                                  download=True,
                                  transform=ToTensor())




class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3),
            nn.ReLU(),
            nn.Conv2d(5, 10, 3),
            nn.ReLU()
            )
        
        self.mlp = nn.Sequential(
            nn.Linear(24*24*10, 10),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        x = torch.flatten(x, 1)

        x = self.mlp(x)
        '''
        ritornare sempre x, ossia logits
        '''
        return x
        

model = OurCNN().to(device)

# test_x = torch.rand((1, 28, 28))
# test_y = model(test_x)

# exit()

epochs = 2
batch_size = 16
learning_rate  = 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

training_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

metric = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    #size = len(dataloader)
    for batch_idx, (X, y) in enumerate(dataloader):
        
        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #print some infos
        if batch_idx % 20 == 0:
            loss_v, current_batch = loss.item(), (batch_idx + 1) * len(X)
            acc = metric(pred, y)
            #loss_v, current_batch = loss.item(), (batch + 1) * len(X)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(X), len(dataloader.dataset),
            100. * batch_idx / len(dataloader), loss_v))

    acc = metric.compute()
    print("Final accuracy", acc)
    metric.reset()
test_losses = []
def test_loop(dataloader, model):
    # get the batch from the dataset
    model.eval()
    test_loss = 0
    correct = 0
        # disable weights update
    with torch.no_grad():
        for X,y in dataloader:
            output = model(X)
            test_loss += F.nll_loss(output, y, size_average=False).item()
            # move data to the correct device
            X = X.to(device)
            y = y.to(device)

            # get the model predictions
            pred = model(X)
            acc = metric(pred,y)

    # print some informations
        test_loss /= len(dataloader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))
        # print(f'loss: {loss_v} [{current_batch}/{size}]')
        # acc = metric(pred,y)
        # print(f'Accuracy on current batch: {acc}')

    # print the final accuracy of the model
    acc = metric.compute()

    print(f'Final Testing accuracy: {acc}')
    metric.reset()

    
for epoch in range(1, epochs + 1):
    #print("epochs: ", epochs)
    train_loop(training_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model)