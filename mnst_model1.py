import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics

train_data = datasets.MNIST(root='data',
                                  train=True,
                                  download=True,
                                  transform=ToTensor())
test_data = datasets.MNIST(root='data',
                                  train=False,
                                  download=True,
                                  transform=ToTensor())
print(train_data[0])
print(type(train_data[0]))
print(type(test_data[0][1]))
exit()
device = ('cuda' if torch.cuda.is_available() else 'cpu')

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

        return x
        

model = OurCNN().to(device)


# test_x = torch.rand((1, 28, 28))
# test_y = model(test_x)

# exit()

epochs = 5
batch_size = 32
learning_rate  = 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

training_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

metric = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        
        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #print some infos
        if batch % 20 == 0:
            loss_v, current_batch = loss.item(), (batch + 1) * len(X)
            acc = metric(pred, y)
            print("#"*20,
                  "\nloss:", loss_v,
                  "\ncurrent batch:", current_batch//size,
                  "\naccuracy:", acc,
                  "\n", "#"*20)

acc = metric.compute()
print("Final accuracy", acc)
metric.reset()

def test_loop(dataloader, model):
    
    model.eval() # per ripristinare connessioni tolte con dropout
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            
            acc = metric(pred, y)

    acc = metric.compute()
    
    print("Final testing accuracy:", acc)
    metric.reset()
    
for epochs in range(epochs):
    print("epochs: ", epochs)
    train_loop(training_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model)
    
torch.save(model.state_dict(),"mnst_model1.pth")