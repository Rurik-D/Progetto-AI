import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
import torchmetrics

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

qmnist = unpickle("MNIST-120k")
data = qmnist['data']
#cv2.imwrite("C:\\Users\giuse\Desktop\Progetto-AI\prova.jpg", data[53050])
#exit()
cv2.imshow("Data[0]",data[0])
cv2.waitKey(0)
data = data.reshape(-1,1, 28, 28)
#print(data[0])

#exit()
labels = qmnist['labels']
def image_attr(img):
    rows = img.shape[0]
    cols = img.shape[1]
    return rows, cols

rows_img, cols_img = image_attr(data[0])

#exit()

device = ("cuda" if torch.cuda.is_available() else "cpu")

#Converti gli array numpy in tensori PyTorch
data_tensor = torch.from_numpy(data).float().to(device)
# Spremi le etichette in un tensore 1D
labels_tensor = torch.from_numpy(labels.squeeze()).long().to(device)
 
# Dividi i dati in training e test set
data_train, data_test, labels_train, labels_test = train_test_split(
    data_tensor, labels_tensor, test_size=0.2, random_state=0
)

# Crea i TensorDataset per training e test set
train_dataset = TensorDataset(data_train, labels_train)
test_dataset = TensorDataset(data_test, labels_test)

# define the hyperparameters
epochs = 10
batch_size = 200
learning_rate = 0.0001

# Crea DataLoader per gestire i batch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#exit()
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
            nn.Linear(24*24*10,10),
            nn.ReLU(),
            nn.Linear(10,10)
        )

    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        x = torch.flatten(x,1)
        #print(x.shape)
        x = self.mlp(x)
        return x

model = OurCNN().to(device)
#test_x = torch.rand((1,1,28,28)).to(device)
#test_y = model(test_x)

#exit()
# define the loss function
loss_fn = nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

# define the accuracy metric

metric = torchmetrics.Accuracy(task='multiclass',num_classes=10).to(device)

#exit()
# defining the training loop
def train_loop(dataloader,model,loss_fn,optimizer):
    size = len(dataloader)

    # get the batch from the dataset
    for batch, (X,y) in enumerate(dataloader):

        # move data to device
        X = X.to(device)
        y = y.to(device)

        # compute the prediction and the loss
        pred = model(X)
        loss = loss_fn(pred,y)

        # let's adjust the weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print some informations
        if batch % 500 == 0:
            loss_v, current_batch = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss_v} [{current_batch}/{size}]')
            acc = metric(pred,y)
            print(f'Accuracy on current batch: {acc}')
    
    # print the final accuracy of the model
    acc = metric.compute()
    print(f'Final Accuracy: {acc}')
    metric.reset()

# define the testing loop
def test_loop(dataloader, model):
    
    # disable weights update
    with torch.no_grad():
        for X,y in dataloader:
            # move data to the correct device
            X = X.to(device)
            y = y.to(device)

            # get the model predictions
            pred = model(X)
            acc = metric(pred,y)
    
    # compute the final accuracy
    acc = metric.compute()

    print(f'Final Testing accuracy: {acc}')
    metric.reset()


# train the model!!!
for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model)

torch.save(model.state_dict(),"digits_rec.pth")

print("Done!")