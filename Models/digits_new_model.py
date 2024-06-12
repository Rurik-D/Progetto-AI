import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
import torchmetrics
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os 
# qmnist = unpickle("MNIST-120k")
# data = qmnist['data']

# # cv2.imwrite("C:\\Users\giuse\Desktop\Progetto-AI\prova.jpg", data[40000])
# # exit()

# data = data.reshape(-1,1, 28, 28)
# #print(data[0])

# #exit()
# labels = qmnist['labels']

# #exit()

device = ("cuda" if torch.cuda.is_available() else "cpu")

# #Converti gli array numpy in tensori PyTorch
# #data_tensor = torch.from_numpy(data).float().to(device)
# # Spremi le etichette in un tensore 1D
# #labels_tensor = torch.from_numpy(labels.squeeze()).long().to(device)
# data_tensor = torch.tensor(data).float().to(device)
# labels_tensor = torch.tensor(labels.squeeze()).long().to(device)

# # Dividi i dati in training e test set
# data_train, data_test, labels_train, labels_test = train_test_split(
#     data_tensor, labels_tensor, test_size=0.2, random_state=0
# )

# # Crea i TensorDataset per training e test set
# train_dataset = TensorDataset(data_train, labels_train)
# test_dataset = TensorDataset(data_test, labels_test)
# Funzione per caricare i file pickle

# Caricamento del dataset QMNIST
csv_input = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\digits\\our_digits.csv'
df = pd.read_csv(csv_input)


train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=100)

path_images = "C:\\Users\\giuse\\Desktop\\Progetto-AI\\digits\\dg_data"

train_dataset['filename'] = train_dataset['filename'].apply(lambda x: os.path.join(path_images, x))
test_dataset['filename'] = test_dataset['filename'].apply(lambda x: os.path.join(path_images, x))

# Definizione del Dataset personalizzato
class QMNISTDataset(Dataset):
    def __init__(self, data, labels):
        self.images = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # Normalizza i pixel
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)
# # Caricamento del dataset
# train_dataset = QMNISTDataset(train_data,train_labels)
# test_dataset = QMNISTDataset(test_data,test_labels)

# define the hyperparameters
epochs = 10
batch_size = 2000
learning_rate = 0.0001

# Crea DataLoader per gestire i batch
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#exit()
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # nn.Conv2d(1, 5, 3),
            # nn.ReLU(),
            # nn.Conv2d(5, 10, 3),
            # nn.ReLU(),
            # nn.Conv2d(10, 1, 3),
            # nn.ReLU()
            nn.Conv2d(1, 32, 3, padding=1),  # Primo strato convoluzionale
            nn.BatchNorm2d(32),               # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Primo strato di pooling
            nn.Conv2d(32, 64, 3, padding=1),  # Secondo strato convoluzionale
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Secondo strato di pooling
            nn.Conv2d(64, 128, 3, padding=1), # Terzo strato convoluzionale
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # Terzo strato di pooling
            )
        self.mlp = nn.Sequential(
            # nn.Linear(22*22,10),
            # nn.ReLU(),
            # nn.Linear(10,10)
            nn.Linear(128 * 3 * 3, 256),      # Primo strato lineare
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout
            nn.Linear(256, 128),              # Secondo strato lineare
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout
            nn.Linear(128, 10)                # Strato di output
        )

    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        #x = torch.flatten(x,1)
        x = x.view(x.size(0), -1)
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
    #size = len(dataloader)
    model.train()  # Metti il modello in modalità di allenamento
    metric.reset()  # 

    # get the batch from the dataset
    for batch, (X,y) in enumerate(dataloader):
    

        X = X.unsqueeze(1)
        y = y.squeeze(1) 
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

        # update the metric
        metric.update(pred, y)
         
        # print some informations
        if batch % 10 == 0:
            #loss_v, current_batch = loss.item(), (batch + 1) * len(X)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch * len(X), len(dataloader.dataset),
            100. * batch/ len(dataloader), loss.item()))
            # print(f'loss: {loss_v} [{current_batch}/{size}]')
            # acc = metric(pred,y)
            # print(f'Accuracy on current batch: {acc}')
    
    # print the final accuracy of the model
    acc = metric.compute()
    print(f'Final Accuracy: {acc}')
    metric.reset()

# define the testing loop
def test_loop(dataloader, model):
    model.eval()  # Metti il modello in modalità di valutazione
    metric.reset()  # Resetta la metrica all'inizio di ogni epoca di test
    # disable weights update
    with torch.no_grad():
        for X,y in dataloader:
            
            X = X.unsqueeze(1)
            y = y.squeeze(1) 
            # move data to the correct device
            X = X.to(device)
            y = y.to(device)

            # get the model predictions
            pred = model(X)
            
            metric.update(pred, y)
    
    # compute the final accuracy
    acc = metric.compute()

    print(f'Final Testing accuracy: {acc}')
    metric.reset()


# train the model!!!
for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model)

#torch.save(model.state_dict(),"digits_rec.pth")

print("Done!")
