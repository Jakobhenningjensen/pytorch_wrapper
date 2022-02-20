# WARNING

This repo is not ready yet, and the `readme` file is under creation, thus it might change several times!



# pytorch_wrapper
A wrapper for training neural-networks using the `pytorch` module which helps to remove a lot of lines.
Note, this is not a wrapper for creating the architecture but for the training part, thus we assume that you know how to create a neural-network using [https://pytorch.org/](pytorch)




# Installation

Simply just clone the repo

`git clone https://github.com/Jakobhenningjensen/pytorch_wrapper/`

# How to

## Create a network architecture
First we need to create a neural-network. We save this network in a folder named `model.py`

```python
#model.py

"""
Creating a simple neural network.
"""
from torch import nn

class Net(nn.Module):
    """
    A simple neural-network.

    inputs:
    -------

    n_classes: int
        - Number of classes to predict

    n_features: int
        - Number of features

    """
    def __init__(self,n_classes,n_features):
        super().__init__()


        self.l1 = nn.Linear(n_features,100)
        self.l2 = nn.Linear(100,50)
        self.l3 = nn.Linear(50,n_classes)


        self.act = nn.ReLU()
        self.dropout50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.dropout50(x)

        x = self.l2(x)
        x = self.act(x)
        x = self.dropout50(x)

        x = self.l3(x)
        x = self.act(x)
        x = self.dropout50(x)
        x = self.softmax(x)
        return x

```

## Create a data-loader

Next thing is to create a data-loader. This is done using pytorchs `Dataset` and `DataLoader`. We save the code below in a file called `create_data_loader.py`

```python
#create_data_loader.py

"""
Helper-function for creating a Dataloader object
"""

from torch.utils.data import Dataset, DataLoader

import torch
class Data(Dataset):
    """
    Constructs a Dataset to be parsed into a DataLoader

    inputs:
    -------
    X: numpy-array (n,k)
      - Numpy-array of size n x k where "n" is number of samples and "k" is number of features
    y: numpy-array(n,)
      - Numpy array of size n where element "i" is the target for row "i" in X
    """

    def __init__(self,X,y):
        X = torch.from_numpy(X).float() #Set correct types
        y = torch.from_numpy(y).long()

        self.X,self.y = X,y

    def __getitem__(self, i):
        return self.X[i],self.y[i]

    def __len__(self):
        return self.X.shape[0]

def create_data_loader(X,y,batch_size):
    """
    Creates a data-loader for the data X and y

    inputs:
    -------

    X: np.array
        - numpy array of size n x k where "n" is samples an "k" is number of features
    y: np.array
        - numpy array of sie "n" containing the target
    batch_size: int
        - batch_size which is the number of (X,y) samples returned each time we iterate over a DataLoader object

    return
    ------

    dl: torch.utils.data.DataLoader object
    """

    data = Data(X, y)

    dl = DataLoader(data, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    return dl
```

## Train the network

Now we have our model and data-loader we can simply train our network by the following

```python
from pytorch_wrapper.classes.network import NeuralNetwork #Load the module needed
from model import Net #Load our neural network
from create_data_loader import create_data_loader #Our function for creating a `DataLoader` object

#Create a dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_features=10, n_classes=3, n_informative = 10,n_redundant=0,random_state=42)

net = NeuralNetwork(Net(3,10)) #Create our net
train_data = create_data_loader(X,y,batch_size=20) #Create our training data

loss_func = torch.nn.CrossEntropyLoss() #Define loss-function

train_info = net.train(train_dataloader=train_data,n_epochs = 50,loss_func = loss_func, n_epoch_print=10) #Train the network for 50 eochs and print values each 10th epoch
# No device specified - defaulting to cpu
# Training on all data
# Epoch       Train-loss
# -------------------------------------------
# 10/50          0.852
# 20/50          0.880
# 30/50          0.852
# 40/50          0.957
# 50/50          0.880
```

`train_info` is a dictionary containing different data depending on the way the network is trained - since we have trained the network without any validation set (see below on how to do that) it just contains `train_loss` which is an array containing the loss for each batch i.e 250 elements (`n_samples`/`batch_size` = 1000/20)

## Train the network with a validation-set
Above is a way to train the network using a full dataset. Often we want to have a validation-set aswell.
This is very to easy; we just need to create a validation-data-loader and parse that as the argument `val_dataloader` in the `net`.
For keeping things nice and simply we just use the `create_data_loader` function we wrote for both the training and validation

```python
from NN_utils.classes.network import NeuralNetwork #Load the module needed
from network_simple import Net #Load our neural network
from create_data_loader import create_data_loader #Our function for creating a `DataLoader` object
import torch


#Create a dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_features=10, n_classes=3, n_informative = 10,n_redundant=0,random_state=42)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42) #split data in train/validation


net = NeuralNetwork(Net(3,10)) #Create our net

train_data = create_data_loader(X_train,y_train,batch_size=20) #create training data-loader
val_data = create_data_loader(X_val,y_val,batch_size=20) #create validation data-loader

loss_func = torch.nn.CrossEntropyLoss() #Define loss-function

train_info = net.train(train_dataloader=train_data,
val_dataloader=val_data, #adding a validation-set
n_epochs = 50,loss_func = loss_func,n_epoch_print=10)


# No device specified - defaulting to cpu
# Training with validation data
# Epoch       Train-loss       Val-loss         Val-score
# --------------------------------------------------------
# 10/50          1.117           1.048           0.245
# 20/50          0.989           1.048           0.239
# 30/50          0.866           1.005           0.285
# 40/50          0.907           1.003           0.319
# 50/50          1.052           1.011           0.278
```

Now `train_info` contains three keys; `train_loss'`, `'val_loss'` and `'val_score`  which are arrays containing training-loss, validation-loss and validtaion-score respectively. If no validation-function is parsed, the [multi-class F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) is used.
