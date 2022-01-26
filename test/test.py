
#%%
import pandas as pd
from NN_utils.classes.network import NN_NLP
from network import NeuralNetwork
from create_data_loader import create_data_loader
from NN_utils.nlp_utils.prep_data import clean_data
from torch.optim.lr_scheduler import StepLR,ConstantLR
import numpy as np

X = pd.DataFrame({"text":["hurtig levering", "alkjsbnfalkjsbnfclkjas aokjnmdxij hjav  ahsdf ", "fucking \n \n asbnbdf aafds a lorte :D hjemmeside"],"y":[1,2,3],"daller":["a","b","c"]})
X_val = pd.DataFrame({"text":["hurtig levering", "alkjsbnfalkjsbnfclkjas aokjnmdxij hjav  ahsdf ", "fucking \n \n asfg lorte :D hjemmeside"],"y":[1,2,3]})

X["clean"] = clean_data(X["text"]) #cleans dat

X.dropna(inplace = True)

#%%

#xx_val,y_val = clean_data(X_val["text"],X_val["y"]) #Validering



train_dl = create_data_loader(np.array([np.array(x,dtype="float") for x in X["clean"].values]),X["y"].values,2) #Lav data-loader (træn)

#val_dl = create_data_loader(xx_val,y_val,2) #Lav data-loader (val)

net = NeuralNetwork(n_words = 30, embed_dim = 500,n_classes =62) #Lav netværk
nn = NN_NLP(net) #lav en NN-instance
"""
class TestFunc:
    def __init__(self,mult=2):
        self.mult = mult

    def __call__(self,pred,true):
        return (pred.argmax(axis=1)-true).abs().sum()+self.mult

test_func = TestFunc(mult=3)

def Test_Func(pred,true,mult=2):
    return (pred.argmax(axis=1)-true).abs().sum()+mult


test_func = TestFunc(mult=3)
#test_func = Test_Func
"""


vals = nn.train(5,train_dl,n_epoch_print=1) #Træn hele lårtet
