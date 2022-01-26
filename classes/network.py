from torch import nn
import torch
from sklearn.metrics import f1_score
import re
from warnings import warn

class NeuralNetwork:
    """
    Class for training a neural-network
    """


    def __init__(self,network):
        """
        Initialize values and check inputs

        params:
        -------

        network: subclass of nn.Module from pytorch
            - must have the function "forward" implemented and contain the design of the network e.g
            class NeuralNetwork(nn.Module):
                def __init__(self,n_input):
                    super(NeuralNetwork, self).__init__()
                    self.l1 = nn.Linear(n_input,3)
                    self.dropout50 = nn.Dropout(0.5)

                def forward(self, x):
                    x = self.dropout50(x)
                    x = self.l1(x)
                    x = self.softmax(x)
                    return x

        """

        #Check all inputs
        self._check_inputs(
            network=network)

        #### Init ####
        self.network = network




    def _check_network(self,network):
        """
        Checks if the network inherits from nn.Module
        """
        if not isinstance(network,nn.Module):
            raise TypeError("'network' must be a sub-class of nn.Module")



    def _check_inputs(self,**kwargs):
        """
        Validates all inputs
        """

        ##### network #####
        self._check_network(kwargs.get("network",""))

    def _default_val_func(self,kwargs = {}):
        """
        Helper function for the standard validation-score (multiclass F1 score;https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) i.e when no function is specified

        params:
        -------
        kwargs: dict
            - additional keyword argurments for the function

        return:
        -------
        score:
            - an instance that calculates the weighted F1 score
        """

        class score:
            def __init__(self,kwargs= {}):
                self.kwargs = kwargs

            def __call__(self,y_score,y_true):
                pred = y_score.argmax(axis=1)
                return f1_score(y_true,pred,**(self.kwargs))

        return score(kwargs)



    def _set_correct_lossfunc_dtype(self,y):
        """
        Converts the target-tensor "y" to the correct dtype for the loss-function during runtime.
        The dtype that "y" is being converted to, is set in _check_loss_func


        params:
        -------

        y: torch.Tensor
            - The target-tensor


        returns
        -------

        torch.Tensor:
            - Either the same "y" (if convert=False) or the converted "y"
        """

        if self.convert_to_correct_lossfunction_target_dtype:
            y_conv = getattr(y,self.correct_lossfunction_target_dtype)()

            return y_conv
        return y





    def _check_loss_func(self):
        """
        Function to check if the dtype is compatible with the loss-function.
        If it is not, get the dtype it should be converted to

        """

        #Test if the data can get through the network and then loss-function
        temp_dl = iter(self.train_dataloader)
        X,y = next(temp_dl)
        pred = self.network(X)

        try:
            self.loss_func(pred,y) #Pass it through the loss-function
        except RuntimeError as e:
            err = str(e)
            from_dtype = re.search(r"(?<=(dtype ))\w+(?=( but))",err).group().lower()

            to_dtype = re.search(r"((\w)+)$",err).group().lower()

            self.correct_lossfunction_target_dtype = to_dtype
            self.convert_to_correct_lossfunction_target_dtype = True
            warn(f"The provided loss-function does not take '{from_dtype}' as input for your target-value.\n We are converting it to '{to_dtype}' in runtime")






    def _train_with_val(self,train_dataloader,val_dataloader):
        """
        Train a network using a validation-set

        params:
        -------

        train_dataloader: torch.utils.data.DataLoader object
            - DataLoader object which provides (X_train,y_train) of correct batch-size when iterating over it

        val_dataloader: torch.utils.data.DataLoader object
            - DataLoader object which provides (X_val,y_val) of correct batch-size when iterating over it


        return: Dict
            - Dictionary with keys ["train_loss","val_loss","val_score"]
                - train_loss: list of last training loss for each epoch
                - val_loss:  list of avg. training loss for each epoch
                - val_score: list of validation-score calculated by self.val_func at each epoch

        """

        ###### START TRAINING  #####

        LOSS_TRAIN = [0]*self.n_epochs
        LOSS_VAL = [0]*self.n_epochs
        VAL_SCORE = [0]*self.n_epochs

        print(f"{'Epoch'.ljust(12)}{'Train-loss'.ljust(17)}{'Val-loss'.ljust(17)}{'Val-score'}")
        print("---------------------------------------------------------")
        for e in range(self.n_epochs):
            self.network.train()
            for x,y in train_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                pred = self.network(x)

                loss = self.loss_func(pred,self._set_correct_lossfunc_dtype(y))

                loss.backward()
                self.optimizer.step()
            LOSS_TRAIN[e] = loss.item()

            #Take a lr_scheduler step if provided
            if self.lr_scheduler:
                self.lr_scheduler.step()

            #Validate
            self.network.eval()
            for x_val, y_val in val_dataloader:
                x_val = x_val.to(self.device)
                pred = self.network(x_val).to("cpu")

            LOSS_VAL[e] = self.loss_func(pred,self._set_correct_lossfunc_dtype(y_val)).item()
            VAL_SCORE[e] = self.val_func(pred.detach(),y_val.detach().int()).item()

            #Print info
            if self.n_epoch_print:
                if ((e+1) % self.n_epoch_print) == 0:
                    print("{:<15}{:.3f} {:>15.3f} {:>15.3f}".format(f"{e+1}/{self.n_epochs}",LOSS_TRAIN[e],LOSS_VAL[e],VAL_SCORE[e]))

        return {"train_loss":LOSS_TRAIN,"val_loss":LOSS_VAL,"val_score":VAL_SCORE}

    def _train_with_full(self,train_dataloader):
        """
        Train a network using no validation-set

        params:
        -------

        train_dataloader: torch.utils.data.DataLoader object
            - DataLoader object which provides (X_train,y_train) of correct batch-size when iterating over it

        returns:
        --------

        Dict:
            - Dictionary with the keys ["train_loss"]
                - train_loss: array of all loss i.e the avg. batch-loss for each iteration
        """

        ###### START TRAINING  #####
        LOSS_TRAIN = []
        print(f"{'Epoch'.ljust(12)}{'Train-loss'}")
        print("--------------------------------------------")
        for e in range(self.n_epochs):
            self.network.train()
            for x,y in train_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                pred = self.network(x)

                loss = self.loss_func(pred,self._set_correct_lossfunc_dtype(y))

                loss.backward()
                self.optimizer.step()
                LOSS_TRAIN.append(loss.item())

            #Take a lr_scheduler step if provided
            if self.lr_scheduler:
                self.lr_scheduler.step()

            #Print info
            if self.n_epoch_print:
                if ((e+1) % self.n_epoch_print) == 0:
                    print("{:<15}{:.3f}".format(f"{e+1}/{self.n_epochs}",LOSS_TRAIN[-1]))


        return {"train_loss":LOSS_TRAIN}



    def train(self,n_epochs,train_dataloader,val_dataloader=None,optim_kwargs ={"lr":0.01}, optimizer = None,lr_scheduler=None, device=None,val_func=None,loss_func=None,ensure_correct_loss_func_dtype = True, n_epoch_print=None):
        """
        trains the network parsed to "network" when creating this object

        params
        ------

        target_col: str
            - name of the column for the training/val-dataset specified. Other columns is assumed to be the plain-text

        n_epoch: int
            - number of epochs

        train_dataloader: pytorch.DataLoader
            - A pytorch.DataLoader of the traning-set. Must return (x,y) pair of a batch_size

        val_dataloader: pytorch.DataLoader
            - A pytorch.DataLoader of the validation-set. Must return (x,y) pair of a batch_size

        optim_kwargs: dict. Default: {"lr":0.01}
            - arguments for the optimizer

        optimizer: {optimizer,None}
            - Optimizer used to update weights. All hyper-parameters such as learning rate etc. should be set already. If None (default) using Adam

        device: {str,None}
            - Which device used for training. If None (default) use GPU if available, else using CPU

        val_func: {call-able,None}. Default: None
            - A callable that calculates a validation measure e.g accuracy, F1, ROC-AUC. Must take (y_score,y_true) with correspoding shapes ([n, c],n) where "n" is number of samples an "c" is number of classes as input
              i.e each row in "y_score" is the score for each class - and return a single float, e.g
              If None, uses the multi-class F1 score with weighted average from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)


        optimizer: {optimizer,None}
            - Optimizer used to update weights. If None (default) using Adam

        lr_scheduler: tuple(torch.optim.lr_scheduler*,kwargs)
            - A tuple where the first element is the learning-rate scheduler and the second is a dictionary containing the arguments to the scheduler
              e.g   lr_scheduler = [torch.optim.lr_scheduler.StepLR, {"step_size":30,"gamma":0.1}]

        loss_func: {call-able,None}. Default: None
            - Callable that serves as loss-function when training. If None (default) it uses cross-entropy (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
              If a function is provided the function must take two inputs 'pred' and 'target' and return a score and be differentiable with PyTorch.

        ensure_correct_loss_func_dtype: Bool. Default=True
            - Checks if the logits can be passed to the loss-function (based on the dtype). If not, converts it automatically to the correct dtype.
            - Note! This might cause unwanted behavior e.g converting float values less than 0 to Long, would make them all 0!

        n_epoch_print: {int,None}
            - Training progress and data is printed after each n_epoch_print i.e if n_epoch_print=5 then after 5,10,15 ... epochs the training-loss (and validtaion-loss if any) is printed


        return
        -------

        Dict:
            - Dictionary with the training-loss (and validtaion-loss if validation set is present)
                - train_loss: array of all loss i.e the avg. batch-loss for each iteration
                - val_loss: array of validtaion loss i.e the avg. batch-loss for each epoch (only if val_dataloader is parsed)

        """


        #Set training functions and parameters
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.n_epochs = n_epochs
        self.val_func = val_func
        self.loss_func = loss_func
        self.n_epoch_print = n_epoch_print
        self.lr_scheduler = None
        self.convert_to_correct_lossfunction_target_dtype = False

        if ensure_correct_loss_func_dtype:
            self._check_loss_func()




        if not optimizer: #Set optimizer
            self.optimizer = torch.optim.Adam(self.network.parameters(),**optim_kwargs)
        else:
            self.optimizer = optimizer(**optim_kwargs)

        if lr_scheduler: #Set scheduler
            lr_kwargs = lr_scheduler[1]
            scheduler = lr_scheduler[0]
            self.lr_scheduler = scheduler(self.optimizer,**lr_kwargs)

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu" #Get device
            self.network.to(self.device)
            print(f"No device specified - defaulting to {self.device}")

        if not loss_func:
            self.loss_func = nn.CrossEntropyLoss()

        if not val_func:
            self.val_func = self._default_val_func(kwargs = {"average":"weighted","zero_division":0})

        #### Train ####
        if val_dataloader is not None: #With validation set
            print("Training with validation data")
            val = self._train_with_val(train_dataloader,val_dataloader)
            return val

        else: #train on all data i.e no validation
            print("Training on all data")
            val = self._train_with_full(train_dataloader)
            return val