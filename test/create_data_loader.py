from torch.utils.data import Dataset, DataLoader
import torch
class Data(Dataset):
    """

    """
    def __init__(self,X,y):
        X = torch.from_numpy(X).float()

        #Transpose to fit dimensions
        X = torch.transpose(X,1,2)
        y = torch.from_numpy(y).float()
        self.X,self.y = X,y

    def __getitem__(self, i):
        return self.X[i],self.y[i]

    def __len__(self):
        return self.X.shape[0]

def create_data_loader(X,y,batch_size):
    """
    Creates a data-loader for the data X and y

    params:
    -------

    X: np.array
        - numpy array of size "n" x k where n is samples an "k" is number of features
    y: np.array
        - numpy array of sie "n"
    batch_size: int
        - Take a wild guess, dumbass

    return
    ------

    dl: torch.utils.data.DataLoader object
    """

    data = Data(X, y)

    dl = DataLoader(data, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    return dl