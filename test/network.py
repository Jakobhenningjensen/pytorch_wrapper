from torch import nn
import torch
class NeuralNetwork(nn.Module):
    def __init__(self,n_words,embed_dim,n_classes):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = embed_dim, #Length of wordembedding
        out_channels = 2, #Number of filters
        kernel_size = 1, #How many words to capture at a time (unigram)
        stride=1) #How many words to stride over


        self.conv2 = nn.Conv1d(in_channels = embed_dim, #Length of wordembedding
        out_channels = 2, #Number of filters
        kernel_size = 2, #How many words to capture at a time (bi-gram)
        stride=1) #How many words to stride over


        self.conv3 = nn.Conv1d(in_channels = embed_dim, #Length of wordembedding
        out_channels = 2, #Number of filters
        kernel_size = 3, #How many words to capture at a time (tri-gram)
        stride=1) #How many words to stride over

        self.l1 = nn.Linear(6,n_classes)


        # Utils
        self.meanpool1 = nn.AvgPool1d(n_words)
        self.meanpool2 = nn.AvgPool1d(n_words-1)
        self.meanpool3 = nn.AvgPool1d(n_words-2)

        self.flatten = nn.Flatten()
        self.act = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout25 = nn.Dropout(0.25)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x1 = self.meanpool1(x1)
        x2 = self.meanpool2(x2)
        x3 = self.meanpool3(x3)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)

        x = torch.cat((x1,x2,x3),1)

        x = self.l1(x)
        x = self.dropout50(x)
        x = self.softmax(x)
        return x