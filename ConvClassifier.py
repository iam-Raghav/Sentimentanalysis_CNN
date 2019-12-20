from __future__ import print_function
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CONVClassifier(nn.Module):
    def __init__(self, kernel_height,out_channels,dropout_prob,embedding_dim, vocab_size, output_size):
        super(CONVClassifier, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.requires_grad= True
        self.kernel_height = kernel_height
        # Defining 3 different kernels to capture different features of the sentence
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels= out_channels, kernel_size= (kernel_height[0],embedding_dim))
        self.Conv2 = nn.Conv2d(in_channels=1, out_channels= out_channels, kernel_size= (kernel_height[1],embedding_dim))
        self.Conv3 = nn.Conv2d(in_channels=1, out_channels= out_channels, kernel_size= (kernel_height[2],embedding_dim))
        self.dropout = nn.Dropout(dropout_prob)
        # self.hidden = nn.Linear(len(kernel_height)*out_channels, hidden_dim)
        self.outspace = nn.Linear(len(kernel_height)*out_channels, output_size)

    def conv_wrapper(self,input,conv):
        #Padding is done if the sentence has no of words less the kernel heights mentioned
        if max(self.kernel_height) > input.size()[2]:
            pad_dim = (0,0,0, max(self.kernel_height) - input.size()[2])
            input = F.pad(input,pad_dim, mode='constant',value =0)
        # embeds = embeds.unsqueeze(0)
        out=conv(input) #dimensions 1 * out_channels * length of sentence * 1
        # out = out
        nl_out = F.relu(out)
        max_out = t.max(nl_out, dim=2) #dimensions 1 * out_channels(hyper parameter) * 1
        max_out = max_out[0]
        return max_out.squeeze(0) #dimensions out_channels (hyper parameter) * 1

    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)#dimensions length of the sentence * embedding dim(hyper parameter)
        self.word_embeddings.weight.requires_grad = True

        embeds.requires_grad_(True)
        embeds = embeds.unsqueeze(0) #dimensions 1 * length of the sentence * embedding dim(hyper parameter)
        embeds = embeds.unsqueeze(0) #dimensions 1 * 1 * length of the sentence * embedding dim(hyper parameter)

        max_out1 = self.conv_wrapper(embeds, self.Conv1) #dimensions out_channels * 1
        max_out2 = self.conv_wrapper(embeds, self.Conv2) #dimensions out_channels * 1
        max_out3 = self.conv_wrapper(embeds, self.Conv3) #dimensions out_channels * 1

        straight_out = t.cat((max_out1,max_out2,max_out3),1) #dimensions out_channels * no of kernels

        fc_out = self.dropout(straight_out)
        fc_out = fc_out.view(1,fc_out.size()[0]*fc_out.size()[1]) #dimensions 1 * (out_channels (*) no of kernels) in this case 1 * 120
#        hidden = F.relu(self.hidden(fc_out))
#         drop_hidden = self.dropout(hidden)
        out = self.outspace(fc_out) #dimensions 1 * output_size(hyper parameter)
        out_vals = F.softmax(out,dim=1)

        return out_vals

class Brain():
    def __init__(self,kernel_height,out_channels,dropout_prob,embedding_dim,vocab_size, output_size):
        self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.kernel_height = kernel_height
        self.dropout_prob = dropout_prob
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.model = CONVClassifier(kernel_height,out_channels,dropout_prob,embedding_dim, vocab_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters())

    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return idxs

    def classify(self, sentence):
        with t.no_grad():
           output = self.model(sentence)
        return output.max(1)[1].view(-1,1)

    def train(self, sentence, labels):
        sentence = t.tensor(sentence, dtype= t.long)
        labels = t.tensor(labels, dtype= t.long)
        self.model.zero_grad()
        output = self.model(sentence)
        labels = labels.unsqueeze(0)
        # print(output)
        # print(labels)

        loss= nn.NLLLoss()
        loss_out = loss(output, t.max(labels,1)[1])
        loss_out.backward()
        self.optimizer.step()


    def pre_trainer(self, inputs, labels, word_to_ix, epochs):
        labels_class = np.zeros(shape=(len(labels), 2))
        labels = labels.values
        for i in range(len(labels)):
            if labels[i] == 0:
                labels_class[i, 0] = 1
            else:
                labels_class[i, 1] = 1

        for epoch in range(epochs):

            for i in range(len(inputs)):
                sentence = self.prepare_sequence(inputs[i].split(), word_to_ix)
                self.model.zero_grad()
                self.train(sentence, labels_class[i])


