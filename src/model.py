"""
Final Project: Neural Image Captioning.
This file contains the CNN-RNN model, the engine of this project.
@author: Abid Hossain
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    Encoder:
        Use the pretrained resnet152 replace the last fc layer and re-train the last fc layer.
    """

    def __init__(self, attention = False, embed_size = 256, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.attention = attention # there's way to implement attention. Focusing on most important part of image
        resnet = models.resnet152(pretrained = True)
        self.enc_image_size = encoded_image_size
        self.resnet = resnet
        # change the output dimension of the last fc and only train for the last layer's weight and bias
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum = 0.01)

    def forward(self, images):
        """
        Forward propagation.
        @param
            images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        return:
            image embeddings: (batch_size, embed_size)
        """
        out = self.resnet(images)  # (batch_size, embed_size)
        out = self.bn(out)
        return out



class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, stateful = False, max_seq_length = 20):
        super(Decoder, self).__init__()
        self.stateful = stateful
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size) # vocab_size: size of dict of embedding , embed_size: size of each embedding(512)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True, dropout = 0.5) # num_layers = 3
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths,train=True):
        """Decode image feature vectors and generates captions."""
        self.train = train
        embeddings = self.embed(captions)  # generate embeddings of all 5 captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        #outputs = self.softmax(outputs)
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
