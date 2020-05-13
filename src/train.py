""" This file trains our model.
    @author: Abid Hossain
"""

import torch
import torchvision.transforms as transforms
import os
import sys
import pickle
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from vocabulary import Vocabulary
from data_loader import CocoDataset, coco_batch
from pycocotools.coco import COCO
import argparse
from model import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn


device = 'cuda'

def main():
    # Create model directory
    ##### arguments #####
    PATH = os.getcwd()
    image_dir = './data/resized2014/'
    caption_path = './data/annotations/captions_train2014.json'
    vocab_path = './data/vocab.pkl'
    model_path = './model'
    crop_size = 224
    batch_size = 128
    num_workers = 4
    learning_rate = 0.001

    # Decoder
    embed_size = 512
    hidden_size = 512
    num_layers = 3 # number of lstm layers
    num_epochs = 10
    start_epoch = 0
    save_step = 3000

    # transfer learning path to pretrained model, load state_dict
    #encoder_path = './model
    #decoder_path = './model


    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    coco = CocoDataset(image_dir, caption_path, vocab, transform)
    dataLoader = torch.utils.data.DataLoader(coco, batch_size, shuffle = True, num_workers = 4, collate_fn = coco_batch)


    # Declare the encoder decoder
    encoder = Encoder(embed_size = embed_size).to(device)
    decoder = Decoder(embed_size = embed_size, hidden_size = hidden_size, vocab_size = len(vocab), num_layers = num_layers).to(device)


    encoder.train()
    decoder.train()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    # For encoder only train the last fc layer
    optimizer = torch.optim.Adam(params, lr = learning_rate)

    # Train the models
    total_step = len(dataLoader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(dataLoader):
            # Set mini-batch dataset
            images = images.cuda()
            captions = captions.cuda()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()

            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if ('step' in state and state['step'] >= 1024):
                        state['step'] = 1000


            loss.backward(retain_graph = True)
            optimizer.step()

            # Print log info
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1+start_epoch, num_epochs+start_epoch, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}-{}.ckpt'.format(epoch+1+start_epoch, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-{}-{}.ckpt'.format(epoch+1+start_epoch, i+1)))

        print('epoch ', epoch+1 , 'loss: ', loss.item())


if __name__ == "__main__":
    print('Start Training')
    main()
