import torch
import torchvision.transforms as transforms
import os
import sys
import pickle
import numpy as np
import pandas as pd
import nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from vocabulary  import Vocabulary
from data_loader import CocoDataset, coco_batch
from pycocotools.coco import COCO
import argparse
from model import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from bleu import get_image_name, bleu1_score, bleu2_score, bleu3_score,  bleu4_score


device = 'cuda'


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def generate_caption(image_path, vocab, encoder, decoder, transform):
    '''
    This is the similary function as in the caption_generator.py
    '''
    # Image preprocessing
    # In generation phase, we need should random crop, just resize

    # Prepare an image
    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<<end>>':
            break

    return sampled_caption


def test(args):
    '''
    compute bleu score on all images, and average its bleu score
    '''
    train_json_path = './data/annotations/captions_train2014.json'
    test_json_path = './data/annotations/captions_val2014.json'
    train_image_dir = './data/train2014'
    test_image_dir = './data/val2014'

    if args.eval=='eval':
        print('eval bleu')
        jsonPath = test_json_path
        image_dir = test_image_dir
    else:
        print('train bleu')
        jsonPath = train_json_path
        image_dir = train_image_dir

    # Image preprocessing
    # In generation phase, we need should not random crop, just resize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    # Load vocabulary wraper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build model
    encoder = Encoder(embed_size=args.embed_size).eval()
    decoder = Decoder(stateful=False, embed_size=args.embed_size, hidden_size=args.hidden_size, vocab_size=len(vocab), num_layers=args.num_layers).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location = device))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location = device))

    name_caption_frame = get_image_name(jsonPath)
    unique_image_names = pd.unique(name_caption_frame['file_name'])

    # Add image directory path train2014 or val2014
    unique_image_names = [os.path.join(image_dir, image_name) for image_name in unique_image_names]

    total_generated_score4 = []
    total_theoratical_score4 = []

    total_generated_score3 = []
    total_theoratical_score3 = []

    total_generated_score2 = []
    total_theoratical_score2 = []

    total_generated_score1 = []
    total_theoratical_score1 = []



    # Parallelize the process
    def score_helper(image_path):
        caption = generate_caption(image_path, vocab, encoder, decoder, transform)

        generated_score4, theoratical_score4 = bleu4_score(image_path, caption, name_caption_frame)
        total_generated_score4.append(generated_score4)
        total_theoratical_score4.append(theoratical_score4)

        generated_score3, theoratical_score3 = bleu3_score(image_path, caption, name_caption_frame)
        total_generated_score3.append(generated_score3)
        total_theoratical_score3.append(theoratical_score3)

        generated_score2, theoratical_score2 = bleu2_score(image_path, caption, name_caption_frame)
        total_generated_score2.append(generated_score2)
        total_theoratical_score2.append(theoratical_score2)

        generated_score1, theoratical_score1 = bleu1_score(image_path, caption, name_caption_frame)
        total_generated_score1.append(generated_score1)
        total_theoratical_score1.append(theoratical_score1)




    _ = pd.Series(unique_image_names).apply(score_helper)

    print('Average bleu-4 score:', sum(total_generated_score4) / len(total_generated_score4),
          ' | Average theoratical bleu-4 score:', sum(total_theoratical_score4)/len(total_theoratical_score4))

    print('Average bleu-3 score:', sum(total_generated_score3) / len(total_generated_score3),
          ' | Average theoratical bleu-3 score:', sum(total_theoratical_score3)/len(total_theoratical_score3))

    print('Average bleu-2 score:', sum(total_generated_score2) / len(total_generated_score2),
          ' | Average theoratical bleu-2 score:', sum(total_theoratical_score2)/len(total_theoratical_score2))

    print('Average bleu-1 score:', sum(total_generated_score1) / len(total_generated_score1),
          ' | Average theoratical bleu-1 score:', sum(total_theoratical_score1)/len(total_theoratical_score1))




if __name__ == '__main__':

    PATH = './'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_dir', type=str, required=True, help='input image directory for generating bleu score')
    parser.add_argument('--eval', type=str, required=True, help='Test bleu or train bleu?')
    parser.add_argument('--encoder_path', type=str, default=PATH + 'model/encoder-10-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default=PATH + 'model/decoder-10-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default=PATH + 'data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers in lstm')
    args = parser.parse_args()
    test(args)
