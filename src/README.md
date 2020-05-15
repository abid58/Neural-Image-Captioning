## This doc explains the functionality of each Python file.

#### 1. data_loader.py:
This file contains data loader for coco dataset. This file is called during training to load data as coco batch.

#### 2. resize.py:
This file is resize all training images in ./data/train2014 to (256, 256) and save it to ./data/resized2014/. We resize all images, so we can have consistent inputs.

#### 3. vocabulary.py:
This file generates vocabulary and add <<padding>>, <<start>>, <<end>>, <<unknown>>, token from captions in training dataset. This file saves vocab.pkl file in data directory.

#### 4. model.py:
This file defines our model. It contains two class Encoder() and Decoder(). 

#### 5. train.py:
This file imports Encoder, Decoder from model.py, CocoDataset and coco_batch from data_loader.py, and Vocabulary from vocabulary.py. This is the training file. All Hyper-parameters is defined at the beginning of the main() function.

#### 6. caption_generator.py
This file load trained encoder and decoder and generates output vectors, then the output is mapped to vocab.idx2word[] dictionary to generate caption. This python takes an image path as argument and prints caption.

#### 7. bleu.py:
bleu_score() compute bleu-1,2,3,4 scores of generated caption using 5 human captions as references and theoretical bleu score by taking the average of bleu score using each human caption as generated caption compares with other 4 human captions as references. It also calculates model bleu scores of a generated caption. 

#### 8. model_scores.py
This file imports get_image_name and bleu_4(), bleu_3(), bleu_2(), and bleu_1() from bleu.py and calculates average bleu scores on training and validation data. 



## How to run: 

 ### Requirements:
      python/2.0.1
      cudatoolkit
      pytorch (compatible version with python 2.0.1) 

#### run 'sh download_data.sh'
#### copy 'vocab.pkl' in 'data' folder: 
      cp ./vocab.pkl ./data/
#### Image resizing: 
      python resize.py
#### Training: 
      python train.py
#### generating example caption: 
      python caption_generator.py --image='test_images/example.jpg'
#### testing BLEU-1,2,3,4
      python model_scores.py --eval='eval'
#### training BLEU-1,2,3,4
      python model_scores.py --eval='train'
 
 
