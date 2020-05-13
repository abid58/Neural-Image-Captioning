### This doc explains the functionality of each Python file.

##### 1. data_loader.py:
The coco_batch() function can be passed into collate_fn in torch.utils.data.DataLoader(). coco_batch() returns images, targets, and caption length in type of torch.LongTensor.


#### 2. resize.py:
This file is resize all training images in ./data/train2014 to (256, 256) and save it to ./data/resized2014/. We resize all images, so we can have consistent inputs.

#### 3. vocabulary.py:
This file generates vocabulary and add <<padding>>, <<start>>, <<end>>, <<unknown>>, token from captions in training dataset. This file saves vocab.pkl file in the current directory.
Vocabulary() class contains two dictionaries: word_to_index and index_to_word. Those two dictionaries help us to map captions to index, vice versa.


#### 4. model.py:
This file defines our model.
Encoder() uses the pre-trained resnet152. We fixed all previously trained parameters and only fine-tune on the last fully connect layer and batch normalization layer. The default dropout rate is set to 0.
Decoder() have two form; one using stateful LSTM with locked dropout and the other uses pytorch nn.LSTM function. Our best model is trained using pytorch nn.LSTM function.


5. train.py:
imports Encoder, Decoder from model.py, CocoDataset and coco_batch from data_loader.py, and Vocabulary from vocabulary.py.
This is the training file. Our training process saves a model for every epoch. Our code can load the pre-trained model state and continue training or tuning. The model used cross entropy loss to train and Adam optimizer for backward propagations All Hyper-parameters is defined at the beginning of the main() function.

#### 6. vocabulary.py
This file load trained encoder and decoder and generates output vectors, then the output is mapped to vocab.idx2word[] dictionary to generate caption. This python takes an image path as argument and prints caption.

python caption_generator.py --image='test_images/example.png'


#### 7. bleu.py:
bleu_score() compute bleu score of generated caption using 5 human captions as references and theoretical bleu score by taking the average of bleu score using each human caption as generated caption compares with other 4 human captions as references.


#### 8. model_scores.py
This file imports get_image_name and bleu_score from bleu.py.

##### testing BLEU-1,2,3,4
python model_bleu.py --eval='eval'
##### training BLEU-4
python model_bleu.py --eval='train'

### How to run 
