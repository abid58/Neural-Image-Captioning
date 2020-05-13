""" This module calculates BLEU-1, BLEU-2, BLEU-3, BLEU-4, CiDER, and ROGUE_L score.
    Author: Abid Hossain
"""
# BLEU = Bilingual Evaluation Understudy
# CIDEr = Consensus-based Image Description Evaluation
# ROGUE = Recall Oriented Understudy for Gisting Evaluation
#         -L: LCS

import json
import pandas as pd
import nltk
from nltk.metrics.scores import precision, recall
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def get_image_name(jsonPath):
    """ get imagename dataframe reference, jsonPath is the validation set json file path """
    with open(jsonPath, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    dataImage = pd.DataFrame.from_dict(data['images'])
    dataAnnotations = pd.DataFrame.from_dict(data['annotations'])
    dataName = pd.merge(dataImage,dataAnnotations, left_on='id',right_on='image_id')
    dataName = dataName[['file_name','caption']]
    return dataName



def bleu1_score(input_imgs_path, generated_captions, name_caption_frame):
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
    references = []

    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)

    candidates = generated_captions[1:-1]
    weights = (1.0,0.0,0.0,0.0)
    generated_score = sentence_bleu(references, candidates, weights, smoothing_function = SmoothingFunction().method4) # model's score

    theoratical_score = 0 # human score
    for i in range(5):
        theoratical_score += sentence_bleu(references[:i]+references[i+1:], references[i], weights, smoothing_function = SmoothingFunction().method4)
    theoratical_score /= 5.0

    return generated_score,theoratical_score



def bleu2_score(input_imgs_path, generated_captions, name_caption_frame):
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
    references = []

    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)

    candidates = generated_captions[1:-1]
    weights = (0.5, 0.5, 0.0, 0.0)
    generated_score = sentence_bleu(references, candidates, weights, smoothing_function = SmoothingFunction().method4) # model's score

    theoratical_score = 0 # human score
    for i in range(5):
        theoratical_score += sentence_bleu(references[:i]+references[i+1:], references[i], weights, smoothing_function = SmoothingFunction().method4)
    theoratical_score /= 5.0

    return generated_score,theoratical_score



def bleu3_score(input_imgs_path, generated_captions, name_caption_frame):
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
    references = []

    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)

    candidates = generated_captions[1:-1]
    weights = (1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0)
    generated_score = sentence_bleu(references, candidates, weights, smoothing_function = SmoothingFunction().method4) # model's score

    theoratical_score = 0 # human score
    for i in range(5):
        theoratical_score += sentence_bleu(references[:i]+references[i+1:], references[i], weights, smoothing_function = SmoothingFunction().method4)
    theoratical_score /= 5.0

    return generated_score,theoratical_score




def bleu4_score(input_imgs_path, generated_captions, name_caption_frame):
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
    references = []

    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)

    candidates = generated_captions[1:-1]
    weights = (0.25, 0.25, 0.25, 0.25,)
    generated_score = sentence_bleu(references, candidates, weights, smoothing_function = SmoothingFunction().method4) # model's score

    theoratical_score = 0 # human score
    for i in range(5):
        theoratical_score += sentence_bleu(references[:i]+references[i+1:], references[i], weights, smoothing_function = SmoothingFunction().method4)
    theoratical_score /= 5.0

    return generated_score,theoratical_score



def CIDEr(input_imgs_path, generated_captions, name_caption_frame):
    """ Not implemented """
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
    references = []

    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)

    candidates = generated_captions[1:-1]

    generated_score =  0

    theoratical_score = 0

    return generated_score,theoratical_score



def  ROGUE_L(input_imgs_path, generated_captions, name_caption_frame):
    """ Not implemented"""
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
    references = []

    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)

    candidates = generated_captions[1:-1]

    generated_score =  0

    theoratical_score = 0

    return generated_score,theoratical_score
