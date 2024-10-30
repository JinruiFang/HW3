import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


random.seed(0)

def synonym_replacement(text, prob=0.2):
    words = word_tokenize(text)
    new_words = words.copy()

    for i, word in enumerate(words):
        if random.random() < prob:
            # Find synonyms for the word using WordNet
            synonyms = wordnet.synsets(word)
            if synonyms:
                lemmas = synonyms[0].lemmas()
                if lemmas:
                    synonym = lemmas[0].name().replace('_', ' ')
                    new_words[i] = synonym

    return TreebankWordDetokenizer().detokenize(new_words)

def typo_introduction(text, prob=0.1):
    qwerty_neighbors = {
        'a': ['s', 'q', 'w', 'z'],
        'e': ['w', 'r', 'd'],
        'i': ['u', 'o', 'k'],
        'o': ['i', 'p', 'l'],
        'u': ['y', 'i', 'j'],
    }
    
    words = word_tokenize(text)
    new_words = words.copy()

    for i, word in enumerate(words):
        if random.random() < prob:
            # Introduce typos by replacing vowels with nearby keys
            new_word = list(word)
            for j, char in enumerate(new_word):
                if char in qwerty_neighbors and random.random() < 0.5:
                    new_word[j] = random.choice(qwerty_neighbors[char])
            new_words[i] = ''.join(new_word)

    return TreebankWordDetokenizer().detokenize(new_words)

def custom_transform(example):
    text = example["text"]

    text = synonym_replacement(text, prob=0.3)

    text = typo_introduction(text, prob=0.3)

    # Update the example with the transformed text
    example["text"] = text

    return example