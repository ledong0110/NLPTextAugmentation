__all__ = ['da_eda']

import pandas as pd
import numpy as np
import re
from nltk import regexp_tokenize
import random
from random import shuffle
from .utils import *


def da_eda_src(proportion, num_aug, synonym_dictionary, stopwords, keep_old=False):
    def wrap(sentences, *args):
        alpha = proportion
        da_pairs = []
        for i in range(len(sentences)):
            tgt_sentence = sentences[i][1]
            src_sentence = sentences[i][0]

            augmented_src_sentences = eda(src_sentence, synonym_dictionary, stopwords,alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
            augmented_sentences = list(set(augmented_tgt_sentences))
            for augmented_sentence in augmented_sentences:
                da_pairs.append([augmented_sentence, tgt_sentence])
        
        if keep_old:
            da_pairs.extend(sentences)
            
        
        return da_pairs
    return wrap

def da_eda_tgt(proportion, num_aug, synonym_dictionary, stopwords, keep_old=False):
    def wrap(sentences, *args):
        alpha = proportion
        da_pairs = []
        for i in range(len(sentences)):
            tgt_sentence = sentences[i][1]
            src_sentence = sentences[i][0]

            augmented_tgt_sentences = eda(tgt_sentence, synonym_dictionary, stopwords,alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
            augmented_sentences = list(set(augmented_tgt_sentences))
            for augmented_sentence in augmented_sentences:
                da_pairs.append([src_sentence, augmented_sentence])
        
        if keep_old:
            da_pairs.extend(sentences)
            
        
        return da_pairs
    return wrap



def synonym_replacement(words, n, synonym_dictionary, stop_words):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word, synonym_dictionary)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(term, synonym_dictionary):
	for ele in synonym_dictionary:
		if term in ele:
			return [_ for _ in ele if _ != term]
	return []

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, synonym_dictionary):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words, synonym_dictionary)
	return new_words

def add_word(new_words, synonym_dictionary):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word, synonym_dictionary)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, synonym_dictionary, stop_words, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	words = sentence.split(' ')
	words = [word for word in words if word != '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4) + 1

	#sr
	if (alpha_sr > 0):
		n_sr = max(1, int(alpha_sr*num_words))
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr, synonym_dictionary, stop_words)
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri, synonym_dictionary)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [sentence for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	# augmented_sentences.append(sentence)
 
	return augmented_sentences