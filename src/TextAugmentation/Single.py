from .utils import *
import random
import pandas as pd
import numpy as np
import math
import re


def da_replaceWord(proportion, keep_old=False):
    def replace_wrap(sentences, loaded_alignments, src_tgt_dict):
        da_pairs = []
        if len(loaded_alignments) < len(sentences):
            loaded_alignments.clear()
            loaded_alignments.extend(build_aligner(sentences=sentences))
        for i in range(len(sentences)):
            ttokens = custom_tokenize(sentences[i][1])
            stokens = custom_tokenize(sentences[i][0])
            alignment = loaded_alignments[i]
            
            n_words_replace=int(len(ttokens)*proportion)
            alg_positions = random.sample(range(len(ttokens)), n_words_replace)
            lexicon_positions = random.sample(range(len(src_tgt_dict)), n_words_replace)

            for alg_pos,lex_pos in zip(alg_positions,lexicon_positions):
                tgt_word, str_src_words = src_tgt_dict.iloc[lex_pos]['tgt'], src_tgt_dict.iloc[lex_pos]['src']
                ls_src_words = str_src_words.split(',')
                src_word = ls_src_words[0]
                src_word = src_word.split('-', 1)[0]
                ttokens[alg_pos] = tgt_word
                for alg_tuple in alignment:
                    if (alg_tuple[1] == alg_pos):
                        stokens[alg_tuple[0]] = src_word

            t_out_str = ' '.join(ttokens)
            s_out_str = ' '.join(stokens)
            da_pairs.append([s_out_str, t_out_str])
        if keep_old:
            loaded_alignments.extend(loaded_alignments)
            da_pairs.extend(sentences)
        return da_pairs
    return replace_wrap

def da_replaceToken(proportion, keep_old=False):
    def replace_wrap(sentences, loaded_alignments, src_tgt_dict): 
        if len(loaded_alignments) < len(sentences):
            loaded_alignments.clear()
            loaded_alignments.extend(build_aligner(sentences=sentences))
        da_pairs = []
        for i in range(len(sentences)):
            ttokens = custom_tokenize(sentences[i][1])
            stokens = custom_tokenize(sentences[i][0])
            alignment = loaded_alignments[i]
            
            n_words_replace = int(len(ttokens)*proportion)
            alg_positions = random.sample(range(len(ttokens)), n_words_replace)
            lexicon_positions = random.sample(range(len(src_tgt_dict)), n_words_replace)

            for alg_pos,lex_pos in zip(alg_positions,lexicon_positions):
                tgt_word, str_src_words = src_tgt_dict.iloc[lex_pos]['tgt'], src_tgt_dict.iloc[lex_pos]['src']
                ls_src_words = str_src_words.split(',')
                src_word = ls_src_words[0]
                src_word = src_word.split('-', 1)[0]
                ttokens[alg_pos] = tgt_word
                for alg_tuple in alignment:
                    if (alg_tuple[1] == alg_pos):
                            stokens[alg_tuple[0]] = src_word

            n_words_token = n_words_replace
            tgt_positions =  random.sample(range(len(ttokens)), n_words_token)
            for position in tgt_positions:
                ttokens[position] = 'UNK'

            t_out_str = ' '.join(ttokens)
            s_out_str = ' '.join(stokens)
            da_pairs.append([s_out_str, t_out_str])
        if keep_old:
            loaded_alignments.extend(loaded_alignments)
            da_pairs.extend(sentences)
        return da_pairs
    return replace_wrap

def da_swap(proportion, keep_old=False):
    def wrap(sentences, *args):
        #NOTE loaded alignments and ba_vi_dict are not used
        da_pairs = []
        for i in range(len(sentences)):
            moved_pos=set()
            tokens = custom_tokenize(sentences[i][1])
            n_words_swap=int(len(tokens)*proportion)
            while(len(moved_pos)<n_words_swap):
                pos1, pos2 = random.sample(range(len(tokens)), 2)
                tokens[pos1], tokens[pos2] = tokens[pos2], tokens[pos1]
                moved_pos.add(pos1)
                moved_pos.add(pos2)
            t_out_str = ' '.join(tokens)
            da_pairs.append([sentences[i][0], t_out_str])
        if keep_old:
            da_pairs.extend(sentences)
        return da_pairs
    return wrap


def da_reverse(proportion=1, keep_old=False):
    def wrap(sentences, *args):
        da_pairs = []
        for i in range(len(sentences)):
            t = custom_tokenize(sentences[i][1])
            t.reverse()
            t_out_str = ' '.join(t)
            da_pairs.append([sentences[i][0], t_out_str])
        if keep_old:
            da_pairs.extend(sentences)
        return da_pairs
    return wrap


def da_source(keep_old=False):
    def wrap(sentences, *args):
        da_pairs = [[sentences[i][0], sentences[i][0]] for i in range(len(sentences))]
        if keep_old:
            da_pairs.extend(sentences)
        return da_pairs

    return wrap