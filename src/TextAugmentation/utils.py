import pandas as pd
import numpy as np
import math
import re
import nltk
from nltk import word_tokenize
from pyvi import ViTokenizer
from simalign import SentenceAligner
from functools import reduce
nltk.download('punkt')

def custom_tokenize(document):
  x = word_tokenize(document)
  for ele in x:
    if re.search('\d{5}', ele):
      idx = x.index(ele)
      return x[:idx] + [_ for _ in ele] + x[idx+1:]
  return x


def build_aligner(sentences, aligned_model="xlm-mlm-100-1280", token_type="bpe", matching_methods="mai", device = 'cuda', layer = 8):
    aligner = SentenceAligner(model=aligned_model, token_type=token_type, matching_methods=matching_methods, device = device, layer = layer)
    sentence_alignments = []
    for i in range(len(sentences)):
        element = sentences[i]
        src_sentence = element[0]
        trg_sentence = element[1]
        alignment_values = aligner.get_word_aligns(src_sentence, trg_sentence)
        sentence_alignments.append(alignment_values['mwmf'])
    return sentence_alignments


def Sequential(*augMethods):
  def wrap(sentences, ba_vi_dict, loaded_alignments=None):
    # if not loaded_alignments:
    # if not ba_vi_dict:
    # loaded_alignments = []
    if loaded_alignments is None:
      print("Starting build aligner...")
      checkReplaceFunction = reduce(lambda base, x: base or ('replace' in x.__name__ or 'da_combine' in x.__name__), augMethods, False)

      if checkReplaceFunction:
        loaded_alignments = build_aligner(sentences)
    return reduce(lambda x, y: y(x, loaded_alignments, ba_vi_dict), augMethods, sentences)
  return wrap


