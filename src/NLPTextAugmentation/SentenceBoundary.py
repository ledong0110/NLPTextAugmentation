from .utils import *
import random
import numpy as np
import math


def da_segmentation(proportion, keep_old=False):
    def wrap(sentences, *args):
        da_pairs = []
        for i in range(0, len(sentences)-1, 2):
            uniform_p = np.random.uniform(high=proportion)
            s1, t1, s2, t2 = custom_tokenize(sentences[i][0]), custom_tokenize(sentences[i][1]), custom_tokenize(sentences[i+1][0]), custom_tokenize(sentences[i+1][1])
            p1S1, p1T1 = math.ceil(uniform_p*len(s1)), math.ceil(uniform_p*len(t1))
            p2S2, p2T2 = math.ceil(uniform_p*len(s2)), math.ceil(uniform_p*len(t2))
            s_out = s1[p1S1:] + s2[:p2S2]
            s_out_str = ' '.join(s_out)
            t_out = t1[p1T1:] + t2[:p2T2]
            t_out_str = ' '.join(t_out)
            da_pairs.append([s_out_str, t_out_str])
        if keep_old:
            da_pairs.extend(sentences)
        return da_pairs
    return wrap