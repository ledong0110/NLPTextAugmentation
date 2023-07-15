from TextAugmentation.MTLCombination import da_combine
from TextAugmentation.Single import da_replaceWord, da_replaceToken, da_swap, da_reverse, da_source
from TextAugmentation.SentenceBoundary import da_segmentation
from TextAugmentation.Word2Vec import da_w2v_src, da_w2v_tgt
from TextAugmentation.EDA import da_eda_src, da_eda_tgt
from TextAugmentation.utils import Sequential
import pandas as pd
import pickle

if __name__ == '__main__':
    

  # Load senetences
  train_new_all_ba_df = pd.read_csv('data/train.ba', sep="/n", header=None, index_col = False, engine='python')
  train_new_all_vi_df = pd.read_csv('data/train.vi', sep="/n", header=None, index_col = False, engine='python')

  N = len(train_new_all_ba_df)
  sentences = []
  for i in range(N):
    src_sentence = train_new_all_vi_df.iloc[i][0]
    tgt_sentence = train_new_all_ba_df.iloc[i][0]
    sentences.append([src_sentence, tgt_sentence])
    


  # Load alignment
  ALIGN_PATH = 'data/'

  alignment_0_8792 = []
  with open(ALIGN_PATH+'aligner_train_updated_0_8792.pkl', 'rb') as f:
    alignment_0_8792 = pickle.load(f)

  alignment_8793_15163 = []
  with open(ALIGN_PATH+'aligner_train_updated_8793_15163.pkl', 'rb') as f:
    alignment_8793_15163 = pickle.load(f)
    
  alignment_15164_16105 = []
  with open(ALIGN_PATH+'aligner_train_updated_15164_16105.pkl', 'rb') as f:
    alignment_15164_16105 = pickle.load(f)
    
  loaded_alignments = alignment_0_8792 + alignment_8793_15163 + alignment_15164_16105

  # Load ba vi dict
  dict_ba_series = train_new_all_ba_df[9275:15161][0]
  dict_vi_series = train_new_all_vi_df[9275:15161][0]
  ba_vi_dict = pd.concat({'tgt': dict_ba_series,
                'src': dict_vi_series}, axis=1)

  # Test
      # Multi-task learning custome
  # mtlCombine = da_combine(proportion=0.5)
  # augmentSentences = mtlCombine(sentences, loaded_alignments, ba_vi_dict)
  # print(augmentSentences)


  #     # Single Method + Sentence Boundary
  # # replaceMethod = da_segmentation(proportion=0.6)
  # # augmentedSentences = replaceMethod(sentences)
  # # print(augmentedSentences)

  #     # Word2Vec
  # replaceMethod = da_w2v_src(proportion=0.6, src_model_path='data/vi-model-CBOW.bin')
  # replaceMethod = da_w2v_tgt(proportion=0.6, tgt_model_path='data/ba-model-CBOW.bin')
  # augmentedSentences = replaceMethod(sentences)
  # print(augmentedSentences)

  #     # EDA bahnar
  # # Build synonym dictionary
  origin_data = pd.read_excel('data/vietbana_dict_v110.xlsx')
  dataframe = origin_data.drop(['Id', 'Vietnamese'], axis=1)
  dataframe = dataframe.applymap(lambda s:s.lower() if isinstance(s, str) else s)
  synonym_dict = []
  for index, row in dataframe.iterrows():
      item = []
      for col in ['BinhDinh', 'Kontum1']:
        value = row[col]
        if value != value: #skip NaN value
          continue
        else:
          # value = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", value)
          value = str(value)
          value = value.replace('/', ',').strip()
          value = value.replace('(', ',').strip()
          value = value.replace(')', '').strip()
          # value = value.replace('?', '').strip()
          if ',' in value:
            elements = value.split(',')
            for element in elements:
              if element not in item and element != '':
                item.append(element)
          else:
            if value not in item and value != '':
              item.append(value)
      
      if (len(item) > 1):
        synonym_dict.append(item)

  # # Get bahnar stopwords
  # dict_ba_series = train_new_all_ba_df[9275:15161][0]
  # dict_vi_series = train_new_all_vi_df[9275:15161][0]

  # ba_vi_dict = pd.concat({'Bahnaric': dict_ba_series,
  #               'Vietnamese': dict_vi_series}, axis=1)

  vi_stop_word_df = pd.read_csv('data/vietnamese_stopwords.txt', sep="/n", header=None, index_col = False, engine='python')  
  dict_vi_list = dict_vi_series.tolist()
  dict_ba_list = dict_ba_series.tolist()
  ba_stop_words = []

  for word in vi_stop_word_df[0]:
    for idx in range(len(dict_ba_series)):
      if word == dict_vi_list[idx]:
        ba_stop_words.append(dict_ba_list[idx])

  # # # Apply EDA

  # # replaceMethod = da_eda(proportion=0.6, num_aug=2, synonym_dictionary=synonym_dict, stopwords=ba_stop_words)
  # # augmentedSentences = replaceMethod(sentences)
  # # print(augmentedSentences)

  # # Compose multiple methods

  seqApply = Sequential(
      da_replaceWord(proportion=0.9),
      da_replaceToken(proportion=0.6, keep_old=True),
      # da_reverse(keep_old=True),
      da_eda_tgt(proportion=0.3, num_aug=2, synonym_dictionary=synonym_dict, stopwords=ba_stop_words),
      # da_segmentation(proportion=0.2, keep_old=True),
  )
  print(seqApply(sentences, ba_vi_dict, loaded_alignments))