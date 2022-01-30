import nltk
import fileinput
#import string
#import re
from vaderSentiment import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sensitive
from sensitive.sensitive import SentimentAnalyzer
from scipy import stats

#os.path

def eval_senti(evalfile):
   base_data = []
   score_list = []
   e_score_list = []
   base_file = open(evalfile).readlines()
   analyzer = SentimentAnalyzer()
   for l in base_file:
      p_score, n_score, sentence = l.strip().split('\t')
      e_score = analyzer.polarity_scores(sentence)['compound']
      score = (int(p_score) - int(n_score))/5
      score_list.append(float(score))
      e_score_list.append(e_score)
      base_data.append((score, e_score, sentence))
   #print(base_data) #output sentiment scores
   print(evalfile)
   print(stats.pearsonr(score_list, e_score_list), stats.spearmanr(score_list, e_score_list, axis=0, nan_policy='propagate'))	


#eval_senti('6humanCodedDataSets\SS_1041MySpace.txt')
#eval_senti('6humanCodedDataSets\SS_twitter4242.txt')
#eval_senti('6humanCodedDataSets\YouTube3407.txt')

def eval_vader(evalfile):
   base_data = []
   diff_list = []
   score_list = []
   e_score_list = []
   base_file = open(evalfile).readlines()
   analyzer = SentimentAnalyzer()
   v_analyzer = SentimentIntensityAnalyzer
   for l in base_file:
      index, score, sentence = l.strip().split('\t')
      e_score = analyzer.polarity_scores(sentence)['compound']
      v_score = v_analyzer.polarity_scores(sentence)['compound']
      score_list.append(float(score))
      e_score_list.append(e_score)
      base_data.append((score, e_score, v_score, sentence))
      if abs(e_score - v_score) > 0.5:
         diff_list.append(sentence)
   print(base_data) #output sentiment scores
   print(diff_list)
   print(evalfile)
   print(stats.pearsonr(score_list, e_score_list), stats.spearmanr(score_list, e_score_list, axis=0, nan_policy='propagate'))	
   
eval_vader('vaderSentiment\_additional_resources\hutto_ICWSM_2014\_amazonReviewSnippets_GroundTruth.txt') 
eval_vader('vaderSentiment\_additional_resources\hutto_ICWSM_2014\movieReviewSnippets_GroundTruth.txt')
eval_vader('vaderSentiment\_additional_resources\hutto_ICWSM_2014\_nytEditorialSnippets_GroundTruth.txt')
eval_vader('vaderSentiment\_additional_resources\hutto_ICWSM_2014\_tweets_GroundTruth.txt')


