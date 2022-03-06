import nltk
import fileinput
#import string
#import re
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sensitive
from sensitive.sensitive import SentimentAnalyzer
from scipy import stats


datadir = '/home/bond/git/vadermulti/additional_resources/hutto_ICWSM_2014/'

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
      score = (int(p_score) - int(n_score))/4
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
   #v_analyzer = SentimentIntensityAnalyzer
   for l in base_file:
      index, score, sentence = l.strip().split('\t')
      e_score = analyzer.polarity_scores(sentence)['compound']
      #v_score = v_analyzer.polarity_scores(sentence)['compound']
      gold= float(score)/4
      score_list.append(gold)
      e_score_list.append(e_score)
      if abs(gold-e_score) > 1:
         print('BAD', index, gold, f'{e_score:.3f}',
         sentence,
         sep='\t')
      base_data.append((score, e_score, sentence))
   #print(base_data) #output sentiment scores
   
   
   print(evalfile, stats.pearsonr(score_list, e_score_list), stats.spearmanr(score_list, e_score_list, axis=0, nan_policy='propagate'))	


files = {'Amazon Reviews': 'amazonReviewSnippets_GroundTruth.txt',
         'Movie Reviews': 'movieReviewSnippets_GroundTruth.txt',
         'NYT Editorials':'nytEditorialSnippets_GroundTruth.txt',
         'Tweets':'tweets_GroundTruth.txt'
}
   
for n,f in files.items():
   print(n)
   eval_vader(f'{datadir}{f}')


