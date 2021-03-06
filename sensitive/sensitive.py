##
## A sense based sentiment analyzer
##
##
## borrows a lot from Vader
##
import sys
import math
import yaml
from importlib.resources import open_text
#import sensitive.wsd
from sensitive.wsd import pos2wn, disambiguate
import wn
from wn.morphy import Morphy

en = wn.Wordnet('omw-en:1.4')
morphy = Morphy(en)

###
### Changes to the scores based on
### * punctuation  DONE
### * capitalization  DONE
### * intensification  DONE
### * negation
### * conjunctions (contrastive)
### * comparatives, superlatives    # DONE, FIXME INCREMENTS
### * morphology based antonyms (hard)
###


###
### Constants
###

# # (empirically derived mean sentiment intensity rating increase for booster words)
# B_INCR = 0.293
# B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for booster words)
COMP_INCR = 0.3
SUPR_INCR = 0.6

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
#N_SCALAR = -0.74

### highlighters
### have a score for before and after
### e.g. (but,      0.5, 1.5) #VADER, SOCAL 1, 2
###      (although, 1.0, 0.5) #SOCAL

contrasts = {'but', 'however'}

# yet
# nevertheless
# nonetheless
# even so
# however
# still
# notwithstanding
# despite that
# in spite of that
# for all that
# all the same
# just the same
# at the same time
# be that as it may
# though
# although
# still and all

#@staticmethod
def increment(valence, increment):
    """
    increment in the same direction as the valence
    """
    if valence == 0.0:
        return valence
    elif valence > 0:
        return valence + increment
    else: # valence < 0
        return valence - increment

#@staticmethod
def stretch(valence, increment):
    """
    stretch the valence
    """
    return valence * increment   

def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score

def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different

def _sift_sentiment_scores(sentiments):
    # want separate positive versus negative sentiment scores
    pos_sum = 0.0
    neg_sum = 0.0
    neu_count = 0
    for sentiment_score in sentiments:
        if sentiment_score > 0:
            pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
        elif sentiment_score < 0:
            neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
        else: #  sentiment_score == 0:
            neu_count += 1
    return pos_sum, neg_sum, neu_count

def score_valence(sentiments, punct_emph_amplifier):
    if sentiments:
        sum_s = float(sum(sentiments))
        # compute and add emphasis from punctuation in text
        sum_s = increment(sum_s, punct_emph_amplifier)

        compound = normalize(sum_s)
        # discriminate between positive, negative and neutral sentiment scores
        pos_sum, neg_sum, neu_count = _sift_sentiment_scores(sentiments)

        if pos_sum > math.fabs(neg_sum):
            pos_sum += punct_emph_amplifier
        elif pos_sum < math.fabs(neg_sum):
            neg_sum -= punct_emph_amplifier

        total = pos_sum + math.fabs(neg_sum) + neu_count
        pos = math.fabs(pos_sum / total)
        neg = math.fabs(neg_sum / total)
        neu = math.fabs(neu_count / total)

    else:
        compound = 0.0
        pos = 0.0
        neg = 0.0
        neu = 0.0

    sentiment_dict = \
        {"neg": round(neg, 3),
         "neu": round(neu, 3),
         "pos": round(pos, 3),
         "compound": round(compound, 4)}

    return sentiment_dict

    
def _amplify_ep(text):
    """
    check for added emphasis resulting from exclamation points (up to 4 of them)
    """
    ep_count = text.count("!")
    if ep_count > 4:
        ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
    ep_amplifier = ep_count * 0.292
    return ep_amplifier

def _amplify_qm(text):
    """
    check for added emphasis resulting from question marks (2 or 3+)
    """
    qm_count = text.count("?")
    qm_amplifier = 0
    if qm_count > 1:
        if qm_count <= 3:
            # (empirically derived mean sentiment intensity rating increase for
            # question marks)
            qm_amplifier = qm_count * 0.18
        else:
            qm_amplifier = 0.96
    return qm_amplifier

def _but_check(senses, sentiments): #could not really understand the code from original VADER and tried to import to there with changes but no change in score in new test
    # check for modification in sentiment due to contrastive conjunction 'but'
    for i, (w, p, l, t)  in enumerate(senses):
        if l in contrasts:
            for j, sentiment in enumerate(sentiments):
                if j < i:
                    sentiments[j]= sentiment * 0.5
                elif j > i:
                    sentiments[j] = sentiment * 1.5
    return sentiments

def punctuation_emphasis(text):
    # add emphasis from exclamation points and question marks
    ep_amplifier = _amplify_ep(text)
    qm_amplifier = _amplify_qm(text)
    punct_emph_amplifier = ep_amplifier + qm_amplifier
    return punct_emph_amplifier


class SentimentAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """
    def __init__(self, model="en_sense"):
        modpath = f"{__package__}.models.{model}"
        datapath = f"{__package__}.data"
        
        self.meta = self.read_meta(modpath, 'meta.yaml')

        ### Valence lexicons
        self.lexicon = dict()

        for lexfile in self.meta['lexicons']:
             self.lexicon.update(self.make_lex_dict(modpath, lexfile))

        self.wlexicon = dict()

        for lexfile in self.meta['wlexicons']:
             self.wlexicon.update(self.make_lex_dict(modpath, lexfile))


        ### Boosters
        self.booster = dict()

        for lexfile in self.meta['boosters']:
             self.booster.update(self.make_lex_dict(modpath, lexfile))

        
        ### Negators
        self.negator = dict()

        for lexfile in self.meta['negators']:
             self.negator.update(self.make_lex_dict(modpath, lexfile))

        self.wnegator = dict()

        for lexfile in self.meta['wnegators']:
             self.wnegator.update(self.make_lex_dict(modpath, lexfile))

        ### make sure boosters and negators carry no sentiment     
        for lex in self.lexicon:                #played with this but only managed to get scores from sensitive.
            if lex in self.negator or lex in self.booster:
                #print('setting to zero:', lex)
                self.lexicon[lex] = 0.0
             
        print(f"""loaded model {model}: 
Lexicons: {self.meta['lexicons']} ({len(self.lexicon):,d} concepts)
Word Lex: {self.meta['wlexicons']} ({len(self.wlexicon):,d} concepts)
Boosters: {self.meta['boosters']} ({len(self.booster):,d} concepts)
Negators: {self.meta['negators']} ({len(self.negator):,d} concepts)""")

        
    def read_meta(self, modpath, meta_file):
        """
        Read meta parameters for the model

        """
        with open_text(modpath, meta_file) as metafh:
            meta = yaml.safe_load(metafh)
        return meta
       

    def make_lex_dict(self, modpath, lexicon_file):
        """
        Convert lexicon file to a dictionary
        Expect a tab separated lexicon
        lemma	score	rest

        Allow comments with hashes
        """
        lex_dict = {}
        fh = open_text(modpath, lexicon_file)
        print(modpath, lexicon_file)
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            row =  line.split('\t')
            if len(row) >= 2:
                (word, measure) = line.strip().split('\t')[0:2]
                lex_dict[word] = float(measure)
        return lex_dict

    
    def lexical_valence(self, w, p, l, t):
        """
        find the lexical valence 
        apply any morphological changes
        """

        if t in self.lexicon:
            valence = self.lexicon[t]

        
        if valence:
            if p == 'JJR':  # comparative
                valence = increment(valence, COMP_INCR)
            elif p == 'JJS':  # superlative
                valence = increment(valence, SUPR_INCR)
 
        return valence
                
    
    def sentiment_valence(self, i, senses, is_cap_diff):
        valence = 0.0
        (w, p, l, t) = senses[i]
        ### get the base valence, with morphological changes
        if t in self.lexicon:
            valence = self.lexical_valence(w, p, l, t)
            ## or get the word lexicon, if we don't know the sense
        elif w in self.wlexicon:
            valence = self.wlexicon[w]
    

        ### CAPITALIZATION
        if valence and is_cap_diff and \
           w.isupper() and not l.isupper():
            valence = increment(valence, C_INCR)
            
        return valence


    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """  
        senses = disambiguate(text, en, morphy)
        print(senses)
        is_cap_diff = allcap_differential([w for (w, p, l, t) in senses])
        ### pad with beginners?

        sentiments = list()

        for i, (w, p, l, t)  in enumerate(senses):
            # see if this word carries sentiment
            local = self.sentiment_valence(i, senses, is_cap_diff)
            if not local: ### if not, keep going
                sentiments.append(0.0)
            else:
                # check for boosts and negation
                for j in range(1,4): # check back three words
                    if i - j < 0:    # if the word exists 
                        continue
                    #print(i, senses[i], senses[i-j])
                    mod = senses[i-j][3] # This is the concept id
                    if mod in  self.booster:
                        local = increment(local,
                                          self.booster[mod]*(1.05 - j*.05))
                    ### diminish a little further back: 1, 0.95, 0.90
                for j in range(1,4): # check back three words
                    if i - j < 0:    # if the word exists 
                        continue
                    #print(i, senses[i], senses[i-j])
                    mod_c = senses[i-j][3] # This is the concept id
                    mod_w = senses[i-j][0] # This is the word
                    if mod_c in  self.negator:
                        local = stretch(local, self.negator[mod_c])
                    elif mod_w in  self.wnegator:
                        local = stretch(local, self.wnegator[mod_w])
                        
                sentiments.append(local)

        ### check punctuation of the whole sentence
        punct_score = punctuation_emphasis(text)
        #print(sentiments)
        _but_check(senses, sentiments)
        print(sentiments)
        valence_dict = score_valence(sentiments, punct_score)


        return valence_dict
                

    
# if __name__ == '__main__':
#     sentences = ["VADER is smart, handsome, and funny.",
#                  "We have some problems."]
#     analyzer = SentimentAnalyzer()
