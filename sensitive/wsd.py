import nltk
#from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer 
from nltk.tag import pos_tag

t = TweetTokenizer()


mfs_tweak = {'fantastic':'i9186'}

def pos2wn (pos):
    """Take PTB POS and return wordnet POS
       a, v, n, r, x or None 'don't tag'
       z becomes z

    FIXME: add language as an attribute
    """
    if pos in "CC EX IN MD POS RP VAX TO".split():
        return None
    elif pos in "PDT DT JJ JJR JJS PRP$ WDT WP$".split():
        return 'a'
    elif pos in "RB RBR RBS WRB".split():
        return 'r'
    elif pos in "VB VBD VBG VBN VBP VBZ".split():
        return 'v'
    elif pos == "UH": # or titles
        return 'x'
    elif pos == "z": # phrase
        return 'z'
    else: #   CD NN NNP NNPS NNS PRP SYM WP
        return 'n';

def w2lemma(word, tag, morphy):
    """
    Given a word and a tag, return the lemma
    """
    pos = pos2wn(tag)
    if pos and (pos in 'avnr'):
        lemmas = morphy(word, pos)
        ### should take most frequent
        if len(lemmas) > 0:
            return lemmas[pos].pop()
        else:
            lemmas = morphy(word.lower(), pos)
            if len(lemmas) > 0:
                return lemmas[pos].pop()
            else:
                return word
    else:
        return word

def lemmatize(tagged, morphy):
    """
    do a tagged sentence
    """
    return [(w, t, w2lemma(w,t, morphy)) for (w,t) in tagged]

def mfs (wordnet, lemmas):
    out = []
    for (w, t, l) in lemmas:
        pos = pos2wn(t)
        sense = None
        ### tweak 
        if l in mfs_tweak:
            sense = mfs_tweak[l]
        elif pos: 
            senses = wordnet.senses(l, pos)
            if len(senses) == 1:
                sense = senses[0].synset().ili.id
            elif len(senses) >1:
                sense = max(senses, key= lambda s: sum(s.counts())).synset().ili.id
            else:
                sense = None
        out.append((w, t, l, sense))
    return out


def disambiguate (sent, wordnet, morphy):
    """
    disambiguate a sentence
    """
    #toks = word_tokenize(sent)
    toks = t.tokenize(sent)
    tags = pos_tag(toks)
    lemmas = lemmatize(tags, morphy)
    senses = mfs(wordnet, lemmas)
    return senses
