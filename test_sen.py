import sensitive
from sensitive.sensitive import SentimentAnalyzer

sentences = [#"VADER is smart, handsome, and funny.",
             "VADER is happy.",
     "VADER is very happy.",
    "VADER is slightly happy.",
    "VADER is very slightly happy.",
    "VADER is very very happy.",
    "VADER is very very very happy.",
     "VADER is not happy.",
     "VADER is not very happy.",
    "VADER is very not happy.",
    # "VADER is happy!",
    # "VADER is happy!!!",
    # "VADER is happy?",
    #          "VADER is happy???",
    #          "VADER is not happy.",
    #          "SENSI is more happy.",
    #          "SENSI is happier.",
    #          "SENSI is very happy.",
    #          "This is unhappy.",
    #          "Wordnet is happiest.",
    "The happy wordnet has some problems.",
    "The happy wordnet has no problems.",
    "The happy wordnet has some PROBLEMS.",
    "The happy wordnet has NO PROBLEMS."
]

analyzer = SentimentAnalyzer()

for s in sentences:
    score = analyzer.polarity_scores(s)['compound']
    print(score, s)
