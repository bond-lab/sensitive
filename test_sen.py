import sensitive
from sensitive.sensitive import SentimentAnalyzer

sentences = ["VADER is smart, handsome, and funny.",
             "VADER is happy.",
             "VADER is happy!",
             "VADER is happy!!!",
             "VADER is happy!!!",
             "VADER is happy?",
             "VADER is happy???",
             "VADER is not happy.",
             "SENSI is more happy.",
             "SENSI is prettier.",
             "SENSI is very happy.",
             "This is unhappy.",
             "Wordnet is prettiest",
             "The happy wordnet has some problems.",
             "The happy wordnet has some PROBLEMS."]

analyzer = SentimentAnalyzer()

for s in sentences:
    score = analyzer.polarity_scores(s)['compound']
    print(score, s)
