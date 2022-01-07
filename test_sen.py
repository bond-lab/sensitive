import sensitive
from sensitive.sensitive import SentimentAnalyzer

sentences = ["VADER is smart, handsome, and funny.",
             "VADER is pretty.",
             "VADER is pretty!",
             "VADER is pretty!!!",
             "VADER is PRETTY!!!",
             "VADER is pretty?",
             "VADER is pretty???",
             "SENSI is more pretty.",
             "SENSI is prettier.",
             "SENSI is very pretty.",
             "This is unpretty.",
             "Wordnet is prettiest",
             "The pretty wordnet has some problems.",
             "The pretty wordnet has some PROBLEMS."]

analyzer = SentimentAnalyzer()

for s in sentences:
    score = analyzer.polarity_scores(s)
    print(score, s)
