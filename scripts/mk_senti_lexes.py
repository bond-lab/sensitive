import wn
en = wn.Wordnet('omw-en:1.4')

mls = '/home/bond/papers/svn/wn/omw/data/mlsenticon/synset-level-pos-neg.txt'
swn = '/home/bond/nltk_data/corpora/sentiwordnet/SentiWordNet_3.0.0.txt'


##
## ML senticon
##
with  open('mlsenticon.tsv', 'w') as out:
    print("""# Sentiment from ML Senticon, converted to ili
# normalized form -4 to +4""", file=out)
    for l in open(mls):
        if l.startswith('#'):
            continue
        (pos, offset, positive, negative) = l.strip().split('\t')
        #print(pos, offset, positive, negative)
        ssid=f'omw-en-{int(offset):08d}-{pos}'
        try:
            ss = en.synset(id=ssid)
            print(ss.ili.id, 4 *(float(positive) - float(negative)),
                  sep='\t', file=out)
        except:
            try:
                ssid = f'omw-en-{int(offset):08d}-s'
                ss = en.synset(id=ssid)
                print(ss.ili.id, 4 *(float(positive) - float(negative)),
                      sep='\t',file=out)
            except:
                print(f"# WARNING NO SYNSET FOR: '{ssid}'")
        
##
## SentiWordnet
##
with  open('sentiwordnet.tsv', 'w') as out:
    print("""# Sentiment from SentiWordNet, converted to ili
# normalized form -4 to +4
# SentiWordNet is distributed under the Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) license.
# http://creativecommons.org/licenses/by-sa/3.0/
#
# For any information about SentiWordNet:
# Web: http://sentiwordnet.isti.cnr.it
#""", file=out)
    for l in open(swn):
        if l.startswith('#'):
            continue
        row = l.strip().split('\t')
        if len(row) != 6:
            continue
        (pos, offset, positive, negative, terms, gloss) = row
        #print(pos, offset, positive, negative)
        ssid=f'omw-en-{int(offset):08d}-{pos}'
        try:
            ss = en.synset(id=ssid)
            print(ss.ili.id, 4 *(float(positive) - float(negative)),
                  sep='\t', file=out)
        except:
            try:
                ssid = f'omw-en-{int(offset):08d}-s'
                ss = en.synset(id=ssid)
                print(ss.ili.id, 4 *(float(positive) - float(negative)),
                      sep='\t',file=out)
            except:
                print(f"# WARNING NO SYNSET FOR: '{ssid}'")
