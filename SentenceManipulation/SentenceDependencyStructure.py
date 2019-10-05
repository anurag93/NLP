import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import Word2Vec
import pandas as pd

nltk.download('punkt')
nlp = spacy.load("en")

doc1 = nlp(u'A strategy in the Affordable Care Act - new york times')

doc2 = nlp(u'Republicans’ 4-Step Plan to Repeal the Affordable Care Act - The New York Times')

for token in doc1:
    print(str(token.text), str(token.pos_))


for token in doc2:
    print(str(token.text), str(token.lemma_), str(token.pos_), str(token.dep_))

tree_pos = dict()

for token in doc1:
    if str(token.pos_)  not in tree_pos.keys():
        value = str(token.pos_)
        pos_value = list()
    else:
        pos_value = tree_pos[str(token.pos_)]
    if pos_value.__contains__(str(token.text)):
        continue
    pos_value.append(str(token.text))
    tree_pos[str(token.pos_)] = pos_value


for token in doc2:
    if str(token.pos_)  not in tree_pos.keys():
        value = str(token.pos_)
        pos_value = list()
    else:
        pos_value = tree_pos[str(token.pos_)]
    if pos_value.__contains__(str(token.text)):
        continue
    pos_value.append(str(token.text))
    tree_pos[str(token.pos_)] = pos_value

print(tree_pos)

#Currently Debugging this section.
articles = pd.read_csv(r'articles1.csv', usecols=['content']).head(1000)
sentences = ''
data = []
for i in range(len(articles)):
    sentences = sentences + articles.iloc[i]

for sentence in sent_tokenize(sentences):
    temp = []

    for word in word_tokenize(sentence):
        temp.append(word.lower())
    data.append(temp)


model = Word2Vec(data, min_count=1, size=100, window=10, sg=1)
print("Similarity between Republicans", model.most_similar('Republicans’'))
