from newspaper import Article
import nltk
from nltk.corpus import stopwords

import math
from textblob import TextBlob as tb


def tf(word,blob):
	return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
	return sum(1 for blob  in bloblist if word in blob.words)

def idf(word, bloblist):
	return math.log(len(bloblist)/(1+n_containing(word,bloblist)))

def tfidf(word,blob,bloblist):
	return tf(word,blob)*idf(word,bloblist)



# url1="http://www.bbc.com/news/world-asia-china-42728251"

# article=Article(url)

# article.download()

# article.html

# article.parse()


# Text=article.text

# print(Text)


document1 = tb("""Former captain Anil Kumble is impressed by the Indian team's performance against the visiting Australians in the ongoing Border-Gavaskar Trophy and is looking forward to a clean sweep in the four-Test series.
"It will be a great thing. To beat Australia, forget Australia, let alone any Test series to win 4-0 is something very special. So India has an opportunity to achieve that. And it is great to see everyone doing really well at the moment. I'm looking forward to a 4-0 victory," Kumble said here.
India are leading the four-match series 3-0, with the final game going on at the Feroz Shah Kotla ground in New Delhi.
The former legspinner lavished praise on Shikhar Dhawan, who notched up the fastest-ever Test century on debut in the third Test.
The former legspinner lavished praise on Delhi opener Shikhar Dhawan, who notched up the fastest-ever Test century on debut in the third Test in Mohali.
"He played brilliantly. It is unfortunate that he is not playing in this game, (in his) hometown (because of injury). I really watched that (Mohali innings).... Fantastic batting."
"To go out there and just play the way he did in his first game was impressive. That really augurs well. There are some tough tours coming up, so this confidence will certainly help the youngsters," he said.
"India has done well to beat Australia (in the series). They (Australia) have put up scores close to 400 in the first innings but haven't backed it up. India has done well. Everyone has bowled well, batted well," Kumble added.
Asked to compare the current tourists with the Australian teams he had played against, Kumble said the earlier squads were highly competitive ones.
"You can't really compare but 2004, 2008, 1996, 1998...I think those teams that I played against were extremely competitive. (They) Probably (were the) number one team for a long time. I think that says a lot.""")

document2 = tb("""The 21-year-old hammer thrower, who reached the Olympic final in London last summer with a British record of 71.98m before finishing 12th, ranked tenth in Spain with a best of 66.69m.
Hitchon produced the distance with her final throw in Castellon with Zalina Marghieva of Moldova taking victory after a throw of 71.98m with her very first attempt.
Meanwhile fellow Brit Rachel Wallader placed 12th in the shot put with a distance of 15.20m with Olympic and recent European indoor silver medallist Yevgeniya Kolodko the winner with 19.04m.
Mark Dry won the B competition of the men's hammer with a season's best of 73.22m while British teammate Alex Smith, who beat him to a place at the London 2012 Olympics, was third with 68.34m.
Sarah Holt also placed third in the B competition of the women's hammer after a best throw of 65.82m while Sophie McKinna was second in the under-23 shot put event.
McKinna recorded a best throw of 16.09m as Emel Dereli of Turkey took victory with 16.78m while Joe Dunderdale placed sixth in the under-23 javelin event with a season's best of 73.02m.""")

document3 = tb("""taly's former 200m world record holder and Olympic champion Pietro Mennea has died, aged 60.
The sprinter won 200m gold at the 1980 Moscow Olympics, beating Britain's 100m champion Alan Wells into second place.
Mennea also won bronze in the 4x400 relay at the same Olympics, to add to a 200m bronze from the 1972 Munich Games.
"British fans will recall Mennea's great duels at 200m with Alan Wells, often decided by a few hundredths of a second.

"Their most memorable duel came at the 1980 Moscow Olympics. Wells raced into a substantial lead and looked set to claim the Olympic sprint double. Then Mennea, one lane outside, clawed ahead with his closing strides to win, and forced his way through the Soviet security throng to complete a lap of honour."

He broke the 200m world record in 1979 with a time of 19.72 seconds, a record that stood for 17 years until Michael Johnson ran 19.66 in 1996.
American Johnson's record was bettered by Jamaica's Usain Bolt at the 2008 Beijing Olympics and now stands at 19.19 - but Mennea's time remains the European record.
Later a Member of the European Parliament following his retirement from athletics, Mennea also collected three European Championship golds at 100m and 200m during his career.
Italian National Olympic Committee president Giovanni Malago said Mennea's body would lie in state at Olympic committee headquarters. No cause of death has been announced.""")

bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

# tokens=nltk.word_tokenize(Text)

# print(tokens)

# stopwords
