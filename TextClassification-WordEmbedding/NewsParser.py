import pandas as pd
from newspaper import Article
import requests


dataset=pd.read_csv(r'C:\Users\Anurag\OneDrive\NLP\uci-news-aggregator.csv')
url=dataset.iloc[:,2]
publisher=dataset.iloc[:,3]


file_name= open('NewsDataSmall.txt',"w+")

for i in range (1,url.size):
	try:
		req=requests.get(url[i])
		if(req.status_code==200):
			article=Article(url[i],language='en')
			article.download()
			article.parse()
			description=article.text
			description=description.replace("\n\n","")
			file_name.write(publisher[i]+"\t"+description+"\n")
			print("Writing "+str(i)+"th article")
	except Exception:
		print("Exception")
		continue
file_name.close	()