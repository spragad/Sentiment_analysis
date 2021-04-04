#importing required packages
import tweepy as tw
import pandas as pd
import preprocessor as p #pip install tweet-preprocessor
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import numpy as ny
from PIL import Image
from textblob import TextBlob

#twitter authentication credentials
consumer_key = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
consumer_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
access_token= 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
access_token_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

notwt=3000 #No. of tweets to scrape
#amazon Data Scrapping, Keyward:AmazonIN
new_search = "AmazonIN" + " -filter:retweets" #neglect Retweets
date_since = "2020-02-01" #Tweets form date
tweets = tw.Cursor(api.search,
              q=new_search,
              lang="en", #Only tweets that are in English
              since=date_since,
              tweet_mode='extended').items(notwt) #No. of tweets to extract is 3000
twlst=[]
for tweet in tweets:
    twlst.append((tweet.full_text))


#Flipkart Data scrapping, Keyward:Flipkart
new_search = "Flipkart" + " -filter:retweets" #neglect Retweets
date_since = "2020-02-01"
tweets = tw.Cursor(api.search,
              q=new_search,
              lang="en", #Only tweets that are in English
              since=date_since,
              tweet_mode='extended').items(notwt) #No. of tweets to extract is 3000
Ftwlst=[]
for tweet in tweets:
    Ftwlst.append((tweet.full_text))

#Snapdeal Data scrapping, Keyward:snapdeal
new_search = "snapdeal" + " -filter:retweets" #neglect Retweets
date_since = "2020-02-01"
tweets = tw.Cursor(api.search,
              q=new_search,
              lang="en", #Only tweets that are in English
              since=date_since,
              tweet_mode='extended').items(notwt) #No. of tweets to extract is 3000
Stwlst=[]
for tweet in tweets:
    Stwlst.append((tweet.full_text))

#Combine all lists and convert into a DataFrame
data=pd.DataFrame({'Amazon_tweets':twlst, 'Flipkart_tweets':Ftwlst, 'Snapdeal_tweets':Stwlst })
data.to_csv('Data/fulltweets.csv')

#Basic cleaning and Preprocessing

p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.SMILEY,p.OPT.MENTION) #Conditions set to removes URLs, Emoji/Smiley, Mentions

clnamz=[]
clnflp=[]
clnsnpd=[]

for twt in data.Amazon_tweets:
    twt=p.clean(twt) #Removes URLs, Emoji/Smiley, Mentions
    twt=re.sub("\d+","", twt) #Removes Numbers
    twt=re.sub(r'[^\w\s]',"",twt) #Removes Punctuations
    clnamz.append(re.sub(" +"," ",twt).strip().lower()) #Removes extra spaces and Converts to lowercase
    
for twt in data.Flipkart_tweets:
    twt=p.clean(twt) #Removes URLs, Emoji/Smiley, Mentions
    twt=re.sub("\d+", "", twt) #Removes Numbers
    twt=re.sub(r'[^\w\s]',"",twt) #Removes Punctuations
    clnflp.append(re.sub(" +"," ",twt).strip().lower()) #Removes extra spaces and Converts to lowercase
    
for twt in data.Snapdeal_tweets:
    twt=p.clean(twt) #Removes URLs, Emoji/Smiley, Mentions
    twt=re.sub("\d+", "", twt) #Removes Numbers
    twt=re.sub(r'[^\w\s]',"",twt) #Removes Punctuations
    clnsnpd.append(re.sub(" +"," ",twt).strip().lower()) #Removes extra spaces and Converts to lowercase

data_cleaned=pd.DataFrame({'Amazon_tweets':clnamz, 'Flipkart_tweets':clnflp, 'Snapdeal_tweets':clnsnpd})
data_cleaned.to_csv('D:/Data Science/CDS/Project/Sentiment analysis/Data/fulltweets_pure.csv')

#Word cloud
mask=ny.array(Image.open('wc.png'))
Stopwords = stopwords.words('english')
Stopwords.append('thi')

#Amazon wordcloud
wordcloud= WordCloud(background_color = "Black", width=800, height=400, max_words = 300, mask=mask,
                     colormap='copper',stopwords = Stopwords).generate(data_cleaned.Amazon_tweets.str.cat(sep='\t'))

plt.figure( figsize=(30,15))
plt.title('AmazonIndia Wordcloud')
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Flipkart wordcloud
wordcloud= WordCloud(background_color = "Black", width=800, height=400,max_words = 300, mask=mask,
                     colormap='YlGnBu',stopwords = Stopwords).generate(data_cleaned.Flipkart_tweets.str.cat(sep='\t'))

plt.figure( figsize=(30,15))
plt.title('Flipkart Wordcloud')
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Snapdeal wordcloud
wordcloud= WordCloud(background_color = "Black", width=800, height=400, max_words = 300, mask=mask,
                     colormap='PuRd', stopwords = Stopwords).generate(data_cleaned.Snapdeal_tweets.str.cat(sep='\t'))

plt.figure( figsize=(30,15))
plt.title('Snapdeal Wordcloud')
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

data_cleaned=pd.read_csv('Data/fulltweets_puredf.csv')
data_cleaned['Flipkart_tweets']=data_cleaned['Flipkart_tweets'].fillna(str('Not available'))

#Analysis
def percentage(part, whole):
    Perc = 100 * float(part) / float(whole)
    return format(Perc, '.2f')

for col in data_cleaned.columns:
    #Creating some variables to store Sentiment polarity
    df=data_cleaned[col]
    polarity = 0
    positive = 0
    wpositive = 0
    spositive = 0
    negative = 0
    wnegative = 0
    snegative = 0
    neutral = 0
    for tweet in df:
        analysis = TextBlob(tweet)
        polarity += analysis.sentiment.polarity
        if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
                neutral += 1
        elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
                wpositive += 1
        elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                positive += 1
        elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
                spositive += 1
        elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
                wnegative += 1
        elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                negative += 1
        elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
                snegative += 1
    
    positive = percentage(positive, notwt)
    wpositive = percentage(wpositive, notwt)
    spositive = percentage(spositive, notwt)
    negative = percentage(negative, notwt)
    wnegative = percentage(wnegative, notwt)
    snegative = percentage(snegative, notwt)
    neutral = percentage(neutral, notwt)


    # finding average reaction
    polarity = polarity / notwt

    # printing Result of analysis
    print("Feedback/Reaction for "+ str(col) + " by analyzing " + str(notwt) + " tweets.")
    print()
    print("General Report: ")
    if (polarity == 0):
        print("Neutral")
    elif (polarity > 0 and polarity <= 0.3):
        print("Weakly Positive")
    elif (polarity > 0.3 and polarity <= 0.6):
        print("Positive")
    elif (polarity > 0.6 and polarity <= 1):
        print("Strongly Positive")
    elif (polarity > -0.3 and polarity <= 0):
        print("Weakly Negative")
    elif (polarity > -0.6 and polarity <= -0.3):
        print("Negative")
    elif (polarity > -1 and polarity <= -0.6):
        print("Strongly Negative")
    print()
    print("Detailed Report: ")
    print(str(spositive) + "% people thought it was strongly positive")
    print(str(positive) + "% people thought it was positive")
    print(str(wpositive) + "% people thought it was weakly positive")
    print(str(neutral) + "% people thought it was neutral")
    print(str(wnegative) + "% people thought it was weakly negative")
    print(str(negative) + "% people thought it was negative")
    print(str(snegative) + "% people thought it was strongly negative")
    print()
    #Plotting the analysed values
    labels = ['Strongly Positive [' + str(spositive) + '%]', 'Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]','Neutral [' + str(neutral) + '%]',
              'Weakly Negative [' + str(wnegative) + '%]', 'Negative [' + str(negative) + '%]', 'Strongly Negative [' + str(snegative) + '%]']
    sizes = [spositive, positive, wpositive, neutral, wnegative, negative, snegative]
    colors = ['darkgreen','yellowgreen','lightgreen','gold','lightsalmon', 'red', 'darkred']
    plt.figure( figsize=(12,10))
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.title("Feedback/Reaction for "+ str(col) + " by analyzing " + str(notwt) + " tweets.")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    print()

'''
Output:
Feedback/Reaction for Amazon_tweets by analyzing 3000 tweets.

General Report: 
Weakly Positive

Detailed Report: 
5.27% people thought it was strongly positive
12.57% people thought it was positive
22.50% people thought it was weakly positive
37.83% people thought it was neutral
14.23% people thought it was weakly negative
5.67% people thought it was negative
1.30% people thought it was strongly negative

Feedback/Reaction for Flipkart_tweets by analyzing 3000 tweets.

General Report: 
Weakly Positive

Detailed Report: 
2.10% people thought it was strongly positive
9.57% people thought it was positive
20.40% people thought it was weakly positive
44.03% people thought it was neutral
14.53% people thought it was weakly negative
6.50% people thought it was negative
1.57% people thought it was strongly negative 	

Feedback/Reaction for Snapdeal_tweets by analyzing 3000 tweets.

General Report: 
Weakly Positive

Detailed Report: 
0.97% people thought it was strongly positive
5.83% people thought it was positive
51.77% people thought it was weakly positive
31.80% people thought it was neutral
7.63% people thought it was weakly negative
1.60% people thought it was negative
0.30% people thought it was strongly negative
'''