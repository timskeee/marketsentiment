import requests
import pandas as pd
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# nltk.download('all') #first time only

# For use with Financial Modeling prep but couldn't get earning call transcripts with free subscription.
# base_url = 'https://financialmodelingprep.com/api'
# data_type = ''
# ticker = 'AAPL'
# API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXX'
# url = f'{base_url}/v3/search?query={ticker}&apikey=xX3HJKM32HkqJJwQ4Nxhejj23Oji01BB'
# url2 = f'https://financialmodelingprep.com/api/v3/search?query=AA&apikey={apikey}'
# ect = f'https://financialmodelingprep.com/api/v3/earning_call_transcript/AAPL?year=2020&quarter=3&apikey={apikey}' #ect - earnings call transcript
# response = requests.get(url2)
# data = response.json()
# print(data)


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    # print(processed_text)
    return processed_text

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
# create get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    print(scores)
    print(sentiment)
    return scores, sentiment


## Using https://discountingcashflows.com/documentation/api-guide/

df = pd.read_csv('companies.csv')
tickerlist = df['Symbol']
scoreslist = []
base_url = 'https://discountingcashflows.com'
quarter = 'Q1' #Q2, Q3, Q4
year = 2024

for index, row in df.iterrows():
    try:
        ticker = row['Symbol']
        company = row ['Security']
        sector = row['GICS Sector']
        subsector = row['GICS Sub-Industry']
        # ticker = 'AMC'
        ect_endpoint = f'{base_url}/api/transcript/{ticker}/{quarter}/{year}'

        response = requests.get(ect_endpoint)
        data = response.json()
        # print(data[0]['content']) #access just the transcript under 'content' *this is a list with dictionary in it

        print(f'company: {company}; ticker:{ticker}; sector: {sector}')
        
        # symbol = data[0]['symbol']
        # quarter = data[0]['quarter']
        # year = data[0]['year']
        date = data[0]['date']
        transcript = data[0]['content']

        proc_transcript = preprocess_text(transcript)
        scores, sentiment = get_sentiment(proc_transcript)

        scoreslist.append([ticker,company,sector,subsector,f'Q{quarter}',year,date,scores,sentiment])
    except:
        print('There was an issue or something')

compiledscores_df = pd.DataFrame(scoreslist)
compiledscores_df.to_csv('compiled_sentiment.csv')
