# %% [markdown]
# ## <font color='green'>S&P500 Treemap</font>

# %% [markdown]
# #### Macbook need this to open NLTK

# %%
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

# %% [markdown]
# #### Load necessary libraries

# %%
# libraries for webscraping, parsing and getting stock data
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import yfinance as yf

# for plotting and data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# NLTK VADER for sentiment analysis
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from datetime import datetime
import warnings
import requests

# %%
# From Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', {'id': 'constituents'})

# Parse the table
df = pd.read_html(str(table))[0]
df = df[['Symbol', 'Security', 'GICS Sector']]
df.columns = ['Ticker', 'Company', 'Sector']

df.head()


# %% [markdown]
# ### <font color='green'>Scrape the Date, Time and News Headlines Data (Some error when reading the tickers)</font>

# %%
# Scrape the Date, Time and News Headlines Data
finwiz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

tickers = df['Ticker'].tolist()

for ticker in tickers:
    print(ticker)
    url = finwiz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    try:
        response = urlopen(req)    
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response, 'html.parser')
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table
        # Avoid HTTP Error 429
        time.sleep(0.3)
    except Exception as e:
        print("Error")
        continue

# %% [markdown]
# ### <font color='green'>Print the Data from news_table</font>

# %%
# Print the Data from news_table (optional)
# Example: Read one single day of headlines for ‘AMZN’ 
amzn = news_tables['AMZN']
# Get all the table rows tagged in HTML with <tr> into ‘amzn_tr’
amzn_tr = amzn.findAll('tr')
for i, table_row in enumerate(amzn_tr):
     # Read the text of the element ‘a’ into ‘link_text’
     a_text = table_row.a.text
     # Read the text of the element ‘td’ into ‘data_text’
     td_text = table_row.td.text
     # Print the contents of ‘link_text’ and ‘data_text’ 
     print(a_text)
     print(td_text)
     # Exit after printing 4 rows of data
     if i == 3:
         break

# %% [markdown]
# ### <font color='green'>Parse the Date, Time and News Headlines into a Python List</font>

# %%
# Parse the Date, Time and News Headlines into a Python List
parsed_news = []
# Iterate through the news
for file_name, news_table in news_tables.items():
    # Iterate through all tr tags in 'news_table'
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        if x.a: # Avoid Null value
            text = x.a.get_text()
        else:
            continue
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element
        if len(date_scrape) == 1:
            time = date_scrape[0]
            date = datetime.now().date()
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        # Extract the ticker from the file name, get the string up to the 1st '_'  
        ticker = file_name
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([ticker, date, time, text])
        
parsed_news[:5] # print first 5 rows of news

# %% [markdown]
# ### <font color='green'>Perform Sentiment Analysis with Vader</font>

# %%
# Perform Sentiment Analysis with Vader
# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()
# Set column names
columns = ['ticker', 'date', 'time', 'headline']
# Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

# Convert today to the date format
parsed_and_scored_news['date'] = parsed_and_scored_news['date'].apply(lambda x: datetime.now().date() if x == 'Today' else x)

# Iterate through the headlines and get the polarity scores using vader
scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)

# Join the DataFrames of the news and the list of dicts
parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
# Convert the date column from string to datetime
parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
parsed_and_scored_news.head()

# Make sure the compound lies between -1 to 1
print(parsed_and_scored_news[['neg', 'neu', 'pos', 'compound']].describe())

# %%
# Visualize the data
parsed_and_scored_news

# %% [markdown]
# ### <font color='green'>Calculate Mean Sentiment for Each Ticker</font>

# %%
# Group by each ticker and get the mean of all sentiment scores
mean_scores = parsed_and_scored_news.groupby(['ticker']).mean()
mean_scores

# %% [markdown]
# ### <font color='green'>Get Current Price, Sector and Industry of each Ticker</font>

# %%
# Fetch additional information from Yahoo Finance
sectors = []
industries = []
prices = []
for ticker in tickers:
    print(ticker)
    tickerdata = yf.Ticker(ticker)
    try:
        prices.append(tickerdata.info['currentPrice'])
        sectors.append(tickerdata.info['sector'])
        industries.append(tickerdata.info['industry'])
    except KeyError:
        prices.append(None)
        sectors.append(None)
        industries.append(None)

# %% [markdown]
# ### <font color='green'>Combine the Information Above and the Corresponding Tickers into a DataFrame</font>

# %%
# Create dataframe with additional information
d = {'Sector': sectors, 'Industry': industries, 'Price': prices}
df_info = pd.DataFrame(data=d, index=tickers)

# %% [markdown]
# ### <font color='green'>Join all the Information into a Single DataFrame</font>

# %%
# Merge sentiment scores and stock information
df = mean_scores.join(df_info)
df = df.rename(columns={"compound": "Sentiment Score", "neg": "Negative", "neu": "Neutral", "pos": "Positive"})
df = df.reset_index()
df

# %% [markdown]
# ### <font color='green'>Choose between Most Positive or Most Negative Stocks' Sentiment Score</font>

# %%
# Define the function to filter sentiment scores
def filter_tickers(df, choice, num_of_company=5):
    filtered_df = pd.DataFrame()
    sectors = df['Sector'].unique()
    
    for sector in sectors:
        sector_df = df[df['Sector'] == sector]
        if choice == 'most positive':
            top_stocks = sector_df.nlargest(num_of_company, 'Sentiment Score')
        elif choice == 'most negative':
            top_stocks = sector_df.nsmallest(num_of_company, 'Sentiment Score')
        filtered_df = pd.concat([filtered_df, top_stocks])
    
    return filtered_df

# Choose between most positive and most negative
choice = 'most negative' 
filtered_df = filter_tickers(df, choice)
filtered_df = filtered_df.reset_index(drop=True)

# %% [markdown]
# ### The Final Treemap

# %%
# Create a treemap (No need to know how large each of ticker is in our position)
# Even through the maximum and minimum sentiment score lies between -1 and 1, 
fig = px.treemap(filtered_df, path=[px.Constant("Sectors"), 'Sector', 'Industry', 'ticker'],
                  color='Sentiment Score', hover_data=['Price', 'Negative', 'Neutral', 'Positive', 'Sentiment Score'],
                  color_continuous_scale=['#FF0000', "#000000", '#00FF00'],
                #   range_color=(-1, 1),
                  color_continuous_midpoint=0)

fig.data[0].customdata = filtered_df[['Price', 'Negative', 'Neutral', 'Positive', 'Sentiment Score']].round(3) # round to 3 decimal places
fig.data[0].texttemplate = "%{label}<br>%{customdata[4]}"

fig.update_traces(textposition="middle center")
fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), font_size=20)

plotly.offline.plot(fig, filename='stock_sentiment.html') # this writes the plot into a html file and opens it
fig.show()

# %%



