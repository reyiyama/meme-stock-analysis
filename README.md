# Reddit Sentiment Analysis: Tracking the Pulse of Financial Subreddits

## Overview

This project, inspired by the 2021 GameStop phenomenon, aims to explore the impact of online discussions on stock market sentiment, especially within communities like r/wallstreetbets. This work focuses on sentiment analysis techniques applied to posts collected from multiple financial subreddits between July and August 2023. The primary objective is to identify patterns and gauge investor sentiment to better understand the dynamics of online investment communities and their influence on market trends.

## Project Inspiration: The GameStop Phenomenon

The 2021 GameStop short squeeze orchestrated by r/wallstreetbets served as the key inspiration for this project. The power of decentralized financial dialogue demonstrated its ability to create significant market shifts, challenging the dominance of institutional investors. This event inspired a deeper dive into online forums as influential financial ecosystems capable of shaping market sentiment and behavior. By applying sentiment analysis techniques to posts from 11 key financial subreddits, this project explores whether similar "meme stock" movements could be predicted or better understood through natural language processing.


## Datasets

The datasets used in this project include:

1. **Reddit Stock Market Data**: Collected using the Reddit API, specifically through Python's Reddit API Wrapper (PRAW). This dataset includes posts and comments scraped from 11 financial subreddits (‘r/investing\_discussion’, ‘r/dividends’, ‘r/wallstreetbets’, etc.). The data was compiled into a CSV file named `reddit_stock_market_data_cleaned.csv`.
2. **Stock Comments Dataset**: A cleaned version of the comment data that was collected, stored in the CSV file `reddit_stock_comments_cleaned.csv`.

Data extraction was handled via PRAW, with rate limits carefully managed to ensure data collection adhered to Reddit's guidelines. Data collected included subreddit names, post titles, post bodies, and up to ten top-level comments per post.

## Project Objectives

- Conduct sentiment analysis of Reddit posts and comments to gauge the public mood around investment topics.
- Apply natural language processing (NLP) to identify positive, negative, and neutral sentiments across multiple subreddits.
- Compare sentiment associated with specific financial instruments and companies.
- Identify conversational themes through topic modeling and track changes over time.

## Analysis Techniques

### Data Cleaning and Pre-Processing

Data cleaning involved removing irrelevant text and noise from the dataset. Comments containing irrelevant content, such as automated messages and pinned posts, were removed. Specific posts mentioning experiences with scams or irrelevant information were filtered to ensure that the final dataset reflected genuine, unbiased sentiment.

Steps taken in the Jupyter Notebook for data cleaning:
- **Removed Unwanted Comments**: Automated or pinned comments were removed to avoid skewing the sentiment analysis.
- **Filtered Specific Subreddit Data**: Posts related to irrelevant topics, such as scams, were filtered out.
- **Text Normalization**: All text was converted to lowercase, punctuation was removed, and stopwords were filtered out using the Natural Language Toolkit (NLTK).
- **Tokenization**: The comments were tokenized to enable efficient analysis, breaking down the text into individual words or tokens.

### Sentiment Analysis Approaches

The sentiment analysis was conducted using a combination of different techniques, focusing on analyzing both the posts and comments collected from multiple subreddits:

1. **VADER Sentiment Analyzer**: Valence Aware Dictionary and sEntiment Reasoner (VADER) was used to classify posts and comments as positive, negative, or neutral. It provided a straightforward and effective method for analyzing the nuances of social media text, particularly for finance-related discussions on Reddit.

   Steps implemented in the Jupyter Notebook:
   - **Applied VADER to Posts and Comments**: The VADER sentiment analyzer was applied to both post titles, bodies, and comments.
   - **Classification**: Sentiment was categorized as Positive, Negative, or Neutral based on VADER's compound score.
   - **Entity Analysis**: Identified key entities like 'VOO', 'SPY', 'IRA', and 'Roth' and analyzed their correlation with sentiment scores.
<img width="1253" alt="Screenshot 2024-10-10 at 4 47 44 PM" src="https://github.com/user-attachments/assets/a2b0a4a5-bf29-46e0-9017-ab4635379ade">

   - Key Findings:
     - Entities like 'VOO', 'SPY', 'IRA', and 'Roth' were positively correlated with positive sentiments.
     - Negative sentiments were more often associated with entities like 'China', 'US', and 'Fed'.
     - The "U.S." context elicited both positive mentions (e.g., stock market performance) and negative mentions (e.g., housing affordability crisis).

2. **TextBlob and NLTK**: Complemented VADER by providing an additional layer of sentiment scoring for validation and cross-referencing, using linguistic tokenization and sentiment tagging.

   Steps implemented in the Jupyter Notebook:
   - **Applied TextBlob for Cross-Validation**: TextBlob was used to validate the sentiment results obtained from VADER.
   - **Tokenization with NLTK**: Tokenization and stopword removal were performed using NLTK to enhance the accuracy of TextBlob's analysis.

3. **TF-IDF Analysis**: Applied Term Frequency-Inverse Document Frequency to better understand the prominence of certain keywords and their contribution to sentiment classification within the discussions.

   Steps implemented in the Jupyter Notebook:
   - **TF-IDF Vectorization**: TF-IDF was applied to identify important keywords and gauge their significance in each document.
   - **Feature Extraction**: The extracted features were used for further analysis, contributing to understanding keyword impact on sentiment.

### Topic Modeling Using LDA

Latent Dirichlet Allocation (LDA) was used for topic modeling, focusing on financial metrics and investment themes like 'PE', 'EBITDA', 'SPY', and 'VOO'. Topic modeling helped reveal key discussion themes among Redditors, such as valuation methods and market dynamics.

Steps implemented in the Jupyter Notebook:
- **Pre-Processed Text for Topic Modeling**: Text data was pre-processed by removing noise, tokenizing, and lemmatizing to ensure high-quality input for LDA.
- **LDA Topic Modeling**: The LDA model was applied using scikit-learn to generate discussion topics.
- **Topic Analysis**: The resulting topics were analyzed to understand dominant themes, and visualizations such as word clouds were generated to display topic significance.

<img width="949" alt="Screenshot 2024-10-10 at 4 49 34 PM" src="https://github.com/user-attachments/assets/9b155123-e513-450c-9951-887611c4ff84">


Example topics included:
- Discussions around company valuation and market opportunities.
- Considerations for ETF timing, buying strategies, and market risk.

### Comparative Sentiment Analysis

Sentiment scores were calculated for different financial metrics and instruments:

Steps implemented in the Jupyter Notebook:
- **Calculated Sentiment Scores**: The sentiment scores for different entities (e.g., VOO, SPY, PE, EBITDA) were calculated based on the compound scores obtained from VADER.
- **Comparison Across Entities**: Visualizations were generated to compare sentiment across various financial instruments and metrics.

- **VOO vs PE**: VOO had a higher sentiment score (65) compared to PE (33), suggesting a generally more favorable outlook towards ETFs over price-to-earnings ratios.
- **SPY vs EBITDA**: SPY's sentiment score (58) was also higher compared to EBITDA (17), emphasizing skepticism or caution towards certain complex financial metrics compared to ETFs.

The analysis highlighted the positive sentiment around ETFs like VOO and SPY compared to financial metrics such as PE and EBITDA. Overall, entities tied to market stability and familiar investment vehicles were perceived more positively, while terms associated with uncertainty (e.g., China, Fed) received more negative feedback.

## Visualizations and Findings

The analysis was supported by various visualizations, such as:

- **Word Clouds**: Generated using the WordCloud library to show the prevalence of positive and negative keywords in the financial subreddits. The Jupyter Notebook included the steps to create word clouds for both positive and negative sentiment.
- **Sentiment Distribution Plots**: Created using Matplotlib to visualize how specific entities correlated with positive, negative, or neutral sentiments. Bar charts and pie charts were also used to display overall sentiment distribution.

## Step-by-Step Instructions to Reproduce This Analysis

If you want to replicate this analysis or conduct a similar sentiment analysis project, follow these steps:

### Prerequisites
- **Python 3.x**: Ensure Python is installed.
- **Jupyter Notebook**: Install Jupyter Notebook to run the provided code interactively.
- **Libraries**: Install the following Python libraries:
  ```bash
  pip install praw pandas nltk matplotlib seaborn wordcloud vaderSentiment scikit-learn textblob
  ```

### Step 1: Data Collection
- **Register Reddit API Application**: Create an application at [Reddit Apps](https://www.reddit.com/prefs/apps) to obtain `client_id`, `client_secret`, and `user_agent`.
- **Extract Data Using PRAW**: Use the Python Reddit API Wrapper (PRAW) to collect subreddit posts and comments. Write the data into a CSV file for further analysis.
  ```python
  import praw
  import pandas as pd

  reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                       client_secret='YOUR_CLIENT_SECRET',
                       user_agent='YOUR_USER_AGENT')

  # Example: Extracting top posts from r/investing
  posts = []
  for post in reddit.subreddit('investing').top(limit=150):
      posts.append([post.title, post.selftext, post.score, post.num_comments])

  # Save data to CSV
  df = pd.DataFrame(posts, columns=['Title', 'Body', 'Score', 'Comments'])
  df.to_csv('reddit_stock_market_data_cleaned.csv', index=False)
  ```

### Step 2: Data Cleaning and Pre-Processing
- **Load Data in Jupyter Notebook**: Load the CSV file into a Pandas DataFrame.
- **Text Cleaning**: Remove unnecessary punctuation, convert text to lowercase, and filter out stopwords using NLTK.
  ```python
  import nltk
  from nltk.corpus import stopwords
  import string

  nltk.download('stopwords')
  stop_words = set(stopwords.words('english'))

  def clean_text(text):
      text = text.lower()
      text = ''.join([char for char in text if char not in string.punctuation])
      text = ' '.join([word for word in text.split() if word not in stop_words])
      return text

  df['Cleaned_Body'] = df['Body'].apply(lambda x: clean_text(x))
  ```

### Step 3: Sentiment Analysis
- **Apply VADER Sentiment Analyzer**: Use VADER to classify each post and comment as positive, negative, or neutral.
  ```python
  from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

  analyzer = SentimentIntensityAnalyzer()

  df['Sentiment'] = df['Cleaned_Body'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
  df['Sentiment_Label'] = df['Sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
  ```

### Step 4: Topic Modeling
- **Prepare Data for LDA**: Pre-process the text to tokenize and lemmatize it.
- **Apply LDA Using Scikit-Learn**: Extract topics using LDA to identify the dominant themes in the discussions.
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.decomposition import LatentDirichletAllocation

  vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
  dtm = vectorizer.fit_transform(df['Cleaned_Body'])

  lda = LatentDirichletAllocation(n_components=5, random_state=42)
  lda.fit(dtm)
  ```

### Step 5: Visualization
- **Word Clouds**: Generate word clouds to visualize the most frequent positive and negative words.
  ```python
  from wordcloud import WordCloud
  import matplotlib.pyplot as plt

  positive_text = ' '.join(df[df['Sentiment_Label'] == 'Positive']['Cleaned_Body'])
  wordcloud = WordCloud(background_color='white').generate(positive_text)

  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show()
  ```

- **Sentiment Distribution Plots**: Create bar and pie charts to visualize the distribution of sentiment labels.
  ```python
  import seaborn as sns

  sns.countplot(x='Sentiment_Label', data=df)
  plt.title('Sentiment Distribution')
  plt.show()
  ```

## Key Insights and Implications

- **Investor Sentiment Towards ETFs**: The positive sentiment towards VOO and SPY underscores their perception as safe and reliable investments, suggesting an ongoing preference for passive index funds.
- **Polarizing Topics**: Entities like 'China' and 'Fed' were highly polarizing, which could reflect uncertainty or risk perceptions among retail investors. This insight may guide future risk assessment strategies for financial institutions.
- **Market Sentiment and Meme Stocks**: The findings highlight the emerging role of meme stocks and retail investors in shaping market sentiment, similar to the GameStop saga. This trend could influence market stability and drive future "meme stock" phenomena.

## Next Steps and Future Research

- **Deep Learning Techniques**: Utilize transformer-based models like BERT to enhance sentiment analysis accuracy and capture deeper nuances in Reddit discussions.
- **Meme Stocks and Financial Stability**: Further explore how meme stocks affect market stability and the mechanisms driving these effects.
- **Longitudinal Analysis**: Conduct a time-series analysis to track changes in sentiment over longer periods, particularly during market-moving events.

## Conclusion

This project demonstrates the power of natural language processing and sentiment analysis in understanding retail investors' perspectives. It provides a roadmap for identifying key market trends and gauging public sentiment in financial discourse. The role of meme stocks and online investor communities continues to be a significant factor in financial market analysis, suggesting that future research in this area could yield actionable insights for both retail and institutional investors.

The project builds a bridge between traditional financial analysis and emerging digital investor dynamics, paving the way for a more comprehensive understanding of modern financial markets.
