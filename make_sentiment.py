import pandas as pd
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

def apply_vader_sentiment(df, review_type):
    tqdm.pandas()  # Initialize tqdm with pandas
    df[f'{review_type}_Sentiment'] = df[review_type].progress_apply(vader_sentiment)
    return df

# Load your dataset
data = pd.read_csv(r'D:\projects\datasets\booking-com-reviews2-europe\Hotel_Reviews.csv')
outfile = r'D:\projects\datasets\booking-com-reviews2-europe\Hotel_Reviews_with_sentiment.csv'

# Apply the VADER sentiment analysis for negative reviews
data = apply_vader_sentiment(data, 'Negative_Review')

# Apply the VADER sentiment analysis for positive reviews
data = apply_vader_sentiment(data, 'Positive_Review')

# Display the updated dataframe
print(data.head())

data.to_csv(outfile, index=False)