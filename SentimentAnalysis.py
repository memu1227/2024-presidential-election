import Reddit
import re
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download stopwords and VADER lexicon
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Define colors for sentiment categories
color_mapping = {
    'Positive': '#4CAF50',    # Green
    'Negative': '#F44336',    # Red
    'Neutral': '#FFC107',     # Yellow
}
# Exclude these words from analysis
exclude_words = {
    "said","say", "donald", "biden", "trump", "harris", "kamala", "joe", "trumps", 
    "bernie", "sanders", "clinton", "us", "election", "president", "says", 
    "thehillcom", "presidential", "bidens", "new", "news", "hillary",
    "cnncom", "washingtonpostcom", "foxnewscom", "nbcnewscom", "msnbccom",
    "nytimescom", "usatodaycom", "reuterscom", "apnewscom", "bloombergcom",
    "nprorg","independentcouk","bbccom","axioscom","cbsnewscom","cnbccom",
    "politicocom","huffpostcom","newsweekcom","abcnewsgocom","voxcom",
    "businessinsidercom","thedailybeastcom","commondreamsorg","theguardiancom",
    "nypostcom","huffingtonpost","buzzfeednews","theweekcom","marketwatchcom",
    "lawandcrimecom"
    
}
news_outlet_patterns = r'cnn|washingtonpost|foxnews|nbcnews|msnbc|nytimes|usatoday|reuters|apnews|bloomberg|npr'
domain_patterns = r'\b\w+\.(com|org|co|net|gov|edu|info|biz|io|me|ly|tv)\b'
generic_tld_patterns = r'\b(com|org|net|gov|edu|info|biz|io|me|ly|tv)\b'

# Function to clean the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions, hashtags, and special characters
    text = re.sub(r'@\w+|#\w+|[^a-zA-Z\s]', '', text)

    # Remove news outlet patterns
    text = re.sub(news_outlet_patterns, '', text)

    # Remove generic TLDs
    text = re.sub(generic_tld_patterns, '', text)

    # Remove domains ending in specified TLDs
    text = re.sub(domain_patterns, '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Function to get sentiment from VADER
def get_vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)
    return score['compound']  # Return the compound score


# Function to perform sentiment analysis
def analyze_sentiments(df, candidate_name):
    # Combine title and selftext into a single text column
    df['combined_text'] = df['title'] + ' ' + df['selftext']  # Concatenate title and selftext

    # Clean the combined text
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    # Sentiment Analysis using VADER
    df['sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)

    # Categorize sentiments with optimum thresholds
    def categorize_sentiment(score):
        if score >= 0.05:  # Adjust threshold as needed
            return 'Positive'
        elif score <= -0.05:  # Adjust threshold as needed
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

    # Check the distribution of sentiment categories
    sentiment_distribution = df['sentiment_category'].value_counts()

    # Collect all words except the excluded ones
    all_words = ' '.join(df['cleaned_text'].dropna()).split()
    filtered_words = [word for word in all_words if word not in stopwords.words('english') and word not in exclude_words]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(10)

    # Convert top_words to a DataFrame for plotting
    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

    # Create a combined figure for sentiment distribution and top words
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))

    # Plot sentiment distribution
    sentiment_distribution.plot(kind='bar', color=[color_mapping[cat] for cat in sentiment_distribution.index], ax=ax[0])
    ax[0].set_title(f'Sentiment Distribution for {candidate_name}')
    ax[0].set_xlabel('Sentiment Category')
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(range(len(sentiment_distribution)))
    ax[0].set_xticklabels(sentiment_distribution.index, rotation=0)

    # Plot top 10 words
    top_words_df.plot(kind='barh', x='Word', y='Frequency', color='skyblue', ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title(f'Top 10 Words for {candidate_name}')
    ax[1].set_xlabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{candidate_name}_SentimentAnalysis.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # Generate a larger word cloud
    text = ' '.join(filtered_words)
    wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate(text)

    # Create a separate figure for the word cloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {candidate_name}')
    plt.savefig(f'{candidate_name}_WordCloud.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

# Perform sentiment analysis for each candidate
analyze_sentiments(Reddit.harris_policy_df, "Kamala Harris")
analyze_sentiments(Reddit.trump_policy_df, "Donald Trump")
analyze_sentiments(Reddit.election_policy_df, "Election Posts")
