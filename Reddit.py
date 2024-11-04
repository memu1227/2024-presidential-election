import os
import praw
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize the Reddit client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="ElectionAnalysis",
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD")
)

# Define queries focused on policies for each candidate
queries = {
    "Kamala Harris": "Kamala Harris policy",
    "Donald Trump": "Donald Trump policy",
    "2024 Presidential Election": "2024 Presidential Election policy"
}
target_post_count = 248

# Define a function to get posts related to a refined query
def fetch_posts(query, subreddit, limit=1000):  # Added subreddit parameter
    posts = []
    after = None  # Initialize the after parameter
    empty_count = 0  # To count filtered out posts due to empty titles

    while len(posts) < limit:
        # Perform the search within the specified subreddit
        submissions = reddit.subreddit(subreddit).search(query, sort="relevance", limit=100, params={'after': after})  # Fetch 100 posts at a time
        submissions_list = list(submissions)  # Convert to a list to access after

        if not submissions_list:  # Break if no more submissions
            break

        for submission in submissions_list:
            # Check if the title is not empty
            if submission.title:
                posts.append({
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "url": submission.url
                })
            else:
                empty_count += 1  # Increment count of filtered posts due to empty title

        after = submissions_list[-1].name  # Set after to the last submission's name

    #print(f"Filtered out {empty_count} posts due to empty titles.")
    return posts

# Fetch posts using refined queries in specific subreddits
harris_policy_posts = fetch_posts(queries["Kamala Harris"], "politics", limit=1000)
trump_policy_posts = fetch_posts(queries["Donald Trump"], "politics", limit=1000)
election_policy_posts = fetch_posts(queries["2024 Presidential Election"], "politics", limit=1000)

# Fetch from r/news as well
harris_policy_news_posts = fetch_posts(queries["Kamala Harris"], "news", limit=1000)
trump_policy_news_posts = fetch_posts(queries["Donald Trump"], "news", limit=1000)
election_policy_news_posts = fetch_posts(queries["2024 Presidential Election"], "news", limit=1000)

# Combine posts from both subreddits and limit to target count
harris_combined_posts = harris_policy_posts + harris_policy_news_posts
trump_combined_posts = trump_policy_posts + trump_policy_news_posts
election_combined_posts = election_policy_posts + election_policy_news_posts

# Limit combined posts to target count
harris_policy_df = pd.DataFrame(harris_combined_posts[:target_post_count])
trump_policy_df = pd.DataFrame(trump_combined_posts[:target_post_count])
election_policy_df = pd.DataFrame(election_combined_posts[:target_post_count])

# Count the number of posts in each DataFrame using len()
harris_post_count = len(harris_policy_df)
trump_post_count = len(trump_policy_df)
election_post_count = len(election_policy_df)

print(f"Number of posts for Kamala Harris: {harris_post_count}")
print(f"Number of posts for Donald Trump: {trump_post_count}")
print(f"Number of posts for 2024 Presidential Election: {election_post_count}")
