import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load in election data
poll_data = pd.read_csv("president_polls.csv")

# Convert start_date to datetime
poll_data['start_date'] = pd.to_datetime(poll_data['start_date'])

# Filter out rows where 'state' is NaN
poll_data = poll_data[poll_data['state'].notna()]

# Exclude congressional districts by checking if 'CD' is not in the state value
poll_data = poll_data[~poll_data['state'].apply(lambda x: 'CD' in str(x))]

# Filter for the latest poll for each candidate in each state
latest_polls = (
    poll_data.sort_values('start_date')
    .groupby(['state', 'candidate_name'])
    .last()  # Keep the last entry for each candidate
    .reset_index()
)

# Filter for candidates
filtered_candidates = latest_polls[latest_polls['candidate_name'].isin(['Kamala Harris', 'Donald Trump'])]

average_pct_per_pollster = filtered_candidates.groupby(['pollster', 'candidate_name'])['pct'].mean().reset_index()
party_colors = {'Kamala Harris': 'blue', 'Donald Trump': 'red'}

#Support Over Time
current_year = datetime.datetime.now().year
filtered_data = poll_data[(poll_data['candidate_name'].isin(['Kamala Harris', 'Donald Trump'])) &
    (poll_data['start_date'].dt.year == current_year)]

# Apply a rolling average of 1 week for reducing noise
filtered_data['pct_smoothed'] = filtered_data.groupby('candidate_name')['pct'].transform(lambda x: x.rolling(7, min_periods=1).mean())

plt.figure(figsize=(14, 8))
sns.lineplot(data=filtered_data, x='start_date', y='pct_smoothed', hue='candidate_name', linewidth=2.5, palette=party_colors)
plt.xlabel("Date")
plt.ylabel("Smoothed Support Percentage (%)")
plt.title(f"Smoothed Support Percentage Over Time for Kamala Harris and Donald Trump ({current_year})")
plt.legend(title="Candidate")
plt.savefig("support_over_time.png", format='png', dpi=300, bbox_inches='tight')

#Top Candidate by State
sorted_polls = poll_data.sort_values(
    by=['pollscore', 'transparency_score', 'end_date'], 
    ascending=[True, False, False]
)

sorted_polls_candidates = sorted_polls[sorted_polls['candidate_name'].isin(['Kamala Harris', 'Donald Trump'])]
latest_per_state_pollster = sorted_polls_candidates.drop_duplicates(subset=['state', 'pollster'], keep='first')
state_avg_pct = latest_per_state_pollster.groupby(['state', 'candidate_name'])['pct'].mean().reset_index()
top_candidate_per_state = state_avg_pct.sort_values(by=['state', 'pct'], ascending=[True, False]).drop_duplicates(subset='state', keep='first')

plt.figure(figsize=(15, 10))
sns.barplot(data=top_candidate_per_state, x='state', y='pct', hue='candidate_name', palette=party_colors)
plt.xticks(rotation=90)
plt.xlabel("State")
plt.ylabel("Average Support Percentage (%)")
plt.title("Top Candidate by Average Support Percentage in Each State (Latest High-Quality Polls)")
plt.legend(title="Candidate")
plt.tight_layout()
plt.savefig("top_candidate_by_state.png", format='png', dpi=300, bbox_inches='tight')

#Swing States Average Support
swing_states = ['Arizona', 'Georgia', 'Michigan', 'Nevada', 'North Carolina', 'Pennsylvania', 'Wisconsin']
swing_state_avg = state_avg_pct[state_avg_pct['state'].isin(swing_states)]

plt.figure(figsize=(12, 8))
sns.barplot(data=swing_state_avg, x='state', y='pct', hue='candidate_name', palette=party_colors)
plt.xticks(rotation=45)
plt.xlabel("Swing State")
plt.ylabel("Average Support Percentage (%)")
plt.title("Average Support Percentage in Swing States for Kamala Harris and Donald Trump (Latest High-Quality Polls)")
plt.legend(title="Candidate")
plt.tight_layout()
plt.savefig("latest_high_quality_swing_states.png", format='png', dpi=300, bbox_inches='tight')

# Calculate average support percentages by methodology
#drop na values
poll_data = poll_data.dropna(subset=['pct', 'methodology'])

methodology_avg_pct = latest_polls[latest_polls['candidate_name'].isin(['Kamala Harris', 'Donald Trump'])].groupby(['methodology', 'candidate_name'])['pct'].mean().reset_index()

# Create a bar plot for average support percentages by methodology
plt.figure(figsize=(16, 10))
sns.barplot(data=methodology_avg_pct, x='methodology', y='pct', hue='candidate_name', palette=party_colors)
plt.xticks(rotation=80)
plt.xlabel("Polling Methodology")
plt.ylabel("Average Support Percentage (%)")
plt.title("Average Support Percentage by Polling Methodology for Kamala Harris and Donald Trump")
plt.legend(title="Candidate")
plt.tight_layout()
plt.savefig("support_by_methodology.png", format='png', dpi=300, bbox_inches='tight')

#plt.show()

# Calculate overall average percentages for both candidates
overall_avg_pct = filtered_candidates.groupby('candidate_name')['pct'].mean().reset_index()

# Display overall average percentages
print("Overall Average Support Percentages:")
print(overall_avg_pct)

# Determine the candidate with the highest overall percentage
leading_candidate = overall_avg_pct.loc[overall_avg_pct['pct'].idxmax()]

print(f"\nLeading Candidate Overall:\n{leading_candidate['candidate_name']} with an average support of {leading_candidate['pct']:.2f}%")