import polls
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import geopandas as gpd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.table import Table

# Define electoral votes by state
electoral_votes = {
    'Alabama': 9, 'Alaska': 3, 'Arizona': 11, 'Arkansas': 6, 'California': 55, 'Colorado': 9,
    'Connecticut': 7, 'Delaware': 3, 'Florida': 29, 'Georgia': 16, 'Hawaii': 4, 'Idaho': 4,
    'Illinois': 20, 'Indiana': 11, 'Iowa': 6, 'Kansas': 6, 'Kentucky': 8, 'Louisiana': 8,
    'Maine': 4, 'Maryland': 10, 'Massachusetts': 11, 'Michigan': 16, 'Minnesota': 10, 
    'Mississippi': 6, 'Missouri': 10, 'Montana': 3, 'Nebraska': 5, 'Nevada': 6, 'New Hampshire': 4,
    'New Jersey': 14, 'New Mexico': 5, 'New York': 29, 'North Carolina': 15, 'North Dakota': 3,
    'Ohio': 18, 'Oklahoma': 7, 'Oregon': 7, 'Pennsylvania': 20, 'Rhode Island': 4,
    'South Carolina': 9, 'South Dakota': 3, 'Tennessee': 11, 'Texas': 38, 'Utah': 6,
    'Vermont': 3, 'Virginia': 13, 'Washington': 12, 'West Virginia': 5, 'Wisconsin': 10,
    'Wyoming': 3
}

# Load poll data
poll_data = polls.poll_data

# Filter for relevant candidates and remove rows with missing values in 'pct' and 'sample_size'
poll_data = poll_data.dropna(subset=['pct', 'sample_size'])
poll_data = poll_data[poll_data['candidate_name'].isin(['Kamala Harris', 'Donald Trump'])]

# Aggregate by state and candidate using average percentage weighted by sample size
weighted_averages = poll_data.groupby(['state', 'candidate_name']).apply(
    lambda x: np.average(x['pct'], weights=x['sample_size'])
).reset_index(name='weighted_pct')

# Normalize to 100% within each state
weighted_averages['total_weighted_pct'] = weighted_averages.groupby('state')['weighted_pct'].transform('sum')
weighted_averages['adjusted_pct'] = weighted_averages['weighted_pct'] / weighted_averages['total_weighted_pct'] * 100

# Convert electoral votes dictionary to DataFrame for easy merging
electoral_votes_df = pd.DataFrame(list(electoral_votes.items()), columns=['state', 'electoral_votes'])

# Merge electoral votes data with weighted averages
weighted_averages = weighted_averages.merge(electoral_votes_df, on='state', how='left')

# Monte Carlo Simulation with Confidence Intervals
num_simulations = 10000
results = {'Kamala Harris': [], 'Donald Trump': []}

for _ in range(num_simulations):
    total_votes = {'Kamala Harris': 0, 'Donald Trump': 0}
    
    for _, row in weighted_averages.iterrows():
        candidate = row['candidate_name']
        state = row['state']
        
        # Apply a normal distribution around the adjusted percentage
        simulated_pct = np.random.normal(row['adjusted_pct'], 2)  # Standard deviation of 2 to add uncertainty
        
        # Assign winner for the state
        if simulated_pct > 50:
            total_votes[candidate] += row['electoral_votes']
    
    # Record the result for each candidate
    results['Kamala Harris'].append(total_votes['Kamala Harris'])
    results['Donald Trump'].append(total_votes['Donald Trump'])

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)

# Confidence intervals for each candidate
kamala_mean = results_df['Kamala Harris'].mean()
kamala_std = results_df['Kamala Harris'].std()
kamala_ci_lower = kamala_mean - 1.96 * kamala_std
kamala_ci_upper = kamala_mean + 1.96 * kamala_std

trump_mean = results_df['Donald Trump'].mean()
trump_std = results_df['Donald Trump'].std()
trump_ci_lower = trump_mean - 1.96 * trump_std
trump_ci_upper = trump_mean + 1.96 * trump_std

#Print the confidence intervals:
print(f'Kamala Harris confidence interval: {kamala_ci_lower} to {kamala_ci_upper}')
print(f'Donald Trump confidence interval: {trump_ci_lower} to {trump_ci_upper}')

# Plot the density of electoral votes with confidence intervals
plt.figure(figsize=(10, 6))

# Density plot for Kamala Harris
sns.kdeplot(results_df['Kamala Harris'], color='blue', label='Kamala Harris', fill=True)
plt.axvline(kamala_mean, color='blue', linestyle='--')
plt.axvspan(kamala_ci_lower, kamala_ci_upper, color='blue', alpha=0.2, label="95% CI - Kamala Harris")

# Density plot for Donald Trump
sns.kdeplot(results_df['Donald Trump'], color='red', label='Donald Trump', fill=True)
plt.axvline(trump_mean, color='red', linestyle='--')
plt.axvspan(trump_ci_lower, trump_ci_upper, color='red', alpha=0.2, label="95% CI - Donald Trump")

# Reference line for winning threshold
plt.axvline(270, color='black', linestyle='--', label="270 Vote Threshold")
plt.xlabel("Electoral Votes")
plt.ylabel("Density")
plt.legend()
plt.title("Electoral Vote Distribution with 95% Confidence Intervals")
plt.savefig('electoral_vote_distribution.png',format='png', dpi=300, bbox_inches='tight')
plt.show()


#visualizing it on the map


gdf = gpd.read_file("cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
gdf = gdf[gdf['NAME'].isin(electoral_votes.keys())]

# Determine winner for each state directly from weighted_averages
state_winners = weighted_averages.loc[weighted_averages.groupby('state')['adjusted_pct'].idxmax()]
state_winners = state_winners[['state', 'candidate_name']].rename(columns={'candidate_name': 'winner'})

# Merge with GeoDataFrame
gdf = gdf.merge(state_winners, left_on='NAME', right_on='state', how='left')

# Define colors and create a better color scheme
color_map = {'Kamala Harris': '#2166ac', 'Donald Trump': '#b2182b'}  # Using ColorBrewer colors
gdf['color'] = gdf['winner'].map(color_map)

# Split the data into continental, Alaska, and Hawaii
continental = gdf[~gdf['NAME'].isin(['Alaska', 'Hawaii'])]
alaska = gdf[gdf['NAME'] == 'Alaska']
hawaii = gdf[gdf['NAME'] == 'Hawaii']

# Create figure with adjusted size ratio
fig = plt.figure(figsize=(15, 8))

# Create GridSpec with better proportions
gs = plt.GridSpec(2, 20, height_ratios=[4, 1])

# Create main map and inset axes with adjusted positions
ax_main = fig.add_subplot(gs[0:1, :20])     
ax_ak = fig.add_subplot(gs[0, 17:18])       
ax_hi = fig.add_subplot(gs[0, 18:19])       

# Plot function remains the same
def plot_state_map(ax, data, title=None):
    data.plot(
        ax=ax,
        color=data['color'],
        edgecolor='white',
        linewidth=0.5
    )
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8, pad=2)

# Plot the states
plot_state_map(ax_main, continental)
plot_state_map(ax_ak, alaska, 'Alaska')
plot_state_map(ax_hi, hawaii, 'Hawaii')

# Add state labels for continental US
for idx, row in continental.iterrows():
    centroid = row['geometry'].centroid
    state_abbrev = row['NAME'][:2]
    electoral_vote = electoral_votes.get(row['NAME'], 0)
    text_color = 'white' if row['color'] == '#2166ac' else 'black'
    
    ax_main.annotate(
        f'{state_abbrev}\n{electoral_vote}',
        xy=(centroid.x, centroid.y),
        xytext=(0, 0),
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=8,
        color=text_color,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.5',
            fc=row['color'],
            ec='none',
            alpha=0.8
        )
    )

# Add labels for Alaska and Hawaii with smaller font
for ax, state, xytext_offset in [(ax_ak, 'Alaska', (-15, -8)), (ax_hi, 'Hawaii', (15, -5))]:
    state_data = gdf[gdf['NAME'] == state].iloc[0]
    centroid = state_data['geometry'].centroid
    electoral_vote = electoral_votes.get(state, 0)
    text_color = 'white' if state_data['color'] == '#2166ac' else 'black'
    
    ax.annotate(
        f'{state[:2]}\n{electoral_vote}',
        xy=(centroid.x, centroid.y),
        xytext=xytext_offset,  # Offset the text position
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=6,
        color=text_color,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.5',
            fc=state_data['color'],
            ec='none',
            alpha=0.8
        )
    )

# Calculate total electoral votes for each candidate
harris_votes = sum(electoral_votes[state] for state, winner in state_winners.set_index('state')['winner'].items() if winner == 'Kamala Harris')
trump_votes = sum(electoral_votes[state] for state, winner in state_winners.set_index('state')['winner'].items() if winner == 'Donald Trump')

# Add summary box
summary_text = (
    f"Projected Electoral Votes\n"
    f"─────────────────\n"
    f"Kamala Harris: {harris_votes}\n"
    f"Donald Trump: {trump_votes}\n"
    f"─────────────────\n"
    f"Winner: {'Kamala Harris' if harris_votes > trump_votes else 'Donald Trump'}\n"
    f"({abs(harris_votes - trump_votes)} vote margin)"
)

# Add summary box with a nice background
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='purple')
ax_main.text(0.02, 0.02, summary_text, transform=ax_main.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=props, family='monospace')

# Add legend with smaller size
legend_elements = [
    Patch(facecolor='#2166ac', label='Kamala Harris', edgecolor='none'),
    Patch(facecolor='#b2182b', label='Donald Trump', edgecolor='none')
]
ax_main.legend(
    handles=legend_elements,
    loc='upper right',
    title='Projected Winner',
    frameon=True,
    framealpha=0.8,
    edgecolor='none',
    fontsize=8,
    title_fontsize=8
)

# Add title directly above the map with less padding
plt.suptitle('Projected Electoral Votes by State', fontsize=14, y=0.95)

# Adjust layout to be more compact
plt.tight_layout()

# Save the figure
plt.savefig('electoral_map.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

