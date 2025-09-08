import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("t-SNE ANIME CLUSTERING ANALYSIS ")
print("="*45)

# Load and prepare the data

anime_df = pd.read_csv('data/anime.csv')
ratings_df = pd.read_csv('data/rating.csv')

# Clean the data

anime_df['rating'] = pd.to_numeric(anime_df['rating'], errors='coerce')
anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce')
anime_df['members'] = pd.to_numeric(anime_df['members'], errors='coerce')

# Remove rows with missing critical data

anime_clean = anime_df.dropna(subset=['rating', 'genre']).copy()
anime_clean = anime_clean[~anime_clean['genre'].str.contains('Hentai', case=False, na=False)].copy()
print(f"Dataset size after cleaning: {len(anime_clean)} anime")

# Calculate additional features from ratings data

rating_stats = ratings_df[ratings_df['rating'] != -1].groupby('anime_id').agg({'rating': ['mean', 'std', 'count']}).round(3)
rating_stats.columns = ['user_avg_rating', 'user_rating_std', 'user_rating_count']
rating_stats = rating_stats.reset_index()

# Merge with anime data

anime_features = anime_clean.merge(rating_stats, on='anime_id', how='left')

# Fill missing user statistics with defaults

anime_features['user_avg_rating'] = anime_features['user_avg_rating'].fillna(anime_features['rating'])
anime_features['user_rating_std'] = anime_features['user_rating_std'].fillna(0)
anime_features['user_rating_count'] = anime_features['user_rating_count'].fillna(1)

print(f"Final dataset size: {len(anime_features)} anime with user statistics")

# Feature Engineering for t-SNE

# 1. Numerical features

numerical_features = ['rating', 'episodes', 'members', 'user_avg_rating', 'user_rating_std', 'user_rating_count']
feature_matrix = anime_features[numerical_features].copy()

# Handle missing episodes (common in movies/specials)

feature_matrix['episodes'] = feature_matrix['episodes'].fillna(1)

# Log transform skewed features

feature_matrix['episodes_log'] = np.log1p(feature_matrix['episodes'])
feature_matrix['members_log'] = np.log1p(feature_matrix['members'])
feature_matrix['user_rating_count_log'] = np.log1p(feature_matrix['user_rating_count'])

# 2. Type encoding (one-hot)

type_dummies = pd.get_dummies(anime_features['type'], prefix='type')
feature_matrix = pd.concat([feature_matrix, type_dummies], axis=1)

# 3. Genre features using TF-IDF

# Clean and prepare genre text

genre_text = anime_features['genre'].fillna('Unknown').str.replace(',', ' ')
tfidf = TfidfVectorizer(max_features=50, stop_words=None)
genre_features = tfidf.fit_transform(genre_text).toarray()
genre_feature_names = [f'genre_{name}' for name in tfidf.get_feature_names_out()]
genre_df = pd.DataFrame(genre_features, columns=genre_feature_names, index=anime_features.index)

# Combine all features

final_features = pd.concat([feature_matrix, genre_df], axis=1)

print(f"Total features for t-SNE: {final_features.shape[1]}")
print("Features include: numerical stats, type encoding, and genre TF-IDF vectors")

# Standardise features

scaler = StandardScaler()
features_scaled = scaler.fit_transform(final_features)

print(f"Standardized feature matrix shape: {features_scaled.shape}")

# Prepare genre labels for coloring

# Extract primary genre (first genre listed)
def get_primary_genre(genre_string):
    if pd.isna(genre_string):
        return 'Unknown'
    genres = [g.strip() for g in str(genre_string).split(',')]
    return genres[0] if genres else 'Unknown'

anime_features['primary_genre'] = anime_features['genre'].apply(get_primary_genre)

# Get top genres for better visualization

top_genres = anime_features['primary_genre'].value_counts().head(12).index.tolist()
anime_features['genre_for_plot'] = anime_features['primary_genre'].apply(
    lambda x: x if x in top_genres else 'Other'
)

print(f"Using top {len(top_genres)} genres + 'Other' category for visualization")
print("Top genres:", top_genres)

# Apply t-SNE

# Sample data if too large (t-SNE can be slow)

samples = len(anime_clean)
if len(features_scaled) > samples:
    print(f"Sampling {samples} anime for t-SNE performance...")
    sample_idx = np.random.choice(len(features_scaled), samples, replace=False)
    features_for_tsne = features_scaled[sample_idx]
    anime_for_plot = anime_features.iloc[sample_idx].copy()
else:
    features_for_tsne = features_scaled
    anime_for_plot = anime_features.copy()

# Run t-SNE with optimized parameters
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42,
    verbose=1
)

tsne_results = tsne.fit_transform(features_for_tsne)

print("t-SNE completed!")
print(f"Final embedding shape: {tsne_results.shape}")

# Create the visualization
plt.figure(figsize=(16, 12))

# Define colors for genres
unique_genres = sorted(anime_for_plot['genre_for_plot'].unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
genre_color_map = dict(zip(unique_genres, colors))

print(f"\nCreating visualization with {len(unique_genres)} genre categories...")

# Create the main scatter plot
for genre in unique_genres:
    mask = anime_for_plot['genre_for_plot'] == genre
    if mask.sum() > 0:  # Only plot if there are points for this genre
        plt.scatter(
            tsne_results[mask, 0], 
            tsne_results[mask, 1],
            c=[genre_color_map[genre]], 
            label=f'{genre} (n={mask.sum()})',
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )

plt.title('t-SNE Visualization of Anime Clusters\nColor-coded by Primary Genre', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)

# Create legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

# Add some annotations for interesting clusters if possible
# Find some well-known anime to highlight
well_known_anime = ['Death Note', 'Attack on Titan', 'One Piece', 'Naruto', 
                   'Dragon Ball Z', 'Studio Ghibli', 'Cowboy Bebop']

for anime_name in well_known_anime:
    matches = anime_for_plot[anime_for_plot['name'].str.contains(anime_name, case=False, na=False)]
    if len(matches) > 0:
        idx = matches.index[0]
        pos_in_tsne = np.where(anime_for_plot.index == idx)[0]
        if len(pos_in_tsne) > 0:
            x, y = tsne_results[pos_in_tsne[0]]
            plt.annotate(matches.iloc[0]['name'][:20], 
                        xy=(x, y), xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=9)

plt.tight_layout()
plt.savefig("t-SNE_clusters.png")
plt.show()

# Additional analysis and insights
print("\nCLUSTERING INSIGHTS")
print("="*30)

# Analyze cluster characteristics
print("Analyzing t-SNE clusters...")

# Calculate cluster statistics by genre
genre_stats = anime_for_plot.groupby('genre_for_plot').agg({
    'rating': ['mean', 'std', 'count'],
    'members': ['mean', 'median'],
    'user_avg_rating': ['mean', 'std'],
    'episodes': ['mean', 'median']
}).round(2)

genre_stats.columns = ['_'.join(col).strip() for col in genre_stats.columns.values]


print("\nGenre Statistics in t-SNE Space:")
print("="*40)
for genre in unique_genres:
    if genre in genre_stats.index:
        stats = genre_stats.loc[genre]
        print(f"\n{genre}:")
        print(f"  • Count: {stats['rating_count']} anime")
        print(f"  • Avg Rating: {stats['rating_mean']} ± {stats['rating_std']}")
        print(f"  • Avg Members: {stats['members_mean']}")
        print(f"  • Avg Episodes: {stats['episodes_mean']}")


# Create a second visualization: Genre distribution in embedding space
plt.figure(figsize=(14, 10))

# Create subplots showing different aspects
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Color by rating
scatter1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                      c=anime_for_plot['rating'], cmap='viridis', 
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax1.set_title('t-SNE colored by Rating Score', fontsize=14, fontweight='bold')
ax1.set_xlabel('t-SNE Component 1')
ax1.set_ylabel('t-SNE Component 2')
plt.colorbar(scatter1, ax=ax1, label='Rating Score')

# Plot 2: Color by popularity (members)
scatter2 = ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                      c=np.log1p(anime_for_plot['members']), cmap='plasma', 
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax2.set_title('t-SNE colored by Popularity (log members)', fontsize=14, fontweight='bold')
ax2.set_xlabel('t-SNE Component 1')
ax2.set_ylabel('t-SNE Component 2')
plt.colorbar(scatter2, ax=ax2, label='Log(Members + 1)')

# Plot 3: Color by type
type_colors = plt.cm.Set1(np.linspace(0, 1, len(anime_for_plot['type'].unique())))
type_color_map = dict(zip(anime_for_plot['type'].unique(), type_colors))
for anime_type in anime_for_plot['type'].unique():
    mask = anime_for_plot['type'] == anime_type
    if mask.sum() > 0:
        ax3.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                   c=[type_color_map[anime_type]], label=anime_type,
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax3.set_title('t-SNE colored by Anime Type', fontsize=14, fontweight='bold')
ax3.set_xlabel('t-SNE Component 1')
ax3.set_ylabel('t-SNE Component 2')
ax3.legend()

# Plot 4: Color by episode count
episodes_for_color = anime_for_plot['episodes'].fillna(1)
scatter4 = ax4.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                      c=np.log1p(episodes_for_color), cmap='coolwarm', 
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax4.set_title('t-SNE colored by Episode Count (log)', fontsize=14, fontweight='bold')
ax4.set_xlabel('t-SNE Component 1')
ax4.set_ylabel('t-SNE Component 2')
plt.colorbar(scatter4, ax=ax4, label='Log(Episodes + 1)')

plt.tight_layout()
plt.suptitle('Multi-Perspective t-SNE Analysis of Anime Dataset', 
             fontsize=16, fontweight='bold', y=1.02)

plt.savefig("Genre_distribution.png")
plt.show()