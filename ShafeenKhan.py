import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import streamlit as st
from wordcloud import WordCloud
import plotly.express as px

# Function to fetch anime image
def get_anime_image(anime_name, anime_df):
    try:
        image_url = anime_df[anime_df['Name'] == anime_name]['Image URL'].values[0]
        return image_url
    except:
        return "https://via.placeholder.com/150"

# Set up Streamlit app title and description
st.title("🎬 Anime Recommendation System Based on FP-Growth Algorithm")

st.markdown("""
This app allows users to select an anime and recommends other animes that are frequently watched together based on the genres of the selected anime.
The recommendations are generated using the FP-Growth algorithm on a dataset of anime and their ratings.
""")

# Load the datasets
anime_df = pd.read_csv("E:\Semester 5\Machine-Learning-Theory\Association-Rule-Mining-Project\FP-Growth\Anime-Recommendation-System-Based-on-FP-Growth-Algorithm\\anime-dataset-2023.csv")
movies_df = pd.read_csv("E:\Semester 5\Machine-Learning-Theory\Association-Rule-Mining-Project\FP-Growth\Anime-Recommendation-System-Based-on-FP-Growth-Algorithm\\ratings_small.csv")

# Display the initial datasets
col1, col2 = st.columns(2)
with col1:
    st.subheader("Initial Movie Ratings Dataset")
    st.dataframe(movies_df.head())
with col2:
    st.subheader("Initial Anime Dataset")
    st.dataframe(anime_df.head())

# Data Preprocessing
movies_df = movies_df.iloc[:24905]
anime_df.replace(['UNKNOWN', 'Unknown'], np.nan, inplace=True)
anime_df = anime_df.ffill()
anime_df.drop(["Other name"], axis=1, inplace=True)

# Merge relevant columns between datasets
anime_df["Movies_id"] = movies_df["movieId"]
anime_df["user_id"] = movies_df["    userId"]
ready_df = anime_df[["Genres", "anime_id", "user_id", "Type", "Score", "Name","Popularity","Image URL"]].copy()

# Preparing data for FP-Growth
genres_list = ready_df['Genres'].apply(lambda x: x.split(', ')).tolist()
te = TransactionEncoder()
te_ary = te.fit(genres_list).transform(genres_list)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Applying FP-Growth algorithm
frequent_itemsets = fpgrowth(df_encoded, min_support=0.04, use_colnames=True)
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: list(x))

# Key Insights Section in Sidebar
st.sidebar.header("Key Insights")
rating_mean = ready_df['Score'].astype(float).mean()
rating_median = ready_df['Score'].astype(float).median()
genre_counts = ready_df['Genres'].str.split(', ', expand=True).stack().value_counts()
top_genre = genre_counts.idxmax()
most_popular_anime = ready_df.sort_values(by='Popularity').iloc[0]['Name']

st.sidebar.write(f"**Average Score:** {rating_mean:.2f}")
st.sidebar.write(f"**Median Score:** {rating_median:.2f}")
st.sidebar.write(f"**Top Genre:** {top_genre} ({genre_counts.max()} animes)")
st.sidebar.write(f"**Most Popular Anime:** {most_popular_anime}")

# Visualizations
st.subheader("Genre Distribution")

# Genre Distribution Bar Chart
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis', ax=ax)
ax.set_xlabel("Number of Animes")
ax.set_ylabel("Genres")
ax.set_title("Top 20 Genres Distribution")
st.pyplot(fig)

# Top-Rated Animes
st.subheader("Top-Rated Animes")
top_rated = ready_df.sort_values(by='Score', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='Score', y='Name', data=top_rated, palette='coolwarm', ax=ax)
ax.set_xlabel("Score")
ax.set_ylabel("Anime Name")
ax.set_title("Top 10 Rated Animes")
st.pyplot(fig)

# Interactive Popularity vs Score Scatter Plot with Plotly
st.subheader("Popularity vs. Score (Interactive)")
fig = px.scatter(ready_df, x='Popularity', y='Score', color='Score',
                 hover_data=['Name', 'Genres'],
                 title="Popularity vs. Score",
                 labels={'Popularity': 'Popularity Rank', 'Score': 'Score'})
st.plotly_chart(fig)

# Genre Word Cloud
st.subheader("Genre Word Cloud")
genres_exploded = ready_df['Genres'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
genre_text = ' '.join(genres_exploded)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(genre_text)

fig, ax = plt.subplots(figsize=(15, 7.5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Recommendation Section
st.subheader("Anime Recommendations")

# Create a selection box for users to select a movie
movie_list = ready_df['Name'].unique()
selected_movie = st.selectbox("Choose an Anime", movie_list)

# Filter the dataset based on the selected movie
selected_genres = ready_df[ready_df['Name'] == selected_movie]['Genres'].values[0].split(", ")
st.write(f"**Genres of the selected anime ({selected_movie}):** {', '.join(selected_genres)}")

# Find frequent patterns that contain the genres of the selected movie
matching_patterns = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: set(selected_genres).issubset(set(x)))]
recommended_genres = matching_patterns['itemsets'].apply(lambda x: ', '.join(x))

# Recommend movies that share the same genres as the selected movie
recommended_movies = ready_df[ready_df['Genres'].apply(lambda x: any(genre in x for genre in selected_genres))]['Name'].unique()

st.subheader("Recommended Animes")

# Display recommended animes in a grid layout with images
cols = st.columns(5)
for idx, anime in enumerate(recommended_movies[:20]):  # Limit to first 20 recommendations
    with cols[idx % 5]:
        image_url = get_anime_image(anime, anime_df)
        st.image(image_url, use_column_width=True)
        st.caption(anime)

# Sidebar Filters
st.sidebar.header("Filter Recommendations")
score_min, score_max = st.sidebar.slider("Score Range", 1.0, 10.0, (5.0, 10.0))
anime_type = st.sidebar.multiselect("Type", options=ready_df['Type'].unique(), default=ready_df['Type'].unique())

# Apply Filters
filtered_df = ready_df[
    (ready_df['Score'].astype(float) >= score_min) &
    (ready_df['Score'].astype(float) <= score_max) &
    (ready_df['Type'].isin(anime_type))
]

# Update genres list and frequent itemsets based on filtered data
genres_list_filtered = filtered_df['Genres'].apply(lambda x: x.split(', ')).tolist()
te_filtered = TransactionEncoder()
te_ary_filtered = te_filtered.fit(genres_list_filtered).transform(genres_list_filtered)
df_encoded_filtered = pd.DataFrame(te_ary_filtered, columns=te_filtered.columns_)

frequent_itemsets_filtered = fpgrowth(df_encoded_filtered, min_support=0.04, use_colnames=True)
frequent_itemsets_filtered['itemsets'] = frequent_itemsets_filtered['itemsets'].apply(lambda x: list(x))


# Enhanced Hover Information with Plotly (if using Plotly for recommendations)
# Alternatively, use Streamlit's built-in features

# Example: Display additional details on click
for idx, anime in enumerate(recommended_movies[:20]):
    with cols[idx % 5]:
        image_url = get_anime_image(anime, anime_df)
        st.image(image_url, use_column_width=True)
        st.caption(anime)
        # Display additional info
        anime_details = ready_df[ready_df['Name'] == anime].iloc[0]
        st.markdown(f"**Score:** {anime_details['Score']}")
        st.markdown(f"**Genres:** {anime_details['Genres']}")

# Optionally, display the Python code
# Optionally, display the Python code with better formatting
with st.expander("Show Python Code"):
    code = '''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import streamlit as st
from wordcloud import WordCloud
import plotly.express as px

# Function to fetch anime image
def get_anime_image(anime_name, anime_df):
    try:
        image_url = anime_df[anime_df['Name'] == anime_name]['Image URL'].values[0]
        return image_url
    except:
        return "https://via.placeholder.com/150"

# Set up Streamlit app title and description
st.title("🎬 Anime Recommendation System Based on FP-Growth Algorithm")

st.markdown("""
This app allows users to select an anime and recommends other animes that are frequently watched together based on the genres of the selected anime.
The recommendations are generated using the FP-Growth algorithm on a dataset of anime and their ratings.
""")

# Load the datasets
anime_df = pd.read_csv("E:/Semester 5/Machine-Learning-Theory/Association-Rule-Mining-Project/anime-dataset-2023.csv/anime-dataset-2023.csv")
movies_df = pd.read_csv("E:/Semester 5/Machine-Learning-Theory/Association-Rule-Mining-Project/anime-dataset-2023.csv/ratings_small.csv")

# Display the initial datasets
col1, col2 = st.columns(2)
with col1:
    st.subheader("Initial Movie Ratings Dataset")
    st.dataframe(movies_df.head())
with col2:
    st.subheader("Initial Anime Dataset")
    st.dataframe(anime_df.head())

# Data Preprocessing
movies_df = movies_df.iloc[:24905]
anime_df.replace(['UNKNOWN', 'Unknown'], np.nan, inplace=True)
anime_df = anime_df.ffill()
anime_df.drop(["Other name"], axis=1, inplace=True)

# Merge relevant columns between datasets
anime_df["Movies_id"] = movies_df["movieId"]
anime_df["user_id"] = movies_df["    userId"]
ready_df = anime_df[["Genres", "anime_id", "user_id", "Type", "Score", "Name","Popularity","Image URL"]].copy()

# Preparing data for FP-Growth
genres_list = ready_df['Genres'].apply(lambda x: x.split(', ')).tolist()
te = TransactionEncoder()
te_ary = te.fit(genres_list).transform(genres_list)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Applying FP-Growth algorithm
frequent_itemsets = fpgrowth(df_encoded, min_support=0.04, use_colnames=True)
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: list(x))

# Key Insights Section in Sidebar
st.sidebar.header("Key Insights")
rating_mean = ready_df['Score'].astype(float).mean()
rating_median = ready_df['Score'].astype(float).median()
genre_counts = ready_df['Genres'].str.split(', ', expand=True).stack().value_counts()
top_genre = genre_counts.idxmax()
most_popular_anime = ready_df.sort_values(by='Popularity').iloc[0]['Name']

st.sidebar.write(f"**Average Score:** {rating_mean:.2f}")
st.sidebar.write(f"**Median Score:** {rating_median:.2f}")
st.sidebar.write(f"**Top Genre:** {top_genre} ({genre_counts.max()} animes)")
st.sidebar.write(f"**Most Popular Anime:** {most_popular_anime}")

# Visualizations
st.subheader("Genre Distribution")

# Genre Distribution Bar Chart
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis', ax=ax)
ax.set_xlabel("Number of Animes")
ax.set_ylabel("Genres")
ax.set_title("Top 20 Genres Distribution")
st.pyplot(fig)

# Top-Rated Animes
st.subheader("Top-Rated Animes")
top_rated = ready_df.sort_values(by='Score', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='Score', y='Name', data=top_rated, palette='coolwarm', ax=ax)
ax.set_xlabel("Score")
ax.set_ylabel("Anime Name")
ax.set_title("Top 10 Rated Animes")
st.pyplot(fig)

# Interactive Popularity vs Score Scatter Plot with Plotly
st.subheader("Popularity vs. Score (Interactive)")
fig = px.scatter(ready_df, x='Popularity', y='Score', color='Score',
                 hover_data=['Name', 'Genres'],
                 title="Popularity vs. Score",
                 labels={'Popularity': 'Popularity Rank', 'Score': 'Score'})
st.plotly_chart(fig)

# Genre Word Cloud
st.subheader("Genre Word Cloud")
genres_exploded = ready_df['Genres'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
genre_text = ' '.join(genres_exploded)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(genre_text)

fig, ax = plt.subplots(figsize=(15, 7.5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Recommendation Section
st.subheader("Anime Recommendations")

# Create a selection box for users to select a movie
movie_list = ready_df['Name'].unique()
selected_movie = st.selectbox("Choose an Anime", movie_list)

# Filter the dataset based on the selected movie
selected_genres = ready_df[ready_df['Name'] == selected_movie]['Genres'].values[0].split(", ")
st.write(f"**Genres of the selected anime ({selected_movie}):** {', '.join(selected_genres)}")

# Find frequent patterns that contain the genres of the selected movie
matching_patterns = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: set(selected_genres).issubset(set(x)))]
recommended_genres = matching_patterns['itemsets'].apply(lambda x: ', '.join(x))

# Recommend movies that share the same genres as the selected movie
recommended_movies = ready_df[ready_df['Genres'].apply(lambda x: any(genre in x for genre in selected_genres))]['Name'].unique()

st.subheader("Recommended Animes")

# Display recommended animes in a grid layout with images
cols = st.columns(5)
for idx, anime in enumerate(recommended_movies[:20]):  # Limit to first 20 recommendations
    with cols[idx % 5]:
        image_url = get_anime_image(anime, anime_df)
        st.image(image_url, use_column_width=True)
        st.caption(anime)

# Sidebar Filters
st.sidebar.header("Filter Recommendations")
score_min, score_max = st.sidebar.slider("Score Range", 1.0, 10.0, (5.0, 10.0))
anime_type = st.sidebar.multiselect("Type", options=ready_df['Type'].unique(), default=ready_df['Type'].unique())

# Apply Filters
filtered_df = ready_df[
    (ready_df['Score'].astype(float) >= score_min) &
    (ready_df['Score'].astype(float) <= score_max) &
    (ready_df['Type'].isin(anime_type))
]

# Update genres list and frequent itemsets based on filtered data
genres_list_filtered = filtered_df['Genres'].apply(lambda x: x.split(', ')).tolist()
te_filtered = TransactionEncoder()
te_ary_filtered = te_filtered.fit(genres_list_filtered).transform(genres_list_filtered)
df_encoded_filtered = pd.DataFrame(te_ary_filtered, columns=te_filtered.columns_)

frequent_itemsets_filtered = fpgrowth(df_encoded_filtered, min_support=0.04, use_colnames=True)
frequent_itemsets_filtered['itemsets'] = frequent_itemsets_filtered['itemsets'].apply(lambda x: list(x))


# Enhanced Hover Information with Plotly (if using Plotly for recommendations)
# Alternatively, use Streamlit's built-in features

# Example: Display additional details on click
for idx, anime in enumerate(recommended_movies[:20]):
    with cols[idx % 5]:
        image_url = get_anime_image(anime, anime_df)
        st.image(image_url, use_column_width=True)
        st.caption(anime)
        # Display additional info
        anime_details = ready_df[ready_df['Name'] == anime].iloc[0]
        st.markdown(f"**Score:** {anime_details['Score']}")
        st.markdown(f"**Genres:** {anime_details['Genres']}")
    '''
    st.code(code, language='python')

