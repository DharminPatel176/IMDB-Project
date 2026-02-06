import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config
st.set_page_config(page_title="IMDb Movie Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Load the cleaned dataset
    df = pd.read_csv('Cleaned_DataSet.csv')
    return df

df = load_data()

# Title
st.title("🎬 IMDb Movie Analysis Dashboard")
st.markdown("Exploring the relationships between movie features, budgets, and IMDb scores.")

# Sidebar - Filters
st.sidebar.header("Filters")
selected_country = st.sidebar.multiselect("Select Country", options=df['country'].dropna().unique(), default=None)
selected_rating = st.sidebar.multiselect("Select Content Rating", options=df['content_rating'].dropna().unique(), default=None)

# Apply filters
filtered_df = df.copy()
if selected_country:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_country)]
if selected_rating:
    filtered_df = filtered_df[filtered_df['content_rating'].isin(selected_rating)]

# Layout: 3 Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Movies", len(filtered_df))
col2.metric("Avg IMDb Score", round(filtered_df['imdb_score'].mean(), 2))
col3.metric("Avg Budget", round(filtered_df['budget'].mean(), 2))

# Row 1: Distribution and Correlation
st.subheader("Data Distributions & Correlations")
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.write("### IMDb Score Distribution")
    fig = px.histogram(filtered_df, x="imdb_score", nbins=30, marginal="box", 
                       title="Distribution of IMDb Scores", color_discrete_sequence=['indianred'])
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    st.write("### Feature Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64']).drop(columns=['title_year'], errors='ignore')
    corr = numeric_df.corr()
    fig_heat, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig_heat)

# Row 2: Financials and Ratings
st.subheader("Financial Performance & Content Ratings")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.write("### Budget vs Gross Revenue")
    fig_scatter = px.scatter(filtered_df, x="budget", y="gross", 
                             hover_name="movie_title", color="imdb_score",
                             title="Budget vs Gross (colored by Score)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with row2_col2:
    st.write("### Avg Score by Content Rating")
    rating_stats = filtered_df.groupby('content_rating')['imdb_score'].mean().reset_index().sort_values('imdb_score', ascending=False)
    fig_bar = px.bar(rating_stats, x='content_rating', y='imdb_score', 
                     title="Average IMDb Score by Content Rating", color='imdb_score')
    st.plotly_chart(fig_bar, use_container_width=True)

# Row 3: Top Performers
st.subheader("Top Performers")
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.write("### Top 10 Directors (Avg Score)")
    dir_counts = filtered_df['director_name'].value_counts()
    eligible_dirs = dir_counts[dir_counts >= 2].index 
    dir_scores = filtered_df[filtered_df['director_name'].isin(eligible_dirs)].groupby('director_name')['imdb_score'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_dir = px.bar(dir_scores, x='imdb_score', y='director_name', orientation='h', 
                     title="Top 10 Directors (Min 2 Movies)", color='imdb_score')
    st.plotly_chart(fig_dir, use_container_width=True)

with row3_col2:
    st.write("### Language Distribution")
    lang_counts = filtered_df['language'].value_counts().head(10).reset_index()
    fig_pie = px.pie(lang_counts, values='count', names='language', title="Top 10 Languages")
    st.plotly_chart(fig_pie, use_container_width=True)

# Data Table
if st.checkbox("Show Raw Data"):
    st.write(filtered_df.head(100))