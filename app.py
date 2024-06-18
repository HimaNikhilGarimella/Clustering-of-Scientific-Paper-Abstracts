import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import umap
from scipy.cluster.hierarchy import dendrogram, linkage

# Function to load data and preprocess
@st.cache_data
def load_data(file_path, sample_size=1000):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()  # Ensure all column names are lowercase
    df = df.dropna(subset=['abstract'])  # Remove rows with missing abstracts
    df = df.sample(n=sample_size, random_state=42)  # Sample a subset of data
    return df

# Function to preprocess text data
@st.cache_data
def preprocess_text(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text)
    return X, vectorizer

# Function to perform KMeans clustering
def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

# Function to perform Agglomerative Clustering (Hierarchical Clustering)
def hierarchical_clustering(X, n_clusters):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agg_clustering.fit_predict(X.toarray())
    return clusters, agg_clustering

# Function to perform Spectral Clustering
def spectral_clustering(X, n_clusters):
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
    clusters = spectral.fit_predict(X)
    return clusters, spectral

# Function to visualize clusters using TruncatedSVD or TSNE
def visualize_clusters(X, clusters, method='TruncatedSVD'):
    if method == 'TruncatedSVD':
        if isinstance(X, np.ndarray) or isinstance(X, list):
            reducer = TruncatedSVD(n_components=2, random_state=42)
        else:
            reducer = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42)
    elif method == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'UMAP':
        reducer = umap.UMAP(random_state=42)

    X_reduced = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    ax.set_title(f'Clustering Visualization using {method}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.colorbar(scatter, ax=ax)

    st.pyplot(fig)

# Function to plot dendrogram
def plot_dendrogram(X):
    linkage_matrix = linkage(X.todense(), 'ward')
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=3, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Sample Index or (Cluster Size)')
    ax.set_ylabel('Distance')
    st.pyplot(fig)

# Function to calculate and plot clustering evaluation metrics
def plot_clustering_metrics(X, clusters):
    if len(np.unique(clusters)) > 1:
        metrics = {
            'Calinski-Harabasz Score': calinski_harabasz_score(X.toarray(), clusters),
            'Davies-Bouldin Score': davies_bouldin_score(X.toarray(), clusters),
            'Silhouette Score': silhouette_score(X.toarray(), clusters)
        }

        st.subheader('Clustering Evaluation Metrics')
        for metric, value in metrics.items():
            st.write(f'{metric}: {value:.4f}')
    else:
        st.warning("Cannot calculate clustering metrics. Ensure more than one cluster is formed.")

# Function to display error plots
def plot_error_plots(X, method):
    if method == 'KMeans':
        inertia_values = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia_values.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(2, 11), inertia_values, marker='o', linestyle='-', color='b')
        ax.set_title('Elbow Method for Optimal K')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia')
        st.pyplot(fig)

# Function to recommend papers based on user input
@st.cache_data
def recommend_papers(user_input, df, _vectorizer, _X):
    user_input_vectorized = _vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vectorized, _X)
    indices = similarities.argsort()[0][::-1]
    recommendations = df.iloc[indices[:5]]  # Get top 5 recommended papers
    scores = similarities[0, indices[:5]]  # Get cosine similarity scores
    return recommendations, scores

def main():
    file_path = "mini_project_dataset.csv"  # Update with your actual file path

    # Sidebar options
    st.sidebar.header('Options')
    sample_size = st.sidebar.slider('Sample Size', min_value=100, max_value=5000, value=1000, step=100)

    # Load dataset and preprocess
    df = load_data(file_path, sample_size)
    abstracts = df['abstract'].tolist()
    X, vectorizer = preprocess_text(abstracts)

    # Clustering methods and parameters
    clustering_methods = {
        'KMeans': kmeans_clustering,
        'Hierarchical Clustering': hierarchical_clustering,
        'Spectral Clustering': spectral_clustering
    }

    # Sidebar options
    st.sidebar.header('Options')
    selected_method = st.sidebar.selectbox('Select Clustering Method', list(clustering_methods.keys()))
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=20, value=5)

    clusters = None
    if selected_method == 'Hierarchical Clustering':
        clusters, model = clustering_methods[selected_method](X, n_clusters)
    else:
        if selected_method == 'KMeans':
            clusters, model = clustering_methods[selected_method](X, n_clusters)
        else:
            clusters, model = clustering_methods[selected_method](X, n_clusters)

    # User input for recommendations
    st.header('Get Paper Recommendations')
    user_input = st.text_area('Enter your abstract here:', value="Enter your abstract here.")
    if st.button('Get Recommendations'):
        if user_input:
            recommendations, scores = recommend_papers(user_input, df, vectorizer, X)
            st.subheader('Top Recommendations:')
            for i, (index, row) in enumerate(recommendations.iterrows()):
                st.write(f"Recommendation {i + 1}:")
                st.write(f"Title: {row['title']}")
                st.write(f"Authors: {row['authors']}")
                st.write(f"Abstract: {row['abstract']}")
                st.write(f"Score: {scores[i]:.4f}")
                st.markdown("---")

    # Display clustering evaluation options
    if st.sidebar.checkbox('Show Clustering Evaluation'):
        st.subheader('Clustering Evaluation')
        st.write(f"Clustering Method: {selected_method}")
        st.write(f"Number of Clusters: {n_clusters}")

        plot_clustering_metrics(X, clusters)

    # Display dendrogram plot if requested
    if st.sidebar.checkbox('Show Dendrogram') and selected_method == 'Hierarchical Clustering':
        st.subheader('Hierarchical Clustering Dendrogram')
        plot_dendrogram(X)

    # Display error plot for KMeans if requested
    if selected_method == 'KMeans' and st.sidebar.checkbox('Show Error Plot'):
        st.subheader('Elbow Method for Optimal K')
        plot_error_plots(X, selected_method)

    # Display clustering visualization
    st.header('Clustering Visualization')
    st.subheader(f'Clustering Visualization using {selected_method}')
    visualize_clusters(X, clusters)

    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable warning for matplotlib global use

if __name__ == "__main__":
    main()
