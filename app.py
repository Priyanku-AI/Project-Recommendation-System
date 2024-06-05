import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path, header=None)
    return data

# Vectorize the dataset
@st.cache_data
def vectorize_dataset(projects):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(projects)
    return tfidf_matrix, vectorizer

# Calculate similarity matrix
@st.cache_data
def calculate_similarity_matrix(_tfidf_matrix):  # Add leading underscore to exclude from hashing
    cosine_sim = cosine_similarity(_tfidf_matrix, _tfidf_matrix)
    return cosine_sim

# Recommendation function
def recommend_projects(projects, cosine_sim, input_projects, num_recommendations=10):
    cumulative_scores = {}
    
    for project_title in input_projects:
        project_idx = projects.index(project_title)
        sim_scores = list(enumerate(cosine_sim[project_idx]))
        sim_scores = [(idx, score) for idx, score in sim_scores if projects[idx] != project_title]

        for idx, score in sim_scores:
            if projects[idx] not in cumulative_scores:
                cumulative_scores[projects[idx]] = score
            else:
                cumulative_scores[projects[idx]] += score

    sorted_scores = sorted(cumulative_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_projects = [project[0] for project in sorted_scores[:num_recommendations]]

    return recommended_projects

# Main function
def main():
    st.title("Project Recommendation System")

    # Load data
    file_path = "Data_RS.xlsx"
    data_load_state = st.text("Loading data...")
    data = load_data(file_path)
    data_load_state.text("Data loaded successfully!")

    # Prepare dataset
    projects = data[0].tolist()
    tfidf_matrix, vectorizer = vectorize_dataset(projects)
    cosine_sim = calculate_similarity_matrix(tfidf_matrix)

    # Get user input
    input_projects = st.text_input("Enter project titles (comma-separated):")

    if input_projects:
        input_projects = [p.strip() for p in input_projects.split(",")]
        recommended_projects = recommend_projects(projects, cosine_sim, input_projects)
        st.subheader("Recommended Projects:")
        for project in recommended_projects:
            st.write(f"- {project}")

if __name__ == "__main__":
    main()
