import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load preprocessed data and models
movies = pd.read_pickle(r"data\movies_with_posters.pkl")
movie_ids = joblib.load(r"models\movie_ids.joblib")
knn_model = joblib.load(r"models\knn_model.joblib")
csr_data = joblib.load(r"data\csr_data.joblib")
tfidf_matrix = joblib.load(r"models\tfidf_matrix.joblib")

# --- Recommendation functions ---

def get_collaborative_recommendation(movie_name, num_recommendations=20):
    matches = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    if matches.empty:
        return []

    movie_id = matches.iloc[0]['movieId']
    try:
        matrix_idx = movie_ids.index(movie_id)
    except ValueError:
        return []

    distances, indices = knn_model.kneighbors(csr_data[matrix_idx], n_neighbors=num_recommendations + 1)
    rec_indices = indices.squeeze().tolist()[1:]
    rec_movie_ids = [movie_ids[i] for i in rec_indices]
    return movies[movies['movieId'].isin(rec_movie_ids)][['title', 'genres', 'poster_url']].to_dict(orient='records')

def get_content_recommendation(movie_name, num_recommendations=20):
    matches = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:num_recommendations + 1]
    return movies.iloc[sim_indices][['title', 'genres', 'poster_url']].to_dict(orient='records')

def get_hybrid_recommendation(movie_name, num_recommendations=10):
    # Get recommendations separately
    content_recs = get_content_recommendation(movie_name, 20)
    collab_recs = get_collaborative_recommendation(movie_name, 20)

    # Assign scores based on rank (higher rank => higher score)
    weight_content = 0.6
    weight_collab = 0.4

    content_scores = {rec['title']: (20 - i) * weight_content for i, rec in enumerate(content_recs)}
    collab_scores = {rec['title']: (20 - i) * weight_collab for i, rec in enumerate(collab_recs)}

    # Merge scores
    all_titles = set(content_scores) | set(collab_scores)
    combined_scores = {
        title: content_scores.get(title, 0) + collab_scores.get(title, 0)
        for title in all_titles
    }

    # Pick top N
    top_titles = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    top_movie_titles = [title for title, _ in top_titles]

    # Return movie details
    recommendations = movies[movies['title'].isin(top_movie_titles)][['title', 'genres', 'poster_url']]
    return recommendations.to_dict(orient='records')


# --- UI handler ---
def recommend_movies_ui(movie_name):
    if not movie_name:
        return "<h3 style='color:red;'>Please select a movie.</h3>"

    selected = movies[movies['title'] == movie_name].iloc[0]
    selected_poster = selected['poster_url']
    selected_html = f"""
        <div style='width: 100%; text-align: left; margin: 30px 0;'>
            <h2>{movie_name}</h2>
            <img src='{selected_poster}' style='width: 140px; height: 210px; object-fit: cover; border-radius: 8px;'>
        </div>
    """

    # Hybrid recommendations
    hybrid_recs = get_hybrid_recommendation(movie_name)
    rec_html = """
        <div style='width: 100%; text-align: left; margin: 40px 0;'>
            <h3><b>🍿 Personalized Recommendations (Hybrid):</b></h3>
            <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 20px;'>
    """
    for rec in hybrid_recs:
        rec_html += f"""
            <div style='text-align: center; width: 160px;'>
                <img src='{rec["poster_url"]}' style='width: 1000px; height: 1200px object-fit: cover; border-radius: 8px;'>
                <p style='margin-top: 8px; font-weight: bold;'>{rec["title"]}</p>
                <p style='font-size: 0.75em; color: white;'>{rec["genres"]}</p>
            </div>
        """
    if not hybrid_recs:
        rec_html += "<p>No recommendations found.</p>"
    rec_html += "</div></div>"

    return selected_html + rec_html

# --- Gradio App ---
safe_movie_ids = set(movie_ids[:csr_data.shape[0]])
available_movies = movies[movies['movieId'].isin(safe_movie_ids)]
all_movie_titles = sorted(available_movies['title'].dropna().unique())


with gr.Blocks(title="📽️ Hybrid Movie Recommendation System") as app:
    gr.Markdown(
    """
    <div style='text-align: center;'>
        <h2 style='font-size: 2.2em;'>📽️ CineMatch</h2>
        <p style='font-size: 1.1em;'>Select a movie to get personalized recommendations powered by hybrid filtering..</p>
    </div>
    """
)


    with gr.Row():
        dropdown = gr.Dropdown(choices=all_movie_titles, label="Select a Movie", scale=3)
        with gr.Column(scale=1):
            clear_btn = gr.Button("Clear")
            submit_btn = gr.Button("Submit")

    output = gr.HTML()

    # Event handlers
    def handle_submit(movie_name):
        return recommend_movies_ui(movie_name)

    def handle_clear():
        return ""

    submit_btn.click(fn=handle_submit, inputs=dropdown, outputs=output)
    clear_btn.click(fn=handle_clear, outputs=output)

app.launch(inbrowser=True)
