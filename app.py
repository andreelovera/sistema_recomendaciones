from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el modelo y los datos
with open('knn_model.pkl', 'rb') as f:
    knn, pca, train_pca_df, movies = pickle.load(f)

def editar_pelicula(movie_id, nuevo_nombre=None, nuevo_genero=None):
    if nuevo_nombre:
        movies.loc[movies['movieId'] == movie_id, 'title'] = nuevo_nombre
    if nuevo_genero:
        movies.loc[movies['movieId'] == movie_id, 'genres'] = nuevo_genero

def recomendar_peliculas(movie_id, n_recommendations=5):
    movie_idx = train_pca_df.index.get_loc(movie_id)
    distances, indices = knn.kneighbors([train_pca_df.iloc[movie_idx]], n_neighbors=n_recommendations + 1)
    recommendations = indices.flatten()[1:]  # Omitir la primera que es la misma película
    recommendation_distances = distances.flatten()[1:]

    results = []
    for idx, distance in zip(recommendations, recommendation_distances):
        rec_movie_id = train_pca_df.index[idx]
        movie_info = movies[movies['movieId'] == rec_movie_id].iloc[0]
        results.append({
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'year': movie_info['year'],
        })

    return pd.DataFrame(results)

@app.route('/')
def index():
    movie_ids = movies['movieId'].tolist()
    genres = movies['genres'].unique().tolist()
    return render_template('index.html', movies=movies.to_dict(orient='records'), movie_ids=movie_ids, genres=genres)

@app.route('/edit', methods=['POST'])
def edit():
    movie_id = int(request.form['movie_id'])
    nuevo_nombre = request.form['nuevo_nombre']
    nuevo_genero = request.form['nuevo_genero']
    editar_pelicula(movie_id, nuevo_nombre, nuevo_genero)
    return redirect(url_for('index'))

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_id = int(request.form['movie_id'])
    recomendaciones = recomendar_peliculas(movie_id)
    return render_template('result.html', recomendaciones=recomendaciones.to_dict(orient='records'))

@app.route('/get_movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    movie = movies[movies['movieId'] == movie_id].iloc[0]
    return jsonify({'title': movie['title'], 'genres': movie['genres']})

if __name__ == '__main__':
    app.run(debug=True)
