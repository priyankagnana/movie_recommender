# movie_recommender.py

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movies = [
    "Action Adventure Hero",
    "Romantic Love Story",
    "Space Sci-fi Adventure",
    "Comedy Fun Drama",
    "Action Thriller Fight"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(movies)

similarity = cosine_similarity(X)

def recommend(movie_index):
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    for i in scores[1:3]:
        print(movies[i[0]])

while True:
    print("\nMovies:")
    for i, m in enumerate(movies):
        print(i, m)

    choice = int(input("Choose movie index: "))
    recommend(choice)