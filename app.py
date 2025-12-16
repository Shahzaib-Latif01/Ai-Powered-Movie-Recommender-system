# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import re
import json
from collections import Counter
import pickle
import os
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class NetflixRecommenderSystem:
    def __init__(self):
        self.data = None
        self.tfidf_matrix = None
        self.indices = None
        self.cosine_sim = None
        # Collaborative filtering attributes
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_ratings = None
        self.svd_model = None
        self.users_df = None
        
    def load_data(self, filepath):
        """Load the Netflix dataset"""
        self.data = pd.read_csv(filepath)
        print(f"Loaded {len(self.data)} Netflix titles")
        return self.data
    
    def generate_synthetic_user_data(self, num_users=1000, num_ratings_per_user=50):
        """Generate synthetic user rating data for collaborative filtering demonstration"""
        print("Generating synthetic user rating data...")
        
        # Create user IDs
        user_ids = range(1, num_users + 1)
        
        # Get all show IDs (using index as show_id)
        show_ids = self.data.index.tolist()
        
        # Generate ratings data
        ratings_data = []
        
        for user_id in user_ids:
            # Each user rates a random subset of shows
            rated_shows = np.random.choice(show_ids, 
                                         size=min(num_ratings_per_user, len(show_ids)), 
                                         replace=False)
            
            for show_id in rated_shows:
                # Generate rating based on show characteristics (to make it somewhat realistic)
                show_info = self.data.iloc[show_id]
                
                # Base rating influenced by release year (newer shows get slightly higher ratings)
                base_rating = 3.0
                if show_info['release_year'] >= 2020:
                    base_rating += 0.5
                elif show_info['release_year'] >= 2015:
                    base_rating += 0.3
                
                # Add some randomness
                rating = base_rating + np.random.normal(0, 1.2)
                rating = max(1, min(5, rating))  # Clamp between 1 and 5
                
                ratings_data.append({
                    'user_id': user_id,
                    'show_id': show_id,
                    'rating': round(rating, 1),
                    'title': show_info['title']
                })
        
        self.user_ratings = pd.DataFrame(ratings_data)
        print(f"Generated {len(self.user_ratings)} user ratings")
        
        # Save synthetic data for future use
        self.user_ratings.to_csv('synthetic_user_ratings.csv', index=False)
        
        return self.user_ratings
    
    def load_user_ratings(self, filepath='synthetic_user_ratings.csv'):
        """Load user ratings data (synthetic or real)"""
        if os.path.exists(filepath):
            self.user_ratings = pd.read_csv(filepath)
            print(f"Loaded {len(self.user_ratings)} user ratings from {filepath}")
        else:
            print("No existing user ratings found. Generating synthetic data...")
            self.generate_synthetic_user_data()
        
        return self.user_ratings
    
    def create_user_item_matrix(self):
        """Create user-item rating matrix for collaborative filtering"""
        if self.user_ratings is None:
            self.load_user_ratings()
        
        # Create pivot table (user-item matrix)
        self.user_item_matrix = self.user_ratings.pivot_table(
            index='user_id', 
            columns='show_id', 
            values='rating', 
            fill_value=0
        )
        
        print(f"Created user-item matrix: {self.user_item_matrix.shape}")
        return self.user_item_matrix
    
    def compute_user_similarity(self):
        """Compute user-user similarity matrix"""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Convert to sparse matrix for efficiency
        user_matrix_sparse = csr_matrix(self.user_item_matrix.values)
        
        # Compute cosine similarity between users
        self.user_similarity = cosine_similarity(user_matrix_sparse)
        
        print("Computed user-user similarity matrix")
        return self.user_similarity
    
    def compute_item_similarity(self):
        """Compute item-item similarity matrix"""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Transpose matrix to get items as rows
        item_matrix = self.user_item_matrix.T
        item_matrix_sparse = csr_matrix(item_matrix.values)
        
        # Compute cosine similarity between items
        self.item_similarity = cosine_similarity(item_matrix_sparse)
        
        print("Computed item-item similarity matrix")
        return self.item_similarity
    
    def user_based_collaborative_filtering(self, user_id, top_n=10):
        """Generate recommendations using user-based collaborative filtering"""
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset")
            return pd.DataFrame()
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Get similar users (excluding the user itself)
        similar_users_scores = self.user_similarity[user_idx]
        similar_users_indices = np.argsort(similar_users_scores)[::-1][1:51]  # Top 50 similar users
        
        # Calculate weighted average ratings for unrated items
        recommendations = {}
        
        for show_id in self.user_item_matrix.columns:
            if user_ratings[show_id] == 0:  # User hasn't rated this show
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_user_idx in similar_users_indices:
                    similar_user_rating = self.user_item_matrix.iloc[similar_user_idx][show_id]
                    
                    if similar_user_rating > 0:  # Similar user has rated this show
                        similarity_score = similar_users_scores[similar_user_idx]
                        weighted_sum += similarity_score * similar_user_rating
                        similarity_sum += abs(similarity_score)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[show_id] = predicted_rating
        
        # Sort recommendations by predicted rating
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create DataFrame with show details
        recommended_shows = []
        for show_id, predicted_rating in sorted_recommendations:
            show_info = self.data.iloc[show_id]
            recommended_shows.append({
                'title': show_info['title'],
                'type': show_info['type'],
                'release_year': show_info['release_year'],
                'listed_in': show_info['listed_in'],
                'description': show_info['description'],
                'predicted_rating': round(predicted_rating, 2),
                'show_id': show_id
            })
        
        return pd.DataFrame(recommended_shows)
    
    def item_based_collaborative_filtering(self, user_id, top_n=10):
        """Generate recommendations using item-based collaborative filtering"""
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset")
            return pd.DataFrame()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        # Calculate recommendations based on item similarity
        recommendations = {}
        
        for show_id in self.user_item_matrix.columns:
            if show_id not in rated_items:  # User hasn't rated this show
                weighted_sum = 0
                similarity_sum = 0
                
                show_idx = self.user_item_matrix.columns.get_loc(show_id)
                
                for rated_show_id in rated_items:
                    rated_show_idx = self.user_item_matrix.columns.get_loc(rated_show_id)
                    similarity_score = self.item_similarity[show_idx][rated_show_idx]
                    user_rating = user_ratings[rated_show_id]
                    
                    weighted_sum += similarity_score * user_rating
                    similarity_sum += abs(similarity_score)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[show_id] = predicted_rating
        
        # Sort recommendations by predicted rating
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create DataFrame with show details
        recommended_shows = []
        for show_id, predicted_rating in sorted_recommendations:
            show_info = self.data.iloc[show_id]
            recommended_shows.append({
                'title': show_info['title'],
                'type': show_info['type'],
                'release_year': show_info['release_year'],
                'listed_in': show_info['listed_in'],
                'description': show_info['description'],
                'predicted_rating': round(predicted_rating, 2),
                'show_id': show_id
            })
        
        return pd.DataFrame(recommended_shows)
    
    def matrix_factorization_recommendations(self, user_id, top_n=10, n_components=50):
        """Generate recommendations using matrix factorization (SVD)"""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Apply SVD
        if self.svd_model is None:
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = self.svd_model.fit_transform(self.user_item_matrix)
            item_factors = self.svd_model.components_
        else:
            user_factors = self.svd_model.transform(self.user_item_matrix)
            item_factors = self.svd_model.components_
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset")
            return pd.DataFrame()
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_factor = user_factors[user_idx]
        
        # Predict ratings for all items
        predicted_ratings = np.dot(user_factor, item_factors)
        
        # Get user's actual ratings
        user_actual_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Create recommendations for unrated items
        recommendations = []
        for i, (show_id, actual_rating) in enumerate(user_actual_ratings.items()):
            if actual_rating == 0:  # User hasn't rated this show
                predicted_rating = predicted_ratings[i]
                show_info = self.data.iloc[show_id]
                recommendations.append({
                    'title': show_info['title'],
                    'type': show_info['type'],
                    'release_year': show_info['release_year'],
                    'listed_in': show_info['listed_in'],
                    'description': show_info['description'],
                    'predicted_rating': round(max(0, predicted_rating), 2),
                    'show_id': show_id
                })
        
        # Sort by predicted rating and return top N
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return pd.DataFrame(recommendations[:top_n])
    
    def hybrid_recommendations(self, user_id=None, title=None, top_n=10, weights=None):
        """Generate hybrid recommendations combining content-based and collaborative filtering"""
        if weights is None:
            weights = {'content': 0.3, 'user_cf': 0.3, 'item_cf': 0.2, 'svd': 0.2}
        
        recommendations = {}
        
        # Content-based recommendations (if title provided)
        if title and title in self.indices:
            content_recs = self.get_recommendations(title, top_n * 2)
            for idx, rec in content_recs.iterrows():
                show_id = self.indices[rec['title']]
                if show_id not in recommendations:
                    recommendations[show_id] = {'score': 0, 'details': rec}
                recommendations[show_id]['score'] += weights['content'] * (len(content_recs) - idx) / len(content_recs)
        
        # Collaborative filtering recommendations (if user_id provided)
        if user_id:
            # User-based CF
            try:
                user_cf_recs = self.user_based_collaborative_filtering(user_id, top_n * 2)
                for idx, rec in user_cf_recs.iterrows():
                    show_id = rec['show_id']
                    if show_id not in recommendations:
                        recommendations[show_id] = {'score': 0, 'details': rec}
                    recommendations[show_id]['score'] += weights['user_cf'] * rec['predicted_rating'] / 5.0
            except Exception as e:
                print(f"User-based CF error: {e}")
            
            # Item-based CF
            try:
                item_cf_recs = self.item_based_collaborative_filtering(user_id, top_n * 2)
                for idx, rec in item_cf_recs.iterrows():
                    show_id = rec['show_id']
                    if show_id not in recommendations:
                        recommendations[show_id] = {'score': 0, 'details': rec}
                    recommendations[show_id]['score'] += weights['item_cf'] * rec['predicted_rating'] / 5.0
            except Exception as e:
                print(f"Item-based CF error: {e}")
            
            # SVD-based recommendations
            try:
                svd_recs = self.matrix_factorization_recommendations(user_id, top_n * 2)
                for idx, rec in svd_recs.iterrows():
                    show_id = rec['show_id']
                    if show_id not in recommendations:
                        recommendations[show_id] = {'score': 0, 'details': rec}
                    recommendations[show_id]['score'] += weights['svd'] * rec['predicted_rating'] / 5.0
            except Exception as e:
                print(f"SVD error: {e}")
        
        # Sort by combined score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]
        
        # Format final recommendations
        final_recommendations = []
        for show_id, rec_data in sorted_recs:
            show_info = self.data.iloc[show_id]
            final_recommendations.append({
                'title': show_info['title'],
                'type': show_info['type'],
                'release_year': show_info['release_year'],
                'listed_in': show_info['listed_in'],
                'description': show_info['description'],
                'hybrid_score': round(rec_data['score'], 3),
                'show_id': show_id
            })
        
        return pd.DataFrame(final_recommendations)

    def preprocess_data(self):
        """Clean and prepare data for recommendation"""
        # Drop rows with missing values in important columns
        self.data = self.data.dropna(subset=['description'])
        
        # Create a combined features column for content-based filtering
        self.data['combined_features'] = ''
        
        # Combine relevant features
        features = ['director', 'cast', 'listed_in', 'description']
        
        for feature in features:
            self.data[feature] = self.data[feature].fillna('')
            
        # Combine all selected features
        self.data['combined_features'] = (
            self.data['director'] + ' ' + 
            self.data['cast'] + ' ' + 
            self.data['listed_in'] + ' ' + 
            self.data['description']
        )
        
        # Create a mapping of indices
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()
        
        print("Data preprocessing completed")
        
    def compute_similarity(self):
        """Compute content similarity between shows"""
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Construct the TF-IDF matrix
        self.tfidf_matrix = tfidf.fit_transform(self.data['combined_features'])
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("Computed content similarity matrix")
        
    def get_recommendations(self, title, top_n=10):
        """Get top N recommendations similar to the title"""
        # Check if title exists in dataset
        if title not in self.indices:
            return pd.DataFrame()
            
        idx = self.indices[title]
        
        # Get similarity scores for all titles
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort titles based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar titles (excluding the input title)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get title indices
        title_indices = [i[0] for i in sim_scores]
        
        # Return the top similar titles with relevant info
        recommendations = self.data.iloc[title_indices][['title', 'type', 'release_year', 'listed_in', 'description']]
        return recommendations
    
    def recommend_by_genre(self, genre, top_n=10):
        """Recommend shows by genre"""
        # Filter shows by genre (case-insensitive partial match)
        genre_pattern = re.compile(genre, re.IGNORECASE)
        matching_shows = self.data[self.data['listed_in'].apply(lambda x: bool(genre_pattern.search(str(x))))]
        
        if matching_shows.empty:
            return pd.DataFrame()
            
        # Sort by release year (newest first) and return top N
        recommendations = matching_shows.sort_values('release_year', ascending=False).head(top_n)
        return recommendations[['title', 'type', 'release_year', 'listed_in', 'description']]
    
    def recommend_by_actor(self, actor, top_n=10):
        """Recommend shows by actor/actress"""
        # Filter shows by actor (case-insensitive partial match)
        actor_pattern = re.compile(actor, re.IGNORECASE)
        matching_shows = self.data[self.data['cast'].apply(lambda x: bool(actor_pattern.search(str(x))))]
        
        if matching_shows.empty:
            return pd.DataFrame()
            
        # Sort by release year (newest first) and return top N
        recommendations = matching_shows.sort_values('release_year', ascending=False).head(top_n)
        return recommendations[['title', 'type', 'release_year', 'listed_in', 'cast']]
    
    def get_all_titles(self):
        """Return list of all titles for dropdown"""
        return sorted(self.indices.index.tolist())
    
    def get_all_genres(self):
        """Extract and return all unique genres"""
        all_genres = []
        for genres in self.data['listed_in'].dropna():
            all_genres.extend([g.strip() for g in genres.split(',')])
        return sorted(list(set(all_genres)))
    
    def get_analytics_data(self):
        """Generate analytics data for visualizations"""
        analytics = {}
        
        # Content type distribution (Movie vs. TV Show)
        type_counts = self.data['type'].value_counts()
        analytics['contentTypes'] = {
            'labels': type_counts.index.tolist(),
            'values': type_counts.values.tolist()
        }
        
        # Top countries by content
        country_data = []
        for countries in self.data['country'].dropna():
            countries_list = [c.strip() for c in countries.split(',')]
            country_data.extend(countries_list)
        
        country_counter = Counter(country_data)
        top_countries = country_counter.most_common(10)
        
        analytics['countries'] = {
            'labels': [country for country, count in top_countries],
            'values': [count for country, count in top_countries]
        }
        
        # Content by release year
        year_counts = self.data['release_year'].value_counts().sort_index()
        # Limit to recent years (last 15 years)
        recent_years = year_counts.tail(15)
        
        analytics['years'] = {
            'labels': [str(year) for year in recent_years.index.tolist()],
            'values': recent_years.values.tolist()
        }
        
        # Top genres
        genre_data = []
        for genres in self.data['listed_in'].dropna():
            genres_list = [g.strip() for g in genres.split(',')]
            genre_data.extend(genres_list)
        
        genre_counter = Counter(genre_data)
        top_genres = genre_counter.most_common(8)
        
        analytics['genres'] = {
            'labels': [genre for genre, count in top_genres],
            'values': [count for genre, count in top_genres]
        }
        
        # Ratings distribution
        rating_counts = self.data['rating'].value_counts().head(10)
        
        analytics['ratings'] = {
            'labels': rating_counts.index.tolist(),
            'values': rating_counts.values.tolist()
        }
        
        return analytics

# Initialize recommender
recommender = NetflixRecommenderSystem()
recommender.load_data('netflix_titles.csv')  # Update with your file path
recommender.preprocess_data()
recommender.compute_similarity()

# Initialize collaborative filtering
print("Initializing collaborative filtering...")
recommender.load_user_ratings()
recommender.create_user_item_matrix()
recommender.compute_user_similarity()
recommender.compute_item_similarity()
print("Collaborative filtering setup completed!")

# App routes
@app.route('/')
def index():
    # Get list of all titles and genres for dropdowns
    titles = recommender.get_all_titles()
    genres = recommender.get_all_genres()
    return render_template('index.html', titles=titles, genres=genres)

@app.route('/analytics')
def analytics():
    """Dedicated analytics page"""
    return render_template('analytics.html')

@app.route('/api/analytics')
def api_analytics():
    """API endpoint for analytics data"""
    analytics_data = recommender.get_analytics_data()
    return jsonify(analytics_data)

@app.route('/recommend_similar', methods=['POST'])
def recommend_similar():
    title = request.form['title']
    recommendations = recommender.get_recommendations(title)
    
    if recommendations.empty:
        return render_template('results.html', 
                               recommendation_type='similar',
                               query=title, 
                               recommendations=None)
    
    # Convert to list of dictionaries for template
    recommendations_list = recommendations.to_dict('records')
    return render_template('results.html', 
                           recommendation_type='similar',
                           query=title, 
                           recommendations=recommendations_list)

@app.route('/recommend_genre', methods=['POST'])
def recommend_genre():
    genre = request.form['genre']
    recommendations = recommender.recommend_by_genre(genre)
    
    if recommendations.empty:
        return render_template('results.html', 
                               recommendation_type='genre',
                               query=genre, 
                               recommendations=None)
    
    # Convert to list of dictionaries for template
    recommendations_list = recommendations.to_dict('records')
    return render_template('results.html', 
                           recommendation_type='genre',
                           query=genre, 
                           recommendations=recommendations_list)

@app.route('/recommend_actor', methods=['POST'])
def recommend_actor():
    actor = request.form['actor']
    recommendations = recommender.recommend_by_actor(actor)
    
    if recommendations.empty:
        return render_template('results.html', 
                               recommendation_type='actor',
                               query=actor, 
                               recommendations=None)
    
    # Convert to list of dictionaries for template
    recommendations_list = recommendations.to_dict('records')
    return render_template('results.html', 
                           recommendation_type='actor',
                           query=actor, 
                           recommendations=recommendations_list)

# New collaborative filtering routes
@app.route('/recommend_collaborative', methods=['POST'])
def recommend_collaborative():
    user_id = int(request.form['user_id'])
    cf_type = request.form.get('cf_type', 'user_based')
    
    if cf_type == 'user_based':
        recommendations = recommender.user_based_collaborative_filtering(user_id)
        cf_name = "User-Based Collaborative Filtering"
    elif cf_type == 'item_based':
        recommendations = recommender.item_based_collaborative_filtering(user_id)
        cf_name = "Item-Based Collaborative Filtering"
    elif cf_type == 'matrix_factorization':
        recommendations = recommender.matrix_factorization_recommendations(user_id)
        cf_name = "Matrix Factorization (SVD)"
    else:
        recommendations = pd.DataFrame()
        cf_name = "Unknown Method"
    
    if recommendations.empty:
        return render_template('results.html', 
                               recommendation_type='collaborative',
                               query=f"User {user_id} ({cf_name})", 
                               recommendations=None)
    
    # Convert to list of dictionaries for template
    recommendations_list = recommendations.to_dict('records')
    return render_template('results.html', 
                           recommendation_type='collaborative',
                           query=f"User {user_id} ({cf_name})", 
                           recommendations=recommendations_list)

@app.route('/recommend_hybrid', methods=['POST'])
def recommend_hybrid():
    user_id = request.form.get('user_id')
    title = request.form.get('title')
    
    # Convert user_id to int if provided
    if user_id:
        user_id = int(user_id)
    
    recommendations = recommender.hybrid_recommendations(
        user_id=user_id, 
        title=title if title else None
    )
    
    query_parts = []
    if user_id:
        query_parts.append(f"User {user_id}")
    if title:
        query_parts.append(f"Similar to '{title}'")
    
    query = " + ".join(query_parts) if query_parts else "Hybrid Recommendations"
    
    if recommendations.empty:
        return render_template('results.html', 
                               recommendation_type='hybrid',
                               query=query, 
                               recommendations=None)
    
    # Convert to list of dictionaries for template
    recommendations_list = recommendations.to_dict('records')
    return render_template('results.html', 
                           recommendation_type='hybrid',
                           query=query, 
                           recommendations=recommendations_list)

@app.route('/api/user_stats/<int:user_id>')
def get_user_stats(user_id):
    """Get statistics for a specific user"""
    if recommender.user_ratings is None:
        return jsonify({'error': 'No user data available'})
    
    user_data = recommender.user_ratings[recommender.user_ratings['user_id'] == user_id]
    
    if user_data.empty:
        return jsonify({'error': 'User not found'})
    
    stats = {
        'user_id': user_id,
        'total_ratings': len(user_data),
        'average_rating': round(user_data['rating'].mean(), 2),
        'favorite_genres': [],
        'recent_ratings': user_data.nlargest(5, 'rating')[['title', 'rating']].to_dict('records')
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)