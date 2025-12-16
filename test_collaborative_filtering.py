#!/usr/bin/env python3
"""
Test script for Netflix Recommender System - Collaborative Filtering
This script demonstrates and tests all collaborative filtering features.
"""

import pandas as pd
import numpy as np
from app import NetflixRecommenderSystem
import time

def test_collaborative_filtering():
    print("🎬 Netflix Recommender System - Collaborative Filtering Test")
    print("=" * 60)
    
    # Initialize the recommender system
    print("\n1. Initializing Recommender System...")
    recommender = NetflixRecommenderSystem()
    
    # Load Netflix data
    try:
        recommender.load_data('netflix_titles.csv')
        recommender.preprocess_data()
        recommender.compute_similarity()
        print("✅ Netflix data loaded successfully!")
    except FileNotFoundError:
        print("❌ Netflix dataset not found. Please ensure 'netflix_titles.csv' is in the directory.")
        return
    
    # Test synthetic user data generation
    print("\n2. Testing Synthetic User Data Generation...")
    start_time = time.time()
    user_ratings = recommender.generate_synthetic_user_data(num_users=100, num_ratings_per_user=20)
    print(f"✅ Generated {len(user_ratings)} ratings for {user_ratings['user_id'].nunique()} users")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # Display sample user data
    print("\n📊 Sample User Ratings:")
    sample_user = user_ratings[user_ratings['user_id'] == 1].head()
    print(sample_user[['user_id', 'title', 'rating']].to_string(index=False))
    
    # Create user-item matrix
    print("\n3. Creating User-Item Matrix...")
    start_time = time.time()
    user_item_matrix = recommender.create_user_item_matrix()
    print(f"✅ User-Item Matrix created: {user_item_matrix.shape}")
    print(f"📈 Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.2f}%")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # Compute similarity matrices
    print("\n4. Computing Similarity Matrices...")
    start_time = time.time()
    user_similarity = recommender.compute_user_similarity()
    item_similarity = recommender.compute_item_similarity()
    print(f"✅ User similarity matrix: {user_similarity.shape}")
    print(f"✅ Item similarity matrix: {item_similarity.shape}")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # Test User-Based Collaborative Filtering
    print("\n5. Testing User-Based Collaborative Filtering...")
    test_user_id = 1
    start_time = time.time()
    user_cf_recs = recommender.user_based_collaborative_filtering(test_user_id, top_n=5)
    print(f"✅ Generated {len(user_cf_recs)} recommendations for User {test_user_id}")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    if not user_cf_recs.empty:
        print("\n🎯 Top User-Based CF Recommendations:")
        for idx, rec in user_cf_recs.iterrows():
            print(f"   • {rec['title']} ({rec['type']}, {rec['release_year']}) - Rating: {rec['predicted_rating']}")
    
    # Test Item-Based Collaborative Filtering
    print("\n6. Testing Item-Based Collaborative Filtering...")
    start_time = time.time()
    item_cf_recs = recommender.item_based_collaborative_filtering(test_user_id, top_n=5)
    print(f"✅ Generated {len(item_cf_recs)} recommendations for User {test_user_id}")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    if not item_cf_recs.empty:
        print("\n🎯 Top Item-Based CF Recommendations:")
        for idx, rec in item_cf_recs.iterrows():
            print(f"   • {rec['title']} ({rec['type']}, {rec['release_year']}) - Rating: {rec['predicted_rating']}")
    
    # Test Matrix Factorization
    print("\n7. Testing Matrix Factorization (SVD)...")
    start_time = time.time()
    svd_recs = recommender.matrix_factorization_recommendations(test_user_id, top_n=5)
    print(f"✅ Generated {len(svd_recs)} recommendations for User {test_user_id}")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    if not svd_recs.empty:
        print("\n🎯 Top SVD Recommendations:")
        for idx, rec in svd_recs.iterrows():
            print(f"   • {rec['title']} ({rec['type']}, {rec['release_year']}) - Rating: {rec['predicted_rating']}")
    
    # Test Hybrid Recommendations
    print("\n8. Testing Hybrid Recommendations...")
    sample_title = recommender.get_all_titles()[0]  # Get first title
    start_time = time.time()
    hybrid_recs = recommender.hybrid_recommendations(user_id=test_user_id, title=sample_title, top_n=5)
    print(f"✅ Generated {len(hybrid_recs)} hybrid recommendations")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    if not hybrid_recs.empty:
        print(f"\n🎯 Top Hybrid Recommendations (User {test_user_id} + Similar to '{sample_title}'):")
        for idx, rec in hybrid_recs.iterrows():
            print(f"   • {rec['title']} ({rec['type']}, {rec['release_year']}) - Score: {rec['hybrid_score']}")
    
    # Performance comparison
    print("\n9. Performance Comparison...")
    print("=" * 60)
    methods = [
        ("Content-Based", lambda: recommender.get_recommendations(sample_title, 5)),
        ("User-Based CF", lambda: recommender.user_based_collaborative_filtering(test_user_id, 5)),
        ("Item-Based CF", lambda: recommender.item_based_collaborative_filtering(test_user_id, 5)),
        ("Matrix Factorization", lambda: recommender.matrix_factorization_recommendations(test_user_id, 5)),
        ("Hybrid", lambda: recommender.hybrid_recommendations(user_id=test_user_id, title=sample_title, top_n=5))
    ]
    
    for method_name, method_func in methods:
        start_time = time.time()
        try:
            results = method_func()
            execution_time = time.time() - start_time
            print(f"{method_name:20} | {len(results):2d} results | {execution_time:.3f}s")
        except Exception as e:
            print(f"{method_name:20} | Error: {str(e)[:30]}...")
    
    # User statistics
    print("\n10. User Statistics...")
    user_stats = {
        'total_users': user_ratings['user_id'].nunique(),
        'total_ratings': len(user_ratings),
        'avg_ratings_per_user': len(user_ratings) / user_ratings['user_id'].nunique(),
        'avg_rating': user_ratings['rating'].mean(),
        'rating_distribution': user_ratings['rating'].value_counts().sort_index()
    }
    
    print(f"📊 Total Users: {user_stats['total_users']}")
    print(f"📊 Total Ratings: {user_stats['total_ratings']}")
    print(f"📊 Avg Ratings per User: {user_stats['avg_ratings_per_user']:.1f}")
    print(f"📊 Average Rating: {user_stats['avg_rating']:.2f}")
    print("\n📊 Rating Distribution:")
    for rating, count in user_stats['rating_distribution'].items():
        print(f"   {rating} stars: {count} ratings ({count/len(user_ratings)*100:.1f}%)")
    
    print("\n🎉 All tests completed successfully!")
    print("\nTo test the web interface:")
    print("1. Run: python app.py")
    print("2. Open: http://localhost:5000")
    print("3. Try the 'Collaborative' and 'Hybrid AI' tabs")

def demonstration_examples():
    """Show specific examples of how collaborative filtering works"""
    print("\n" + "="*60)
    print("📚 COLLABORATIVE FILTERING EXPLANATION")
    print("="*60)
    
    print("""
🔍 How Collaborative Filtering Works:

1. USER-BASED COLLABORATIVE FILTERING:
   - Finds users with similar rating patterns
   - Recommends items liked by similar users
   - Example: If users A and B both rated "Stranger Things" 
     and "Breaking Bad" highly, and user A also likes "Dark",
     then "Dark" will be recommended to user B.

2. ITEM-BASED COLLABORATIVE FILTERING:
   - Finds items that are frequently rated similarly
   - Recommends items similar to what the user has liked
   - Example: If "The Office" and "Friends" are often 
     rated similarly by users, and you like "The Office",
     then "Friends" will be recommended.

3. MATRIX FACTORIZATION (SVD):
   - Uses mathematical techniques to find hidden patterns
   - Decomposes the user-item matrix into lower dimensions
   - Can discover latent factors like genre preferences

4. HYBRID APPROACH:
   - Combines content-based and collaborative filtering
   - Uses weighted scoring to balance different methods
   - Provides more robust and diverse recommendations
    """)

if __name__ == "__main__":
    test_collaborative_filtering()
    demonstration_examples() 