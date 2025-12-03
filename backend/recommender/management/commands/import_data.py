import csv
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings
from recommender.models import Movie, Rating
import pytz

class Command(BaseCommand):
    help = 'Import data from CSV files'

    def handle(self, *args, **kwargs):
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        
        # 1. Import Movies
        self.stdout.write("Importing Movies...")
        movies_path = os.path.join(data_dir, 'movies.csv')
        links_path = os.path.join(data_dir, 'links.csv')
        
        # Load Links first to map movieId -> imdb/tmdb
        links_map = {}
        with open(links_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                links_map[row['movieId']] = {
                    'imdbId': row['imdbId'],
                    'tmdbId': row['tmdbId']
                }

        movies_to_create = []
        with open(movies_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = row['movieId']
                link_data = links_map.get(movie_id, {})
                
                movies_to_create.append(Movie(
                    id=int(movie_id),
                    title=row['title'],
                    genres=row['genres'],
                    imdb_id=link_data.get('imdbId'),
                    tmdb_id=link_data.get('tmdbId')
                ))
        
        # Bulk Create Movies (ignore conflicts if run multiple times)
        Movie.objects.bulk_create(movies_to_create, ignore_conflicts=True)
        self.stdout.write(f"Imported {len(movies_to_create)} movies.")

        # 2. Import Ratings
        self.stdout.write("Importing Ratings (this may take a while)...")
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        
        ratings_to_create = []
        users_cache = {} # Cache created users to avoid DB hits
        
        # Pre-fetch existing users to avoid duplicates
        existing_users = User.objects.values_list('username', flat=True)
        for u in existing_users:
            users_cache[u] = True # Just mark as existing

        # We need actual User objects for ForeignKey
        # Strategy: 
        # 1. Collect all unique userIds from CSV
        # 2. Bulk create missing Users
        # 3. Load all Users into a dictionary {id: UserObject}
        
        user_ids_in_csv = set()
        with open(ratings_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_ids_in_csv.add(int(row['userId']))
        
        # Create missing users
        new_users = []
        for uid in user_ids_in_csv:
            username = f"user_{uid}"
            if username not in users_cache:
                new_users.append(User(username=username))
        
        if new_users:
            User.objects.bulk_create(new_users)
            self.stdout.write(f"Created {len(new_users)} new users.")
            
        # Load all users for mapping
        # Assuming username format 'user_{id}'
        all_users = User.objects.filter(username__startswith='user_')
        user_map = {int(u.username.split('_')[1]): u for u in all_users}

        # Create Ratings
        with open(ratings_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = int(row['userId'])
                mid = int(row['movieId'])
                
                # Timestamp conversion
                dt = datetime.fromtimestamp(int(row['timestamp']), tz=pytz.UTC)
                
                if uid in user_map:
                    ratings_to_create.append(Rating(
                        user=user_map[uid],
                        movie_id=mid, # Use _id to avoid fetching Movie object
                        rating=float(row['rating']),
                        timestamp=dt
                    ))
                    
                if len(ratings_to_create) >= 5000:
                    Rating.objects.bulk_create(ratings_to_create, ignore_conflicts=True)
                    ratings_to_create = []
                    self.stdout.write(".", ending='')
        
        if ratings_to_create:
            Rating.objects.bulk_create(ratings_to_create, ignore_conflicts=True)
            
        self.stdout.write("\nDone!")
