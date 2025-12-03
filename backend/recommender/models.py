from django.db import models
from django.contrib.auth.models import User

class Movie(models.Model):
    id = models.IntegerField(primary_key=True)  # movieId from CSV
    title = models.CharField(max_length=255)
    genres = models.CharField(max_length=255)
    imdb_id = models.CharField(max_length=20, null=True, blank=True)
    tmdb_id = models.CharField(max_length=20, null=True, blank=True)

    def __str__(self):
        return self.title

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.FloatField()
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.user.username} - {self.movie.title}: {self.rating}"
