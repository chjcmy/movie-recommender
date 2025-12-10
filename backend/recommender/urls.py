from django.urls import path
from . import views

urlpatterns = [
    path('similar/<int:movie_id>/', views.find_similar_movies, name='find_similar_movies'),
]
