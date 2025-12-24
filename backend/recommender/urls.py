from django.urls import path
from . import views

urlpatterns = [
    path('similar/<int:movie_id>/', views.find_similar_movies, name='find_similar_movies'),
    path('history/mf/', views.recommend_mf, name='recommend_mf'),
    path('history/ncf/', views.recommend_ncf, name='recommend_ncf'),
    path('history/wd/', views.recommend_wd, name='recommend_wd'),
    path('history/sasrec/', views.recommend_sasrec, name='recommend_sasrec'),
    path('search/', views.search_movies, name='search_movies'),
]
