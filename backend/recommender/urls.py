from django.urls import path
from . import views

urlpatterns = [
    path('mf/<int:user_id>/', views.recommend_mf, name='recommend_mf'),
    path('ncf/<int:user_id>/', views.recommend_ncf, name='recommend_ncf'),
    path('wide_deep/<int:user_id>/', views.recommend_wide_deep, name='recommend_wide_deep'),
    path('sasrec/', views.recommend_sasrec, name='recommend_sasrec'),
]
