import pytest
from rest_framework.test import APIClient
from django.urls import reverse
from unittest.mock import patch, MagicMock
import torch
import json

# Dummy data for tests
MOCK_ITEM_MAP = {1: 0, 2: 1, 3: 2}
MOCK_REVERSE_ITEM_MAP = {0: 1, 1: 2, 2: 3}

@pytest.fixture
def client():
    return APIClient()

@pytest.fixture
def mock_item_map():
    with patch('recommender.views.ITEM_MAP', MOCK_ITEM_MAP) as _mock:
        yield _mock

@pytest.fixture
def mock_model():
    """
    Mocks the get_or_load_model to return a dummy PyTorch-like model.
    """
    mock_model = MagicMock()
    
    # Mock for MF/NCF/WD (Attribute access)
    # They access .item_embedding.weight or .gmf_item_embedding.weight
    # Shape: (NumItems, Dim) -> Let's say (20 items + 1 padding, 4 dim)
    dummy_embeddings = torch.rand(21, 4)
    
    mock_model.item_embedding.weight = dummy_embeddings
    mock_model.gmf_item_embedding.weight = dummy_embeddings
    
    # Mock for SASRec (Forward pass)
    # Output shape: (Batch, SeqLen, NumItems) -> (1, 50, 4) -> No, NumItems needs to match dummy_embeddings size (21)
    # We allow the model to be called nicely
    def forward_side_effect(input_tensor):
        # input_tensor shape (1, 50)
        return torch.rand(1, 50, 21)
        
    mock_model.side_effect = forward_side_effect
    
    with patch('recommender.views.get_or_load_model', return_value=mock_model) as _mock:
        yield _mock

@pytest.mark.django_db
class TestRecommendationAPIs:
    
    def test_recommend_mf_success(self, client, mock_item_map, mock_model):
        url = reverse('recommend_mf')
        data = {"movie_ids": [1, 2]}
        response = client.post(url, data, format='json')
        
        if response.status_code != 200:
            print(f"\nResponse Error: {response.data}")
        assert response.status_code == 200
        assert "results" in response.data
        assert len(response.data["results"]) > 0

    def test_recommend_ncf_success(self, client, mock_item_map, mock_model):
        url = reverse('recommend_ncf')
        data = {"movie_ids": [2]}
        response = client.post(url, data, format='json')
        
        if response.status_code != 200:
            print(f"\nResponse Error: {response.data}")
        assert response.status_code == 200
        assert "results" in response.data

    def test_recommend_wd_success(self, client, mock_item_map, mock_model):
        url = reverse('recommend_wd')
        data = {"movie_ids": [1, 3]}
        response = client.post(url, data, format='json')
        
        if response.status_code != 200:
            print(f"\nResponse Error: {response.data}")
        assert response.status_code == 200
        assert "results" in response.data

    def test_recommend_sasrec_success(self, client, mock_item_map, mock_model):
        url = reverse('recommend_sasrec')
        data = {"movie_ids": [1, 2, 3]}
        response = client.post(url, data, format='json')
        
        if response.status_code != 200:
            print(f"\nResponse Error: {response.data}")
        assert response.status_code == 200
        assert "results" in response.data

    def test_invalid_input_empty(self, client):
        # Testing generic error handling for empty list or invalid keys
        for url_name in ['recommend_mf', 'recommend_ncf', 'recommend_wd', 'recommend_sasrec']:
            url = reverse(url_name)
            data = {"movie_ids": []} # Empty list should mostly be ignored or handled gracefully (here likely 400 or empty results depending on logic)
            # Logic says: if not input_indices: return 400
            response = client.post(url, data, format='json')
            assert response.status_code == 400

    def test_input_unknown_movies(self, client, mock_item_map):
        # Movie ID 999 is not in MOCK_ITEM_MAP
        url = reverse('recommend_mf')
        data = {"movie_ids": [999]}
        response = client.post(url, data, format='json')
        assert response.status_code == 400 # "Need valid movies"

@pytest.mark.django_db
def test_search_api(client):
    from recommender.models import Movie
    # Create dummy movie
    Movie.objects.create(id=1, title="Toy Story", genres="Animation")
    
    url = reverse('search_movies')
    response = client.get(url, {'query': 'Toy'})
    
    assert response.status_code == 200
    assert len(response.data['results']) == 1
    assert response.data['results'][0]['title'] == "Toy Story"
