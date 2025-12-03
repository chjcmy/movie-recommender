# 🎬 Advanced Movie Recommender System

> **"단순한 별점 예측을 넘어, 유저의 맥락(Context)과 순서(Sequence)까지 이해하는 추천 시스템"**

이 프로젝트는 **Django**와 **PyTorch**를 기반으로 구축된 **하이브리드 영화 추천 시스템**입니다.
Matrix Factorization부터 최신 Transformer 기반의 SASRec까지, 다양한 추천 알고리즘을 직접 구현하고 비교/분석할 수 있도록 설계되었습니다.

---

## 🚀 Key Features

### 1. 다양한 추천 알고리즘 구현 (Algorithms)
현업에서 사용되는 핵심 알고리즘 4가지를 모두 구현했습니다.
-   **Matrix Factorization (MF)**: 기본적인 잠재 요인 협업 필터링
-   **Neural Collaborative Filtering (NCF)**: 비선형적 관계를 학습하는 딥러닝 모델
-   **Wide & Deep**: 암기(Memorization)와 일반화(Generalization)의 장점을 결합
-   **SASRec (Self-Attentive Sequential Recommendation)**: 유저의 행동 순서(Sequence)를 반영한 Transformer 기반 모델

### 2. 하이브리드 아키텍처 (Hybrid Architecture)
-   **Retrieval (후보 추출)**: Vector Search (FAISS/pgvector)를 활용한 고속 후보군 선정
-   **Ranking (정밀 정렬)**: 딥러닝 모델(SASRec/Wide&Deep)을 활용한 개인화 랭킹

### 3. 확장 가능한 백엔드 (Scalable Backend)
-   **Django MVT 패턴**: 견고한 API 서버 구축
-   **PostgreSQL**: 대용량 데이터 처리를 위한 RDB
-   **Docker**: 배포 용이성을 위한 컨테이너화 (예정)

---

## 🛠 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Backend** | Python, Django, Django REST Framework |
| **AI / ML** | PyTorch, Pandas, Scikit-learn, Numpy |
| **Database** | PostgreSQL (Production), SQLite (Dev) |
| **Vector DB** | FAISS (Local), pgvector (Optional) |
| **DevOps** | Docker (Planned) |

---

## 🏗 Architecture

```mermaid
graph TD
    User[Client / User] -->|API Request| Django[Django API Server]
    
    subgraph "Backend (Django)"
        Django -->|Query| DB[(PostgreSQL / SQLite)]
        Django -->|Vector Search| FAISS[FAISS (Vector Index)]
        Django -->|Inference| Model[PyTorch Models]
    end
    
    subgraph "AI Models"
        Model --> MF[Matrix Factorization]
        Model --> NCF[Neural CF]
        Model --> WD[Wide & Deep]
        Model --> SAS[SASRec]
    end
    
    DB -->|Training Data| Model
```

---

## 📚 Model Intuition (학습 노트)

이 프로젝트는 단순 구현을 넘어, 각 모델의 **수학적 원리와 직관**을 깊이 있게 이해하는 것을 목표로 합니다.
아래 링크에서 각 모델에 대한 상세한 설명을 확인하실 수 있습니다.

-   [**Matrix Factorization (MF)**](MF_INTUITION.md): "취향의 지도 그리기"
-   [**Neural Collaborative Filtering (NCF)**](NCF_INTUITION.md): "비선형적 관계의 탐정"
-   [**Wide & Deep**](WIDE_AND_DEEP_INTUITION.md): "암기왕과 응용왕의 협업"
-   [**SASRec**](SASREC_INTUITION.md): "맥락을 읽는 독심술사 (Transformer)"
-   [**Vector DB & Embedding**](VECTOR_DB_INTUITION.md): "고속 검색의 비밀"

---

## ⚡️ Getting Started

### 1. Prerequisites
-   Python 3.8+
-   Virtualenv

### 2. Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender/backend

# 2. Create & Activate Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 3. Install Dependencies
pip install -r requirements.txt
```

### 3. Data Setup (Import)

MovieLens 데이터를 DB에 적재합니다.

```bash
# 1. Migrate Database
python manage.py makemigrations
python manage.py migrate

# 2. Import Data (Movies & Ratings)
# data/ 폴더에 movies.csv, ratings.csv가 있어야 합니다.
python manage.py import_data
```

### 4. Run Server

```bash
python manage.py runserver
```

---

## 🔌 API Usage

### 1. 추천 받기 (Recommendation)
-   **URL**: `/api/recommend/`
-   **Method**: `POST`
-   **Body**:
    ```json
    {
        "user_id": 1,
        "model": "sasrec" // or "mf", "ncf", "wide_deep"
    }
    ```
-   **Response**:
    ```json
    {
        "recommendations": [
            {"id": 1, "title": "Toy Story (1995)", "score": 0.98},
            {"id": 260, "title": "Star Wars: Episode IV (1977)", "score": 0.95}
        ]
    }
    ```

---

## 📝 License

This project is licensed under the MIT License.
