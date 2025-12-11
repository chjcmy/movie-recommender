# 🎬 AI 기반 영화 추천 시스템 구축 프로젝트

## 1. 프로젝트 개요 (Overview)
사용자의 평점 및 시청 기록(MovieLens 데이터)을 활용하여 개인화된 영화 추천 서비스를 구축했습니다.  
단순한 DB 쿼리가 아닌, **딥러닝 모델(PyTorch)**을 통해 사용자/영화의 잠재적 특성(Latent Factor)을 학습하고, 이를 **Vector DB(Qdrant)**에 임베딩하여 **실시간 유사도 검색**이 가능한 시스템을 구현했습니다.

*   **진행 기간**: 2025.11 ~ 2025.12
*   **핵심 목표**: 대용량 데이터 환경에서도 실시간 추론(Inference) 부하를 줄이고, 다양한 추천 알고리즘을 유연하게 서빙하는 아키텍처 수립.

---

## 2. 사용 기술 (Tech Stack)

### **Backend & API**
*   **Python 3.13**, **Django 5.1**
*   **Django REST Framework**: RESTful API 구현

### **AI & ML**
*   **PyTorch**: 추천 모델 설계 및 학습
*   **Pandas / NumPy**: 데이터 전처리 및 분석
*   **Scikit-learn**: 데이터 분할 및 평가

### **Database & Infrastructure**
*   **Qdrant**: 고성능 벡터 검색 엔진 (Vector Database)
*   **PostgreSQL**: 메타데이터(영화 정보, 유저 정보) 관리
*   **Docker**: Qdrant 컨테이너 운영

---

## 3. 구현된 추천 모델 (Implemented Models)

다양한 추천 관점을 제공하기 위해 4가지 핵심 모델을 구현하고 비교/적용했습니다.

| 모델명 | 특징 및 역할 | 활용 데이터 | Qdrant 컬렉션 |
| :--- | :--- | :--- | :--- |
| **MF (Matrix Factorization)** | 가장 기본적인 협업 필터링. 사용자와 아이템의 상호작용 행렬 분해. | User ID, Movie ID, Rating | `movies_mf` |
| **NCF (Neural Collab Filtering)** | MF의 선형적 한계를 극복하기 위해 MLP(다층 퍼셉트론)를 결합하여 비선형 관계 학습. | User ID, Movie ID | `movies_ncf` |
| **Wide & Deep** | '기억(Wide)'과 '일반화(Deep)'를 동시에 수행. 희소한 특성과 밀집된 임베딩을 함께 학습. | User ID, Movie ID, **Genres** | `movies_wide_deep` |
| **SASRec (Sequential)** | 사용자의 시청 **순서(Sequence)**를 고려하여 '다음에 볼 영화'를 예측 (Transformer 기반). | User History (Time-ordered) | `movies_sasrec` |

---

## 4. 시스템 아키텍처 및 핵심 로직

### **Hybrid Serving Architecture**
본 프로젝트는 **"학습(Training) - 추출(Extraction) - 검색(Retrieval)"**의 파이프라인을 채택했습니다.

1.  **Training**: PyTorch로 각 모델 학습 (GPU/MPS 가속 활용).
2.  **Embedding Extraction**: 학습된 모델의 `Embedding Layer` 가중치를 추출.
    *   *문제 해결*: 학습 시점의 ID와 DB의 ID가 달라지는 문제 발견 -> `item_map.json`을 통한 **ID 전역 동기화(Global ID Synchronization)** 로직 구현.
3.  **Vector Indexing**: 추출된 임베딩을 Qdrant에 업로드 (Batch Processing & Retry Logic).
4.  **Serving (API)**:
    *   사용자 요청 -> Django API -> Qdrant (`recommend` 쿼리) -> 유사 벡터 ID 반환 -> PostgreSQL 상세 정보 매핑 -> JSON 응답.

---

## 5. 트러블슈팅 (Problem Solving)

### **1. ID 불일치 문제 (The "Lost Map" Problem)**
*   **상황**: 모델을 개별적으로 학습시키다 보니, 내부 인덱스(0~N)와 실제 DB의 `movieId`가 모델마다 서로 다르게 매핑되는 현상 발생. A모델의 1번 영화가 B모델에서는 5번 영화로 인식됨.
*   **해결**: `models/item_map.json`을 생성하여 모든 모델이 **동일한 ID-Index 매핑**을 강제하도록 파이프라인 재설계. 이를 위해 전용 관리 명령어(`upload_embeddings_*.py`)를 모듈화하여 개발.

### **2. 대량 벡터 업로드 타임아웃**
*   **상황**: 수십만 개의 고차원(64-dim) 벡터를 한 번에 업로드 시 HTTP Timeout 및 메모리 오버헤드 발생.
*   **해결**:
    *   `batch_size`를 20~50으로 동적 조절.
    *   Qdrant Client의 Timeout 설정 값을 300초로 증설.
    *   업로드 중단 시 이어서 올릴 수 있도록 진행 상황 로깅 및 예외 처리(Try-Catch) 강화.

---

## 6. 주요 코드 구조 (Directory Structure)

```
backend/
├── recommender/
│   ├── algorithms.py          # PyTorch 모델 클래스 정의 (MF, NCF, SASRec, W&D)
│   ├── vector_db.py           # Qdrant 클라이언트 연결 및 컬렉션 관리
│   ├── views.py               # 추천 API 로직 (유사도 검색 + DB 조회)
│   └── management/commands/   # 커스텀 관리 명령어
│       ├── upload_embeddings.py         # MF 학습 및 업로드
│       ├── upload_embeddings_sasrec.py  # SASRec 학습 및 업로드
│       ├── upload_embeddings_ncf.py     # NCF 학습 및 업로드
│       └── upload_embeddings_wd.py      # Wide & Deep 학습 및 업로드
├── models/                    # 학습된 가중치(.pth) 및 매핑 파일 저장소
└── config/                    # Django 설정
```

## 7. 향후 발전 계획
*   **실시간 유저 피드백 반영**: API를 통해 들어온 평점 데이터를 Redis에 캐싱 후 배치 학습에 반영.
*   **A/B 테스트**: `find_similar_movies` API 호출 시 `model` 파라미터를 통해 여러 모델의 클릭률(CTR) 비교.
