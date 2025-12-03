# 행렬 분해(Matrix Factorization) 모델 설명 가이드

이 문서는 `mf.ipynb`에 구현된 추천 시스템의 핵심 로직을 포트폴리오나 면접에서 설명할 수 있도록 정리한 자료입니다.

## 1. 핵심 개념: "취향과 특성의 매칭"

이 모델의 기본 아이디어는 **"사용자의 취향"**과 **"영화의 특성"**을 숫자로 표현하고, 이 둘이 얼마나 잘 맞는지를 계산하는 것입니다.

- **User Embedding (사용자 임베딩)**: 사용자의 취향을 나타내는 벡터입니다. (예: [로맨스 선호도, 액션 선호도, ...])
- **Item Embedding (아이템 임베딩)**: 영화의 특성을 나타내는 벡터입니다. (예: [로맨스 정도, 액션 정도, ...])
- **Dot Product (내적)**: 두 벡터를 곱해서 더하는 연산으로, 두 벡터가 비슷할수록 큰 값이 나옵니다. 즉, **예상 평점**이 됩니다.

---

## 2. 데이터 흐름 (Data Flow)

### 1단계: 데이터 준비 (Indexing)
컴퓨터는 `userId: 123` 같은 숫자의 의미를 모릅니다. 그래서 이를 0부터 시작하는 고유한 번호(Index)로 바꿔줍니다.

```python
# 코드 예시 (mf.ipynb)
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {m: i for i, m in enumerate(item_ids)}
```
- **Input**: `userId=10`, `movieId=500`
- **Process**: 매핑 테이블(Dictionary)을 통해 변환
- **Output**: `user_idx=3`, `item_idx=15` (3번째 유저, 15번째 영화)

### 2단계: 임베딩 조회 (Lookup)
변환된 인덱스를 사용해 각 유저와 영화에 해당하는 "특성 벡터(Embedding)"를 꺼냅니다.

```python
# 코드 예시 (MF 클래스)
self.user_embedding = nn.Embedding(num_users, embed_dim)
self.item_embedding = nn.Embedding(num_items, embed_dim)

# forward 함수 내부
u = self.user_embedding(users) # 유저 3번의 벡터를 가져옴
i = self.item_embedding(items) # 영화 15번의 벡터를 가져옴
```
- **User Vector**: `[0.1, -0.5, 0.8, ...]` (32차원)
- **Item Vector**: `[0.2, 0.1, 0.9, ...]` (32차원)

### 3단계: 예측 평점 계산 (Dot Product)
두 벡터의 각 요소를 곱한 뒤 모두 더합니다.

```python
# 코드 예시
return (u * i).sum(1)
```
- **수식**: $(0.1 \times 0.2) + (-0.5 \times 0.1) + (0.8 \times 0.9) + \dots$
- **결과**: `4.2` (이 유저가 이 영화에 줄 것으로 예상되는 평점)

---

## 3. 학습 과정 (Training)

모델은 처음에 랜덤한 값으로 초기화되어 있습니다. 학습 데이터(실제 평점)를 보면서 점점 정확해집니다.

1. **예측**: 모델이 평점을 예측합니다. (예: 4.2점)
2. **비교**: 실제 평점과 비교합니다. (예: 실제는 5.0점 → 오차 -0.8)
3. **수정 (Backpropagation)**: 오차를 줄이는 방향으로 유저 벡터와 아이템 벡터의 숫자를 조금씩 수정합니다.
   - *"아, 이 유저는 생각보다 로맨스를 더 좋아하는구나"* (유저 벡터 수정)
   - *"아, 이 영화는 생각보다 로맨스 성향이 강하구나"* (아이템 벡터 수정)

## 4. 요약 (면접용 멘트)

> "제 프로젝트의 MF 모델은 사용자와 영화를 각각 32차원의 벡터로 임베딩하여 표현했습니다.
> PyTorch의 `nn.Embedding`을 사용하여 각 유저와 영화의 잠재 요인(Latent Factor)을 학습했고,
> 이 두 벡터의 내적(Dot Product)을 통해 평점을 예측했습니다.
> 학습 과정에서는 MSE Loss를 사용하여 예측 평점과 실제 평점 간의 차이를 줄이는 방향으로 임베딩 벡터들을 업데이트했습니다."
