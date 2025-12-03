# Django 아키텍처 설명 가이드 (MVT Pattern)

"장고는 MVC 패턴이구나"라고 하셨는데, **정확합니다!**
다만, 장고에서는 이름을 조금 다르게 부릅니다. 이를 **MVT (Model-View-Template)** 패턴이라고 합니다.

## 1. MVC vs MVT 용어 비교

가장 헷갈리는 부분이 바로 **"View"**입니다.

| 역할 (Role) | 일반적인 MVC | Django (MVT) | 우리 프로젝트 매핑 |
| :--- | :--- | :--- | :--- |
| **데이터 (Data)** | **M**odel | **M**odel | `models.py` (DB) / `csv` 파일 |
| **로직 (Logic)** | **C**ontroller | **V**iew | `views.py` (핵심 로직) |
| **화면 (Screen)** | **V**iew | **T**emplate | `JSON Response` (API) / `html` |

### 왜 이름을 이렇게 지었을까요?
장고 개발자들은 **"데이터를 보여주는 방식(View)을 결정하는 것은 템플릿(Template)이다"**라고 생각했기 때문입니다.
그래서 **Controller(제어자)** 역할을 하는 녀석을 **View(보는 관점)**라고 부르는 조금 독특한 네이밍이 되었습니다.

## 2. 우리 프로젝트의 흐름 (Request Flow)

사용자가 추천을 요청했을 때의 흐름을 MVT로 풀면 다음과 같습니다.

1.  **URL (Dispatcher)**: `urls.py`
    -   "어? `/recommend/mf/` 주소로 요청이 왔네? 이건 `views.py`의 `recommend_mf` 함수로 보내야겠다."
2.  **View (Controller)**: `views.py` (**지금 보고 계신 파일!**)
    -   "주문이 들어왔군. `MF` 모델을 가져와서 계산 좀 해볼까?"
    -   데이터 로딩, 모델 추론, 결과 가공 등 **모든 업무 처리**를 담당합니다.
3.  **Model (Data)**: `algorithms.py` / `ratings.csv`
    -   실제 데이터와 알고리즘이 있는 곳입니다. View가 요청하면 데이터를 줍니다.
4.  **Template (Presentation)**: `JsonResponse`
    -   우리는 웹페이지(HTML) 대신 **JSON 데이터**를 돌려주므로, 템플릿 파일은 따로 없지만 이 부분이 "화면" 역할을 합니다.

## 3. 면접용 요약 멘트

> "Django는 전통적인 MVC 패턴을 따르지만, 명칭상으로는 **MVT(Model-View-Template)** 패턴을 사용합니다.
> 특히 **View**가 비즈니스 로직(Controller 역할)을 담당하고, **Template**이 프레젠테이션(View 역할)을 담당한다는 점이 특징입니다.
> 제 프로젝트에서는 `views.py`에서 추천 알고리즘을 호출하여 결과를 처리하는 **Controller** 역할을 수행하도록 설계했습니다."
