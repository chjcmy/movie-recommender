import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (Mac OS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_embeddings():
    # 1. 가상의 임베딩 좌표 정의 (2차원으로 축소했다고 가정)
    # 액션/히어로 영화를 좋아하는 그룹 (1사분면)
    users = {
        '철수': (0.8, 0.7),
        '민수': (0.75, 0.8)
    }
    
    items = {
        '아이언맨': (0.9, 0.6),
        '어벤져스': (0.85, 0.85),
        '노트북': (-0.5, -0.3) # 로맨스 영화 (3사분면, 멀리 떨어짐)
    }

    plt.figure(figsize=(10, 8))
    
    # 2. 유저 그리기 (파란색 점)
    for name, (x, y) in users.items():
        plt.scatter(x, y, c='blue', s=200, label='User' if name == '철수' else "", alpha=0.6)
        plt.text(x, y+0.05, name, fontsize=12, ha='center', fontweight='bold', color='blue')

    # 3. 아이템 그리기 (빨간색 별)
    for name, (x, y) in items.items():
        plt.scatter(x, y, c='red', marker='*', s=300, label='Movie' if name == '아이언맨' else "", alpha=0.6)
        plt.text(x, y+0.05, name, fontsize=12, ha='center', fontweight='bold', color='red')

    # 4. 유사도 선 그리기 (점선)
    # 철수 <-> 민수 (유저 유사도)
    plt.plot([users['철수'][0], users['민수'][0]], [users['철수'][1], users['민수'][1]], 'b--', alpha=0.3)
    plt.text(0.77, 0.75, "유사함", fontsize=10, color='blue', alpha=0.7)

    # 철수 <-> 아이언맨 (선호)
    plt.plot([users['철수'][0], items['아이언맨'][0]], [users['철수'][1], items['아이언맨'][1]], 'g-', alpha=0.3)
    
    # 민수 <-> 어벤져스 (추천 예측)
    plt.arrow(users['민수'][0], users['민수'][1], 
              items['어벤져스'][0] - users['민수'][0], items['어벤져스'][1] - users['민수'][1],
              color='purple', width=0.005, head_width=0.02, alpha=0.5)
    plt.text(0.8, 0.82, "추천!", fontsize=12, color='purple', fontweight='bold')

    # 5. 그래프 스타일링
    plt.title("사용자 및 영화 임베딩 공간 시각화 (예시)", fontsize=15)
    plt.xlabel("잠재 요인 1 (예: 액션 성향)", fontsize=12)
    plt.ylabel("잠재 요인 2 (예: 히어로물 성향)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    plt.xlim(-1.0, 1.5)
    plt.ylim(-1.0, 1.5)
    plt.legend(loc='upper left')

    # 저장
    output_path = 'embedding_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    plot_embeddings()
