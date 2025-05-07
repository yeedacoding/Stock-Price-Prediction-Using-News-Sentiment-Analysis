import os
import duckdb
import joblib
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -----------------------------------
# 설정: 모델, 윈도우, 피처
# -----------------------------------
models = ['lightgbm', 'xgboost', 'randomforest', 'decisiontree', 'gradientboosting']
windows = [5, 15, 30]   # 슬라이딩 윈도우 너비 지정
feature_cols = ['news_count', 'avg_negative', 'avg_neutral', 'avg_positive', 'open', 'high', 'low', 'close', 'volume']

param_grid = {
    'lightgbm': [
        {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31}
    ],
    'xgboost': [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
    ],
    'randomforest': [
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 10}
    ],
    'decisiontree': [
        {'max_depth': 3},
        {'max_depth': 5}
    ],
    'gradientboosting': [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
    ]
}

# -----------------------------------
# 슬라이딩 윈도우 피처 생성
# -----------------------------------
def generate_flatten_features(df, window_size, feature_cols, label_col='label'):
    X, y = [], []
    for i in range(window_size, len(df)):
        # 과거 window_size만큼의 구간을 잘라서 피처들을 flatten
        features = df.iloc[i-window_size:i][feature_cols].values.flatten()

        # 예측대상 : 윈도우 이후 날짜(t)의 label값
        label = df.iloc[i][label_col]   
        X.append(features)
        y.append(label)
    return pd.DataFrame(X), pd.Series(y)

# -----------------------------------
# 모델 학습 및 평가
# -----------------------------------
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, params):
    if model_name == 'lightgbm':
        model = LGBMClassifier(**params)
    elif model_name == 'xgboost':
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'randomforest':
        model = RandomForestClassifier(**params)
    elif model_name == 'decisiontree':
        model = DecisionTreeClassifier(**params)
    elif model_name == 'gradientboosting':
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)   # 클래스 예측
    probas = model.predict_proba(X_test)[:, 1]  # 확률 예측

    return {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probas)
    }

# -----------------------------------
# 단일 실험 실행
# -----------------------------------
def run_one(args):
    df, model_name, window_size, param, feature_cols, date_name = args
    # 피처셋 생성 (flatten 방식)
    X, y = generate_flatten_features(df, window_size, feature_cols)

    # 레이블이 단일값(예 : 전부 0)이면 평가 의미 없음 -> skip
    if len(set(y)) < 2:
        return None

    # 훈련/테스트 분할 (시간 순 8:2 split)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 모델 학습/평가
    model, result = train_and_evaluate(model_name, X_train, X_test, y_train, y_test, param)

    # 모델 저장 경로
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}_window{window_size}.pkl"
    joblib.dump(model, model_path)

    return {
        'model': model_name,
        'window': window_size,
        'params': param,
        **result
    }

# -----------------------------------
# 전체 병렬 실험 실행
# -----------------------------------
def run_all_experiments(df):
    tasks = []  # 각 조합별 (df, model, window, param, feature_cols) 튜플 생성
    for model in models:
        for window in windows:
            for param in param_grid[model]:
                tasks.append((df.copy(), model, window, param, feature_cols))

    results = []
    # task별 병렬 실행
    with Pool(processes=os.cpu_count() - 1) as pool:    # 가용가능한 cpu수 - 1 사용
        for result in tqdm(pool.imap_unordered(run_one, tasks), total=len(tasks)):
            if result is not None:
                results.append(result)

    return results

# -----------------------------------
# Markdown 형식으로 실험 결과 요약 파일 저장
# -----------------------------------
def save_markdown_summary(csv_path: str, md_path: str):
    # 실험 결과 CSV 불러오기
    df = pd.read_csv(csv_path)

    # 필요한 컬럼만 추출
    summary = df[['model', 'window', 'params', 'accuracy', 'f1', 'roc_auc']]
    summary = summary.sort_values(by=['model', 'window']).reset_index(drop=True)

    # Markdown 테이블 헤더 작성
    md_lines = [
        "# 실험 결과 요약",
        "",
        "| Model | Window | Params | Accuracy | F1 Score | ROC AUC |",
        "|-------|--------|--------|----------|----------|---------|"
    ]

    # 각 실험 결과를 Markdown 테이블 형식으로 포맷팅
    for _, row in summary.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['window']} | `{row['params']}` | "
            f"{row['accuracy']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} |"
        )

    # Markdown 파일로 저장
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"마크다운 요약 파일 저장 완료 → {md_path}")


# 메인 실행
if __name__ == "__main__":
    today = datetime.today()
    date_name = today.strftime('%y%m%d')  # ex: 250504
    db_path = f"{date_name}_sentiment_stock.duckdb"

    con = duckdb.connect(db_path)
    df = con.execute("SELECT * FROM merged_data ORDER BY date").fetchdf()

    # 뉴스 원문 컬럼 제거 (숫자 형태의 컬럼 제거)
    cols_to_remove = [col for col in df.columns if col.isdigit()]
    df = df.drop(columns=cols_to_remove)

    results = run_all_experiments(df)

    results_df = pd.DataFrame(results)
    csv_path = f"{date_name}_experiment_results.csv"
    md_path = f"{date_name}_experiment_summary.md"
    results_df.to_csv(csv_path, index=False)
    save_markdown_summary(csv_path, md_path)

    print("실험 완료")
