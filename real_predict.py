import os
import joblib
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------------
# 설정: 모델, 윈도우, 피처
# -----------------------------------
models = ['lightgbm', 'xgboost', 'randomforest', 'decisiontree', 'gradientboosting']
windows = [5, 15, 30]
feature_cols = ['news_count', 'avg_negative', 'avg_neutral', 'avg_positive', 'open', 'high', 'low', 'close', 'volume']

# -----------------------------------
# 오늘 데이터의 입력 피처 생성 함수
# -----------------------------------
def get_today_feature(df, window_size, feature_cols):
    # 라벨이 NaN인 오늘 데이터 제외하고 window 확보
    if len(df) < window_size:
        print("Not enough data for window size", window_size)
        return None

    # 마지막 window_size 만큼 슬라이싱
    feature_window = df.iloc[-window_size:][feature_cols]
    if feature_window.isnull().any().any():
        print("Missing values in feature window")
        return None

    # flatten
    return feature_window.values.flatten().reshape(1, -1)

# -----------------------------------
# 메인 실행
# -----------------------------------
if __name__ == "__main__":
    today = datetime.today()
    date_name = today.strftime('%y%m%d')  # 예: 250507
    db_path = f"{date_name}_sentiment_stock.duckdb"

    # DB에서 데이터 불러오기
    con = duckdb.connect(db_path)
    df = con.execute("SELECT * FROM merged_data ORDER BY date").fetchdf()

    # 뉴스 원문 컬럼 제거 (숫자 형태 컬럼 제외)
    cols_to_remove = [col for col in df.columns if col.isdigit()]
    df = df.drop(columns=cols_to_remove)

    # 오늘 데이터 (label == NaN) 기준으로 예측
    today_row = df[df['label'].isna()].copy()
    if today_row.empty:
        print("오늘 예측할 데이터가 없습니다.")
        exit()

    print("오늘 날짜:", today_row['date'].values[0])

    predictions = []
    for model_name in models:
        for window in windows:
            model_path = f"models/{model_name}_window{window}.pkl"
            if not os.path.exists(model_path):
                print(f"모델 파일 없음: {model_path}")
                continue

            X_today = get_today_feature(df, window, feature_cols)
            if X_today is None:
                continue

            model = joblib.load(model_path)
            prob = model.predict_proba(X_today)[0][1]  # 클래스 1 (상승) 확률

            predictions.append({
                'model': model_name,
                'window': window,
                'rise_probability': round(prob, 4),
                'prediction': int(prob > 0.5)
            })

    if predictions:
        print("\n--- 오늘 종가 예측 결과 ---")
        for pred in predictions:
            print(f"[{pred['model']:^17}] 윈도우={pred['window']:>2}일 → 상승 확률={pred['rise_probability']:.4f} → 예측: {'상승' if pred['prediction']==1 else '하락/유지'}")
    else:
        print("모델 예측 불가능 (데이터 부족 또는 모델 누락)")