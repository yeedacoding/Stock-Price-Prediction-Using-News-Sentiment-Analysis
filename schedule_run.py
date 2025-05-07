import schedule
import time
import subprocess
from datetime import datetime

def run_pipeline():
    today = datetime.today().strftime('%Y-%m-%d')
    print(f"[{today}] 실행 시작")

    try:
        # 1. 데이터 수집 및 DuckDB 저장
        subprocess.run(["python3", "update_today_data.py"], check=True)

        # 2. 슬라이딩 윈도우 모델링 실행
        subprocess.run(["python3", "sliding_window.py"], check=True)

        print(f"[{today}] 전체 파이프라인 완료")

    except subprocess.CalledProcessError as e:
        print(f"[{today}] 실행 중 오류 발생: {e}")

# 매일 23:30 실행
schedule.every().day.at("23:30").do(run_pipeline)

print("스케줄러 시작됨: 매일 23:30에 자동 실행됩니다.")
while True:
    schedule.run_pending()
    time.sleep(1)

