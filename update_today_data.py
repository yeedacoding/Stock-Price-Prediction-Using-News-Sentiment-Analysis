import sys
import time
import duckdb
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------
# 1. 뉴스 크롤링
# -----------------------------------
def crawl_news(date) :
    # webdriver 연동
    driver = webdriver.Chrome()
    driver.get("https://www.bigkinds.or.kr/v2/news/index.do")

    ## 언론사 필터링
    find_media = ['매일경제', '서울경제', '아주경제', '한국경제']
    media_list = driver.find_element(By.CSS_SELECTOR, "#category_provider_list > li")
    for media in media_list :
        label = media.find_element(By.CSS_SELECTOR, "span > label")
        label_text = label.text.strip()
        if label_text in find_media :
            driver.execute_script("arguments[0].click();", label)
        else :
            continue

    ## 기간 설정
    duration_news = driver.find_element(By.CSS_SELECTOR, "#collapse-step-1-body > div.srch-detail.v2 > div > div.tab-btn-wp1 > div.tab-btn-inner.tab1 > a")
    duration_news.click()

    time.sleep(0.5)

    # 기간 직접 입력
    start_date = driver.find_element(By.CSS_SELECTOR, "#search-begin-date")
    start_date.send_keys(Keys.COMMAND + "a")
    time.sleep(1)
    start_date.send_keys(Keys.DELETE)
    time.sleep(0.2)
    start_date.send_keys(date)
    time.sleep(1)

    end_date = driver.find_element(By.CSS_SELECTOR, "#search-end-date")
    end_date.send_keys(Keys.COMMAND + "a")
    time.sleep(1)
    end_date.send_keys(Keys.DELETE)
    time.sleep(0.2)
    end_date.send_keys(date)
    time.sleep(0.5)
    end_date.send_keys(Keys.ENTER)

    ## 카테고리 필터링 : 경제, 국제, IT_과학
    general_category = driver.find_element(By.CSS_SELECTOR, "#collapse-step-1-body > div.srch-detail.v2 > div > div.tab-btn-wp2 > div.tab-btn-inner.tab3 > a")
    general_category.click()

    # 경제, 국제, IT_과학 체크박스 클릭
    economy_checkbox = driver.find_element(By.CSS_SELECTOR, "#srch-tab3 > ul > li:nth-child(2) > div > span:nth-child(3)")
    economy_checkbox.click()

    world_checkbox = driver.find_element(By.CSS_SELECTOR, "#srch-tab3 > ul > li:nth-child(5) > div > span:nth-child(3)")
    world_checkbox.click()

    it_checkbox = driver.find_element(By.CSS_SELECTOR, "#srch-tab3 > ul > li:nth-child(8) > div > span:nth-child(3)")
    it_checkbox.click()

    ## 상세검색 조건 설정
    detail_search = driver.find_element(By.CSS_SELECTOR, "#collapse-step-1-body > div.srch-detail.v2 > div > div.tab-btn-wp3 > div.tab-btn-inner.tab5 > a")
    detail_search.click()

    # 검색어 범위 설정
    search_range = driver.find_element(By.CSS_SELECTOR, "#search-scope-type")
    search_range.click()

    # 제목 검색 설정
    filter_title = driver.find_element(By.CSS_SELECTOR, "#search-scope-type > option:nth-child(2)")
    filter_title.click()

    # 단어 중 1개 이상 포함 : "삼성전자"
    filter_samsung = driver.find_element(By.CSS_SELECTOR, "#orKeyword1")
    filter_samsung.send_keys("삼성전자")

    # 제외 단어 설정
    filter_except = driver.find_element(By.CSS_SELECTOR, "#notKeyword1")
    filter_except.send_keys("[속보] OR [스팟] OR 칼럼")

    ## 최종 검색 버튼 클릭
    search_btn = driver.find_element(By.CSS_SELECTOR, "#detailSrch1 > div.srch-foot > div > button.btn.btn-md.btn-primary.news-search-btn")
    search_btn.click()

    time.sleep(5)

    ## 분석 기사 클릭
    label = driver.find_element(By.CSS_SELECTOR, 'label[for="filter-tm-use"]')

    # JS로 강제 클릭 (보통 이게 가장 확실함)
    driver.execute_script("arguments[0].click();", label)

    time.sleep(5)

    ## 보기 정렬 : 과거순
    sort_articles = driver.find_element(By.CSS_SELECTOR, "#select1")
    sort_articles.click()

    sort_ascending = driver.find_element(By.CSS_SELECTOR, "#select1 > option:nth-child(3)")
    sort_ascending.click()

    time.sleep(5)

    ## 보기 개수 : 100개
    view_articles = driver.find_element(By.CSS_SELECTOR, "#select2")
    view_articles.click()

    view_hunnit = driver.find_element(By.CSS_SELECTOR, "#select2 > option:nth-child(4)")
    view_hunnit.click()

    time.sleep(5)

    ## 뉴스 크롤링
    # 전체 보기 페이지 개수
    total_page = int(driver.find_element(By.CSS_SELECTOR, "#news-results-tab > div.data-result-btm.m-only.paging-v3-wrp > div.btm-pav-wrp > div > div > div > div:nth-child(6) > div").text)
    news_by_date = defaultdict(list)

    for i in range(1, total_page+1) :
        print(f"=========={i} 페이지 뉴스 기사 크롤링 시작==========")
        
        # 한 페이지 전체 뉴스 기사 리스트
        article_list = driver.find_elements(By.CSS_SELECTOR, "#news-results > div")
        
        for j in range(1, len(article_list)+1) :
            tmp_article = driver.find_element(By.CSS_SELECTOR, f"#news-results > div:nth-child({j}) > div > div.cont > a")

            # 기사 제목
            title = tmp_article.find_element(By.TAG_NAME, "span").text
            # 본문
            summary_element = tmp_article.find_element(By.TAG_NAME, "p")
            summary_html = summary_element.get_attribute("innerHTML")

            # <br>를 줄바꿈 문자로 변환
            parts = summary_html.replace('<br>', '\n').replace('<br/>', '\n').split('\n')

            # 각 줄 공백 정리
            parts = [part.strip() for part in parts if part.strip()]

            # 마지막 줄 점검
            if parts and '..' in parts[-1]:
                parts = parts[:-1]  # 마지막 문장이 ".." 포함이면 버림

            # 최종 텍스트 생성
            summary_text = ' '.join(parts)

            # 제목 + 구분자 + 본문
            SEPERATOR = " ||| "
            final_article = title + SEPERATOR + summary_text

            # 날짜
            dt = driver.find_element(By.CSS_SELECTOR, f"#news-results > div:nth-child({j}) > div > div.cont > div > p:nth-child(2)").text
            # datetime 객체로 변환
            date = datetime.strptime(dt, '%Y/%m/%d')

            # 원하는 형식의 문자열로 변환
            article_date = date.strftime('%Y-%m-%d')

            # 딕셔너리 저장
            news_by_date[date].append(final_article)

        # 현재 페이지
        current_input = driver.find_element(By.CSS_SELECTOR, "#paging_news_result")
        current_page = int(current_input.get_attribute("value"))

        if current_page == total_page :
            print("마지막 페이지에 도달했습니다.")
            break

        # 다음 페이지 : 입력창에 페이지 번호 직접 입력
        next_button = driver.find_element(By.CSS_SELECTOR, "#news-results-tab > div.data-result-btm.m-only.paging-v3-wrp > div.btm-pav-wrp > div > div > div > div:nth-child(7) > a")
        driver.execute_script("arguments[0].click();", next_button)

        WebDriverWait(driver, 200).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, "#collapse-step-2-body > div > div.data-result.loading-cont > div.news-loader.loading > div")))

        new_input = driver.find_element(By.CSS_SELECTOR, "#paging_news_result")
        new_page = int(new_input.get_attribute("value"))

        # date_name = date.split('-')[0][2:] + date.split('-')[1] + date.split('-')[2]

        # 최대 기사 수를 가진 데이터의 컬럼 개수 지정
        max_articles = max(len(articles) for articles in news_by_date.values())

        # 최대 기사 수를 컬럼 수로 지정
        columns = ['date'] + [str(i) for i in range(1, max_articles+1)] + ['news_count']

        data_rows = []
        for date, articles in news_by_date.items() :
            date_str = date.strftime('%Y-%m-%d')
            news_count = len(articles)

            row = [date_str] + articles
            # 최대 기사 수보다 적은 기사 수를 가진 날에는 빈 컬럼값에 NaN 처리
            if len(articles) < max_articles :
                row += [pd.NA] * (max_articles - len(articles))

            row += [news_count]
            data_rows.append(row)

    driver.close()

    # 데이터프레임 생성
    news_df = pd.DataFrame(data_rows, columns = columns)

    return news_df
    

# -----------------------------------
# 2. sentiment analysis : 뉴스 데이터로부터 감성 확률 계산
# -----------------------------------
def analyze_sentiment(text : str) :
    # 모델 이름
    MODEL_NAME = "snunlp/KR-FinBert-SC"

    # tokenizer initialize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 사전학습된 감성 분류 모델 불러오기
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # 추론 모드
    model.eval()

    # 입력 문장 토크나이징 및 텐서 변환
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # 추론 모드에서 모델 실행 (기울기 미계산)
    with torch.no_grad() :
        outputs = model(**inputs)

    # softmax를 통해 확률값 벡터화
    probs = softmax(outputs.logits, dim=1).squeeze().tolist()

    # 결과를 딕셔너리로 반환 (부정/중립/긍정 확률)
    return {
        'negative' : probs[0],
        'neutral' : probs[1],
        'positive' : probs[2]
    }

def news_analyze(date) :
    news_df = crawl_news(date)
    sentiment_daily = []

    # 전체 날짜 순회
    for idx, row in tqdm(news_df.iterrows(), total=len(news_df)) :
        date = row['date']
        news_count = row['news_count']

        # 기사 컬럼 자동 추출 (숫자 컬럼만)
        article_cols = [col for col in row.index if col.isdigit()]

        daily_vectors = []

        for col in article_cols :
            headline = row[col]
            if isinstance(headline, str) and len(headline.strip()) > 0 :
                sentiment = analyze_sentiment(headline)
                daily_vectors.append([
                    sentiment['negative'],
                    sentiment['neutral'],
                    sentiment['positive']
                ])

        # 기본 딕셔너리 구성
        sentiment_record = {
            'date' : date,
            'news_count' : news_count,
            **{col : row[col] for col in article_cols} # 기사 원문도 저장
        }

        # 감성 벡터 평균 게산
        if daily_vectors :
            arr = np.array(daily_vectors)
            avg_vector = arr.mean(axis=0)   # 일별 평균 감성확률
        else :
            # 뉴스가 하나도 없을 경우 -> avg_neutral = 1.0
            avg_vector = [0.0, 1.0, 0.0]

        sentiment_record.update({
            'avg_negative' : round(avg_vector[0], 4),
            'avg_neutral' : round(avg_vector[1], 4),
            'avg_positive' : round(avg_vector[2], 4)
        })

        sentiment_daily.append(sentiment_record)

    sentiment_df = pd.DataFrame(sentiment_daily)

    return sentiment_df

# -----------------------------------
# 3. 주가 데이터 수집
# -----------------------------------
def stock_data(date) :
    # 수집 날짜 설정
    target_date = date
    next_date = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    # datetime 객체로 변환
    date_obj = datetime.strptime(target_date, '%Y-%m-%d')

    # weekday()는 월=0, ..., 일=6
    is_weekend = date_obj.weekday() >= 5 # 토=5, 일=6
    print(f"Is weekend? {is_weekend}") # True면 주말

    # 삼성전자 티커
    samsung = yf.Ticker("005930.KS")
    # yfinance는 end 날짜를 포함하지 않으므로 다음 날을 end 날짜로 설정
    hist = samsung.history(start=target_date, end=next_date, auto_adjust=False)

    if is_weekend :
        # 주말인 경우 : 데이터 없음 처리
        hist = pd.DataFrame([{
            'date' : target_date,
            'open' : np.nan,
            'high' : np.nan,
            'low' : np.nan,
            'close' : np.nan,
            'volume' : np.nan
        }])

    else :
        # 평일인 경우 : yfinance로 실제 데이터 요청
        next_date = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        
        if not hist.empty :
            hist = hist.reset_index()
            hist.columns = hist.columns.str.lower()
            hist['date'] = pd.to_datetime(hist['date']).dt.date.astype(str)
            hist = hist[['date', 'open', 'high', 'low', 'close', 'volume']]
        else :
            # 거래일인데 데이터가 없을 경우 (ex. 한국 공휴일)
            hist = pd.DataFrame([{
                'date' : target_date,
                'open' : np.nan,
                'high' : np.nan,
                'low' : np.nan,
                'close' : np.nan,
                'volume' : np.nan
            }])

    stock_df = hist.copy()

    return stock_df


# -----------------------------------
# 4. 뉴스 데이터(감성확률 포함) + 주가 데이터 merge : sentiment_df + stock_df
# -----------------------------------
def merge_sentiment_stock(date) :
    previous_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    # 파일명 호출을 위한 날짜명 처리 : "2025-05-04" -> "250504"
    date_name = previous_date.split('-')[0][2:] + previous_date.split('-')[1] + previous_date.split('-')[2]

    # DuckDB에 저장된 가장 최신의 데이터를 불러옴
    con = duckdb.connect(f'{date_name}_sentiment_stock.duckdb')

    prep = con.execute("SELECT * FROM merged_data").fetchdf()
    con.close()

    # 오늘 수집된 sentiment, stock 불러오기
    sentiment = news_analyze(date)
    stock = stock_data(date)

    # 날짜 형식 정리
    sentiment['date'] = pd.to_datetime(sentiment['date']).dt.normalize()
    sentiment['date'] = sentiment['date'].dt.strftime('%Y-%m-%d')

    stock['date'] = pd.to_datetime(stock['date']).dt.normalize()
    stock['date'] = stock['date'].dt.strftime('%Y-%m-%d')

    # sentiment + stock merge
    merged = pd.merge(sentiment, stock, on='date', how='left')

    # 준비된 이전날까지의 데이터(prep)와 오늘 수집된 데이터(merged)를 concat
    merged = pd.concat([prep, merged], ignore_index=True)

    # 정렬 후 forward fill
    merged = merged.sort_values('date').reset_index(drop=True)
    merged[['open', 'high', 'low', 'close', 'volume']] = merged[['open', 'high', 'low', 'close', 'volume']].ffill()

    ## 레이블(target) 생성
    # target = 이전 날 대비 주가 등락 여부(상승 : 1 / 하락or유지 : 0)
    merged['next_close'] = merged['close'].shift(-1)    # 다음날 종가 생성

    merged['label'] = (merged['next_close'] > merged['close']).astype(int)

    # 필요없는 컬럼 삭제
    merged = merged.drop(columns=['next_close'])

    return merged

# -----------------------------------
# 5. DuckDB 적재
# -----------------------------------
import os

def save_db(date):
    # 날짜 객체 변환
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    previous_date = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')

    # 파일명 처리
    previous_name = previous_date.split('-')[0][2:] + previous_date.split('-')[1] + previous_date.split('-')[2]
    today_name = date.split('-')[0][2:] + date.split('-')[1] + date.split('-')[2]

    previous_file = f'{previous_name}_sentiment_stock.duckdb'
    today_file = f'{today_name}_sentiment_stock.duckdb'

    # 1. 이전 날 파일에서 데이터 로드
    con_prev = duckdb.connect(previous_file)
    prep = con_prev.execute("SELECT * FROM merged_data").fetchdf()
    con_prev.close()

    # 2. 오늘 날짜 기준 데이터 수집 및 병합
    merged = merge_sentiment_stock(date)

    # 3. 오늘 날짜 파일에 저장
    con = duckdb.connect(today_file)
    con.execute("DROP TABLE IF EXISTS merged_data")
    con.register("df", merged)
    con.execute("CREATE TABLE merged_data AS SELECT * FROM df")
    con.close()

    # 4. 이전 파일 삭제
    if os.path.exists(previous_file):
        os.remove(previous_file)
        print(f"이전 파일 삭제 완료: {previous_file}")
    else:
        print(f"이전 파일이 존재하지 않아 삭제하지 않음: {previous_file}")



# -----------------------------------
# main 실행
# -----------------------------------
if __name__ == "__main__" :
    # 자동 schedule 외 직접 날짜 지정시 사용
    # python update_today_data.py 2025-05-05
    if len(sys.argv) > 1 :
        target_date = sys.argv[1]
    else :
        target_date = datetime.today().strftime('%Y-%m-%d') # 오늘 날짜

    # 전체 실행
    try :
        save_db(target_date)
        print(f"{target_date} 데이터 저장 완료")
    except Exception as e :
        print(f"에러발생 : {e}")