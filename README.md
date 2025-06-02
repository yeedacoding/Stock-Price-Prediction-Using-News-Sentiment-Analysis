# 💹 감성은 시장을 움직일까? : 뉴스 감성분석 기반 삼성전자 주가 등락 방향 예측 프로젝트
<p align="center"><img width="600" alt="image" src="https://github.com/user-attachments/assets/8699150a-443f-4d38-bb84-9b9a0d0b77c6" />
</p>
<br>


## 💡 소개
### 주제
- 뉴스 감성분석을 통해 삼성전자의 다음날 종가 등락 방향을 예측
### 목표
- 삼성전자 뉴스 기사의 감성분석 결과와 주가 데이터를 결합하여 다음날 종가 등락 방향을 예측
- 주가 데이터로만 이루어진 데이터와 감성분석 결과+주가가 결합된 데이터 간의 예측 성능을 비교하여, 감성확률이라는 심리 반영 데이터가 시장에 영향을 미치는지 분석
### 데이터 출처
- 뉴스 데이터 : BigKinds 뉴스 아카이빙 사이트 크롤링 [BigKinds](https://www.bigkinds.or.kr/v2/news/index.do)
- 주가 데이터 : yfinance (Python open source) 활용 데이터 수집
<br>

## 👥 주요 수행 역할 (개인 프로젝트)
| 김태헌 |
|:---:|
| 데이터 수집(뉴스 기사 크롤링, 주가 데이터 수집) <br> 일별 기사 감성 분석 <br> EDA <br> DB 및 수집/저장 자동화 프로세스 구축 <br> 모델링 <br> 방향 예측 모델 평가 |
<br>

## ⏰ 개발 기간
- 25.04.28 ~ 25.05.21

<br>

## 🏔️ 개발 과정
<img width="1177" alt="image" src="https://github.com/user-attachments/assets/23956090-3a3b-4a6a-9b8f-8f8549198207" />
<br>

#### 1️⃣ 1차 : 문제정의
- "**주식 시장은 투자 심리에 민감하게 반응하지 않을까?**"라는 투자 심리와 시장과의 연관성에 대한 궁금증 발생
- 기존 금융 주가 데이터 기반 예측 모델은 시장 심리를 반영하지 못하는 한계점 존재

> **기대 효과** : 분석 및 모델링 결과를 리스크 경보 시스템 등의 **단기 투자 전략 보조 도구로 활용** 기대

#### 2️⃣ 2차 : 데이터 수집
- 수집 기간 : 2021년 1월 1일 ~ 2025년 5월 20일
  - 뉴스 데이터 : 삼성전자 관련 기사 수집 ([code](https://github.com/yeedacoding/Stock-Price-Prediction-Using-Sentiment-Analysis-from-News-Headlines/blob/master/news_crawling.ipynb))
    - 언론사 : 한국 대형 경제 언론사 4곳 (매일경제, 서울경제, 아주경제, 한국경제)
    - 카테고리 : 경제, 국제, IT_과학
  - 주가 데이터 : yfinanace 활용 OHLCV 데이터 수집 (Open, High, Low, Close, Volume) ([code](https://github.com/yeedacoding/Stock-Price-Prediction-Using-Sentiment-Analysis-from-News-Headlines/blob/master/stock_package.ipynb))

#### 3️⃣ 3차 : 뉴스 기사 감성 분석
<p align="center"><img width="701" alt="image" src="https://github.com/user-attachments/assets/e7930ec7-8284-4d20-aaa8-ff458fde4638" /></p>

- 감성 분석 : **KR-FinBERT-SC** 모델 활용 일별 평균 감성확률 값 계산 ([code](https://github.com/yeedacoding/Stock-Price-Prediction-Using-Sentiment-Analysis-from-News-Headlines/blob/master/sentiment_analysis.ipynb))
  1) 일별 수집된 각 뉴스 기사에 대한 감성 확률 계산 (부정, 중립, 긍정)
  2) 일별 모든 뉴스 기사에 대한 감성 확률 평균 계산 (avg_negative, avg_neutral, avg_positive)

#### 4️⃣ 4차 : 데이터 merge 및 EDA
- 뉴스 원문, 감성 확률, 주가 데이터를 날짜 기준으로 merge ([code](https://github.com/yeedacoding/Stock-Price-Prediction-Using-Sentiment-Analysis-from-News-Headlines/blob/master/sentiment_analysis.ipynb))
- 주말, 공휴일에 기사가 게재된 날이 있으므로 주말/공휴일 포함 vs 주말/공휴일 미포함 데이터로 나누어 추가 분석 진행
  1) 주말, 공휴일 포함 데이터 : weekend_df
  2) 주말, 공휴일 미포함 데이터 : weekday_df
- EDA code : [weekend_df](https://github.com/yeedacoding/Stock-Price-Prediction-Using-Sentiment-Analysis-from-News-Headlines/blob/master/eda_weekend.ipynb), 
  [weekday_df](https://github.com/yeedacoding/Stock-Price-Prediction-Using-Sentiment-Analysis-from-News-Headlines/blob/master/eda_weekday.ipynb)

#### 5️⃣ 5차 : 모델링 및 방향 예측 모델 평가
- 예측 대상 : **다음날 종가(Close)의 등락 여부** (label 0 : 하락 / label 1 : 상승)
- 비교 실험
<p align="center"><img width="665" alt="image" src="https://github.com/user-attachments/assets/1ce1550d-3141-487d-94dc-4101c89d94f3" /></p>

- 기본 모델 설계
<p align="center"><img width="690" alt="image" src="https://github.com/user-attachments/assets/41f07783-bc92-4bf9-b8f1-19509d678680" /></p>

- 방향 예측 모델 평가 : [code](https://github.com/yeedacoding/Stock-Price-Prediction-Using-Sentiment-Analysis-from-News-Headlines/blob/master/modeling.ipynb)
<br>

## 결과
<p align="center"><img width="711" alt="image" src="https://github.com/user-attachments/assets/06c7b810-cb30-4814-b2b1-19d49b96cd3e" />
</p>

<p align="center"><img width="711" alt="image" src="https://github.com/user-attachments/assets/088b346a-cf31-4e40-baad-5be79bca26e8" /></p>


## 📒 PPT 및 개발일지

