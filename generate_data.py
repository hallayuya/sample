import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 데이터 생성 설정
np.random.seed(42)

# 사업장 목록
facilities = [
    '서울 지점', '부산 지점', '대구 지점', '인천 지점', '광주 지점',
    '대전 지점', '울산 지점', '경기 지점', '강원 지점', '충북 지점',
    '충남 지점', '전북 지점', '전남 지점', '경북 지점', '경남 지점', '제주 지점'
]

# 고객 카테고리
categories = ['주택용', '상업용', '산업용', '발전용']

# 데이터 디렉토리 생성
os.makedirs('C:\\YSJ\\data', exist_ok=True)

# 1. 월별 판매량 데이터 (2022-2025)
print("[1] 월별 판매량 데이터 생성 중...")
start_date = datetime(2022, 1, 1)
months = pd.date_range(start=start_date, periods=36, freq='MS')

data_monthly = []
for month in months:
    for facility in facilities:
        for category in categories:
            base_sales = {
                '주택용': np.random.uniform(15000, 35000),
                '상업용': np.random.uniform(8000, 20000),
                '산업용': np.random.uniform(20000, 50000),
                '발전용': np.random.uniform(10000, 30000)
            }

            sales_volume = base_sales[category] * np.random.uniform(0.8, 1.2)

            data_monthly.append({
                '연월': month.strftime('%Y-%m'),
                '사업장': facility,
                '지역': facility.replace(' 지점', ''),
                '고객_카테고리': category,
                '판매량_m3': round(sales_volume, 2),
                '판매액_만원': round(sales_volume * np.random.uniform(0.8, 1.2) * 100, 0),
                '고객수': np.random.randint(100, 5000),
                '평균_사용량': round(sales_volume / np.random.randint(100, 5000), 2)
            })

df_monthly = pd.DataFrame(data_monthly)
df_monthly.to_csv('C:\\YSJ\\data\\월별_판매량.csv', index=False, encoding='utf-8-sig')
print(f"OK - 월별_판매량.csv ({len(df_monthly)} 행)")

# 2. 사업장별 상세 데이터 (사업장당 1개 파일)
print("[2] 사업장별 상세 데이터 생성 중...")
for facility in facilities:
    data_facility = []
    for month in months:
        for category in categories:
            sales_volume = np.random.uniform(5000, 40000)
            for day in range(1, 28):  # 일별 데이터
                daily_variance = np.random.uniform(0.7, 1.3)
                data_facility.append({
                    '일시': f"{month.strftime('%Y-%m')}-{day:02d}",
                    '고객_카테고리': category,
                    '판매량_m3': round(sales_volume * daily_variance / 27, 2),
                    '판매액_만원': round(sales_volume * daily_variance * 100 / 27, 0),
                    '고객수': np.random.randint(10, 500)
                })

    df_facility = pd.DataFrame(data_facility)
    filename = f"C:\\YSJ\\data\\{facility.replace(' ', '_')}.csv"
    df_facility.to_csv(filename, index=False, encoding='utf-8-sig')

print(f"OK - 사업장별 파일 16개 생성 완료")

# 3. 분기별 요약 데이터
print("[3] 분기별 요약 데이터 생성 중...")
quarters = pd.period_range(start='2022Q1', periods=12, freq='Q')
data_quarterly = []

for quarter in quarters:
    for facility in facilities:
        quarter_data = df_monthly[
            (df_monthly['사업장'] == facility) &
            (df_monthly['연월'].str[:4] == str(quarter.year)) &
            (df_monthly['연월'].str[5:7].astype(int).isin([quarter.quarter*3-2, quarter.quarter*3-1, quarter.quarter*3]))
        ]

        if len(quarter_data) > 0:
            data_quarterly.append({
                '분기': f"{quarter.year}Q{quarter.quarter}",
                '사업장': facility,
                '지역': facility.replace(' 지점', ''),
                '총_판매량_m3': round(quarter_data['판매량_m3'].sum(), 2),
                '총_판매액_만원': round(quarter_data['판매액_만원'].sum(), 0),
                '총_고객수': quarter_data['고객수'].sum(),
                '평균_판매량': round(quarter_data['판매량_m3'].mean(), 2)
            })

df_quarterly = pd.DataFrame(data_quarterly)
df_quarterly.to_csv('C:\\YSJ\\data\\분기별_요약.csv', index=False, encoding='utf-8-sig')
print(f"OK - 분기별_요약.csv ({len(df_quarterly)} 행)")

# 4. 지역별 카테고리 통계
print("[4] 지역별 카테고리 통계 생성 중...")
data_region_category = []
for region in set([f.replace(' 지점', '') for f in facilities]):
    region_data = df_monthly[df_monthly['지역'] == region]
    for category in categories:
        category_data = region_data[region_data['고객_카테고리'] == category]
        if len(category_data) > 0:
            data_region_category.append({
                '지역': region,
                '고객_카테고리': category,
                '총_판매량_m3': round(category_data['판매량_m3'].sum(), 2),
                '총_판매액_만원': round(category_data['판매액_만원'].sum(), 0),
                '총_고객수': category_data['고객수'].sum(),
                '데이터_개수': len(category_data)
            })

df_region_category = pd.DataFrame(data_region_category)
df_region_category.to_csv('C:\\YSJ\\data\\지역_카테고리_통계.csv', index=False, encoding='utf-8-sig')
print(f"OK - 지역_카테고리_통계.csv ({len(df_region_category)} 행)")

# 5. 일일 판매 추이 데이터
print("[5] 일일 판매 추이 데이터 생성 중...")
data_daily = []
current_date = start_date
while current_date < datetime(2025, 1, 1):
    for facility in facilities:
        daily_sales = np.random.uniform(10000, 60000)
        data_daily.append({
            '날짜': current_date.strftime('%Y-%m-%d'),
            '요일': ['월', '화', '수', '목', '금', '토', '일'][current_date.weekday()],
            '사업장': facility,
            '판매량_m3': round(daily_sales, 2),
            '판매액_만원': round(daily_sales * 100, 0)
        })
    current_date += timedelta(days=1)

df_daily = pd.DataFrame(data_daily)
df_daily.to_csv('C:\\YSJ\\data\\일일_판매량.csv', index=False, encoding='utf-8-sig')
print(f"OK - 일일_판매량.csv ({len(df_daily)} 행)")

print("\n" + "="*60)
print("SUCCESS - 모든 CSV 파일 생성 완료!")
print("="*60)
print(f"Save location: C:\\YSJ\\data\\")
print(f"\nGenerated files:")
print(f"  - 월별_판매량.csv : 월별 판매 현황")
print(f"  - 분기별_요약.csv : 분기별 요약 통계")
print(f"  - 지역_카테고리_통계.csv : 지역별·카테고리별 분석")
print(f"  - 일일_판매량.csv : 일일 판매 추이")
print(f"  - [사업장별].csv : 16개 사업장 상세 데이터")
print(f"\nTotal files: 21")
print(f"Total rows: {len(df_monthly) + len(df_quarterly) + len(df_region_category) + len(df_daily):,}")
