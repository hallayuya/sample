import io
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

st.set_page_config(
    page_title="AI 데이터 분석",
    page_icon="🤖",
    layout="wide",
)

DATA_DIR = Path("C:/YSJ/data")
MODEL = "gemini-flash-latest"


@st.cache_data
def load_data():
    monthly = pd.read_csv(DATA_DIR / "월별_판매량.csv", encoding="utf-8-sig", parse_dates=["연월"])
    monthly["연도"] = monthly["연월"].dt.year.astype(str)
    region = pd.read_csv(DATA_DIR / "지역_카테고리_통계.csv", encoding="utf-8-sig")
    quarterly = pd.read_csv(DATA_DIR / "분기별_요약.csv", encoding="utf-8-sig")
    return monthly, region, quarterly


@st.cache_data
def build_system_prompt(monthly_json: str, quarterly_json: str) -> str:
    monthly = pd.read_json(io.StringIO(monthly_json))
    quarterly = pd.read_json(io.StringIO(quarterly_json))

    monthly["연월"] = pd.to_datetime(monthly["연월"])
    monthly["연도"] = monthly["연월"].dt.year.astype(str)

    years = sorted(monthly["연도"].unique().tolist())
    regions = sorted(monthly["지역"].unique().tolist())
    categories = monthly["고객_카테고리"].unique().tolist()

    total_vol = monthly["판매량_m3"].sum()
    total_rev = monthly["판매액_만원"].sum()
    avg_price = (total_rev / total_vol) if total_vol > 0 else 0

    top_regions = (
        monthly.groupby("지역")["판매량_m3"]
        .sum()
        .nlargest(5)
        .reset_index()
        .rename(columns={"판매량_m3": "판매량(m³)"})
    )

    cat_summary = (
        monthly.groupby("고객_카테고리")["판매량_m3"]
        .sum()
        .reset_index()
        .rename(columns={"판매량_m3": "판매량(m³)"})
    )

    recent = (
        monthly.groupby("연월")
        .agg(판매량=("판매량_m3", "sum"), 판매액=("판매액_만원", "sum"))
        .tail(24)
    )
    recent["단가"] = (recent["판매액"] / recent["판매량"]).round(2)
    recent.index = recent.index.strftime("%Y-%m")

    yearly = (
        monthly.groupby("연도")
        .agg(판매량=("판매량_m3", "sum"), 판매액=("판매액_만원", "sum"))
        .reset_index()
    )

    return f"""당신은 도시가스 판매 데이터 분석 전문가입니다. 반드시 한국어로 답변하세요.

## 데이터셋 개요
- 분석 기간: {years[0]}년 ~ {years[-1]}년
- 대상 지역 ({len(regions)}개): {', '.join(regions)}
- 고객 카테고리: {', '.join(categories)}
- 전체 판매량: {total_vol/1e6:.2f}백만 m³
- 전체 판매액: {total_rev/1e4:.1f}억 원
- 평균 단가: {avg_price:.2f} 원/m³

## 연도별 판매량
{yearly.to_string(index=False)}

## 카테고리별 총 판매량
{cat_summary.to_string(index=False)}

## 판매량 상위 5개 지역
{top_regions.to_string(index=False)}

## 최근 24개월 월별 판매량 · 단가
{recent.to_string()}

## 분기별 요약
{quarterly.to_string(index=False)}

사용자 질문에 위 데이터를 근거로 구체적인 수치와 함께 명확하게 답변하세요.
데이터에 없는 내용은 솔직히 모른다고 하세요.
"""


def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── 페이지 헤더 ──────────────────────────────────
st.title("🤖 AI 데이터 분석")
st.caption("도시가스 판매 데이터에 대해 자유롭게 질문하세요.")

with st.spinner("데이터 불러오는 중..."):
    monthly_df, region_df, quarterly_df = load_data()
    system_prompt = build_system_prompt(
        monthly_df.to_json(), quarterly_df.to_json()
    )

# ── 사이드바: 데이터 요약 ────────────────────────
with st.sidebar:
    st.header("📊 데이터 요약")
    years = sorted(monthly_df["연도"].unique())
    st.info(f"**기간:** {years[0]} ~ {years[-1]}")
    st.info(f"**지역 수:** {monthly_df['지역'].nunique()}개")
    st.info(f"**총 판매량:** {monthly_df['판매량_m3'].sum()/1e6:.1f}백만 m³")
    st.divider()
    st.caption("질문 예시")
    examples = [
        "가장 판매량이 많은 지역은?",
        "주택용과 산업용 판매량 비교해줘",
        "최근 3년간 판매 트렌드는?",
        "단가가 가장 높은 카테고리는?",
        "전년 대비 성장률이 가장 높은 지역은?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_question = ex

# ── 세션 상태 초기화 ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 대화 기록 출력 ────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 입력 처리 ─────────────────────────────────────
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("데이터에 대해 질문하세요...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            client = get_client()

            history_contents = [system_prompt + "\n\n대화를 시작합니다."]
            for m in st.session_state.messages[:-1]:
                prefix = "사용자: " if m["role"] == "user" else "어시스턴트: "
                history_contents.append(prefix + m["content"])
            history_contents.append("사용자: " + user_input)

            resp = client.models.generate_content(
                model=MODEL,
                contents="\n\n".join(history_contents),
            )
            answer = resp.text

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── 대화 초기화 버튼 ──────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ 대화 초기화", type="secondary"):
        st.session_state.messages = []
        st.rerun()
