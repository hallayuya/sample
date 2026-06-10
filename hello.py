import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

# ──────────────────────────────────────────
# [섹션 A] 상수 / 설정
# ──────────────────────────────────────────
st.set_page_config(
    page_title="도시가스 판매 분석",
    page_icon="🔥",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
FONT = "Malgun Gothic, NanumGothic, sans-serif"
CATEGORY_ORDER = ["주택용", "상업용", "산업용", "발전용"]
CATEGORY_COLORS = {"주택용": "#4C72B0", "상업용": "#DD8452", "산업용": "#55A868", "발전용": "#C44E52"}


def _layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        font=dict(family=FONT, size=12),
        height=height,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# ──────────────────────────────────────────
# [섹션 B] 데이터 로딩
# ──────────────────────────────────────────
@st.cache_data
def load_all_data():
    monthly = pd.read_csv(DATA_DIR / "월별_판매량.csv", encoding="utf-8-sig", parse_dates=["연월"])
    monthly["연도"] = monthly["연월"].dt.year.astype(str)
    monthly["월"] = monthly["연월"].dt.month

    region = pd.read_csv(DATA_DIR / "지역_카테고리_통계.csv", encoding="utf-8-sig")
    quarterly = pd.read_csv(DATA_DIR / "분기별_요약.csv", encoding="utf-8-sig")

    return {"monthly": monthly, "region": region, "quarterly": quarterly}


@st.cache_data
def build_price_series(monthly_json: str):
    """월별_판매량 → 전국 월별 단가 시계열"""
    df = pd.read_json(io.StringIO(monthly_json))
    df["연월"] = pd.to_datetime(df["연월"])
    agg = df.groupby("연월").agg(판매액=("판매액_만원", "sum"), 판매량=("판매량_m3", "sum"))
    agg["단가"] = agg["판매액"] / agg["판매량"]
    series = agg["단가"].sort_index()
    series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
    series = series.groupby(series.index).mean()
    series = series.asfreq("MS").interpolate(method="time")
    return series


@st.cache_data
def run_forecast(monthly_json: str, horizon: int = 12):
    """ARIMA(2,1,2)로 예측 + 80% 신뢰구간"""
    df = pd.read_json(io.StringIO(monthly_json))
    df["연월"] = pd.to_datetime(df["연월"])
    agg = df.groupby("연월").agg(판매액=("판매액_만원", "sum"), 판매량=("판매량_m3", "sum"))
    agg["단가"] = agg["판매액"] / agg["판매량"]
    series = agg["단가"].sort_index()
    series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
    series = series.groupby(series.index).mean()
    series = series.asfreq("MS").interpolate(method="time")

    fit = ARIMA(series, order=(2, 1, 2)).fit()
    pred = fit.get_forecast(horizon)
    forecast = pred.predicted_mean
    ci = pred.conf_int(alpha=0.20)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    resid = fit.resid.astype(float)

    return forecast, lower, upper, resid


# ──────────────────────────────────────────
# [섹션 C] 전처리 헬퍼
# ──────────────────────────────────────────
def filter_monthly(df, years, categories):
    mask = pd.Series(True, index=df.index)
    if years:
        mask &= df["연도"].isin(years)
    if categories:
        mask &= df["고객_카테고리"].isin(categories)
    return df[mask]


def calc_yoy_by_region(df):
    """지역별 최근 2개 연도 YoY 성장률"""
    years = sorted(df["연도"].unique())
    if len(years) < 2:
        return pd.DataFrame(columns=["지역", "YoY_%"])
    cur_year, prev_year = years[-1], years[-2]
    cur = df[df["연도"] == cur_year].groupby("지역")["판매량_m3"].sum()
    prev = df[df["연도"] == prev_year].groupby("지역")["판매량_m3"].sum()
    yoy = ((cur - prev) / prev * 100).round(1).reset_index()
    yoy.columns = ["지역", "YoY_%"]
    return yoy.sort_values("YoY_%")


# ──────────────────────────────────────────
# [섹션 D] 차트 함수
# ──────────────────────────────────────────
def fig_region_yoy_bar(df):
    yoy = calc_yoy_by_region(df)
    colors = ["#EF553B" if v < 0 else "#3366CC" for v in yoy["YoY_%"]]
    fig = go.Figure(go.Bar(
        x=yoy["YoY_%"], y=yoy["지역"], orientation="h",
        marker_color=colors,
        text=yoy["YoY_%"].astype(str) + "%", textposition="outside",
    ))
    fig.add_vline(x=0, line_color="gray", line_width=1)
    return _layout(fig, "지역별 전년 대비 판매량 증감률 (%)", height=480)


def fig_forecast_mini(series, forecast, lower, upper):
    """홈 탭용 미니 예측 차트"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.round(1),
        name="실적 단가", line=dict(color="#3366CC", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=list(upper.index) + list(lower.index[::-1]),
        y=list(upper.round(1)) + list(lower.round(1)[::-1]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="80% 신뢰구간", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.round(1),
        name="예측 단가", line=dict(color="#FF7F0E", width=2.5, dash="dash"),
    ))
    fig.add_vline(x=str(series.index[-1]), line_dash="dot", line_color="#aaaaaa",
                  annotation_text="예측 시작", annotation_position="top left")
    return _layout(fig, "도시가스 단가 예측 (향후 12개월)", height=480)


def fig_region_line(df, regions):
    agg = df[df["지역"].isin(regions)].groupby(["연월", "지역"])["판매량_m3"].sum().reset_index()
    fig = px.line(agg, x="연월", y="판매량_m3", color="지역",
                  labels={"판매량_m3": "판매량 (m³)", "연월": "연월"})
    fig.update_xaxes(rangeslider_visible=True)
    return _layout(fig, "지역별 월별 판매량 추이", height=450)


def fig_region_grouped_bar(df):
    agg = df.groupby(["지역", "연도"])["판매량_m3"].sum().reset_index()
    order = (agg[agg["연도"] == agg["연도"].max()]
             .sort_values("판매량_m3", ascending=False)["지역"].tolist())
    fig = px.bar(agg, x="지역", y="판매량_m3", color="연도", barmode="group",
                 category_orders={"지역": order},
                 labels={"판매량_m3": "판매량 (m³)"})
    fig.update_xaxes(tickangle=-30)
    return _layout(fig, "연도별 지역 판매량 비교", height=420)


def fig_heatmap_region_category(df_region):
    pivot = df_region.pivot_table(
        index="지역", columns="고객_카테고리", values="총_판매량_m3", aggfunc="sum"
    ).fillna(0)
    fig = px.imshow(pivot, text_auto=".2s", aspect="auto",
                    color_continuous_scale="Blues",
                    labels={"color": "판매량 (m³)"})
    return _layout(fig, "지역 × 카테고리 판매량 히트맵", height=500)


def fig_region_bubble(df):
    yearly = df.groupby(["지역", "연도"]).agg(
        판매량=("판매량_m3", "sum"), 고객수=("고객수", "sum")
    ).reset_index()
    years = sorted(yearly["연도"].unique())
    if len(years) < 2:
        return go.Figure()
    cur = yearly[yearly["연도"] == years[-1]].set_index("지역")
    prev = yearly[yearly["연도"] == years[-2]].set_index("지역")
    bubble = cur.copy()
    bubble["YoY_%"] = ((cur["판매량"] - prev["판매량"]) / prev["판매량"] * 100).round(1)
    bubble = bubble.reset_index()
    fig = px.scatter(bubble, x="판매량", y="YoY_%", size="고객수", color="지역",
                     hover_name="지역", size_max=45,
                     labels={"판매량": "총 판매량 (m³)", "YoY_%": "YoY 성장률 (%)"},
                     text="지역")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_traces(textposition="top center")
    return _layout(fig, "지역별 판매량 vs YoY 성장률 (버블=고객수)", height=450)


def fig_full_forecast(series, forecast, lower, upper):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.round(2),
        name="실적 단가", line=dict(color="#3366CC", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=list(upper.index) + list(lower.index[::-1]),
        y=list(upper.round(2)) + list(lower.round(2)[::-1]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="80% 신뢰구간", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.round(2),
        name="예측 단가", line=dict(color="#FF7F0E", width=2.5, dash="dash"),
        mode="lines+markers",
    ))
    fig.add_vline(x=str(series.index[-1]), line_dash="dot", line_color="#aaaaaa",
                  annotation_text="예측 시작", annotation_position="top left")
    fig.update_xaxes(rangeslider_visible=True)
    return _layout(fig, "전국 도시가스 단가 예측 (향후 12개월)", height=460)


def fig_region_price_line(df, regions):
    agg = df[df["지역"].isin(regions)].groupby(["연월", "지역"]).agg(
        판매액=("판매액_만원", "sum"), 판매량=("판매량_m3", "sum")
    ).reset_index()
    agg["단가"] = (agg["판매액"] / agg["판매량"]).round(2)
    fig = px.line(agg, x="연월", y="단가", color="지역",
                  labels={"단가": "단가 (만원/m³)", "연월": "연월"})
    return _layout(fig, "지역별 단가 추이", height=400)


def fig_residual(resid):
    r = resid.astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.index, y=r.round(4),
                             mode="lines", name="잔차", line=dict(color="#3366CC", width=1.2)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return _layout(fig, "모델 잔차 시계열", height=300)


def fig_residual_hist(resid):
    r = resid.astype(float)
    fig = px.histogram(x=r, nbins=20, labels={"x": "잔차"},
                       color_discrete_sequence=["#3366CC"])
    return _layout(fig, "잔차 분포 (정규분포에 가까울수록 모델 적합)", height=300)


# ──────────────────────────────────────────
# [섹션 E] 사이드바
# ──────────────────────────────────────────
def render_sidebar(data):
    st.sidebar.header("필터")

    all_years = sorted(data["monthly"]["연도"].unique().tolist())
    selected_years = st.sidebar.multiselect("연도 (Tab 1·2 적용)", all_years, default=all_years)

    selected_cats = st.sidebar.multiselect(
        "고객 카테고리 (Tab 2 적용)", CATEGORY_ORDER, default=CATEGORY_ORDER
    )

    st.sidebar.divider()
    st.sidebar.subheader("요금 예측 설정 (Tab 3)")
    all_regions = sorted(data["monthly"]["지역"].unique().tolist())
    selected_pred_regions = st.sidebar.multiselect(
        "단가 비교 지역", all_regions, default=all_regions[:4]
    )

    return {
        "years": selected_years,
        "categories": selected_cats,
        "pred_regions": selected_pred_regions,
    }


# ──────────────────────────────────────────
# [섹션 F] 탭 렌더링
# ──────────────────────────────────────────
def render_tab_home(data, filters, series, forecast, lower, upper):
    df = filter_monthly(data["monthly"], filters["years"], filters["categories"])

    # KPI
    total_vol = df["판매량_m3"].sum()
    total_rev = df["판매액_만원"].sum()
    cur_price = series.iloc[-1]
    pred_price = forecast.mean()
    price_delta = pred_price - cur_price

    years = sorted(df["연도"].unique())
    if len(years) >= 2:
        cur_y = df[df["연도"] == years[-1]]["판매량_m3"].sum()
        prev_y = df[df["연도"] == years[-2]]["판매량_m3"].sum()
        yoy_pct = (cur_y - prev_y) / prev_y * 100
        yoy_label = f"{yoy_pct:+.1f}%"
    else:
        yoy_label = "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 판매량", f"{total_vol/1e6:.1f}백만 m³")
    c2.metric("전년 대비 증감", yoy_label)
    c3.metric("현재 평균 단가", f"{cur_price:.1f} 만원/m³")
    c4.metric("12개월 예측 단가", f"{pred_price:.1f} 만원/m³",
              delta=f"{price_delta:+.1f} 만원/m³")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(fig_region_yoy_bar(df), use_container_width=True)
    with col_r:
        st.plotly_chart(fig_forecast_mini(series, forecast, lower, upper), use_container_width=True)


def render_tab_region(data, filters):
    df = filter_monthly(data["monthly"], filters["years"], filters["categories"])

    all_regions = sorted(df["지역"].unique().tolist())
    top5 = (df.groupby("지역")["판매량_m3"].sum()
            .nlargest(5).index.tolist())
    selected_regions = st.multiselect("지역 선택", all_regions, default=top5, key="region_select")

    if selected_regions:
        st.plotly_chart(fig_region_line(df, selected_regions), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_region_grouped_bar(df), use_container_width=True)
    with col2:
        st.plotly_chart(fig_region_bubble(df), use_container_width=True)

    st.plotly_chart(fig_heatmap_region_category(data["region"]), use_container_width=True)


def render_tab_forecast(data, filters, series, forecast, lower, upper, resid):
    st.plotly_chart(fig_full_forecast(series, forecast, lower, upper), use_container_width=True)

    # 예측 수치 테이블
    with st.expander("예측 수치 보기"):
        pred_df = pd.DataFrame({
            "연월": forecast.index.strftime("%Y-%m"),
            "예측 단가": forecast.round(2).values,
            "하한 (80% CI)": lower.round(2).values,
            "상한 (80% CI)": upper.round(2).values,
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

    st.divider()

    if filters["pred_regions"]:
        st.plotly_chart(
            fig_region_price_line(data["monthly"], filters["pred_regions"]),
            use_container_width=True,
        )

    st.divider()
    st.subheader("모델 잔차 진단")
    st.caption("잔차가 0 주변에 고르게 분포하고 히스토그램이 정규분포에 가까울수록 예측 모델이 적합합니다.")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_residual(resid), use_container_width=True)
    with col2:
        st.plotly_chart(fig_residual_hist(resid), use_container_width=True)


# ──────────────────────────────────────────
# [섹션 F-4] AI 분석 탭
# ──────────────────────────────────────────
def render_tab_ai():
    st.markdown("""
    <style>
    .ai-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .ai-card h2 { font-size: 2rem; margin-bottom: 10px; }
    .ai-card p  { font-size: 1.1rem; opacity: 0.9; margin-bottom: 24px; }
    .open-btn {
        display: inline-block;
        background: white;
        color: #764ba2;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 14px 36px;
        border-radius: 50px;
        text-decoration: none;
        transition: transform 0.2s;
    }
    .open-btn:hover { transform: scale(1.05); }
    </style>
    <div class="ai-card">
        <h2>🤖 AI 데이터 분석</h2>
        <p>도시가스 판매 데이터에 대해 자유롭게 질문하세요.<br>
        지역별 트렌드, 카테고리 비교, 단가 분석 등 무엇이든 물어보세요.</p>
        <a class="open-btn" href="/ai_chat" target="_blank">✨ AI 채팅 새 탭으로 열기</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 이런 질문을 해보세요")
    examples = [
        ("📍", "가장 판매량이 많은 지역은 어디인가요?"),
        ("📈", "최근 3년간 판매량 트렌드를 알려주세요"),
        ("🏠", "주택용과 산업용 판매량을 비교해주세요"),
        ("💰", "단가가 가장 높은 카테고리는?"),
        ("🚀", "전년 대비 성장률이 가장 높은 지역은?"),
        ("📅", "계절별 판매량 패턴을 분석해주세요"),
    ]
    cols = st.columns(3)
    for i, (icon, q) in enumerate(examples):
        with cols[i % 3]:
            st.markdown(
                f'<a href="/ai_chat" target="_blank" style="text-decoration:none;">'
                f'<div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px; '
                f'margin:6px 0; cursor:pointer; background:#fafafa;">'
                f'<span style="font-size:1.3rem">{icon}</span> '
                f'<span style="font-size:0.9rem; color:#333;">{q}</span>'
                f'</div></a>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────
# [섹션 G] 메인
# ──────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매 분석 대시보드")
    st.caption("핵심 목표 ① 지역별 판매 추이  |  ② 도시가스 요금 예측")

    with st.spinner("데이터 로딩 중..."):
        data = load_all_data()

    filters = render_sidebar(data)

    # 예측 사전 계산 (전체 데이터 기준, 필터 무관)
    series = build_price_series(data["monthly"].to_json())
    forecast, lower, upper, resid = run_forecast(
        data["monthly"].to_json(), horizon=12
    )

    tab1, tab2, tab3, tab4 = st.tabs(["핵심 요약", "지역별 판매 추이", "요금 예측", "🤖 AI 분석"])

    with tab1:
        render_tab_home(data, filters, series, forecast, lower, upper)
    with tab2:
        render_tab_region(data, filters)
    with tab3:
        render_tab_forecast(data, filters, series, forecast, lower, upper, resid)
    with tab4:
        render_tab_ai()


main()
