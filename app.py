"""
Outlier Detection Platform — Streamlit App
Run:  streamlit run outlier_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import shapiro, levene
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Outlier Detection Platform",
    page_icon="🔬",
    layout="wide",
)

# ── CSS 스타일 ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 2rem; }

h1 { color: #1A1A2E; letter-spacing: -0.5px; }
h2, h3 { color: #2C3E50; }

/* 카드 스타일 */
.stat-card {
    background: white;
    border-radius: 10px;
    padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 5px solid;
    margin-bottom: 10px;
}
.card-mz    { border-color: #E74C3C; }
.card-euc   { border-color: #9B59B6; }
.card-cos   { border-color: #E67E22; }

.badge-outlier { background:#FDDEDE; color:#C0392B; padding:3px 10px; border-radius:12px; font-weight:600; font-size:12px; }
.badge-normal  { background:#D5F5E3; color:#1A7A4A; padding:3px 10px; border-radius:12px; font-weight:600; font-size:12px; }
.badge-t       { background:#D5F5E3; color:#1A7A4A; padding:2px 8px; border-radius:8px; font-weight:700; font-size:11px; }
.badge-f       { background:#FDDEDE; color:#C0392B; padding:2px 8px; border-radius:8px; font-weight:700; font-size:11px; }

.section-header {
    background: linear-gradient(90deg, #2C3E50 0%, #34495E 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    margin: 24px 0 16px 0;
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 분석 함수
# ═══════════════════════════════════════════════════════════
def modified_z_score(df, threshold):
    x = df.values
    median = np.median(x, axis=0)
    mad    = np.median(np.abs(x - median), axis=0)
    mad[mad == 0] = 1e-10
    mz     = 0.6745 * np.abs(x - median) / mad
    scores = np.max(mz, axis=1)
    mask   = scores > threshold
    return scores, mask, df.loc[~mask].copy()

def euclidean_outliers(df, n):
    x      = MinMaxScaler().fit_transform(df.values)
    dist   = pairwise_distances(x, metric="euclidean")
    scores = dist.sum(axis=1)
    top    = np.argsort(scores)[::-1][:n]
    mask   = np.zeros(len(df), dtype=bool)
    mask[top] = True
    return scores, mask, df.loc[~mask].copy()

def cosine_outliers(df, n):
    x      = MinMaxScaler().fit_transform(df.values)
    sim    = np.clip(cosine_similarity(x), -1, 1)
    scores = np.arccos(sim).sum(axis=1)
    top    = np.argsort(scores)[::-1][:n]
    mask   = np.zeros(len(df), dtype=bool)
    mask[top] = True
    return scores, mask, df.loc[~mask].copy()

def calc_cv(df):
    return (df.std() / df.mean().abs() * 100).round(4)

def run_stat_tests(df_dict, alpha):
    variables = list(df_dict["Original"].columns)
    rows = []
    for var in variables:
        row = {"Variable": var}
        for name, dff in df_dict.items():
            _, p = shapiro(dff[var].dropna())
            row[f"Shapiro p ({name})"] = round(p, 4)
            row[f"Normal ({name})"]    = p >= alpha
        groups = [dff[var].dropna().values for dff in df_dict.values()]
        _, p = levene(*groups)
        row["Levene p"]      = round(p, 4)
        row["Equal Var"]     = p >= alpha
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variable")

def make_rank_fig(mz_scores, mz_mask, euc_scores, euc_mask, cos_scores, cos_mask, mz_thresh):
    COLOR_OUT   = "#E74C3C"
    COLOR_NRM   = "#3498DB"
    COLOR_TLINE = "#F39C12"

    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len(mz_scores)*0.45 + 2)))
    fig.patch.set_facecolor("#FAFAFA")

    configs = [
        ("Modified Z-Score",   mz_scores,  mz_mask,  mz_thresh),
        ("Euclidean Distance",  euc_scores, euc_mask, None),
        ("Cosine Angular",      cos_scores, cos_mask, None),
    ]

    for ax, (title, scores, mask, thresh) in zip(axes, configs):
        ax.set_facecolor("#FAFAFA")
        order  = np.argsort(scores)[::-1]
        labels = [f"S{i+1}" for i in order]
        vals   = scores[order]
        colors = [COLOR_OUT if mask[i] else COLOR_NRM for i in order]

        ax.barh(range(len(vals)), vals[::-1], color=colors[::-1],
                edgecolor="white", linewidth=0.5, height=0.65)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels[::-1], fontsize=10, fontfamily="monospace")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#2C3E50", pad=10)
        ax.set_xlabel("Score", fontsize=10, color="#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, (v, c) in enumerate(zip(vals[::-1], colors[::-1])):
            ax.text(v + max(vals)*0.01, i, f"{v:.3f}",
                    va="center", ha="left", fontsize=8.5, color="#333")

        if thresh is not None:
            ax.axvline(thresh, color=COLOR_TLINE, linewidth=1.8,
                       linestyle="--", label=f"threshold = {thresh}")

        patches = [mpatches.Patch(color=COLOR_OUT, label="이상치"),
                   mpatches.Patch(color=COLOR_NRM,  label="정상")]
        ax.legend(handles=patches, fontsize=8, loc="lower right")
        ax.set_xlim(0, max(vals) * 1.20)

    plt.tight_layout()
    return fig

def make_cv_fig(cv_table):
    methods = cv_table.columns
    x = np.arange(len(cv_table))
    w = 0.2
    colors = ["#95A5A6", "#2ECC71", "#9B59B6", "#E67E22"]

    fig, ax = plt.subplots(figsize=(max(8, len(cv_table)*1.2), 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for i, (col, c) in enumerate(zip(methods, colors)):
        ax.bar(x + i*w, cv_table[col], width=w, label=col,
               color=c, alpha=0.85, edgecolor="white")

    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(cv_table.index, fontsize=11)
    ax.set_ylabel("CV (%)", fontsize=12)
    ax.set_title("변수별 CV 비교 (이상치 제거 전/후)", fontsize=13,
                 fontweight="bold", color="#2C3E50")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════
st.title("🔬 Outlier Detection Platform")
st.caption("Modified Z-Score · Euclidean Distance · Cosine Angular Distance")
st.divider()

# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader("Excel 파일 업로드 (.xlsx)", type=["xlsx"])
    st.divider()
    mz_thresh = st.slider("Modified Z-Score 기준값", 1.0, 10.0, 3.5, 0.1)
    remove_n  = st.slider("Euclidean / Cosine 제거 개수 (상위 N)", 1, 10, 2)
    alpha     = st.select_slider("유의수준 α", options=[0.01, 0.05, 0.10], value=0.05)
    st.divider()
    st.info("💡 파일 업로드 후 분석이 자동 실행됩니다.")

# ── 메인 ────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 사이드바에서 Excel 파일을 업로드해주세요.")
    st.stop()

# 데이터 로드
df_raw = pd.read_excel(uploaded)
df_raw = df_raw.select_dtypes(include=[np.number])
if "Sample" in df_raw.columns:
    df_raw = df_raw.drop(["Sample"], axis=1)

st.markdown('<div class="section-header">📂 원본 데이터</div>', unsafe_allow_html=True)
st.write(f"**{df_raw.shape[0]}개 샘플 × {df_raw.shape[1]}개 변수**")
st.dataframe(df_raw.style.format("{:.4f}"), use_container_width=True)

# ── 분석 실행 ────────────────────────────────────────────
mz_s,  mz_m,  df_mz  = modified_z_score(df_raw, mz_thresh)
euc_s, euc_m, df_euc = euclidean_outliers(df_raw, remove_n)
cos_s, cos_m, df_cos = cosine_outliers(df_raw, remove_n)

# ── 요약 카드 ────────────────────────────────────────────
st.markdown('<div class="section-header">📊 탐지 요약</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

for col, title, mask, scores, card_cls, color in [
    (c1, "Modified Z-Score", mz_m,  mz_s,  "card-mz",  "#E74C3C"),
    (c2, "Euclidean Distance", euc_m, euc_s, "card-euc", "#9B59B6"),
    (c3, "Cosine Angular",  cos_m,  cos_s,  "card-cos",  "#E67E22"),
]:
    outlier_idx = list(np.where(mask)[0])
    col.markdown(f"""
    <div class="stat-card {card_cls}">
        <div style="font-size:13px; color:#888; margin-bottom:4px">{title}</div>
        <div style="font-size:28px; font-weight:700; color:{color}">{mask.sum()}</div>
        <div style="font-size:12px; color:#555; margin-top:4px">이상치 탐지 / 잔존 {len(df_raw)-mask.sum()}개</div>
        <div style="margin-top:8px; font-size:12px; color:#555">
            인덱스: <code>{outlier_idx}</code>
        </div>
    </div>""", unsafe_allow_html=True)

# ── 이상치 스코어 순위표 ─────────────────────────────────
st.markdown('<div class="section-header">📋 이상치 스코어 순위표</div>', unsafe_allow_html=True)

idx_labels = [f"Sample_{i+1}" for i in df_raw.index]
score_df = pd.DataFrame({
    "Modified Z-Score"   : mz_s,
    "MZ Outlier"         : mz_m,
    "Euclidean Distance" : euc_s,
    "Euc Outlier"        : euc_m,
    "Cosine Angular"     : cos_s,
    "Cos Outlier"        : cos_m,
}, index=idx_labels)

tab1, tab2, tab3 = st.tabs(["🔴 Modified Z-Score 순", "🟣 Euclidean 순", "🟠 Cosine 순"])

def render_score_tab(tab, sort_col, bool_cols, float_cols):
    with tab:
        df_sorted = score_df.sort_values(sort_col, ascending=False).copy()

        def _style(row):
            styles = [""] * len(row)
            # outlier bool 열 색칠
            for i, col in enumerate(df_sorted.columns):
                if col in bool_cols:
                    if row[col] is True or row[col] == True:
                        styles[i] = "background:#FDDEDE; color:#C0392B; font-weight:bold; text-align:center"
                    else:
                        styles[i] = "background:#D5F5E3; color:#1A7A4A; font-weight:bold; text-align:center"
            return styles

        display_df = df_sorted.copy()
        for c in bool_cols:
            display_df[c] = display_df[c].map(lambda v: "⚠️ YES" if v else "✅ NO")

        st.dataframe(
            display_df.style
            .format("{:.4f}", subset=float_cols)
            .apply(lambda x: [
                "background:#FDDEDE; color:#C0392B; font-weight:bold; text-align:center" if v == "⚠️ YES"
                else "background:#D5F5E3; color:#1A7A4A; font-weight:bold; text-align:center" if v == "✅ NO"
                else "" for v in x
            ], subset=bool_cols),
            use_container_width=True,
        )

bool_cols  = ["MZ Outlier", "Euc Outlier", "Cos Outlier"]
float_cols = ["Modified Z-Score", "Euclidean Distance", "Cosine Angular"]
render_score_tab(tab1, "Modified Z-Score",   bool_cols, float_cols)
render_score_tab(tab2, "Euclidean Distance",  bool_cols, float_cols)
render_score_tab(tab3, "Cosine Angular",      bool_cols, float_cols)

# Modified Z 기준선 안내
st.info(f"🔔 Modified Z-Score: **{mz_thresh}** 초과 시 이상치로 판정 (현재 설정값)")

# ── 시각화: 순위 비교 ────────────────────────────────────
st.markdown('<div class="section-header">📈 방법별 이상치 순위 비교</div>', unsafe_allow_html=True)
fig_rank = make_rank_fig(mz_s, mz_m, euc_s, euc_m, cos_s, cos_m, mz_thresh)
st.pyplot(fig_rank, use_container_width=True)

# ── CV 비교 ──────────────────────────────────────────────
st.markdown('<div class="section-header">📉 CV(변동계수) 비교 — 이상치 제거 전/후</div>', unsafe_allow_html=True)

cv_table = pd.DataFrame({
    "Original (%)":  calc_cv(df_raw),
    "Modified Z (%)": calc_cv(df_mz),
    "Euclidean (%)":  calc_cv(df_euc),
    "Cosine (%)":     calc_cv(df_cos),
})
for col in ["Modified Z (%)", "Euclidean (%)", "Cosine (%)"]:
    dcol = col.replace("(%)", "Δ(%)")
    cv_table[dcol] = (cv_table[col] - cv_table["Original (%)"]).round(4)

delta_cols = [c for c in cv_table.columns if "Δ" in c]
score_cols = [c for c in cv_table.columns if "Δ" not in c]

def _delta_style(v):
    if isinstance(v, float) and v < 0: return "color:#1A7A4A; font-weight:bold"
    if isinstance(v, float) and v > 0: return "color:#C0392B; font-weight:bold"
    return ""

st.dataframe(
    cv_table.style
    .map(_delta_style, subset=delta_cols)
    .format("{:.4f}", subset=score_cols)
    .format(lambda v: f"▼ {v:.4f}" if v < 0 else (f"▲ {v:.4f}" if v > 0 else "—"), subset=delta_cols),
    use_container_width=True,
)
st.caption("Δ = 제거 후 − 원본 | ▼ 파란색: CV 감소(개선), ▲ 빨간색: CV 증가(악화)")

fig_cv = make_cv_fig(cv_table[score_cols])
st.pyplot(fig_cv, use_container_width=True)

# ── 통계 검정 ────────────────────────────────────────────
st.markdown('<div class="section-header">🧪 통계 검정 — 정규성 & 등분산성</div>', unsafe_allow_html=True)

df_dict = {"Original": df_raw, "Modified Z": df_mz, "Euclidean": df_euc, "Cosine": df_cos}
stat_df = run_stat_tests(df_dict, alpha)

tf_cols = [c for c in stat_df.columns if "Normal" in c or "Equal Var" in c]
p_cols  = [c for c in stat_df.columns if c not in tf_cols]

def _tf_style(v):
    if v is True  or v == True:  return "background:#D5F5E3; color:#1A7A4A; font-weight:700; text-align:center"
    if v is False or v == False: return "background:#FDDEDE; color:#C0392B; font-weight:700; text-align:center"
    return "text-align:center"

st.dataframe(
    stat_df.style
    .map(_tf_style, subset=tf_cols)
    .format("{:.4f}", subset=p_cols)
    .format(lambda v: "✅ T" if v else "❌ F", subset=tf_cols),
    use_container_width=True,
)
st.caption(f"판단 기준: p ≥ {alpha} → T (가정 충족) | p < {alpha} → F (가정 위반)")

# ── 정제 데이터 다운로드 ─────────────────────────────────
st.markdown('<div class="section-header">💾 정제 데이터 다운로드</div>', unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)

for col, name, df_clean in [(d1,"Modified_Z",df_mz),(d2,"Euclidean",df_euc),(d3,"Cosine",df_cos)]:
    buf = df_clean.to_csv(index=True).encode("utf-8-sig")
    col.download_button(
        label=f"⬇️ {name} 제거 후 데이터",
        data=buf,
        file_name=f"clean_{name}.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Outlier Detection Platform · Built with Streamlit")
