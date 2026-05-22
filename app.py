"""
Multivariate Outlier Detection Platform
Run: streamlit run outlier_app_v2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import shapiro, levene
import warnings
warnings.filterwarnings("ignore")

# matplotlib font stabilization
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'

# ═══════════════════════════════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Multivariate Outlier Detection Platform",
    page_icon="🔬",
    layout="wide",
)

# ── 글로벌 CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* 섹션 헤더 */
.section-header {
    background: linear-gradient(90deg, #1A3A5C 0%, #2C5F8A 100%);
    color: white;
    padding: 11px 22px;
    border-radius: 8px;
    margin: 28px 0 16px 0;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* 탐지 요약 카드 */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.08);
    border-left: 6px solid;
    margin-bottom: 10px;
    min-height: 140px;
}
.card-mz  { border-color: #E74C3C; }
.card-euc { border-color: #8E44AD; }
.card-cos { border-color: #E67E22; }

/* 설명 박스 */
.info-box {
    background: #F0F6FF;
    border: 1px solid #C6DEFF;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0 16px 0;
    font-size: 14px;
    color: #2C3E50;
    line-height: 1.8;
}
.info-box-warn {
    background: #FFF8EE;
    border: 1px solid #FFD9A0;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0 16px 0;
    font-size: 14px;
    color: #7D4E00;
    line-height: 1.8;
}

/* 방법 설명 카드 */
.method-card {
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
    font-size: 13.5px;
    line-height: 1.75;
}
.method-mz  { background:#FEF0EF; border-left:4px solid #E74C3C; }
.method-euc { background:#F5EEF8; border-left:4px solid #8E44AD; }
.method-cos { background:#FEF9EE; border-left:4px solid #E67E22; }

/* 엑셀 예시 테이블 */
.excel-table {
    border-collapse: collapse;
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    width: 100%;
    margin-top: 10px;
}
.excel-table th, .excel-table td {
    border: 1px solid #C5D5E8;
    padding: 7px 14px;
    text-align: center;
}
.excel-table th {
    background: #D6E8FA;
    color: #1A3A5C;
    font-weight: 600;
}
.excel-table td:first-child {
    background: #EAF3FB;
    font-weight: 600;
    color: #1A3A5C;
}
.excel-table td { background: #FAFCFF; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 분석 함수
# ═══════════════════════════════════════════════════════════════════
def modified_z_score(df, threshold):
    x      = df.values
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
        row["Levene p"]  = round(p, 4)
        row["Equal Var"] = p >= alpha
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variable")


# ── 시각화 함수들 ─────────────────────────────────────────────────
def make_rank_fig(sample_names, mz_scores, mz_mask, euc_scores, euc_mask,
                  cos_scores, cos_mask, mz_thresh):
    COLOR_OUT   = "#E74C3C"
    COLOR_NRM   = "#5DADE2"
    COLOR_TLINE = "#F39C12"
    n = len(sample_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, n * 0.5 + 2)))
    fig.patch.set_facecolor("#F8FAFC")

    configs = [
        ("① Modified Z-Score",  mz_scores,  mz_mask,  mz_thresh),
        ("② Euclidean Distance", euc_scores, euc_mask, None),
        ("③ Cosine Angular",     cos_scores, cos_mask, None),
    ]

    for ax, (title, scores, mask, thresh) in zip(axes, configs):
        ax.set_facecolor("#F8FAFC")
        order  = np.argsort(scores)[::-1]
        labels = [sample_names[i] for i in order]
        vals   = scores[order]
        bar_colors = [COLOR_OUT if mask[i] else COLOR_NRM for i in order]

        bars = ax.barh(range(n), vals[::-1], color=bar_colors[::-1],
                       edgecolor="white", linewidth=0.6, height=0.62)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels[::-1], fontsize=9.5, fontfamily="monospace")
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="#1A3A5C", pad=12)
        ax.set_xlabel("Score", fontsize=10, color="#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=0)

        for i, v in enumerate(vals[::-1]):
            ax.text(v + max(vals) * 0.012, i, f"{v:.4f}",
                    va="center", ha="left", fontsize=8.5, color="#333")

        if thresh is not None:
            ax.axvline(thresh, color=COLOR_TLINE, linewidth=2,
                       linestyle="--", zorder=5)
            ax.text(thresh + max(vals)*0.01, n - 0.5,
                    f"threshold\n= {thresh}", color=COLOR_TLINE,
                    fontsize=8, va="top")

        patches = [mpatches.Patch(color=COLOR_OUT, label="Outlier"),
                   mpatches.Patch(color=COLOR_NRM,  label="Normal")]
        ax.legend(handles=patches, fontsize=8.5, loc="lower right",
                  framealpha=0.85)
        ax.set_xlim(0, max(vals) * 1.22)

    fig.suptitle("Outlier Score Ranking Comparison\n(Higher position = higher outlier possibility)",
                 fontsize=13, fontweight="bold", color="#1A3A5C", y=1.01)
    plt.tight_layout()
    return fig


def make_cv_fig(cv_table):
    cols   = cv_table.columns.tolist()
    vars_  = cv_table.index.tolist()
    x      = np.arange(len(vars_))
    n_cols = len(cols)
    w      = 0.18
    colors = ["#95A5A6", "#E74C3C", "#8E44AD", "#E67E22"]

    fig, ax = plt.subplots(figsize=(max(9, len(vars_) * 1.4), 5.5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for i, (col, c) in enumerate(zip(cols, colors)):
        offset = (i - (n_cols - 1) / 2) * w
        rects = ax.bar(x + offset, cv_table[col], width=w, label=col,
                       color=c, alpha=0.88, edgecolor="white", linewidth=0.6)
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                    color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(vars_, fontsize=11, fontweight="500")
    ax.set_ylabel("CV (%)", fontsize=12)
    ax.set_title("CV Comparison Before/After Outlier Removal", fontsize=13,
                 fontweight="bold", color="#1A3A5C", pad=12)
    ax.legend(fontsize=9.5, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_ylim(0, cv_table.values.max() * 1.18)
    plt.tight_layout()
    return fig


def make_pca_fig(df, sample_names, mz_mask, euc_mask, cos_mask):
    """2D + 3D PCA 나란히 표시. 각 방법의 이상치 색으로 구분."""
    x_scaled = MinMaxScaler().fit_transform(df.values)
    n = len(sample_names)

    # ── 색상: 3개 방법 중 하나라도 이상치면 빨강, 아니면 파랑 ──
    any_outlier = mz_mask | euc_mask | cos_mask

    def _color(i):
        if mz_mask[i] and euc_mask[i] and cos_mask[i]: return "#C0392B"  # 전부
        if mz_mask[i]:  return "#E74C3C"
        if euc_mask[i]: return "#8E44AD"
        if cos_mask[i]: return "#E67E22"
        return "#2980B9"

    pt_colors = [_color(i) for i in range(n)]

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#F8FAFC")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ────────────────────────────────────────────
    # 2D PCA
    # ────────────────────────────────────────────
    pca2 = PCA(n_components=min(2, df.shape[1]))
    xp2  = pca2.fit_transform(x_scaled)
    var2 = pca2.explained_variance_ratio_ * 100

    ax2 = fig.add_subplot(gs[0])
    ax2.set_facecolor("#F8FAFC")

    for i in range(n):
        ax2.scatter(xp2[i, 0], xp2[i, 1], color=pt_colors[i],
                    s=110, zorder=4, edgecolors="white", linewidths=1.2)
        ax2.annotate(sample_names[i],
                     xy=(xp2[i, 0], xp2[i, 1]),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=8.5, color="#1A3A5C",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               ec="#C5D5E8", alpha=0.75, lw=0.7))

    ax2.set_xlabel(f"PC 1  ({var2[0]:.1f}% Explained)", fontsize=11, color="#444")
    ax2.set_ylabel(f"PC 2  ({var2[1]:.1f}% Explained)" if len(var2) > 1 else "PC 2",
                   fontsize=11, color="#444")
    ax2.set_title("PCA — 2D Projection", fontsize=13, fontweight="bold",
                  color="#1A3A5C", pad=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, linestyle="--", alpha=0.4, color="#AABBCC")
    ax2.tick_params(axis="both", length=0)

    # ── 여백 확보 ──
    xpad = (xp2[:, 0].max() - xp2[:, 0].min()) * 0.18 or 0.1
    ypad = (xp2[:, 1].max() - xp2[:, 1].min()) * 0.18 or 0.1
    ax2.set_xlim(xp2[:, 0].min() - xpad, xp2[:, 0].max() + xpad)
    ax2.set_ylim(xp2[:, 1].min() - ypad, xp2[:, 1].max() + ypad)

    # ────────────────────────────────────────────
    # 3D PCA
    # ────────────────────────────────────────────
    n_comp3 = min(3, df.shape[1])
    pca3    = PCA(n_components=n_comp3)
    xp3     = pca3.fit_transform(x_scaled)
    var3    = pca3.explained_variance_ratio_ * 100

    ax3 = fig.add_subplot(gs[1], projection="3d")
    ax3.set_facecolor("#F8FAFC")
    ax3.patch.set_alpha(0)

    for i in range(n):
        z_val = xp3[i, 2] if n_comp3 >= 3 else 0
        ax3.scatter(xp3[i, 0], xp3[i, 1], z_val,
                    color=pt_colors[i], s=110, zorder=4,
                    edgecolors="white", linewidths=1.0)
        ax3.text(xp3[i, 0] + 0.01, xp3[i, 1] + 0.01,
                 z_val + 0.01,
                 sample_names[i], fontsize=8, color="#1A3A5C",
                 fontfamily="monospace")

    ax3.set_xlabel(f"PC1 ({var3[0]:.1f}%)", fontsize=9, color="#444", labelpad=6)
    ax3.set_ylabel(f"PC2 ({var3[1]:.1f}%)" if len(var3) > 1 else "PC2",
                   fontsize=9, color="#444", labelpad=6)
    ax3.set_zlabel(f"PC3 ({var3[2]:.1f}%)" if len(var3) > 2 else "PC3",
                   fontsize=9, color="#444", labelpad=6)
    ax3.set_title("PCA — 3D Projection", fontsize=13, fontweight="bold",
                  color="#1A3A5C", pad=12)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.view_init(elev=22, azim=130)
    ax3.tick_params(axis="both", labelsize=8)

    # ── 축 비율 균일 ──
    for arr, center_fn in [(xp3[:, 0], np.mean), (xp3[:, 1], np.mean)]:
        pass
    all_vals = [xp3[:, 0], xp3[:, 1], xp3[:, 2] if n_comp3 >= 3 else np.zeros(n)]
    centers  = [v.mean() for v in all_vals]
    max_r    = max((v.max() - v.min()) for v in all_vals) / 2 * 1.25 or 0.5
    ax3.set_xlim(centers[0] - max_r, centers[0] + max_r)
    ax3.set_ylim(centers[1] - max_r, centers[1] + max_r)
    ax3.set_zlim(centers[2] - max_r, centers[2] + max_r)

    # ── 공통 범례 ──
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
               markersize=9, label="Modified Z-Score Outlier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8E44AD",
               markersize=9, label="Euclidean Outlier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E67E22",
               markersize=9, label="Cosine Outlier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B",
               markersize=9, label="Detected by All Methods"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980B9",
               markersize=9, label="Normal"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("PCA-based Multivariate Distribution", fontsize=14,
                 fontweight="bold", color="#1A3A5C", y=1.01)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# ── 사이드바 ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    uploaded = st.file_uploader("📂 Excel 파일 업로드 (.xlsx)", type=["xlsx"])
    st.divider()

    st.markdown("**Modified Z-Score 기준값 (threshold)**")
    mz_thresh = st.slider("기본값: 3.5 (높을수록 덜 엄격)", 1.0, 10.0, 3.5, 0.1,
                           label_visibility="collapsed")
    st.caption(f"현재: **{mz_thresh}** — 이 값 초과 시 이상치 판정")

    st.divider()
    st.markdown("**Euclidean / Cosine 이상치 제거 개수 (상위 N개)**")
    remove_n = st.slider("기본값: 2", 1, 10, 2, label_visibility="collapsed")
    st.caption(
        f"현재: 상위 **{remove_n}개** 제거  \n"
        "⚠️ 이 값은 참고용입니다. 스코어 순위표를 확인하고 연구자가 직접 판단하세요."
    )

    st.divider()
    st.markdown("**유의수준 α (통계 검정)**")
    alpha = st.select_slider("α 선택", options=[0.01, 0.05, 0.10], value=0.05,
                              label_visibility="collapsed")
    st.caption(f"현재: α = {alpha}")

    st.divider()
    st.info("💡 파일을 업로드하면 분석이 자동으로 실행됩니다.")


# ═══════════════════════════════════════════════════════════════════
# ── 메인 헤더 & 플랫폼 소개 ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════
st.title("🔬 다변량 이상치 탐지 플랫폼")
st.caption("Multivariate Outlier Detection · Modified Z-Score · Euclidean · Cosine Angular")

# ── 플랫폼 소개 ───────────────────────────────────────────────────
with st.expander("📖 이 플랫폼에 대하여 — 처음 사용하신다면 꼭 읽어주세요", expanded=True):
    st.markdown("""
    <div class="info-box">
    <b>🎯 왜 만들었나요?</b><br>
    식품 연구에서 기기 측정값은 시료의 특성 상 편차가 매우 크게 나타납니다.
    이 때문에 이상치(Outlier)로 판단되는 샘플을 제거한 뒤 분석하는 것이 일반적입니다.<br><br>
    기존 방식은 <b>단변량(Univariate)</b> 접근 — 즉 특정 한 가지 컬럼의 최댓값·최솟값만 기준으로
    이상치를 판단했습니다. 이 방법은 다른 컬럼의 정보를 무시하기 때문에 중요한 패턴을 놓칠 수 있습니다.<br><br>
    본 플랫폼은 <b>모든 측정 컬럼(기기적 특성)을 동시에 고려하는 3가지 다변량(Multivariate)
    이상치 탐지 방법</b>을 제공합니다. 각 방법의 결과를 비교하여 연구자가 보다 근거 있는
    의사결정을 내릴 수 있도록 돕는 것이 목적입니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧪 3가지 탐지 방법 소개")
    st.markdown("""
    <div class="method-card method-mz">
    <b>🔴 ① Modified Z-Score</b><br>
    각 샘플이 전체 데이터의 중앙값(Median)에서 얼마나 떨어져 있는지를 측정합니다.
    점수가 설정한 <b>기준값(threshold, 기본 3.5)을 초과</b>하면 이상치로 판정합니다.
    기준값을 직접 조절할 수 있으므로 엄격도를 연구자가 결정할 수 있습니다.
    </div>
    <div class="method-card method-euc">
    <b>🟣 ② Euclidean Distance (유클리드 거리)</b><br>
    각 샘플과 다른 모든 샘플 사이의 <b>"직선 거리"의 합</b>을 계산합니다.
    다른 샘플들과 거리가 가장 멀리 떨어진 샘플일수록 이상치 가능성이 높습니다.
    마치 반 친구들 사이에서 혼자 멀리 앉아 있는 학생을 찾는 것과 같습니다.
    </div>
    <div class="method-card method-cos">
    <b>🟠 ③ Cosine Angular Distance (코사인 각도 거리)</b><br>
    각 샘플의 <b>"방향성"</b>을 비교합니다. 값의 크기가 아니라 측정값들 간의
    상대적 비율 패턴이 다른 샘플과 크게 다를 때 이상치로 판단합니다.
    값이 크지 않아도 패턴이 다르면 이상치로 감지할 수 있습니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 엑셀 파일 준비 방법")
    st.markdown("""
    <div class="info-box">
    아래 형식과 같이 엑셀 파일을 만들어 업로드하세요.<br>
    • <b>A열 (첫 번째 열)</b>: 샘플명 (예: Sample_1, Sample_2 …)<br>
    • <b>1행 (첫 번째 행)</b>: 측정 특성 이름 (예: Peak, Trough, Final …)<br>
    • 첫 번째 셀(A1)은 비워두거나 라벨을 적어도 됩니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="excel-table">
      <tr>
        <th></th><th>Peak</th><th>Trough</th><th>Final</th><th>Breakdown</th><th>Setback</th>
      </tr>
      <tr><td>Sample_1</td><td>1.456</td><td>1.103</td><td>2.003</td><td>0.353</td><td>0.900</td></tr>
      <tr><td>Sample_2</td><td>1.459</td><td>1.083</td><td>2.003</td><td>0.376</td><td>0.920</td></tr>
      <tr><td>Sample_3</td><td>1.424</td><td>1.048</td><td>1.996</td><td>0.376</td><td>0.948</td></tr>
      <tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
    </table>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 파일 미업로드 상태
# ═══════════════════════════════════════════════════════════════════
if uploaded is None:
    st.divider()
    st.info("👈 **왼쪽 사이드바에서 Excel 파일을 업로드**하면 분석이 시작됩니다.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# 데이터 로드 & 샘플명 추출
# ═══════════════════════════════════════════════════════════════════
df_all = pd.read_excel(uploaded)

# 첫 번째 열이 문자열(샘플명)인지 확인
first_col = df_all.columns[0]
if not pd.api.types.is_numeric_dtype(df_all[first_col]):
    sample_names = df_all[first_col].astype(str).tolist()
    df_raw = df_all.drop(columns=[first_col])
else:
    sample_names = [f"Sample_{i+1}" for i in range(len(df_all))]
    df_raw = df_all.copy()

# 숫자형 컬럼만 사용
df_raw = df_raw.select_dtypes(include=[np.number])
df_raw = df_raw.reset_index(drop=True)

st.markdown('<div class="section-header">📂 원본 데이터</div>', unsafe_allow_html=True)
st.write(f"**{df_raw.shape[0]}개 샘플 × {df_raw.shape[1]}개 변수**")

display_df = df_raw.copy()
display_df.index = sample_names
st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# 분석 실행
# ═══════════════════════════════════════════════════════════════════
mz_s,  mz_m,  df_mz  = modified_z_score(df_raw, mz_thresh)
euc_s, euc_m, df_euc = euclidean_outliers(df_raw, remove_n)
cos_s, cos_m, df_cos = cosine_outliers(df_raw, remove_n)

# 마스크를 numpy array로 확보
mz_m  = np.array(mz_m)
euc_m = np.array(euc_m)
cos_m = np.array(cos_m)


# ═══════════════════════════════════════════════════════════════════
# 탐지 요약 카드
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 탐지 요약</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

for col_widget, title, mask, card_cls, color, note in [
    (c1, "Modified Z-Score",   mz_m,  "card-mz",  "#E74C3C",
     f"기준값 {mz_thresh} 초과 시 이상치"),
    (c2, "Euclidean Distance", euc_m, "card-euc", "#8E44AD",
     f"전체 거리 합 상위 {remove_n}개"),
    (c3, "Cosine Angular",     cos_m, "card-cos", "#E67E22",
     f"각도 거리 합 상위 {remove_n}개"),
]:
    outlier_names = [sample_names[i] for i in np.where(mask)[0]]
    names_str = ", ".join(outlier_names) if outlier_names else "없음"
    col_widget.markdown(f"""
    <div class="stat-card {card_cls}">
        <div style="font-size:13px;color:#888;margin-bottom:6px">{title}</div>
        <div style="font-size:32px;font-weight:700;color:{color};line-height:1.1">
            {mask.sum()}
        </div>
        <div style="font-size:12px;color:#555;margin-top:4px">
            이상치 탐지 / 잔존 <b>{len(df_raw)-mask.sum()}</b>개
        </div>
        <div style="margin-top:10px;font-size:12.5px;color:#333">
            <b>이상치 샘플:</b><br>
            <code style="font-size:12px;color:{color}">{names_str}</code>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#aaa">{note}</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 이상치 스코어 순위표
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 이상치 스코어 순위표</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
각 탭에서 해당 방법의 스코어 <b>내림차순(높은 점수 → 낮은 점수)</b>으로 정렬된 결과를 확인할 수 있습니다.<br>
• <b>테이블 컬럼 헤더를 클릭</b>하면 오름차순 ↔ 내림차순으로 정렬을 바꿀 수 있습니다.<br>
• <span style="color:#C0392B;font-weight:600">⚠️ YES</span> = 해당 방법에서 이상치로 판정된 샘플 &nbsp;|&nbsp;
  <span style="color:#1A7A4A;font-weight:600">✅ NO</span> = 정상<br>
• Euclidean · Cosine 방법은 <b>상위 N개를 이상치로 표시</b>합니다.
  스코어 차이가 크지 않을 경우 N을 줄여서 보수적으로 판단하는 것을 권장합니다.
</div>
""", unsafe_allow_html=True)

score_df = pd.DataFrame({
    "Modified Z-Score"   : mz_s,
    "MZ Outlier"         : mz_m,
    "Euclidean Distance" : euc_s,
    "Euc Outlier"        : euc_m,
    "Cosine Angular"     : cos_s,
    "Cos Outlier"        : cos_m,
}, index=sample_names)

bool_cols  = ["MZ Outlier", "Euc Outlier", "Cos Outlier"]
float_cols = ["Modified Z-Score", "Euclidean Distance", "Cosine Angular"]

def render_score_tab(tab, sort_col):
    with tab:
        df_sorted = score_df.sort_values(sort_col, ascending=False).copy()
        display_df2 = df_sorted.copy()
        for c in bool_cols:
            display_df2[c] = display_df2[c].map(lambda v: "⚠️ YES" if v else "✅ NO")

        st.dataframe(
            display_df2.style
            .format("{:.4f}", subset=float_cols)
            .apply(lambda col: [
                "background:#FDDEDE;color:#C0392B;font-weight:bold;text-align:center"
                if v == "⚠️ YES"
                else "background:#D5F5E3;color:#1A7A4A;font-weight:bold;text-align:center"
                if v == "✅ NO"
                else ""
                for v in col
            ], subset=bool_cols),
            use_container_width=True,
            height=min(500, (len(score_df) + 1) * 38 + 10),
        )

tab1, tab2, tab3 = st.tabs(["🔴 Modified Z-Score 순", "🟣 Euclidean Distance 순", "🟠 Cosine Angular 순"])
render_score_tab(tab1, "Modified Z-Score")
render_score_tab(tab2, "Euclidean Distance")
render_score_tab(tab3, "Cosine Angular")

st.markdown(f"""
<div class="info-box-warn">
🔔 <b>Modified Z-Score</b>: 현재 기준값 = <b>{mz_thresh}</b>
— 이 값을 초과하는 샘플만 이상치로 자동 판정됩니다. (사이드바에서 조절 가능)<br>
🔔 <b>Euclidean / Cosine</b>: 상위 <b>{remove_n}개</b>를 이상치 후보로 표시합니다.
스코어 분포를 보고 연구자가 적절한 제거 기준을 직접 결정하세요.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 순위 비교 시각화
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 방법별 이상치 순위 비교 차트</div>', unsafe_allow_html=True)
st.caption("막대가 길수록, 위에 위치할수록 이상치 가능성이 높습니다. 빨간 막대 = 이상치 판정.")
fig_rank = make_rank_fig(sample_names, mz_s, mz_m, euc_s, euc_m, cos_s, cos_m, mz_thresh)
st.pyplot(fig_rank, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PCA 시각화
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔭 PCA 다변량 분포 시각화</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
PCA(주성분 분석)는 여러 측정값을 2~3개의 축으로 압축하여 샘플의 전체적인 분포를 시각적으로 보여줍니다.<br>
• 다른 샘플들과 <b>멀리 떨어져 있거나 방향이 크게 다른 점</b>이 이상치 가능성이 높습니다.<br>
• 점의 색상은 어떤 방법에서 이상치로 탐지되었는지를 나타냅니다.
</div>
""", unsafe_allow_html=True)

if df_raw.shape[1] >= 2:
    fig_pca = make_pca_fig(df_raw, sample_names, mz_m, euc_m, cos_m)
    st.pyplot(fig_pca, use_container_width=True)
else:
    st.warning("PCA 시각화는 변수가 2개 이상일 때 가능합니다.")


# ═══════════════════════════════════════════════════════════════════
# CV 비교
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📉 CV(변동계수) 비교 — 이상치 제거 전/후</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<b>CV(Coefficient of Variation, 변동계수)</b>는 측정값의 분산 정도를 나타냅니다.
CV가 낮을수록 데이터가 안정적입니다.<br>
• <b>Δ 값이 음수(▼ 파란색)</b>: 이상치 제거 후 CV가 감소 → 데이터 안정성 <b>개선</b><br>
• <b>Δ 값이 양수(▲ 빨간색)</b>: 이상치 제거 후 CV가 증가 → 오히려 분산 <b>증가</b>
</div>
""", unsafe_allow_html=True)

# ── 안전하게 인덱스 재설정 후 CV 계산 ───────────────────────────
df_mz_cv  = df_mz.reset_index(drop=True)
df_euc_cv = df_euc.reset_index(drop=True)
df_cos_cv = df_cos.reset_index(drop=True)

cv_table = pd.DataFrame({
    "Original (%)":   calc_cv(df_raw),
    "Modified Z (%)": calc_cv(df_mz_cv),
    "Euclidean (%)":  calc_cv(df_euc_cv),
    "Cosine (%)":     calc_cv(df_cos_cv),
})

for col_name in ["Modified Z (%)", "Euclidean (%)", "Cosine (%)"]:
    dcol = col_name.replace("(%)", "Δ(%)")
    cv_table[dcol] = (cv_table[col_name] - cv_table["Original (%)"]).round(4)

delta_cols = [c for c in cv_table.columns if "Δ" in c]
score_cols = [c for c in cv_table.columns if "Δ" not in c]

def _delta_style(v):
    if isinstance(v, (int, float)) and v < 0:
        return "color:#1A7A4A;font-weight:bold"
    if isinstance(v, (int, float)) and v > 0:
        return "color:#C0392B;font-weight:bold"
    return ""

st.dataframe(
    cv_table.style
    .map(_delta_style, subset=delta_cols)
    .format("{:.4f}", subset=score_cols)
    .format(lambda v: f"▼ {v:.4f}" if v < 0 else (f"▲ {v:.4f}" if v > 0 else "—"),
            subset=delta_cols),
    use_container_width=True,
)
st.caption("Δ = 이상치 제거 후 CV − 원본 CV")

fig_cv = make_cv_fig(cv_table[score_cols])
st.pyplot(fig_cv, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# 통계 검정
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🧪 통계 가정 검정 — 정규성 & 등분산성</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="info-box">
이상치 제거 전/후 각 변수가 통계 분석의 기본 가정을 만족하는지 확인합니다.<br>
• <b>Shapiro-Wilk 검정 (정규성)</b>: p ≥ {alpha} → <span style="color:#1A7A4A;font-weight:700">✅ T (정규 분포 가정 충족)</span>
  / p &lt; {alpha} → <span style="color:#C0392B;font-weight:700">❌ F (위반)</span><br>
• <b>Levene 검정 (등분산성)</b>: 이상치 제거 전/후 그룹 간 분산이 동일한지 검정합니다.<br>
• 결과가 T가 될수록 모수 통계 방법(t-test, ANOVA 등) 적용이 적합합니다.
</div>
""", unsafe_allow_html=True)

df_dict_stat = {
    "Original"  : df_raw,
    "Modified Z": df_mz_cv,
    "Euclidean" : df_euc_cv,
    "Cosine"    : df_cos_cv,
}
stat_df = run_stat_tests(df_dict_stat, alpha)

tf_cols = [c for c in stat_df.columns if "Normal" in c or "Equal Var" in c]
p_cols  = [c for c in stat_df.columns if c not in tf_cols]

def _tf_style(v):
    if v is True  or v == True:
        return "background:#D5F5E3;color:#1A7A4A;font-weight:700;text-align:center"
    if v is False or v == False:
        return "background:#FDDEDE;color:#C0392B;font-weight:700;text-align:center"
    return "text-align:center"

st.dataframe(
    stat_df.style
    .map(_tf_style, subset=tf_cols)
    .format("{:.4f}", subset=p_cols)
    .format(lambda v: "✅ T" if v else "❌ F", subset=tf_cols),
    use_container_width=True,
)
st.caption(f"판단 기준: p ≥ {alpha} → T (가정 충족) | p < {alpha} → F (가정 위반)")


# ═══════════════════════════════════════════════════════════════════
# 정제 데이터 다운로드
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💾 정제 데이터 다운로드</div>', unsafe_allow_html=True)
st.caption("각 방법으로 이상치를 제거한 데이터를 CSV 파일로 다운로드할 수 있습니다.")

d1, d2, d3 = st.columns(3)
for col_w, name, df_clean, mask_used in [
    (d1, "Modified_Z",  df_mz,     mz_m),
    (d2, "Euclidean",   df_euc,    euc_m),
    (d3, "Cosine",      df_cos,    cos_m),
]:
    kept_names = [sample_names[i] for i in range(len(df_raw)) if not mask_used[i]]
    out_df = df_clean.copy()
    out_df.index = kept_names
    buf = out_df.to_csv(index=True).encode("utf-8-sig")
    col_w.download_button(
        label=f"⬇️ {name} — 제거 후 데이터\n({len(kept_names)}개 샘플)",
        data=buf,
        file_name=f"clean_{name}.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Multivariate Outlier Detection Platform · Built with Streamlit · 🔬 Food Science Lab")
