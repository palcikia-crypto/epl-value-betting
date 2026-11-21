"""Графики для отчёта и дашборда."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style(ax, title):
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)


def make_charts(out_dir, res_test, res_baseline, wf_rows, rel, rel_t, model_name):
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    m = [p["bin"] for p in rel_t if p["count"] > 0]
    c = [p["mean_conf"] for p in rel_t if p["count"] > 0]
    a = [p["accuracy"] for p in rel_t if p["count"] > 0]
    ax.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")
    ax.plot(c, a, "o-", label=f"{model_name} (Test)")
    _style(ax, "Reliability Diagram (Test, 2024/25)")
    ax.set_xlabel("Уверенность модели"); ax.set_ylabel("Фактическая точность")
    ax.legend(); save(fig, "reliability.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    m = [p["bin"] for p in rel if p["count"] > 0]
    c = [p["mean_conf"] for p in rel if p["count"] > 0]
    a = [p["accuracy"] for p in rel if p["count"] > 0]
    ax.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")
    ax.plot(c, a, "o-", label=f"{model_name} (Val)")
    _style(ax, "Reliability Diagram (Validation, 2023/24)")
    ax.set_xlabel("Уверенность модели"); ax.set_ylabel("Фактическая точность")
    ax.legend(); save(fig, "reliability_val.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res_test["curve"], label="Value betting (модель)")
    bl = res_baseline.get("curve")
    if bl:
        ax.plot(bl, label="Baseline: всегда на фаворита", alpha=0.8)
    ax.axhline(10000, color="gray", ls=":")
    _style(ax, "Кривая банка — Test 2024/25 (старт 10 000)")
    ax.set_xlabel("Ставка №"); ax.set_ylabel("Банк")
    ax.legend(); save(fig, "bank_curve.png")

    if wf_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [r["test"] for r in wf_rows]
        ax.bar(xs, [r["roi"] * 100 for r in wf_rows], color="#4c72b0")
        ax.axhline(0, color="black", lw=1)
        _style(ax, "Walk-Forward: ROI по окнам")
        ax.set_ylabel("ROI, %"); ax.set_xlabel("Тест-сезон")
        save(fig, "walkforward.png")
