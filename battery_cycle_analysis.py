"""バッテリ サイクル試験データ解析デモ：充放電データ → 劣化解析 → 合否レポート。

リチウムイオンセルの充放電サイクル試験データを読み込み、サイクルごとの
放電容量・容量維持率・クーロン効率・エネルギー効率・内部抵抗(DCIR)を算出し、
劣化カーブと電圧プロファイルを描画して、規格合否つきレポートを自動生成する。

データ生成 simulate_cycles() は、デモを誰でも動かせるよう実セルの挙動を模擬。
実案件では、ここを充放電試験機(サイクラ)の出力CSV読み込みに差し替えるだけで、
解析・レポート部はそのまま使える。

    python battery_cycle_analysis.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "out"

# numpy>=2.0 は trapz を trapezoid に改名（旧版互換のフォールバック）
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ── セル仕様・試験条件 ───────────────────────────────────────────────────────
CAP_NOM = 3.0          # 公称容量 [Ah]
V_MAX, V_MIN = 4.2, 3.0
C_RATE = 1.0           # 放電レート [C]
N_CYCLES = 300
SOC_PTS = 60           # 1サイクルの電圧サンプル点数
RNG = np.random.default_rng(3)

SPEC = dict(
    retention_min=80.0,    # 最終サイクル 容量維持率 [%] 下限
    ce_min=99.0,           # 平均クーロン効率 [%] 下限
    dcir_max=45.0,         # 内部抵抗 [mΩ] 上限
)


def _ocv(soc: np.ndarray) -> np.ndarray:
    """開回路電圧の近似カーブ（SOC 0→1 で V_MIN→V_MAX、実セル風の非線形）。"""
    return V_MIN + (V_MAX - V_MIN) * (0.15 * soc + 0.85 * soc ** 0.55)


def simulate_cycles():
    """実セルの劣化挙動を模擬。実案件ではサイクラ出力CSVの読み込みに置換。
    返り値: per-cycle の指標配列＋代表サイクルの放電曲線。"""
    soc = np.linspace(1.0, 0.0, SOC_PTS)
    fade = 0.00055      # 1サイクルあたり容量劣化率
    r0 = 0.030          # 初期内部抵抗 [Ω]
    cycles = np.arange(1, N_CYCLES + 1)

    dis_cap = np.zeros(N_CYCLES)
    chg_cap = np.zeros(N_CYCLES)
    dcir = np.zeros(N_CYCLES)
    energy_eff = np.zeros(N_CYCLES)
    curves = {}     # cycle -> (capacity[Ah], voltage[V]) 放電
    for k in cycles:
        retention = (1 - fade) ** (k - 1)
        cap_k = CAP_NOM * retention * (1 + RNG.normal(0, 0.0007))
        r_k = r0 * (1 + 0.0008 * (k - 1))               # 劣化で内部抵抗が増加
        i_dis = C_RATE * cap_k
        v_dis = _ocv(soc) - i_dis * r_k                 # IRドロップ込み放電電圧
        cap_axis = (1 - soc) * cap_k                    # 放出容量 [Ah]

        dis_cap[k - 1] = cap_k
        ce = 0.997 - 0.000004 * (k - 1) + RNG.normal(0, 0.0003)  # クーロン効率
        chg_cap[k - 1] = cap_k / ce
        dcir[k - 1] = r_k * 1000.0                       # [mΩ]
        e_dis = _trapz(v_dis, cap_axis)
        v_chg = _ocv(soc) + i_dis * r_k                  # 充電は逆にIR上乗せ
        e_chg = _trapz(v_chg, cap_axis) / ce
        energy_eff[k - 1] = 100.0 * e_dis / e_chg
        if k in (1, 100, 200, 300):
            curves[k] = (cap_axis, v_dis)
    return dict(cycles=cycles, dis_cap=dis_cap, chg_cap=chg_cap, dcir=dcir,
                ce=100.0 * dis_cap / chg_cap, energy_eff=energy_eff, curves=curves)


def analyse(d: dict) -> dict:
    retention = d["dis_cap"] / d["dis_cap"][0] * 100.0
    final_ret = float(retention[-1])
    fade_per_100 = float((100.0 - final_ret) / (len(d["cycles"]) / 100.0))
    mean_ce = float(np.mean(d["ce"]))
    dcir_end = float(d["dcir"][-1])
    checks = [
        ("最終容量維持率", f"{final_ret:.1f}% (cycle {len(d['cycles'])})",
         f">= {SPEC['retention_min']}%", final_ret >= SPEC["retention_min"]),
        ("平均クーロン効率", f"{mean_ce:.2f}%", f">= {SPEC['ce_min']}%",
         mean_ce >= SPEC["ce_min"]),
        ("内部抵抗(終)", f"{dcir_end:.1f} mΩ", f"<= {SPEC['dcir_max']} mΩ",
         dcir_end <= SPEC["dcir_max"]),
    ]
    return dict(retention=retention, final_ret=final_ret, fade_per_100=fade_per_100,
                mean_ce=mean_ce, dcir_end=dcir_end, checks=checks,
                passed=all(c[3] for c in checks))


def plot_retention(d, a):
    plt.figure(figsize=(7, 4.2))
    plt.plot(d["cycles"], a["retention"], color="#1f77b4")
    plt.axhline(SPEC["retention_min"], color="r", ls="--", lw=0.8,
                label=f"規格 {SPEC['retention_min']:g}%")
    plt.xlabel("サイクル数"); plt.ylabel("容量維持率 [%]")
    plt.title("容量維持率 vs サイクル数"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "capacity_retention.png", dpi=130); plt.close()


def plot_curves(d):
    plt.figure(figsize=(7, 4.2))
    for k, (cap, v) in d["curves"].items():
        plt.plot(cap, v, label=f"cycle {k}")
    plt.xlabel("放電容量 [Ah]"); plt.ylabel("セル電圧 [V]")
    plt.title("放電電圧カーブの推移（劣化）"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "discharge_curves.png", dpi=130); plt.close()


def plot_efficiency(d):
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(d["cycles"], d["ce"], color="#2ca02c", label="クーロン効率")
    ax1.set_xlabel("サイクル数"); ax1.set_ylabel("クーロン効率 [%]", color="#2ca02c")
    ax2 = ax1.twinx()
    ax2.plot(d["cycles"], d["dcir"], color="#d62728", label="内部抵抗")
    ax2.set_ylabel("内部抵抗 DCIR [mΩ]", color="#d62728")
    plt.title("クーロン効率・内部抵抗 vs サイクル数")
    fig.tight_layout(); plt.savefig(OUT / "efficiency_dcir.png", dpi=130); plt.close()


def write_csv(d):
    with open(OUT / "cycle_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "discharge_cap_Ah", "charge_cap_Ah",
                    "coulombic_eff_pct", "energy_eff_pct", "dcir_mohm"])
        for i in range(len(d["cycles"])):
            w.writerow([int(d["cycles"][i]), round(d["dis_cap"][i], 4),
                        round(d["chg_cap"][i], 4), round(d["ce"][i], 3),
                        round(d["energy_eff"][i], 3), round(d["dcir"][i], 2)])


def write_report(d, a):
    verdict = "✅ PASS" if a["passed"] else "❌ FAIL"
    lines = [
        "# バッテリ サイクル試験 解析レポート（自動生成）", "",
        f"**総合判定: {verdict}**", "",
        f"- 解析サイクル数: {len(d['cycles'])}",
        f"- 初期容量: {d['dis_cap'][0]:.3f} Ah → 最終容量: {d['dis_cap'][-1]:.3f} Ah",
        f"- 最終容量維持率: **{a['final_ret']:.1f}%**（劣化 {a['fade_per_100']:.2f}%/100cyc）",
        f"- 平均クーロン効率: {a['mean_ce']:.2f}%",
        f"- 内部抵抗(終): {a['dcir_end']:.1f} mΩ", "",
        "## 規格合否", "", "| 項目 | 測定 | 規格 | 判定 |", "|---|---|---|---|",
    ]
    for name, meas, spec, ok in a["checks"]:
        lines.append(f"| {name} | {meas} | {spec} | {'✅' if ok else '❌'} |")
    lines += [
        "", "## グラフ", "",
        "![容量維持率](capacity_retention.png)", "",
        "![放電カーブ](discharge_curves.png)", "",
        "![効率・内部抵抗](efficiency_dcir.png)", "",
        "---", "*simulate_cycles() を充放電試験機の出力CSV読み込みに差し替えると実データ解析になります。*",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "sans-serif"]
    d = simulate_cycles()
    a = analyse(d)
    plot_retention(d, a); plot_curves(d); plot_efficiency(d)
    write_csv(d); write_report(d, a)
    print(f"総合判定: {'PASS' if a['passed'] else 'FAIL'}")
    print(f"最終容量維持率 {a['final_ret']:.1f}% / 劣化 {a['fade_per_100']:.2f}%/100cyc")
    print(f"平均クーロン効率 {a['mean_ce']:.2f}% / 内部抵抗(終) {a['dcir_end']:.1f} mΩ")
    print(f"出力先: {OUT}")


if __name__ == "__main__":
    main()
