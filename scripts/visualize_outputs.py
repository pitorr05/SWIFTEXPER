"""Audit recorded runs and render mechanism figures; never loads an LLM.

Requires numpy and matplotlib. Input files are read-only. Run from the repo root:
python scripts/visualize_outputs.py
"""
import argparse
import csv
import hashlib
from html import escape
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"baseline", "swift", "adaptive", "fixed_first", "native_control", "raw_dp", "rank_dp"}
NAMES = {"baseline": "AR", "swift": "SWIFT", "adaptive": "AdaptiveCache",
         "fixed_first": "Fixed-first", "native_control": "Native control",
         "raw_dp": "Raw DP", "rank_dp": "Rank DP"}
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#CC6677"
PARTS = [("optimization_host_seconds", "Optimization", ORANGE),
         ("embedding_seconds", "Embedding", BLUE),
         ("search_seconds", "Cache search", "#56B4E9"),
         ("cache_update_seconds", "Cache update", "#AA4499")]
REGIONS = ["M down / slower", "M down / faster", "M equal / slower",
           "M equal / faster", "M up / slower", "M up / faster"]
SOURCES = {
    "SWIFT (ICLR 2025)": "https://proceedings.iclr.cc/paper_files/paper/2025/file/d74d002a9154b4cc433a234feb27c5f4-Paper-Conference.pdf",
    "KNN-SSD (Findings EACL 2026)": "https://aclanthology.org/2026.findings-eacl.31/",
    "Sequoia (NeurIPS 2024)": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract.html",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 220, "axes.axisbelow": True})


def heading(fig, title, subtitle):
    fig.suptitle(title, x=.055, y=.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(.055, .943 if fig.get_figheight() >= 8 else .90, subtitle, fontsize=10, color="#4B5563")


def footer(fig, text):
    fig.text(.055, .018, text, fontsize=8, color="#4B5563", va="bottom")


def dump_figure(fig, out, name, report=None):
    fig.savefig(out / (name + ".pdf"))
    fig.savefig(out / (name + ".png"))
    if report is not None:
        report.savefig(fig)
    plt.close(fig)


def canonical_mask(value):
    if value is None:
        return None
    return tuple(tuple(sorted(map(int, part))) for part in value)


def load_runs(root):
    runs_root = root / "_local" / "runs" if (root / "_local" / "runs").is_dir() else root
    groups, audit, files = {}, [], []
    for run in sorted(runs_root.iterdir()):
        if not run.is_dir() or not (run / "results.csv").is_file():
            continue
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as f:
            summary = {(r["dataset"], r["method"]): r for r in csv.DictReader(f)}
        for path in sorted(run.rglob("*.jsonl")):
            method = path.stem.split(".")[-1]
            if method not in METHODS:
                continue
            with path.open(encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            if not records or "choices" not in records[0]:
                continue
            meta = read_json(Path(str(path) + ".metadata.json"))
            manifest = read_json(Path(str(path) + ".manifest.json"))
            if meta["manifest_sha256"] != record_digest(manifest):
                raise ValueError(f"Manifest hash mismatch: {path}")
            config = meta["config"]
            dataset = path.stem.rsplit(".", 1)[0]
            group_key = (run.name, str(path.parent.relative_to(run)), dataset)
            group = groups.setdefault(group_key, {"run": run.name, "dataset": dataset, "methods": {},
                                                  "layers": meta["model_config"]["num_hidden_layers"]})
            group["model"] = config["model_path"]
            group["limit"] = config["max_new_tokens"]
            group["split"] = config.get("research_split", "test")
            group["stage"] = config.get("stage", "diagnose" if "swift_tree_" in run.name else "evaluate")
            ids = [str(r["task_id"]) for r in records]
            if len(set(ids)) != len(ids) or ids != list(map(str, manifest["task_ids"])):
                raise ValueError(f"Task ID order/uniqueness failure: {path}")
            values = []
            for i, row in enumerate(records):
                if record_digest(row["input_token_ids"]) != row["input_sha256"]:
                    raise ValueError(f"Input hash mismatch: {path}, row {i}")
                c, fm = row["choices"][0], row.get("forward_metrics", {})
                seconds, tokens = sum(c["wall_time"]), sum(c["new_tokens"])
                lengths = c["accept_lengths"]
                if seconds <= 0 or tokens != len(row["output_token_ids"]):
                    raise ValueError(f"Invalid timing/token count: {path}, row {i}")
                v = {"index": i + 1, "id": str(row["task_id"]), "time": seconds, "tokens": tokens,
                     "M": float(np.mean(lengths)) if lengths else 1., "input_length": len(row["input_token_ids"]),
                     "opt": fm.get("optimization_host_seconds", 0.), "hit": fm.get("cache_hit"),
                     "steps_total": fm.get("opt_iter_total", 0),
                     "mask": canonical_mask(fm.get("used_layers", fm.get("incumbent_layers"))),
                     "incumbent": canonical_mask(fm.get("incumbent_layers")), "metrics": fm, "row": row}
                # Host intervals are nested in request time, not GPU phase durations.
                if sum(fm.get(k, 0.) for k, _, _ in PARTS) > seconds + 1e-6:
                    raise ValueError(f"Host partition exceeds request wall time: {path}, row {i}")
                values.append(v)
            measured = {"samples": len(values), "tokens": sum(v["tokens"] for v in values),
                        "inference_seconds": sum(v["time"] for v in values),
                        "tokens_per_second": np.mean([v["tokens"] / v["time"] for v in values]),
                        "mean_accepted_length": np.mean([x for r in records for x in r["choices"][0]["accept_lengths"]])}
            original = summary[(dataset, method)]
            for field, value in measured.items():
                if not np.isclose(float(original[field]), value, rtol=1e-8, atol=1e-7):
                    raise ValueError(f"CSV mismatch {path}: {field}: {value} != {original[field]}")
            group["methods"][method] = values
            group.setdefault("summaries", {})[method] = {**original, **measured}
            group.setdefault("metadata", {})[method] = meta
            audit.append({"run": run.name, "dataset": dataset, "method": method,
                          "requests": len(values), "csv_recomputed": True,
                          "python": meta["python"], "model_revision": meta.get("model_revision"),
                          "source_commit": meta["source"]["commit"], "input": str(path.relative_to(root))})
            for item in (path, Path(str(path) + ".metadata.json"), Path(str(path) + ".manifest.json")):
                files.append({"path": str(item.relative_to(root)),
                              "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
    if not groups:
        raise ValueError("No supported run outputs found")
    for g in groups.values():
        ref = g["methods"].get("baseline")
        if ref is None or "swift" not in g["methods"]:
            raise ValueError("AR and SWIFT are required in each comparison")
        for method, values in g["methods"].items():
            if len(values) != len(ref):
                raise ValueError("Paired sample count differs")
            for a, b in zip(values, ref):
                if a["id"] != b["id"] or a["row"]["input_sha256"] != b["row"]["input_sha256"]:
                    raise ValueError(f"Input pairing failed: {g['run']} {method}")
            cm, cr = g["metadata"][method], g["metadata"]["baseline"]
            for key in ("model_path", "max_new_tokens", "dtype", "seed", "data_seed", "temperature", "top_p"):
                if cm["config"].get(key) != cr["config"].get(key):
                    raise ValueError(f"Config pairing failed: {key}")
            if cm.get("model_revision") != cr.get("model_revision") or cm["packages"] != cr["packages"]:
                raise ValueError("Model revision/package mismatch")
        speed = float(g['summaries']['baseline']['tokens_per_second'])
        for method, summary in g['summaries'].items():
            if not np.isclose(float(summary['speedup_vs_baseline']), float(summary['tokens_per_second']) / speed, rtol=1e-8):
                raise ValueError("AR speedup mismatch")
    return list(groups.values()), audit, files


def label(g):
    task = "CNN/DM" if g["dataset"].startswith("cnndm") else "HumanEval" if g["dataset"] == "humaneval" else "GSM8K"
    size = "13B" if "13b" in g["model"].lower() else "7B"
    return f"{task}-{g['limit']} | {size}"


def case_order(g):
    return ("adaptive" not in g["methods"], g["model"].lower().find("13b") >= 0,
            {"cnndm": 0, "humaneval": 1, "gsm8k": 2}.get(g["dataset"].split(".")[0], 3), g["limit"])


def comparisons(groups):
    return [(g, "adaptive" if "adaptive" in g["methods"] else "fixed_first")
            for g in sorted(groups, key=case_order) if "adaptive" in g["methods"] or "fixed_first" in g["methods"]]


def paired(g, method):
    for a, s in zip(g["methods"][method], g["methods"]["swift"]):
        dm = 100 * (a["M"] / s["M"] - 1)
        gain = 100 * (1 - a["time"] / s["time"])
        equal = abs(a["M"] - s["M"]) < 1e-10
        region = ("M equal" if equal else "M up" if dm > 0 else "M down") + (" / faster" if gain > 0 else " / slower")
        yield a, s, dm, gain, region


def export_analysis(groups, out):
    requests, regions, partitions, overview = [], [], [], []
    for g, method in comparisons(groups):
        buckets = defaultdict(list)
        previous, previous_steps = None, 0
        for a, s, dm, gain, region in paired(g, method):
            ham = sum(len(set(x) ^ set(y)) for x, y in zip(a["mask"], previous)) if previous and a["mask"] else 0
            previous = a["mask"]
            baseline = g["methods"]["baseline"][a["index"] - 1]
            r = {"run": g["run"], "case": label(g), "method": method, "request": a["index"],
                 "task_id": a["id"], "region": region, "M": a["M"], "M_swift": s["M"], "delta_M_pct": dm,
                 "latency_seconds": a["time"], "swift_seconds": s["time"], "latency_reduction_pct": gain,
                 "faster_vs_AR": a["time"] < baseline["time"], "faster_vs_SWIFT": gain > 0,
                 "input_tokens": a["input_length"], "cache_hit": a["hit"], "optimization_seconds": a["opt"],
                 "optimization_share_pct": 100 * a["opt"] / a["time"], "opt_iter_total": a["steps_total"],
                 "optimization_iterations": a["steps_total"] - previous_steps,
                 "mask_hamming_to_previous": ham,
                 "output_equal_swift": a["row"]["output_token_ids"] == s["row"]["output_token_ids"]}
            requests.append(r)
            buckets[region].append(r)
            previous_steps = a["steps_total"]
        for region in REGIONS:
            rows = buckets[region]
            r = {"run": g["run"], "case": label(g), "method": method, "region": region,
                 "n": len(rows), "fraction": len(rows) / len(g["methods"][method])}
            for k in ("input_tokens", "optimization_share_pct", "optimization_iterations", "mask_hamming_to_previous", "latency_reduction_pct"):
                for q, tag in ((.25, "q25"), (.5, "median"), (.75, "q75")):
                    r[f"{k}_{tag}"] = float(np.quantile([v[k] for v in rows], q)) if rows else ""
            hits = [v["cache_hit"] for v in rows if v["cache_hit"] is not None]
            r["hit_fraction"] = np.mean(hits) if hits else ""
            regions.append(r)
    for g in groups:
        for method, values in g["methods"].items():
            t = sum(v["time"] for v in values)
            part = {k: sum(v["metrics"].get(k, 0) for v in values) for k, _, _ in PARTS}
            part["other_inference_seconds"] = t - sum(part.values())
            partitions.append({"run": g["run"], "case": label(g), "method": method, "total_seconds": t, **part})
            masks = [v["mask"] for v in values if v["mask"]]
            overview.append({"run": g["run"], "case": label(g), "split": g["split"], "method": method,
                             "n": len(values), "distinct_request_end_masks": len(set(masks)),
                             "last_request_with_optimization": max([v["index"] for v in values if v["opt"] > 0], default=0),
                             "optimization_requests": sum(v["opt"] > 0 for v in values),
                             "total_seconds": t, "macro_tokens_per_second": g["summaries"][method]["tokens_per_second"]})
    save_csv(out / "paired_requests.csv", requests)
    save_csv(out / "region_characteristics.csv", regions)
    save_csv(out / "timing_partitions.csv", partitions)
    save_csv(out / "stream_summary.csv", overview)
    return regions


def timing_figure(groups, out, report):
    cases = comparisons(groups)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, left=.055, right=.98, bottom=.16, top=.87, hspace=.48, wspace=.48)
    for i, (g, method) in enumerate([x for x in cases if x[1] == "adaptive"]):
        ax = fig.add_subplot(gs[0, i])
        vals = g["methods"][method]
        t = sum(v["time"] for v in vals)
        parts = [sum(v["metrics"].get(k, 0) for v in vals) for k, _, _ in PARTS]
        remainder = t - sum(parts)
        ax.pie(parts + [remainder], colors=[c for _, _, c in PARTS] + ["#E5E7EB"],
               startangle=90, counterclock=False, wedgeprops={"width": .28, "edgecolor": "white", "linewidth": .5})
        ax.text(0, .10, f"{t:,.1f} s", ha="center", fontsize=13, fontweight="bold")
        ax.text(0, -.17, "request wall time", ha="center", fontsize=8)
        ax.set_title(label(g), pad=9)
        ax.text(.5, -.08, f"Opt {parts[0]:.2f}s ({100*parts[0]/t:.2f}%)\n"
                f"Embed {parts[1]:.2f}s ({100*parts[1]/t:.2f}%)\n"
                f"Search/update {sum(parts[2:]):.2f}s", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
    for col, (kind, title) in enumerate((("adaptive", "AdaptiveCache runs"), ("fixed_first", "Fixed-first runs"))):
        ax = fig.add_subplot(gs[1, col*2:col*2+2])
        ax.set_position([.16 if col == 0 else .65, .16, .32, .31])
        selected = [(g, m) for g, m in cases if m == kind]
        ticks, pos = [], []
        y = 0
        for g, m in selected:
            for method in ("swift", m):
                vals = g["methods"][method]; total = sum(v["time"] for v in vals)
                parts = [sum(v["metrics"].get(k, 0) for v in vals) / total * 100 for k, _, _ in PARTS]
                parts.append(100 - sum(parts)); left = 0
                for v, color in zip(parts, [c for _, _, c in PARTS] + ["#E5E7EB"]):
                    ax.barh(y, v, left=left, height=.72, color=color)
                    left += v
                ax.text(101, y, f"{total:,.0f}s", va="center", fontsize=8)
                ticks.append(f"{label(g)}\n{NAMES[method]}"); pos.append(y); y += 1
            y += .35
        ax.set_yticks(pos, ticks, fontsize=7); ax.invert_yaxis()
        ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Share of recorded request wall time (%)")
        ax.set_title(title)
    fig.legend(handles=[Patch(color=c, label=n) for _, n, c in PARTS] + [Patch(color="#E5E7EB", label="Other inference")],
               loc="lower center", bbox_to_anchor=(.5, .075), ncol=5, frameon=False)
    heading(fig, "01  |  Where is inference time spent?", "Recorded host wall-time intervals; residual is not a draft/verify decomposition.")
    footer(fig, "Intervals are subtracted once from synchronized request wall time. GPU events are never added to this partition.\n"
           "One seed (2024); separate sequential executions. AdaptiveCache: 7B test N=1000/164 and HumanEval-13B N=164.")
    dump_figure(fig, out, "01_timing_partition", report)


def hit_figure(groups, out, report):
    cases = [(g, m) for g, m in comparisons(groups) if m == "adaptive"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), sharex="col")
    counts = []
    for col, (g, method) in enumerate(cases):
        vals = g["methods"][method]
        for row, ref in enumerate(("baseline", "swift")):
            ax = axes[row, col]
            y = np.log([b["time"] / a["time"] for a, b in zip(vals, g["methods"][ref])])
            extent = max(float(np.max(abs(y))) * 1.10, .10)
            ax.axhspan(0, extent, color=GREEN, alpha=.09)
            ax.axhspan(-extent, 0, color=RED, alpha=.09)
            ax.axhline(0, color="#444444", lw=.8)
            text = []
            for hit, marker, color in ((True, "o", BLUE), (False, "X", ORANGE)):
                sel = np.array([v["hit"] == hit for v in vals])
                ax.scatter(np.arange(1, len(vals)+1)[sel], y[sel], marker=marker, s=12 if hit else 40,
                           color=color, alpha=.55 if hit else .95, linewidths=.4, rasterized=True)
                n, fast = int(sum(sel)), int(sum(y[sel] > 0))
                text.append(f"{'Hit' if hit else 'Miss'}: {fast}/{n} faster")
                counts.append({"case": label(g), "reference": ref, "hit": hit, "n": n, "faster": fast})
            ax.set_title(label(g) if row == 0 else f"Relative to {NAMES[ref]}", fontweight="bold" if row == 0 else "normal")
            ax.text(.03, .97, "\n".join(text), transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
            ax.set_ylim(-extent, extent)
            if row == 1: ax.set_xlabel("Request order")
            if col == 0: ax.set_ylabel(f"log(T {NAMES[ref]} / T AdaptiveCache)\npositive = faster")
    fig.subplots_adjust(left=.075, right=.98, bottom=.16, top=.85, hspace=.35, wspace=.28)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="Cache hit"),
                        Line2D([], [], marker="X", color=ORANGE, ls="", label="Cache miss"),
                        Patch(color=GREEN, alpha=.2, label="Observed faster"), Patch(color=RED, alpha=.2, label="Observed slower")],
               loc="lower center", bbox_to_anchor=(.5, .075), ncol=4, frameon=False)
    heading(fig, "02  |  Cache hits and observed latency benefits", "Same-input pairs within each run. AR reference above; SWIFT reference below.")
    footer(fig, "Per-request ratios are descriptive; sequential methods were not interleaved. They do not isolate cache causality.\n"
           "Symmetric log scale preserves all outliers. Cache hit/miss is an observed grouping, not randomized treatment.")
    save_csv(out / "cache_hit_regions.csv", counts)
    dump_figure(fig, out, "02_cache_hit_regions", report)


def region_figure(groups, out, report):
    cases = comparisons(groups)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    norm = Normalize(0, 100)
    for ax, (g, method) in zip(axes.flat, cases):
        pairs = list(paired(g, method))
        x = np.array([p[2] for p in pairs]); y = np.array([p[3] for p in pairs])
        xmin, xmax = min(-5, np.min(x)*1.12), max(5, np.max(x)*1.12)
        ymin, ymax = min(-5, np.min(y)*1.15), max(5, np.max(y)*1.15)
        for xlo, xhi, ylo, yhi, color in ((xmin, 0, 0, ymax, "#E8F3EC"), (0, xmax, 0, ymax, "#D6EBDD"),
                                         (xmin, 0, ymin, 0, "#F4DCE1"), (0, xmax, ymin, 0, "#F7ECD8")):
            ax.fill_between([xlo, xhi], ylo, yhi, color=color, zorder=0)
        ax.axvspan(-.25, .25, color="#BBBBBB", alpha=.3)
        for hit, marker in ((True, "o"), (False, "X")) if method == "adaptive" else ((None, "o"),):
            ix = [i for i, p in enumerate(pairs) if p[0]["hit"] == hit]
            colors = [100 * pairs[i][0]["opt"] / pairs[i][0]["time"] for i in ix]
            ax.scatter(x[ix], y[ix], c=colors, cmap="viridis", norm=norm, marker=marker,
                       s=16 if marker == "o" else 36, alpha=.6, linewidths=.25, edgecolors="#555555", rasterized=True)
        ax.axhline(0, color="#555555", lw=.6); ax.axvline(0, color="#555555", lw=.6)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        counts = Counter(p[4] for p in pairs)
        for xp, yp, reg, ha in ((.02, .97, "M down / faster", "left"), (.98, .97, "M up / faster", "right"),
                               (.02, .03, "M down / slower", "left"), (.98, .03, "M up / slower", "right")):
            ax.text(xp, yp, f"{reg}\n{100*counts[reg]/len(pairs):.1f}%", transform=ax.transAxes,
                    va="top" if yp > .5 else "bottom", ha=ha, fontsize=7.5,
                    bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": 2})
        eq = counts["M equal / faster"] + counts["M equal / slower"]
        ax.set_title(f"{NAMES[method]} | {label(g)}\nExact M tie: {eq}/{len(pairs)}; faster: {counts['M equal / faster']}", fontsize=9)
        ax.set_xlabel("Change in per-request M vs SWIFT (%)", fontsize=8)
        ax.set_ylabel("Latency reduction vs SWIFT (%)", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.065, right=.91, bottom=.12, top=.86, wspace=.33, hspace=.53)
    cax = fig.add_axes([.935, .29, .012, .42])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    cb.set_label("Optimization / request time (%)")
    heading(fig, "03  |  Request-level trade-off regions", "Four trade-off regions plus the exact-M-tie line. All points shown; no jitter, clipping, or fitted clusters.")
    footer(fig, "Grey band: +/-0.25% visual guide around zero; ties counted with absolute M tolerance 1e-10. Timing signs are not significance tests.\n"
           "Positive y = lower latency. Circles = hit (or Fixed); X = miss. Feature medians and IQRs: region_characteristics.csv.")
    dump_figure(fig, out, "03_M_latency_regions", report)


def mask_image(values, layers):
    image = np.zeros((layers * 2, len(values)))
    for j, v in enumerate(values):
        if v["mask"]:
            for part, skipped in enumerate(v["mask"]):
                image[np.array(skipped, dtype=int) + part * layers, j] = 1
    return image


def rolling(y, width=25):
    # Trailing window; never crosses a run boundary or fabricates future data.
    return [np.median(y[max(0, i-width+1):i+1]) for i in range(len(y))]


def timeline_figure(cases, out, name, report=None):
    detail = len(cases) == 1
    fig, axes = plt.subplots(7 if detail else 4, len(cases), figsize=(11, 18) if detail else (16, 11), squeeze=False)
    cmap = ListedColormap(["#F3F4F6", BLUE])
    for col, (g, method) in enumerate(cases):
        n = len(g["methods"][method]); layer = g["layers"]
        for row, m in enumerate(("swift", method)):
            ax = axes[row, col]; vals = g["methods"][m]
            ax.imshow(mask_image(vals, layer), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                      extent=[.5, n+.5, layer*2-.5, -.5], rasterized=True)
            ax.axhline(layer-.5, color="white", lw=2)
            changed = [i+1 for i in range(1, n) if vals[i]["mask"] != vals[i-1]["mask"]]
            for i in changed: ax.axvline(i, color=ORANGE, lw=1)
            ax.set_yticks([layer/2, 1.5*layer], ["Attn", "MLP"])
            ax.set_title(f"{label(g)}\n{NAMES[m]}: {len(set(v['mask'] for v in vals))} end masks", fontsize=10)
            ax.tick_params(labelbottom=False)
        ax = axes[2, col]
        for m, color in (("swift", BLUE), (method, GREEN)):
            vals = g["methods"][m]; x = np.arange(1, n+1)
            ax.plot(x, np.cumsum([v["opt"] for v in vals]), color=color, label=NAMES[m], lw=1.7)
            active = [v["index"] for v in vals if v["opt"] > 0]
            ax.scatter(active, np.interp(active, x, np.cumsum([v["opt"] for v in vals])), color=color, s=12)
        ax.set_ylabel("Cumulative optimization (s)", fontsize=9); ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=8)
        a, s = g["methods"][method], g["methods"]["swift"]
        if detail:
            ax = axes[3, col]
            hamming = [sum(len(set(x)^set(y)) for x,y in zip(v['mask'],b['mask'])) for v,b in zip(a,s)]
            ax.plot(np.arange(1,n+1), hamming, color=ORANGE)
            ax.set_ylabel("Hamming distance\nvs SWIFT end mask")
            ax.set_title("Same mask is zero; this does not recover within-request candidates",fontsize=9)
            for r, field, ylabel in ((4,"M","Mean committed length M"),(5,"time","Request latency (s)")):
                ax=axes[r,col]
                for vals,color,nm in ((s,BLUE,"SWIFT"),(a,GREEN,NAMES[method])):
                    raw=[v[field] for v in vals]
                    ax.plot(np.arange(1,n+1),raw,color=color,alpha=.18,lw=.6)
                    ax.plot(np.arange(1,n+1),rolling(raw),color=color,lw=1.5,label=nm)
                ax.set_ylabel(ylabel);ax.legend(frameon=False,fontsize=8)
                ax.set_title("Faint: raw requests; solid: trailing median (up to 25 requests)",fontsize=9)
        ax = axes[6 if detail else 3, col]
        saving = np.cumsum([b["time"] - v["time"] for v, b in zip(a, s)])
        x = np.arange(1, n+1)
        ax.fill_between(x, 0, saving, where=saving>=0, color=GREEN, alpha=.25)
        ax.fill_between(x, 0, saving, where=saving<0, color=RED, alpha=.25)
        ax.plot(x, saving, color="#333333", lw=1)
        ax.axhline(0, color="#555555", lw=.6); ax.set_xlabel("Request order")
        ax.set_ylabel("Cumulative SWIFT - method (s)", fontsize=9)
        ax.set_title("Positive = method saved wall time", fontsize=9)
        if col == 0:
            inset = axes[2, col].inset_axes([.49, .13, .48, .48])
            end = min(50, n)
            for m, color in (("swift", BLUE), (method, GREEN)):
                inset.plot(np.arange(1, end+1), np.cumsum([v["opt"] for v in g["methods"][m]])[:end], color=color)
            inset.set_title("First 50 requests", fontsize=7); inset.tick_params(labelsize=6)
    fig.subplots_adjust(left=.13 if detail else .065, right=.98, bottom=.07 if detail else .10,
                        top=.91 if detail else .86, hspace=.65, wspace=.34)
    heading(fig, "04  |  Configuration evolution and the cost of adaptation", "Blue cells = skipped layer; grey = retained; orange lines = changes between request-end masks.")
    footer(fig, "Masks are request-end observations, not within-request search traces. Optimization dots mark requests with nonzero recorded host time.\n"
           "Cumulative differences compare independent sequential executions; they do not estimate counterfactual adaptation value.")
    dump_figure(fig, out, name, report)


def region_profiles(groups, out):
    # Distribution pages supplement the scatter: common features are visible, not inferred from a single mean.
    for g, method in comparisons(groups):
        pairs = list(paired(g, method))
        bins = defaultdict(list)
        for a, s, dm, gain, region in pairs: bins[region].append(a)
        present = [r for r in REGIONS if bins[r]]
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        features = [("Input tokens", lambda a: a["input_length"]),
                    ("Optimization share (%)", lambda a: 100*a["opt"]/a["time"]),
                    ("Optimization iterations this request", None)]
        steps = g["methods"][method]; diffs = np.diff([0]+[v["steps_total"] for v in steps])
        for ax, (title, fn) in zip(axes, features):
            data = [[fn(v) if fn else diffs[v["index"]-1] for v in bins[r]] for r in present]
            bp = ax.boxplot(data, orientation="horizontal", tick_labels=[f"{r}\n(n={len(bins[r])})" for r in present],
                            patch_artist=True, showfliers=True, flierprops={"markersize": 2, "alpha": .25})
            for patch in bp["boxes"]: patch.set_facecolor("#CCE4EF")
            ax.set_xlabel(title); ax.grid(axis="x", alpha=.2)
        heading(fig, "Region profiles | " + NAMES[method] + " | " + label(g), "Medians, interquartile ranges, and outliers. Regions are descriptive, not learned clusters.")
        fig.subplots_adjust(left=.15, right=.98, bottom=.16, top=.80, wspace=.65)
        footer(fig, "A shared characteristic is a candidate explanation, not evidence of causation. Small groups are retained and labelled.")
        dump_figure(fig, out, "profile_" + method + "_" + g["run"] + "_" + g["dataset"])


def tree_figure(groups, out, report):
    g = next((g for g in groups if "rank_dp" in g["methods"]), None)
    if g is None: return
    methods = ["swift", "native_control", "raw_dp", "rank_dp"]
    colors = [BLUE, "#999999", ORANGE, GREEN]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    rank = g["methods"]["rank_dp"]
    labels = [v for r in rank for e in r["metrics"]["tree_events"] for v in e["rank_labels"] if v.get("support", 0) >= 32]
    learned = np.mean([v["brier"] for v in labels]); raw = np.mean([v["raw_brier"] for v in labels])
    ax = axes[0, 0]
    ax.bar(["Raw on same labels", "Learned"], [raw, learned], color=[ORANGE, GREEN], width=.6)
    for i, v in enumerate([raw, learned]): ax.text(i, v+.003, f"{v:.4f}", ha="center")
    ax.set_ylim(0, max(raw, learned)*1.22); ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(f"(a) Paired supported labels (n={len(labels):,})")
    tree_rows = []
    for method in methods:
        vals = g["methods"][method]; events = [e for r in vals for e in r["metrics"].get("tree_events", [])]
        summary = g["summaries"][method]
        tree_rows.append({"method": method, "nodes": np.mean([e["verified_nodes"] for e in events]) if events else np.nan,
                          "depth": np.mean([e["draft_depth"] for e in events]) if events else np.nan,
                          "M": float(summary["mean_accepted_length"]), "macro_speedup_AR": float(summary["speedup_vs_baseline"]),
                          "global_speedup_AR": float(summary["global_tokens_per_second"])/float(g["summaries"]["baseline"]["global_tokens_per_second"]),
                          "time": float(summary["inference_seconds"]),
                          "controller": sum(r["metrics"].get("controller_seconds",0) for r in vals)})
    x = np.arange(len(methods)); names = [NAMES[m] for m in methods]
    for ax, key, title in ((axes[0,1], "nodes", "(b) Mean verified nodes"), (axes[0,2], "M", "(c) Mean committed length M")):
        vals = [r[key] for r in tree_rows]; ax.bar(x, vals, color=colors, width=.6)
        for i,v in enumerate(vals):
            if np.isfinite(v): ax.text(i, v+.02, f"{v:.3f}", ha="center", fontsize=9)
            else: ax.text(i, .6, "not\nrecorded", ha="center", fontsize=8)
        ax.set_xticks(x, names, rotation=15); ax.set_title(title); ax.set_ylim(0, np.nanmax(vals)*1.2)
    ax = axes[1,0]; vals = [r["controller"] for r in tree_rows]
    ax.bar(x, vals, color=colors, width=.6)
    for i,v in enumerate(vals): ax.text(i,v+.3,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_xticks(x,names,rotation=15); ax.set_ylabel("Recorded controller host seconds")
    ax.set_title("(d) Controller overhead (not a timing partition)")
    ax = axes[1,1]
    ax.bar(x-.18,[r['macro_speedup_AR'] for r in tree_rows],width=.35,color=BLUE,label="Mean request tok/s ratio")
    ax.bar(x+.18,[r['global_speedup_AR'] for r in tree_rows],width=.35,color=ORANGE,label="Global throughput ratio")
    ax.axhline(1,color="#444444",ls="--",lw=1); ax.set_ylim(0,1.4);ax.set_xticks(x,names,rotation=15)
    ax.set_title("(e) Two aggregations; AR = 1 in both"); ax.legend(frameon=False,fontsize=8)
    ax = axes[1,2]; ax.axis("off")
    ax.text(0, .95, "Read the chain of evidence",fontsize=13,fontweight="bold",va="top")
    ax.text(0,.79,"Better rank prediction\n        ↓\nDifferent tree / greater M?\n        ↓\nEnough benefit to pay controller cost?",fontsize=12,va="top",linespacing=1.8)
    ax.text(0,.16,"Observed: learned Brier improves;\nM is nearly unchanged; no speed win.\nNext: inspect decision sensitivity\nand overhead before a larger run.",fontsize=10,va="top",color="#555555")
    heading(fig,"05  |  From rank prediction to execution cost", "Tree-control diagnostic only | Llama-2-7B | GSM8K-512 | train N=128 | seed 2024")
    fig.subplots_adjust(left=.065,right=.98,bottom=.13,top=.85,hspace=.5,wspace=.35)
    footer(fig,"Raw and learned Brier use exactly the same supported labels within Rank DP, not different policies' label populations.\n"
           "Controller time may overlap phase timers; it is never stacked with GPU time. Training diagnostic is not a test benchmark.")
    save_csv(out/"tree_diagnostics.csv",tree_rows)
    dump_figure(fig,out,"05_tree_diagnostics",report)


def write_notes(out, audit, files, groups):
    data = {"input_files": files, "audit": audit, "sources": SOURCES,
            "scope": "Fixed-first, AdaptiveCache, tree-control and their own AR/SWIFT references only",
            "region_rule": "exact M ties atol=1e-10; faster iff recorded method latency < paired reference latency",
            "limitations": ["one seed", "sequential executions, not randomized interleaving", "no causal effect from paired timing alone",
                            "request-end mask only", "unsupported model revisions remain unverified, not assumed pinned"]}
    (out/"analysis_provenance.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    text = """# Phân tích cơ chế từ output H100

Hình chính: `mechanism_report.pdf` (5 trang); mỗi hình có PDF vector và PNG riêng.
`profile_*.pdf/png` là phân bố đặc điểm trong các vùng; `timeline_detail_*.pdf/png` là chi tiết từng cấu hình.

## Cách đọc và giới hạn

- Hình 1: các khoảng host wall time không chồng lấn về phạm vi gọi, được trừ một lần; phần còn lại không phải decoding thuần hay draft/verify. Không cộng GPU event với CPU wall time.
- Hình 2: dấu latency chỉ mô tả từng cặp input trong các execution tuần tự; không phải tác động nhân quả của cache hit.
- Hình 3: bốn vùng cộng đường M bằng nhau. M theo request là mean(accept_lengths); M trong bảng tổng là mean của mọi vòng, không phải mean của các M request.
- Hình 4: mask là trạng thái cuối request; không quan sát được mọi mask thử bên trong request. SWIFT thường dừng optimization sớm, không chạy xuyên stream.
- Hình 5: train/128 bài riêng, không ghép với test. Brier raw/learned được so trên cùng nhãn đủ support trong Rank DP.
- Không vẽ confidence interval từ một stream để giả làm nhiều lần lặp độc lập. Không dùng accuracy/pass@1 trong hình motivation.
- Model revision null là thiếu bằng chứng pin revision; kiểm tra các config và phiên bản bằng nhau không khắc phục được giới hạn đó.

## Quan sát và phép thử tiếp theo

1. Cả bốn cấu hình AdaptiveCache ghi nhận một mask dùng cuối request duy nhất; hit cao không chứng minh định tuyến nhiều mask. Cần thử đối chứng cùng mask, có/không embedding và lookup, áp dụng cùng runtime.
2. GSM8K-512 và HumanEval-7B có các trường hợp M/output không đổi nhưng latency khác. Cần interleave hoặc đảo thứ tự phương pháp và lặp seed; không quy mọi khác biệt cho cache hoặc optimization.
3. HumanEval-13B có vùng M tăng/latency giảm; CNN-64 có vùng M giảm/latency tăng. Cần kiểm tra mức độ phụ thuộc mask đầu và thứ tự request trước khi thiết kế điều kiện tái sử dụng.
4. CNN-512 Fixed có nhiều điểm M giảm nhưng latency giảm. So CNN-64/512 phải kiểm tra input hash: cùng task không đảm bảo cùng prompt sau clipping.
5. Rank DP cải thiện prediction nhưng M gần như không đổi. Kiểm tra độ nhạy quyết định tree đối với predictor và giữ chi phí controller trước khi mở rộng benchmark.

## Nền tảng thiết kế hình

Không sao chép hình hay coi thiết kế là chứng nhận chất lượng hội nghị. Học cách nối phân tích cơ chế với hiệu năng từ các nguồn chính:

"""
    for name,url in SOURCES.items(): text+=f"- [{name}]({url})\n"
    text += "\nSWIFT: tách phân tích sparsity/optimization và hiệu năng. KNN-SSD: kiểm tra chuyển giao cấu hình và overhead. Sequoia: đối chiếu cấu trúc cây, số token chấp nhận và chi phí phần cứng. Các phân vùng hiện tại là phân tích riêng của output này, chưa chứng minh novelty.\n"
    (out/"figure_notes_vi.txt").write_text(text,encoding="utf-8")
    cards = [
        ("01_timing_partition", "1. Phân bổ thời gian", "Đọc lát optimization, embedding/search và phần còn lại. Không gọi phần còn lại là draft/verify."),
        ("02_cache_hit_regions", "2. Cache hit có đi cùng lợi ích?", "Mỗi điểm là một request. Nền xanh: nhanh hơn; nền đỏ: chậm hơn. AR ở hàng trên, SWIFT ở hàng dưới."),
        ("03_M_latency_regions", "3. Bản đồ phân vùng hành vi", "Trục ngang: thay đổi M. Trục dọc: phần trăm giảm latency. Vùng trên-trái đặc biệt đáng xem: M giảm nhưng chạy nhanh hơn. Đường M=0 là nhóm riêng, không loại bỏ."),
        ("04_mask_timeline", "4. Mask và chi phí thích nghi theo stream", "Xanh: layer bị skip. Cam: đổi mask cuối request. Đường tích lũy cho thấy thời điểm phát sinh chi phí; không phải toàn bộ trace tối ưu."),
        ("05_tree_diagnostics", "5. Tree-control: từ prediction đến chi phí thực thi", "Prediction tốt hơn chưa chuyển thành M hay tốc độ tốt hơn. Đây là train/128 bài, giữ riêng với các benchmark test."),
    ]
    html = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Phân tích output SWIFT</title>
<style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#172536;background:#f7f8fa}
h1,h2{line-height:1.25}section{background:white;padding:26px;margin:28px 0;border:1px solid #dce2e8;border-radius:10px}
img{width:100%;height:auto}a{color:#0072b2}small{color:#526170}li{margin:8px 0}</style>
<h1>Phân tích cơ chế từ output H100</h1><p>Fixed-first · AdaptiveCache · Tree-control · AR/SWIFT đối chứng</p>
<p><a href="mechanism_report.pdf">Mở PDF báo cáo 5 trang</a> · <a href="figure_notes_vi.txt">Chú giải và giả thuyết cần kiểm chứng</a></p>
<small>32 JSONL / 20.116 request · seed 2024 · dữ liệu quan sát, chưa phải chứng minh nhân quả hay novelty.</small>'''
    for stem, title, description in cards:
        html += f'<section><h2>{title}</h2><p>{description}</p><a href="{stem}.pdf">PDF vector</a><a href="{stem}.png"><img src="{stem}.png" alt="{escape(title)}"></a></section>'
    html += '<section><h2>Chi tiết để tìm đặc điểm chung</h2><ul>'
    for p in sorted(out.glob('profile_*.pdf')):
        html += f'<li><a href="{escape(p.name)}">{escape(p.stem)}</a></li>'
    html += '</ul><h2>Diễn biến từng cấu hình</h2><ul>'
    for p in sorted(out.glob('timeline_detail_*.pdf')):
        html += f'<li><a href="{escape(p.name)}">{escape(p.stem)}</a></li>'
    html += '</ul><p>Bảng vùng: <a href="region_characteristics.csv">region_characteristics.csv</a>. Dữ liệu ghép cặp: <a href="paired_requests.csv">paired_requests.csv</a>.</p></section></html>'
    (out/'index.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"_local/visualization_selected_outputs_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"_local/analysis")
    args=parser.parse_args()
    groups,audit,files=load_runs(args.input.resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    setup_style()
    export_analysis(groups,args.output)
    with PdfPages(args.output/"mechanism_report.pdf") as report:
        timing_figure(groups,args.output,report)
        hit_figure(groups,args.output,report)
        region_figure(groups,args.output,report)
        cases=comparisons(groups)
        examples=[next((g,m) for g,m in cases if m=="fixed_first" and g['dataset']=="cnndm.quality512"),
                  next((g,m) for g,m in cases if m=="fixed_first" and g['dataset']=="gsm8k.quality512"),
                  next((g,m) for g,m in cases if m=="adaptive" and "13b" in g['model'].lower())]
        timeline_figure(examples,args.output,"04_mask_timeline",report)
        tree_figure(groups,args.output,report)
    region_profiles(groups,args.output)
    for g,m in comparisons(groups):
        timeline_figure([(g,m)],args.output,"timeline_detail_"+m+"_"+g['run']+"_"+g['dataset'])
    write_notes(args.output,audit,files,groups)
    print(f"Audited {len(audit)} JSONL files / {sum(r['requests'] for r in audit):,} requests.")
    print(f"Figures and report: {args.output.resolve()}")


if __name__=="__main__":
    main()
