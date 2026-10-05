import json
import os
import math
import io
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
PER_CLASS_METRICS = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

def replace_nans_with_none(obj):
    """Recursively converts NaNs to None so standard json.dump won't fail."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    elif isinstance(obj, dict):
        return {k: replace_nans_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nans_with_none(v) for v in obj]
    return obj

def per_class_breakdown(coco_eval, coco_gt):
    precision = coco_eval.eval["precision"]
    cat_ids = coco_eval.params.catIds
    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    def _ap(k, t_idx, area_idx):
        p = precision[t_idx, :, k, area_idx, -1] if t_idx is not None else precision[:, :, k, area_idx, -1]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    per_class = {}
    for k, cat_id in enumerate(cat_ids):
        name = id_to_name.get(cat_id, str(cat_id))
        per_class[name] = {
            "AP":   _ap(k, None, 0),  # mean over IoU .50:.95, all areas
            "AP50": _ap(k, 0, 0),     # IoU = .50 exactly
            "AP75": _ap(k, 5, 0),     # IoU = .75 exactly
            "APs":  _ap(k, None, 1),  # small objects
            "APm":  _ap(k, None, 2),  # medium objects
            "APl":  _ap(k, None, 3),  # large objects
        }

    return per_class

def run_coco_eval(coco_gt, coco_dt, iou_type, use_cats):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.useCats = 1 if use_cats else 0

    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return coco_eval, buf.getvalue()

def evaluate_predictions(prediction_file, predictions, iou_type="bbox"):
    if len(predictions) == 0:
        raise ValueError("Error: The `predictions` list is completely empty!")

    coco_gt = COCO(prediction_file)
    coco_dt = coco_gt.loadRes(predictions)

    overall_eval, overall_summary = run_coco_eval(coco_gt, coco_dt, iou_type, use_cats=True)
    agnostic_eval, agnostic_summary = run_coco_eval(coco_gt, coco_dt, iou_type, use_cats=False)

    return {
        "overall": {
            "stats": {name: float(v) for name, v in zip(STAT_NAMES, overall_eval.stats)},
            "summary_text": overall_summary,
        },
        "class_agnostic": {
            "stats": {name: float(v) for name, v in zip(STAT_NAMES, agnostic_eval.stats)},
            "summary_text": agnostic_summary,
            "note": "Every detection/GT box is treated as one 'object' class -- measures pure localization ability.",
        },
        "per_class": per_class_breakdown(overall_eval, coco_gt),
    }

def format_result(result):
    lines = []
    lines.append("=" * 70)
    lines.append("           SARDet-100K Object Detection Evaluation")
    lines.append("=" * 70)
    
    # Summary Table for Main Metrics
    ov_stats = result["overall"]["stats"]
    ag_stats = result["class_agnostic"]["stats"]

    summary_headers = ["Evaluation Metric", "Overall (Class-Aware)", "Class-Agnostic"]
    summary_rows = [
        ["mAP @ IoU=0.50:0.95 (AP)", f"{ov_stats['AP']:.4f}", f"{ag_stats['AP']:.4f}"],
        ["mAP @ IoU=0.50 (AP50)",     f"{ov_stats['AP50']:.4f}", f"{ag_stats['AP50']:.4f}"],
        ["mAP @ IoU=0.75 (AP75)",     f"{ov_stats['AP75']:.4f}", f"{ag_stats['AP75']:.4f}"],
        ["AP Small (APs)",             f"{ov_stats['APs']:.4f}", f"{ag_stats['APs']:.4f}"],
        ["AP Medium (APm)",            f"{ov_stats['APm']:.4f}", f"{ag_stats['APm']:.4f}"],
        ["AP Large (APl)",             f"{ov_stats['APl']:.4f}", f"{ag_stats['APl']:.4f}"],
        ["AR @ 100 Detections",        f"{ov_stats['AR100']:.4f}", f"{ag_stats['AR100']:.4f}"],
    ]

    lines.append("\n--- METRICS OVERVIEW ---")
    if HAS_TABULATE:
        lines.append(tabulate(summary_rows, headers=summary_headers, tablefmt="github"))
    else:
        # Simple ASCII Fallback Table
        fmt = "{:<30} | {:<20} | {:<20}"
        lines.append(fmt.format(*summary_headers))
        lines.append("-" * 76)
        for row in summary_rows:
            lines.append(fmt.format(*row))

    lines.append("\n--- PER-CLASS BREAKDOWN ---")
    per_class = result["per_class"]
    headers = ["Category"] + PER_CLASS_METRICS
    rows = []
    
    for name, ap in per_class.items():
        row = [name]
        for m in PER_CLASS_METRICS:
            val = ap[m]
            row.append(f"{val:.4f}" if (isinstance(val, (int, float)) and not math.isnan(val)) else "N/A")
        rows.append(row)

    if HAS_TABULATE:
        lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        fmt = "{:<16}" + " {:<10}" * len(PER_CLASS_METRICS)
        lines.append(fmt.format(*headers))
        lines.append("-" * 76)
        for row in rows:
            lines.append(fmt.format(*row))

    return "\n".join(lines)

def save_result(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_text = format_result(result)

    txt_path = os.path.join(output_dir, "metrics.txt")
    json_path = os.path.join(output_dir, "metrics.json")

    with open(txt_path, "w") as f:
        f.write(report_text + "\n")

    # Clean NaNs before serializing to JSON
    cleaned_result = replace_nans_with_none(result)
    with open(json_path, "w") as f:
        json.dump(cleaned_result, f, indent=2)

    print(f"\nSaved formatted metrics to: {txt_path}")
    print(f"Saved clean metrics JSON to: {json_path}")