import json
import io
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
PER_CLASS_METRICS = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

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
            "AP50": _ap(k, 0, 0),     # IoU = .50 exactly (index 0 of the 10 thresholds)
            "AP75": _ap(k, 5, 0),     # IoU = .75 exactly (index 5)
            "APs":  _ap(k, None, 1),  # mean over IoU, small objects only
            "APm":  _ap(k, None, 2),  # medium objects only
            "APl":  _ap(k, None, 3),  # large objects only
        }

    return per_class

def run_coco_eval(coco_gt, coco_dt, iou_type, use_cats):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.useCats =1 if use_cats else 0

    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return coco_eval, buf.getvalue()

def evaluate_predictions(prediction_file, predictions, iou_type="bbox"):
    if len(predictions) == 0:
        raise ValueError("Error: The `predictions` list is completely empty! "
                         "Your model generated zero bounding boxes above the score threshold.")

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
            "note": "Every detection/GT box is treated as one 'object' class here -- this "
                    "measures pure localization ability, ignoring whether the predicted "
                    "category was correct.",
        },
        "per_class": per_class_breakdown(overall_eval, coco_gt),
    }

def format_result(result):
    lines = []
    lines.append("=" * 70)
    lines.append("SARDet-100K Object Detection Evaluation")
    lines.append("=" * 70)
 
    lines.append("\n--- Overall (all classes, standard COCO mAP) ---")
    lines.append(result["overall"]["summary_text"].strip())
 
    lines.append("\n--- Class-agnostic (any object vs. no object, ignores category) ---")
    lines.append(result["class_agnostic"]["note"])
    lines.append(result["class_agnostic"]["summary_text"].strip())
 
    lines.append("\n--- Per-class breakdown (matches SARDet-100K paper Table S13 format) ---")
    per_class = result["per_class"]
    name_width = max(len(n) for n in per_class) + 2
    header = f"{'category':<{name_width}}" + "".join(f"{m:>8}" for m in PER_CLASS_METRICS)
    lines.append(header)
    for name, ap in per_class.items():
        row = f"{name:<{name_width}}" + "".join(f"{ap[m]:>8.3f}" for m in PER_CLASS_METRICS)
        lines.append(row)
 
    valid_aps = [ap["AP"] for ap in per_class.values() if ap["AP"] == ap["AP"]]  # drop NaNs
    if valid_aps:
        mean_ap = sum(valid_aps) / len(valid_aps)
        lines.append(f"\nUnweighted mean of per-class AP (should match overall AP above): {mean_ap:.3f}")
 
    return "\n".join(lines)

def save_result(result, output_file):
    report = format_result(result)

    with open(output_file, "w") as f:
        f.write(report + "\n")

    json_path = output_file.rsplit(".", 1)[0] + ".json"
    json.dump(json_path, f, indent=2, allow_nan=False)
 
    # return output_file, json_path