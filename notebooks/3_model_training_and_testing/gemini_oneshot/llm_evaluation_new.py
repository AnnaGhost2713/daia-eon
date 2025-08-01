import json
import re
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

Entity = Tuple[int, int, str]  # (start, end, label)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_gt_mapping(ground_truth_records: List[Dict]) -> Dict[str, Dict[str, List[Entity]]]:
    """
    Returns: { filename: { label: [ (start, end, label), ... sorted by start ] } }
    """
    mapping = {}
    for rec in ground_truth_records:
        file = rec["file"]
        labels = rec.get("labels", [])
        per_label = defaultdict(list)
        for ent in labels:
            start, end, label = ent["start"], ent["end"], ent["label"]
            per_label[label].append((start, end, label))
        for label in per_label:
            per_label[label].sort(key=lambda x: x[0])
        mapping[file] = per_label
    return mapping


def extract_predictions_from_anonymized(anonymized_records: List[Dict],
                                       gt_mapping: Dict[str, Dict[str, List[Entity]]]
                                      ) -> Dict[str, List[Entity]]:
    """
    For each file, extract placeholders like <<LABEL_2>> and map them to spans in ground truth via ordinal.
    Returns: { filename: [ (start, end, label), ... ] } predicted entities
    """
    placeholder_re = re.compile(r"<<([A-Z_]+)_(\d+)>>")
    predictions = {}

    for rec in anonymized_records:
        file = rec["file"]
        text = rec.get("anonymized_text", "")
        preds = []
        gt_labels_for_file = gt_mapping.get(file, {})
        for match in placeholder_re.finditer(text):
            label = match.group(1)
            index = int(match.group(2))  # 1-based
            gt_list = gt_labels_for_file.get(label, [])
            if 1 <= index <= len(gt_list):
                span = gt_list[index - 1]  # ordinal mapping
                preds.append(span)
            else:
                # No corresponding ground truth span: sentinel to count as FP
                preds.append((-1, -1, label))
        predictions[file] = preds
    return predictions


def compute_confusion_single_file(gt_labels: Dict[str, List[Entity]],
                                  pred_entities: List[Entity]
                                 ) -> Tuple[Counter, Counter, Counter]:
    """
    Compute TP/FP/FN for one file.
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()

    gt_entities_all = []
    for label, lst in gt_labels.items():
        gt_entities_all.extend(lst)
    gt_set = set(gt_entities_all)
    pred_set = set(pred_entities)

    for ent in pred_set:
        if ent in gt_set and ent[0] != -1:
            tp[ent[2]] += 1
        else:
            fp[ent[2]] += 1

    for ent in gt_set:
        if ent not in pred_set:
            fn[ent[2]] += 1

    return tp, fp, fn


def compute_confusion(gt_mapping: Dict[str, Dict[str, List[Entity]]],
                      pred_mapping: Dict[str, List[Entity]]
                     ) -> Tuple[Counter, Counter, Counter]:
    tp_total = Counter()
    fp_total = Counter()
    fn_total = Counter()

    # Files present in either set
    all_files = set(gt_mapping.keys()) | set(pred_mapping.keys())
    for file in all_files:
        gt_labels = gt_mapping.get(file, {})  # could be empty
        pred_entities = pred_mapping.get(file, [])
        tp, fp, fn = compute_confusion_single_file(gt_labels, pred_entities)
        tp_total.update(tp)
        fp_total.update(fp)
        fn_total.update(fn)

    return tp_total, fp_total, fn_total


def precision_recall_f1(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def aggregate_metrics(tp: Counter, fp: Counter, fn: Counter):
    per_label = {}
    for label in sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))):
        p, r, f1 = precision_recall_f1(tp[label], fp[label], fn[label])
        per_label[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
        }

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p, micro_r, micro_f1 = precision_recall_f1(total_tp, total_fp, total_fn)

    macro_f1 = sum(v["f1"] for v in per_label.values()) / max(len(per_label), 1)

    return {
        "per_label": per_label,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro_f1": macro_f1,
        "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }


def print_summary(overall_metrics: Dict, per_file_metrics: Dict[str, Dict]):
    def fmt(x):
        return f"{x:.3f}"

    print("\n=== OVERALL ===")
    print("Micro precision:  ", fmt(overall_metrics["micro"]["precision"]))
    print("Micro recall:     ", fmt(overall_metrics["micro"]["recall"]))
    print("Micro F1:         ", fmt(overall_metrics["micro"]["f1"]))
    print("Macro F1:         ", fmt(overall_metrics["macro_f1"]))
    print("Totals (TP/FP/FN):", overall_metrics["totals"]["tp"],
          overall_metrics["totals"]["fp"], overall_metrics["totals"]["fn"])
    print("\nPer-label breakdown:")
    headers = ["Label", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    print(f"{headers[0]:20} {headers[1]:>3} {headers[2]:>3} {headers[3]:>3} {headers[4]:>9} {headers[5]:>7} {headers[6]:>6}")
    for label, stats in overall_metrics["per_label"].items():
        print(f"{label:20} {stats['tp']:3} {stats['fp']:3} {stats['fn']:3} "
              f"{fmt(stats['precision']):>9} {fmt(stats['recall']):>7} {fmt(stats['f1']):>6}")

    print("\n=== PER FILE ===")
    for fname, metrics in sorted(per_file_metrics.items()):
        print(f"\nFile: {fname}")
        print("  Micro P / R / F1:",
              fmt(metrics["micro"]["precision"]), "/",
              fmt(metrics["micro"]["recall"]), "/",
              fmt(metrics["micro"]["f1"]))
        print("  Totals (TP/FP/FN):", metrics["totals"]["tp"],
              metrics["totals"]["fp"], metrics["totals"]["fn"])
        # optionally per-label for that file:
        for label, stats in metrics["per_label"].items():
            print(f"    {label:15} TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} "
                  f"P={fmt(stats['precision'])} R={fmt(stats['recall'])} F1={fmt(stats['f1'])}")


def compute_per_file(gt_mapping: Dict[str, Dict[str, List[Entity]]],
                     pred_mapping: Dict[str, List[Entity]]
                    ) -> Dict[str, Dict]:
    per_file = {}
    all_files = set(gt_mapping.keys()) | set(pred_mapping.keys())
    for file in all_files:
        gt_labels = gt_mapping.get(file, {})
        pred_entities = pred_mapping.get(file, [])
        tp, fp, fn = compute_confusion_single_file(gt_labels, pred_entities)
        metrics = aggregate_metrics(tp, fp, fn)
        per_file[file] = metrics
    return per_file


def main(gt_json_path: str, anonymized_json_path: str):
    anonymized = load_json(anonymized_json_path)
    ground_truth = load_json(gt_json_path)

    gt_map = build_gt_mapping(ground_truth)
    pred_map = extract_predictions_from_anonymized(anonymized, gt_map)
    tp, fp, fn = compute_confusion(gt_map, pred_map)
    overall_metrics = aggregate_metrics(tp, fp, fn)
    per_file_metrics = compute_per_file(gt_map, pred_map)

    print_summary(overall_metrics, per_file_metrics)



if __name__ == "__main__":
    # === hardcode your paths here ===
    gt_path = "../../../data/original/ground_truth_split/test_norm.json"
    anon_path = "../../../data/testing/gemini_results/anonymized_text_results/combined_results_1.5.json"
    main(gt_path, anon_path)