#!/usr/bin/env python3
"""
Evaluation Script for Gemini PII Detection Results

Compares Gemini-detected entities against ground truth labels and computes
precision, recall, and F1-score metrics at both entity-level and token-level.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EntityEvaluator:
    """Evaluates entity detection performance against ground truth."""

    def __init__(self, ground_truth_path: str, predictions_path: str):
        """Initialize with paths to ground truth and predictions."""
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path

    def load_ground_truth(self) -> Dict[str, Dict]:
        """Load ground truth data indexed by filename."""
        data = json.loads(Path(self.ground_truth_path).read_text(encoding='utf-8'))
        gt_by_file = {}
        for entry in data:
            filename = entry.get("file", "")
            gt_by_file[filename] = {
                "text": entry.get("text", ""),
                "entities": entry.get("labels", [])  # Ground truth uses 'labels' key
            }
        return gt_by_file

    def load_predictions(self) -> Dict[str, Dict]:
        """Load prediction data indexed by filename."""
        data = json.loads(Path(self.predictions_path).read_text(encoding='utf-8'))
        pred_by_file = {}
        for entry in data:
            filename = entry.get("file", "")
            pred_by_file[filename] = {
                "text_length": entry.get("text_length", 0),
                "entities": entry.get("entities", [])
            }
        return pred_by_file

    def normalize_entity(self, entity: Dict) -> Tuple[int, int, str]:
        """Normalize entity to (start, end, label) tuple."""
        return (entity["start"], entity["end"], entity["label"])

    def get_entity_spans(self, entities: List[Dict]) -> Set[Tuple[int, int, str]]:
        """Convert entity list to set of (start, end, label) tuples."""
        return {self.normalize_entity(e) for e in entities}

    def get_token_level_spans(self, entities: List[Dict], text: str) -> Set[Tuple[int, str]]:
        """Convert entities to character-level (position, label) pairs."""
        token_spans = set()
        for entity in entities:
            start, end, label = entity["start"], entity["end"], entity["label"]
            # Validate bounds
            if 0 <= start < end <= len(text):
                for pos in range(start, end):
                    token_spans.add((pos, label))
        return token_spans

    def compute_metrics(self, true_set: Set, pred_set: Set) -> Dict[str, float]:
        """Compute precision, recall, and F1-score for two sets."""
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    def compute_label_metrics(self, gt_entities: List[Dict], pred_entities: List[Dict],
                              text: str) -> Dict[str, Dict]:
        """Compute per-label metrics."""
        # Group entities by label
        gt_by_label = defaultdict(list)
        pred_by_label = defaultdict(list)

        for entity in gt_entities:
            gt_by_label[entity["label"]].append(entity)

        for entity in pred_entities:
            pred_by_label[entity["label"]].append(entity)

        # Get all labels
        all_labels = set(gt_by_label.keys()) | set(pred_by_label.keys())

        label_metrics = {}
        for label in all_labels:
            gt_spans = self.get_entity_spans(gt_by_label[label])
            pred_spans = self.get_entity_spans(pred_by_label[label])
            label_metrics[label] = self.compute_metrics(gt_spans, pred_spans)

        return label_metrics

    def evaluate_file(self, filename: str, gt_data: Dict, pred_data: Dict) -> Dict:
        """Evaluate predictions for a single file."""
        gt_text = gt_data["text"]
        gt_entities = gt_data["entities"]
        pred_entities = pred_data["entities"]

        # Entity-level evaluation (exact span + label match)
        gt_entity_spans = self.get_entity_spans(gt_entities)
        pred_entity_spans = self.get_entity_spans(pred_entities)
        entity_metrics = self.compute_metrics(gt_entity_spans, pred_entity_spans)

        # Token-level evaluation (character position + label match)
        gt_token_spans = self.get_token_level_spans(gt_entities, gt_text)
        pred_token_spans = self.get_token_level_spans(pred_entities, gt_text)
        token_metrics = self.compute_metrics(gt_token_spans, pred_token_spans)

        # Per-label metrics
        label_metrics = self.compute_label_metrics(gt_entities, pred_entities, gt_text)

        return {
            "filename": filename,
            "entity_metrics": entity_metrics,
            "token_metrics": token_metrics,
            "label_metrics": label_metrics,
            "gt_entity_count": len(gt_entities),
            "pred_entity_count": len(pred_entities),
            "text_length": len(gt_text)
        }

    def aggregate_metrics(self, file_results: List[Dict]) -> Dict:
        """Aggregate metrics across all files."""
        # Aggregate entity-level metrics
        total_entity_tp = sum(r["entity_metrics"]["tp"] for r in file_results)
        total_entity_fp = sum(r["entity_metrics"]["fp"] for r in file_results)
        total_entity_fn = sum(r["entity_metrics"]["fn"] for r in file_results)

        # Aggregate token-level metrics
        total_token_tp = sum(r["token_metrics"]["tp"] for r in file_results)
        total_token_fp = sum(r["token_metrics"]["fp"] for r in file_results)
        total_token_fn = sum(r["token_metrics"]["fn"] for r in file_results)

        # Aggregate label-level metrics
        label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for result in file_results:
            for label, metrics in result["label_metrics"].items():
                label_stats[label]["tp"] += metrics["tp"]
                label_stats[label]["fp"] += metrics["fp"]
                label_stats[label]["fn"] += metrics["fn"]

        # Compute overall metrics
        overall_entity = self.compute_metrics(
            set(range(total_entity_tp + total_entity_fn)),
            set(range(total_entity_tp + total_entity_fp))
        )
        overall_entity.update({"tp": total_entity_tp, "fp": total_entity_fp, "fn": total_entity_fn})

        overall_token = self.compute_metrics(
            set(range(total_token_tp + total_token_fn)),
            set(range(total_token_tp + total_token_fp))
        )
        overall_token.update({"tp": total_token_tp, "fp": total_token_fp, "fn": total_token_fn})

        # Compute per-label aggregated metrics
        aggregated_label_metrics = {}
        for label, stats in label_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            aggregated_label_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }

        return {
            "overall_entity_metrics": overall_entity,
            "overall_token_metrics": overall_token,
            "label_metrics": aggregated_label_metrics,
            "total_files": len(file_results),
            "total_gt_entities": sum(r["gt_entity_count"] for r in file_results),
            "total_pred_entities": sum(r["pred_entity_count"] for r in file_results),
            "total_characters": sum(r["text_length"] for r in file_results)
        }

    def run_evaluation(self) -> Dict:
        """Run complete evaluation and return results."""
        logger.info("Loading ground truth data...")
        gt_data = self.load_ground_truth()

        logger.info("Loading prediction data...")
        pred_data = self.load_predictions()

        # Find common files
        common_files = set(gt_data.keys()) & set(pred_data.keys())
        missing_in_pred = set(gt_data.keys()) - set(pred_data.keys())
        extra_in_pred = set(pred_data.keys()) - set(gt_data.keys())

        if missing_in_pred:
            logger.warning(f"Files in ground truth but not in predictions: {missing_in_pred}")
        if extra_in_pred:
            logger.warning(f"Files in predictions but not in ground truth: {extra_in_pred}")

        logger.info(f"Evaluating {len(common_files)} common files...")

        # Evaluate each file
        file_results = []
        for filename in sorted(common_files):
            result = self.evaluate_file(filename, gt_data[filename], pred_data[filename])
            file_results.append(result)

            # Log per-file summary
            entity_f1 = result["entity_metrics"]["f1"]
            token_f1 = result["token_metrics"]["f1"]
            logger.info(f"{filename}: Entity F1={entity_f1:.3f}, Token F1={token_f1:.3f}")

        # Aggregate results
        aggregated = self.aggregate_metrics(file_results)

        return {
            "file_results": file_results,
            "aggregated_metrics": aggregated,
            "evaluation_summary": {
                "total_files_evaluated": len(common_files),
                "files_missing_in_predictions": len(missing_in_pred),
                "files_extra_in_predictions": len(extra_in_pred)
            }
        }

    def print_summary(self, results: Dict):
        """Print evaluation summary to console."""
        agg = results["aggregated_metrics"]

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"Total files evaluated: {agg['total_files']}")
        print(f"Total ground truth entities: {agg['total_gt_entities']}")
        print(f"Total predicted entities: {agg['total_pred_entities']}")
        print(f"Total characters: {agg['total_characters']}")

        print("\nOVERALL ENTITY-LEVEL METRICS (exact span + label match):")
        entity_metrics = agg["overall_entity_metrics"]
        print(f"  Precision: {entity_metrics['precision']:.3f}")
        print(f"  Recall:    {entity_metrics['recall']:.3f}")
        print(f"  F1-Score:  {entity_metrics['f1']:.3f}")
        print(f"  TP: {entity_metrics['tp']}, FP: {entity_metrics['fp']}, FN: {entity_metrics['fn']}")

        print("\nOVERALL TOKEN-LEVEL METRICS (character position + label match):")
        token_metrics = agg["overall_token_metrics"]
        print(f"  Precision: {token_metrics['precision']:.3f}")
        print(f"  Recall:    {token_metrics['recall']:.3f}")
        print(f"  F1-Score:  {token_metrics['f1']:.3f}")
        print(f"  TP: {token_metrics['tp']}, FP: {token_metrics['fp']}, FN: {token_metrics['fn']}")

        print("\nPER-LABEL METRICS (entity-level):")
        label_metrics = agg["label_metrics"]
        for label in sorted(label_metrics.keys()):
            metrics = label_metrics[label]
            print(f"  {label:15s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f} " +
                  f"(TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})")


def main():
    """Main evaluation function."""
    ground_truth_path = "../../../data/original/ground_truth_split/test_norm.json"
    gemini_path = "../../../data/testing/gemini_results/combined_results.json"
    output_path = "../../../data/testing/gemini_results/evaluation_results.json"

    # Check if input files exist
    if not Path(ground_truth_path).exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return 1

    if not Path(gemini_path).exists():
        logger.error(f"Predictions file not found: {gemini_path}")
        return 1

    # Run evaluation
    evaluator = EntityEvaluator(ground_truth_path, gemini_path)
    results = evaluator.run_evaluation()

    # Print summary
    evaluator.print_summary(results)

    # Save detailed results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Detailed evaluation results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())