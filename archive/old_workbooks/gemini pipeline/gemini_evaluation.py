#!/usr/bin/env python3
"""
Improved span extraction and evaluation for Gemini anonymization results.

The main issues with the current approach:
1. The anchor-based alignment often misaligns when text structure changes
2. Single characters or short tokens get misaligned easily
3. The evaluation doesn't account for semantic correctness of anonymization

This version provides:
1. A more robust span extraction using multiple alignment strategies
2. A semantic evaluation that checks if the right type of content was anonymized
3. Better handling of repeated entities and edge cases
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from difflib import SequenceMatcher
from dataclasses import dataclass
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntitySpan:
    start: int
    end: int
    label: str
    text: str = ""


# Regex patterns for different entity types to validate semantic correctness
VALIDATION_PATTERNS = {
    'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'TELEFONNUMMER': re.compile(r'[\d\s\-\+\(\)\/]{7,}'),
    'POSTLEITZAHL': re.compile(r'\b\d{4,5}\b'),
    'DATUM': re.compile(
        r'\b\d{1,2}[.\-\/]\d{1,2}[.\-\/]\d{2,4}\b|\b\d{4}[.\-\/]\d{1,2}[.\-\/]\d{1,2}\b|\b\d{1,2}\.\s*\w+\s*\d{4}\b'),
    'IBAN': re.compile(r'\b[A-Z]{2}\d{2}[\s\d]{15,32}\b'),
    'VERTRAGSNUMMER': re.compile(r'\b\d{6,}\b'),
    'ZÄHLERNUMMER': re.compile(r'\b\d{8,}\b'),
    'LINK': re.compile(r'https?://[^\s]+|www\.[^\s]+'),
}


class ImprovedSpanExtractor:
    def __init__(self):
        self.placeholder_re = re.compile(r'<<([A-Z_]+)(?:_\d+)?>>')

    def extract_spans_improved(self, original: str, anonymized: str) -> List[Dict]:
        """
        Improved span extraction using multiple strategies:
        1. Direct text replacement detection
        2. Semantic validation of extracted content
        3. Context-aware alignment
        """
        # First, try the direct replacement approach
        spans = self._extract_by_replacement(original, anonymized)

        # If that fails or gives poor results, try alignment-based approach
        if not spans or self._spans_quality_low(spans, original):
            spans = self._extract_by_alignment(original, anonymized)

        # Validate and clean up spans
        spans = self._validate_and_clean_spans(spans, original)

        return spans

    def _extract_by_replacement(self, original: str, anonymized: str) -> List[Dict]:
        """
        Try to extract spans by finding what text was replaced by placeholders.
        This works well when the anonymization is mostly 1:1 replacement.
        """
        spans = []

        # Find all placeholders in anonymized text
        placeholders = list(self.placeholder_re.finditer(anonymized))

        if not placeholders:
            return spans

        # Try to map each placeholder back to original content
        for i, ph_match in enumerate(placeholders):
            label = ph_match.group(1)
            ph_start_anon = ph_match.start()
            ph_end_anon = ph_match.end()

            # Find the context before and after the placeholder
            context_before = anonymized[max(0, ph_start_anon - 20):ph_start_anon]
            context_after = anonymized[ph_end_anon:min(len(anonymized), ph_end_anon + 20)]

            # Find this context in the original text
            orig_pos = self._find_context_in_original(original, context_before, context_after)

            if orig_pos:
                start_orig, end_orig = orig_pos
                # Validate that the extracted content makes sense for this label
                extracted_text = original[start_orig:end_orig]
                if self._validate_entity_content(extracted_text, label):
                    spans.append({
                        'start': start_orig,
                        'end': end_orig,
                        'label': label
                    })

        return spans

    def _find_context_in_original(self, original: str, context_before: str, context_after: str) -> Optional[
        Tuple[int, int]]:
        """
        Find the position in original text where context_before and context_after
        surround some content that was replaced by a placeholder.
        """
        # Clean up contexts
        context_before = context_before.strip()
        context_after = context_after.strip()

        if not context_before and not context_after:
            return None

        # Find all possible positions for the before context
        before_positions = []
        if context_before:
            start = 0
            while True:
                pos = original.find(context_before, start)
                if pos == -1:
                    break
                before_positions.append(pos + len(context_before))
                start = pos + 1
        else:
            before_positions = [0]

        # For each before position, try to find the after context
        for before_pos in before_positions:
            if context_after:
                after_pos = original.find(context_after, before_pos)
                if after_pos != -1:
                    return (before_pos, after_pos)
            else:
                # If no after context, we need to guess the end
                # Look for word boundaries or common separators
                end_pos = self._find_likely_entity_end(original, before_pos)
                return (before_pos, end_pos)

        return None

    def _find_likely_entity_end(self, text: str, start: int) -> int:
        """
        When we don't have after-context, try to find where an entity likely ends.
        """
        # Look for natural word boundaries
        remaining = text[start:]

        # Find next whitespace, punctuation, or special character
        for i, char in enumerate(remaining):
            if char in ' \t\n\r.,;:!?()[]{}"\'-<>':
                return start + i

        # If no natural boundary found, take next 20 characters or end of string
        return min(start + 20, len(text))

    def _extract_by_alignment(self, original: str, anonymized: str) -> List[Dict]:
        """
        Fallback to alignment-based extraction with improvements.
        """
        spans = []
        tokens = self._tokenize_anonymized(anonymized)

        # Use SequenceMatcher for better alignment
        matcher = SequenceMatcher(None, original, anonymized)
        opcodes = matcher.get_opcodes()

        # Track position in original text
        orig_pos = 0
        anon_pos = 0

        for token in tokens:
            if token['type'] == 'TXT':
                # Skip this text in both strings
                text_len = len(token['text'])
                # Find this text in original starting from current position
                found_pos = original.find(token['text'], orig_pos)
                if found_pos != -1:
                    orig_pos = found_pos + text_len
                anon_pos += text_len

            elif token['type'] == 'PLH':
                # This is where we need to extract a span
                placeholder_len = len(token['text'])

                # Find the next text token to know where this span ends
                next_text = self._find_next_text_token(tokens, tokens.index(token))

                if next_text:
                    # Find where the next text appears in original
                    next_pos = original.find(next_text, orig_pos)
                    if next_pos != -1:
                        spans.append({
                            'start': orig_pos,
                            'end': next_pos,
                            'label': token['label']
                        })
                        orig_pos = next_pos
                    else:
                        # Couldn't find next text, make a reasonable guess
                        end_pos = self._find_likely_entity_end(original, orig_pos)
                        spans.append({
                            'start': orig_pos,
                            'end': end_pos,
                            'label': token['label']
                        })
                        orig_pos = end_pos
                else:
                    # Last token, consume rest of original
                    spans.append({
                        'start': orig_pos,
                        'end': len(original),
                        'label': token['label']
                    })
                    orig_pos = len(original)

                anon_pos += placeholder_len

        return spans

    def _tokenize_anonymized(self, text: str) -> List[Dict]:
        """Split anonymized text into text and placeholder tokens."""
        tokens = []
        pos = 0

        for match in self.placeholder_re.finditer(text):
            # Add text before placeholder
            if match.start() > pos:
                tokens.append({
                    'type': 'TXT',
                    'text': text[pos:match.start()]
                })

            # Add placeholder
            tokens.append({
                'type': 'PLH',
                'text': match.group(0),
                'label': match.group(1)
            })

            pos = match.end()

        # Add remaining text
        if pos < len(text):
            tokens.append({
                'type': 'TXT',
                'text': text[pos:]
            })

        return tokens

    def _find_next_text_token(self, tokens: List[Dict], current_index: int) -> Optional[str]:
        """Find the next text token after the current placeholder."""
        for i in range(current_index + 1, len(tokens)):
            if tokens[i]['type'] == 'TXT':
                return tokens[i]['text'].strip()
        return None

    def _validate_entity_content(self, text: str, label: str) -> bool:
        """
        Validate that the extracted text content makes sense for the given label.
        """
        text = text.strip()

        if not text:
            return False

        # Use regex patterns where available
        if label in VALIDATION_PATTERNS:
            return bool(VALIDATION_PATTERNS[label].search(text))

        # Additional heuristic validations
        if label in ['VORNAME', 'NACHNAME']:
            # Names should be alphabetic (possibly with umlauts)
            return bool(re.match(r'^[A-Za-zÄÖÜäöüß\s\-\.]+$', text)) and len(text) > 1

        elif label == 'FIRMA':
            # Company names often contain letters, numbers, and common symbols
            return len(text) > 2 and not text.isdigit()

        elif label == 'STRASSE':
            # Street names typically contain letters and may have numbers
            return bool(re.search(r'[A-Za-zÄÖÜäöüß]', text)) and len(text) > 2

        elif label == 'WOHNORT':
            # City names are typically alphabetic
            return bool(re.match(r'^[A-Za-zÄÖÜäöüß\s\-]+$', text)) and len(text) > 2

        elif label == 'HAUSNUMMER':
            # House numbers are typically short and contain digits
            return bool(re.search(r'\d', text)) and len(text) < 10

        elif label in ['ZAHLUNG', 'ZÄHLERSTAND']:
            # These should contain numbers
            return bool(re.search(r'\d', text))

        # Default: accept if not empty and not too long
        return len(text) < 100

    def _spans_quality_low(self, spans: List[Dict], original: str) -> bool:
        """
        Determine if the extracted spans are of low quality.
        """
        if not spans:
            return True

        # Check for common quality issues
        total_span_length = sum(span['end'] - span['start'] for span in spans)
        text_length = len(original)

        # If spans cover more than 80% of text, probably wrong
        if total_span_length > 0.8 * text_length:
            return True

        # If average span length is too short (< 2 chars), probably wrong
        avg_span_length = total_span_length / len(spans)
        if avg_span_length < 2:
            return True

        # Check for overlapping spans (shouldn't happen with good extraction)
        sorted_spans = sorted(spans, key=lambda x: x['start'])
        for i in range(len(sorted_spans) - 1):
            if sorted_spans[i]['end'] > sorted_spans[i + 1]['start']:
                return True

        return False

    def _validate_and_clean_spans(self, spans: List[Dict], original: str) -> List[Dict]:
        """
        Final validation and cleanup of extracted spans.
        """
        cleaned_spans = []

        for span in spans:
            start = max(0, span['start'])
            end = min(len(original), span['end'])

            if end <= start:
                continue

            extracted_text = original[start:end]

            # Skip if extracted text doesn't make sense for this label
            if not self._validate_entity_content(extracted_text, span['label']):
                logger.warning(f"Skipping invalid span for {span['label']}: '{extracted_text}'")
                continue

            cleaned_spans.append({
                'start': start,
                'end': end,
                'label': span['label']
            })

        # Sort by start position
        cleaned_spans.sort(key=lambda x: x['start'])

        # Remove overlaps by keeping the first span in case of overlap
        final_spans = []
        for span in cleaned_spans:
            if not final_spans or span['start'] >= final_spans[-1]['end']:
                final_spans.append(span)
            else:
                logger.warning(f"Removing overlapping span: {span}")

        return final_spans


class ImprovedEvaluator:
    """
    Improved evaluation that considers both exact matches and semantic correctness.
    """

    def __init__(self):
        self.extractor = ImprovedSpanExtractor()

    def _validate_input_files(self, results: List[Dict], ground_truth: List[Dict]) -> None:
        """
        Sanity-check the structure of the loaded result and ground truth lists. Logs missing keys.
        """
        for item in ground_truth:
            if 'file' not in item:
                logger.warning("Ground truth item missing 'file', skipping it in lookups.")
                continue
            if 'text' not in item:
                logger.warning(f"Ground truth for {item.get('file','<unknown>')} missing 'text'.")

            # Normalize legacy key 'labels' to 'entities' so evaluation downstream works.
            if 'entities' not in item:
                if 'labels' in item:
                    item['entities'] = item['labels']
                    logger.info(f"Normalized ground truth 'labels' to 'entities' for {item['file']}.")
                else:
                    logger.warning(f"Ground truth for {item['file']} missing both 'entities' and 'labels'.")

        for result in results:
            if 'file' not in result:
                logger.warning("Result item missing 'file', skipping it in evaluation.")
            if 'anonymized_text' not in result:
                logger.warning(f"Result for {result.get('file','<unknown>')} missing 'anonymized_text'.")
            if 'entities' not in result:
                logger.info(f"Result for {result.get('file','<unknown>')} has no 'entities' key; initializing to empty list.")
                result['entities'] = []

    def evaluate_results(self, results_file: str, ground_truth_file: str) -> Dict:
        """
        Evaluate the anonymization results against ground truth.
        """
        # Load data
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        # Sanity-check input structure to avoid KeyError later
        self._validate_input_files(results, ground_truth)

        # Create lookup for ground truth by filename
        gt_lookup = {item['file']: item for item in ground_truth}

        # Re-extract spans using improved method
        logger.info("Re-extracting spans with improved method...")
        for result in results:
            if result['file'] in gt_lookup:
                # Get original text from ground truth
                gt_item = gt_lookup[result['file']]
                original_text = gt_item.get('text', '')  # Ground truth has 'text', not 'anonymized_text'

                if original_text and result.get('anonymized_text'):
                    # Re-extract spans using improved method
                    new_spans = self.extractor.extract_spans_improved(
                        original_text,
                        result['anonymized_text']
                    )
                    result['entities'] = new_spans
                    logger.info(f"Re-extracted {len(new_spans)} spans for {result['file']}")

        # Calculate metrics
        return self._calculate_metrics(results, gt_lookup)

    def _reconstruct_original_text(self, anonymized: str, entities: List[Dict]) -> str:
        """
        Attempt to reconstruct original text from anonymized text and entity spans.
        This is a best-effort approach and may not be perfect.
        """
        # This is tricky since we don't have the original entity texts
        # For now, return None to indicate we need the original text from elsewhere
        return None

    def _calculate_metrics(self, results: List[Dict], gt_lookup: Dict) -> Dict:
        """
        Calculate precision, recall, and F1 scores with tolerant span matching.
        """
        def normalize_label(label: str) -> str:
            return unicodedata.normalize("NFC", label.strip().upper())

        def iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
            a0, a1 = span1
            b0, b1 = span2
            inter = max(0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            return inter / union if union > 0 else 0

        total_tp = 0  # True positives
        total_fp = 0  # False positives
        total_fn = 0  # False negatives

        label_metrics = {}
        label_confusion = {}  # true_label -> {pred_label: count}

        for result in results:
            filename = result['file']
            if filename not in gt_lookup:
                continue

            gt_item = gt_lookup[filename]
            if 'entities' not in gt_item:
                logger.warning(f"Ground truth for {filename} missing 'entities'; skipping this file in metrics.")
                continue

            # Build normalized true and predicted entity lists
            true_raw = gt_item['entities']
            pred_raw = result.get('entities') or []

            true_list = []
            for e in true_raw:
                norm = normalize_label(e['label'])
                true_list.append({'start': e['start'], 'end': e['end'], 'label': norm})

            pred_list = []
            for e in pred_raw:
                norm = normalize_label(e['label'])
                pred_list.append({'start': e['start'], 'end': e['end'], 'label': norm})

            matched_true = set()
            matched_pred = set()

            file_tp = 0
            file_fp = 0
            file_fn = 0

            # Exact matches (span + label)
            for ti, t in enumerate(true_list):
                for pi, p in enumerate(pred_list):
                    if ti in matched_true or pi in matched_pred:
                        continue
                    if t['start'] == p['start'] and t['end'] == p['end'] and t['label'] == p['label']:
                        matched_true.add(ti)
                        matched_pred.add(pi)
                        file_tp += 1
                        label_metrics.setdefault(t['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                        label_metrics[t['label']]['tp'] += 1

            # Partial overlap with same label (IoU >= 0.5)
            for ti, t in enumerate(true_list):
                if ti in matched_true:
                    continue
                for pi, p in enumerate(pred_list):
                    if pi in matched_pred:
                        continue
                    if t['label'] == p['label'] and iou((t['start'], t['end']), (p['start'], p['end'])) >= 0.5:
                        matched_true.add(ti)
                        matched_pred.add(pi)
                        file_tp += 1
                        label_metrics.setdefault(t['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                        label_metrics[t['label']]['tp'] += 1
                        break

            # Overlaps with label mismatches (IoU >= 0.5)
            for ti, t in enumerate(true_list):
                if ti in matched_true:
                    continue
                for pi, p in enumerate(pred_list):
                    if pi in matched_pred:
                        continue
                    if iou((t['start'], t['end']), (p['start'], p['end'])) >= 0.5:
                        matched_true.add(ti)
                        matched_pred.add(pi)
                        # label mismatch: count as FN for true label and FP for predicted label
                        file_fn += 1
                        file_fp += 1
                        label_metrics.setdefault(t['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                        label_metrics.setdefault(p['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                        label_metrics[t['label']]['fn'] += 1
                        label_metrics[p['label']]['fp'] += 1
                        # record confusion
                        label_confusion.setdefault(t['label'], {})
                        label_confusion[t['label']].setdefault(p['label'], 0)
                        label_confusion[t['label']][p['label']] += 1
                        break

            # Unmatched true spans -> false negatives
            for ti, t in enumerate(true_list):
                if ti in matched_true:
                    continue
                file_fn += 1
                label_metrics.setdefault(t['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                label_metrics[t['label']]['fn'] += 1

            # Unmatched predicted spans -> false positives
            for pi, p in enumerate(pred_list):
                if pi in matched_pred:
                    continue
                file_fp += 1
                label_metrics.setdefault(p['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                label_metrics[p['label']]['fp'] += 1

            total_tp += file_tp
            total_fp += file_fp
            total_fn += file_fn

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate per-label metrics
        label_results = {}
        for label, counts in label_metrics.items():
            label_precision = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
            label_recall = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
            label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall) if (
                    label_precision + label_recall) > 0 else 0

            label_results[label] = {
                'precision': label_precision,
                'recall': label_recall,
                'f1': label_f1,
                'true_positives': counts['tp'],
                'false_positives': counts['fp'],
                'false_negatives': counts['fn']
            }

        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            },
            'per_label': label_results,
            'label_confusion': label_confusion
        }


def analyze_anonymization_quality(results_file: str, sample_size: int = 10):
    """
    Analyze the quality of anonymization by examining actual examples.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"\n=== ANONYMIZATION QUALITY ANALYSIS ===")
    print(f"Total files processed: {len(results)}")

    # Count placeholders vs original text ratio
    placeholder_counts = []
    for result in results[:sample_size]:
        anonymized = result['anonymized_text']
        if anonymized:
            placeholders = len(re.findall(r'<<[A-Z_]+>>', anonymized))
            placeholder_counts.append(placeholders)

            print(f"\nFile: {result['file']}")
            print(f"Placeholders found: {placeholders}")
            print(f"Text length: {result['text_length']}")
            print(f"Anonymized preview: {anonymized[:200]}...")

    if placeholder_counts:
        avg_placeholders = sum(placeholder_counts) / len(placeholder_counts)
        print(f"\nAverage placeholders per document: {avg_placeholders:.1f}")


if __name__ == "__main__":
    # Example usage
    results_file = "../../../data/testing/gemini_results_2.5/combined_results.json"
    ground_truth_file = "../../../data/original/ground_truth_split/test_norm.json"

    # Analyze quality first
    analyze_anonymization_quality(results_file)

    # Then evaluate if you have ground truth
    evaluator = ImprovedEvaluator()
    metrics = evaluator.evaluate_results(results_file, ground_truth_file)
    print(json.dumps(metrics, indent=2))