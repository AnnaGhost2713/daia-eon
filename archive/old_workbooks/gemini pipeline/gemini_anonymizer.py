#!/usr/bin/env python3
"""
Gemini Email Anonymization Pipeline (Anchor-Align Version)

This version sends raw text to Gemini, which returns anonymized text where
PII spans are replaced by placeholders like <<VORNAME>>. We then recover the
character spans of the original PII by aligning the anonymized text back to
the original using a forward, monotonic, anchor-based algorithm.

Features:
- Handles repeated entities (each <<LABEL>> becomes its own span).
- More robust to whitespace changes and mild rephrasing than raw difflib.
- Outputs one combined JSON list with records containing:
    {
      "file": <filename>,
      "text_length": <len(original_text)>,
      "anonymized_text": <Gemini_output>,
      "entities": [{"start":int,"end":int,"label":str}, ...]
    }

NOTE: The recovered spans are heuristic; for strict benchmarking prefer an LLM
mode that returns spans directly.
"""

import os
import sys
import json
import re
import getpass
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(r"<<([A-Z_]+)(?:_[0-9]+)?>>")  # accept optional numeric suffix
NBSP = "\xa0"


@dataclass
class EntitySpan:
    start: int
    end: int
    label: str


# ---------------------------------------------------------------------------
# Alignment utilities
# ---------------------------------------------------------------------------

def _normalize_ws(s: str) -> str:
    """Light normalization for fallback search (replace NBSP with space)."""
    return s.replace(NBSP, " ")


def _find_forward(haystack: str, needle: str, start: int) -> int:
    """Exact substring search from `start`; return index or -1."""
    return haystack.find(needle, start)


def _find_forward_norm(haystack: str, needle: str, start: int) -> int:
    """Substring search under light whitespace normalization; maps index via length diff.

    Because NBSP->space preserves string length, we can reuse index.
    If you extend normalization, adjust mapping logic!
    """
    hs = _normalize_ws(haystack)
    nd = _normalize_ws(needle)
    return hs.find(nd, start)


def _find_forward_fuzzy(haystack: str, needle: str, start: int, min_ratio: float = 0.6) -> int:
    """Fuzzy forward search using difflib longest match; returns start index or -1."""
    if not needle:
        return start
    sub = haystack[start:]
    if not sub:
        return -1
    sm = SequenceMatcher(None, needle, sub)
    m = sm.find_longest_match(0, len(needle), 0, len(sub))
    if m.size == 0:
        return -1
    ratio = m.size / max(len(needle), 1)
    if ratio < min_ratio:
        return -1
    return start + m.b  # map back into full haystack


def _tokenize_anonymized(text: str) -> List[Dict]:
    """Split anonymized text into [{'type':'TXT'|'PLH','text':..., 'label':...}] tokens."""
    tokens: List[Dict] = []
    pos = 0
    for m in PLACEHOLDER_RE.finditer(text):
        if m.start() > pos:
            tokens.append({"type": "TXT", "text": text[pos:m.start()]})
        tokens.append({"type": "PLH", "text": m.group(0), "label": m.group(1)})
        pos = m.end()
    if pos < len(text):
        tokens.append({"type": "TXT", "text": text[pos:]})
    return tokens


def extract_spans_anchor(
    original: str,
    anonymized: str,
    *,
    fuzzy: bool = True,
    min_fuzzy_ratio: float = 0.6,
) -> List[Dict]:
    """Recover PII spans in `original` from anonymized Gemini output.

    Algorithm (monotonic forward alignment):
      - Tokenize anonymized into literal text + placeholders.
      - Maintain pointer `o_ptr` in original.
      - For each token:
          * TXT: align this literal in original at/after o_ptr -> advance o_ptr.
          * PLH: span start = o_ptr; find next literal token (suffix) to anchor span end.
                  If suffix aligns at index `idx`, span end = idx.
                  Else span end = len(original) (consume rest).
                  Record span {start,end,label}.  Do not advance o_ptr yet; the next TXT
                  token will align and advance.  To prevent loops when suffix missing,
                  set o_ptr = span_end.
    """
    tokens = _tokenize_anonymized(anonymized)
    spans: List[Dict] = []
    n = len(original)
    o_ptr = 0  # where we are in original

    # Pre-scan indexes of TXT tokens for quick lookahead
    txt_indexes = [i for i, t in enumerate(tokens) if t["type"] == "TXT"]

    def align_literal(lit: str, start_idx: int) -> Optional[int]:
        """Return end index in original after aligning literal starting at start_idx.
        Returns None if alignment fails."""
        if not lit:
            return start_idx
        idx = _find_forward(original, lit, start_idx)
        if idx == -1:
            idx = _find_forward_norm(original, lit, start_idx)
        if idx == -1 and fuzzy:
            idx = _find_forward_fuzzy(original, lit, start_idx, min_ratio=min_fuzzy_ratio)
        if idx == -1:
            return None
        return idx + len(lit)

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok["type"] == "TXT":
            lit = tok["text"]
            new_ptr = align_literal(lit, o_ptr)
            if new_ptr is None:
                # best-effort resync: search first char
                if lit:
                    ch = lit[0]
                    idx = original.find(ch, o_ptr)
                    if idx != -1:
                        new_ptr = idx + len(lit)
                        logger.warning("Literal alignment fallback on char match.")
                    else:
                        logger.warning("Failed to align literal; forcing end-of-original.")
                        new_ptr = n
                else:
                    new_ptr = o_ptr
            o_ptr = min(new_ptr, n)
            i += 1
            continue

        # Placeholder token
        label = tok["label"]
        span_start = o_ptr

        # Find the next TXT token to use as suffix anchor
        suffix_txt = None
        for j in range(i+1, len(tokens)):
            if tokens[j]["type"] == "TXT":
                suffix_txt = tokens[j]["text"]
                break

        if suffix_txt is not None:
            # attempt to locate suffix in original starting at o_ptr
            idx = _find_forward(original, suffix_txt, o_ptr)
            if idx == -1:
                idx = _find_forward_norm(original, suffix_txt, o_ptr)
            if idx == -1 and fuzzy:
                idx = _find_forward_fuzzy(original, suffix_txt, o_ptr, min_ratio=min_fuzzy_ratio)
            if idx == -1:
                logger.warning("Suffix literal not found; expanding placeholder to end-of-original.")
                span_end = n
            else:
                span_end = idx  # end at start of suffix
        else:
            # no suffix -> consume rest of original
            span_end = n

        if span_end < span_start:
            logger.warning("Correcting negative-length span.")
            span_end = span_start

        spans.append({"start": span_start, "end": span_end, "label": label})

        # Advance o_ptr to span_end; the suffix TXT (if any) will advance further in its own turn
        o_ptr = min(span_end, n)
        i += 1

    # Clip & drop empties
    clean: List[Dict] = []
    for s in spans:
        st = max(0, min(n, s["start"]))
        en = max(st, min(n, s["end"]))
        if en > st:
            clean.append({"start": st, "end": en, "label": s["label"]})
        else:
            logger.warning(f"Dropping zero-length span: {s}")
    # Sort by start for consistency
    clean.sort(key=lambda x: x["start"])
    return clean


# ---------------------------------------------------------------------------
# Gemini anonymizer wrapper
# ---------------------------------------------------------------------------
class GeminiAnonymizer:
    """Handle Gemini API-based email anonymization and span recovery."""

    TARGET_LABELS = [
        'NACHNAME', 'VORNAME', 'STRASSE', 'POSTLEITZAHL', 'WOHNORT',
        'HAUSNUMMER', 'VERTRAGSNUMMER', 'DATUM', 'ZÄHLERNUMMER',
        'TELEFONNUMMER', 'GESENDET_MIT', 'ZAHLUNG', 'FIRMA', 'TITEL',
        'EMAIL', 'ZÄHLERSTAND', 'LINK', 'IBAN', 'BANK', 'BIC', 'FAX'
    ]

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        # Force low temperature for determinism if supported; fall back gracefully
        try:
            self.model = genai.GenerativeModel(model_name, generation_config={"temperature": 0})
        except Exception:  # older SDKs may not support generation_config here
            self.model = genai.GenerativeModel(model_name)

    def _create_anonymization_prompt(self, text: str) -> str:
        labels_str = "', '".join(self.TARGET_LABELS)
        # Encourage model to output placeholders *for each occurrence*.
        return f"""You are an expert text anonymization system. Your task is to identify and replace sensitive entities in German email text with anonymization placeholders.\n\nIMPORTANT RULES:\n1. Only replace entities that match these 21 labels: ['{labels_str}']\n2. Replace *each* detected entity occurrence with <<LABEL>> (e.g., <<VORNAME>>, <<NACHNAME>>). If the same type appears multiple times, repeat the placeholder each time.\n3. Preserve the original text structure, formatting, and whitespace exactly where possible.\n4. Do not modify any text that doesn't contain sensitive entities.\n5. Return ONLY the anonymized text, no explanations or metadata.\n\nLABEL DEFINITIONS:\n- NACHNAME: Last names/surnames\n- VORNAME: First names/given names\n- STRASSE: Street names\n- POSTLEITZAHL: Postal codes\n- WOHNORT: City/town names\n- HAUSNUMMER: House numbers\n- VERTRAGSNUMMER: Contract numbers, all other sensitive numbers that are not defined in other categories\n- DATUM: Dates in any format\n- ZÄHLERNUMMER: Meter numbers\n- TELEFONNUMMER: Phone numbers\n- GESENDET_MIT: \"Sent with\" messages\n- ZAHLUNG: Payment information\n- FIRMA: Company names\n- TITEL: Titles (e.g., Dr., Dipl.)\n- EMAIL: Email addresses\n- ZÄHLERSTAND: Meter readings, could be in m3 or kWh or without units\n- LINK: URLs and web links\n- IBAN: Bank account numbers\n- BANK: Bank names\n- BIC: Bank identifier codes\n- FAX: Fax numbers\n\nTEXT TO ANONYMIZE:\n{text}\n\nANONYMIZED TEXT:"""

    def call_gemini_api(self, text: str) -> str:
        """Call the Gemini API to anonymize the given text. On error, return original."""
        try:
            prompt = self._create_anonymization_prompt(text)
            response = self.model.generate_content(prompt)
            # Avoid .strip() which can remove leading/trailing newlines that may matter for alignment.
            if response and hasattr(response, 'text') and response.text is not None:
                # Soft trim trailing carriage returns; keep leading spacing
                return response.text.rstrip("\r")
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
        # fallback
        return text

    # Override extract_spans with anchor algorithm
    def extract_spans(self, original: str, anonymized: str) -> List[Dict]:
        return extract_spans_anchor(original, anonymized)

    def process_file(self, path: str) -> Dict:
        """Process a single .txt file and return its result dict."""
        logger.info(f"Processing file: {path}")
        original = Path(path).read_text(encoding='utf-8')
        anonymized = self.call_gemini_api(original)
        spans = self.extract_spans(original, anonymized)
        return {
            "file": Path(path).name,
            "text_length": len(original),
            "anonymized_text": anonymized,
            "entities": spans,
        }

    def process_json_file(self, json_path: str) -> List[Dict]:
        """Process a JSON file of entries; return list of result dicts."""
        logger.info(f"Processing JSON file: {json_path}")
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))
        results: List[Dict] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text", "")
            fname = entry.get("file", "unknown.txt")
            anonymized = self.call_gemini_api(text)
            spans = self.extract_spans(text, anonymized)
            results.append({
                "file": fname,
                "text_length": len(text),
                "anonymized_text": anonymized,
                "entities": spans,
            })
        return results

    def process_directory_parallel(self, dir_path: str, max_workers: int = 5) -> List[Dict]:
        """Process all .txt files in a directory in parallel."""
        files = list(Path(dir_path).glob("*.txt"))
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, str(fp)): fp for fp in files}
            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error processing {fp.name}: {e}")
                    results.append({
                        "file": fp.name,
                        "text_length": 0,
                        "anonymized_text": None,
                        "entities": [],
                        "error": str(e),
                    })
        return results

    def process_json_parallel(self, json_path: str, max_workers: int = 5) -> List[Dict]:
        """Process JSON entries in parallel."""
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))
        entries = [e for e in data if isinstance(e, dict) and "text" in e and "file" in e]
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {executor.submit(self._process_entry, entry): entry for entry in entries}
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    fname = entry.get("file", "unknown.txt")
                    logger.error(f"Error on entry {fname}: {e}")
                    results.append({
                        "file": fname,
                        "text_length": len(entry.get("text", "")),
                        "anonymized_text": None,
                        "entities": [],
                        "error": str(e),
                    })
        return results

    def _process_entry(self, entry: Dict) -> Dict:
        """Helper for JSON parallel processing."""
        fname = entry["file"]
        text = entry["text"]
        anonymized = self.call_gemini_api(text)
        spans = self.extract_spans(text, anonymized)
        return {
            "file": fname,
            "text_length": len(text),
            "anonymized_text": anonymized,
            "entities": spans,
        }


def main():
    # 1) Load API key (prompt if needed)
    print("Start Gemini Anonymization Process")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Gemini API key: ")
        if not api_key:
            logger.error("No API key provided.")
            return 1

    # 2) Hard-coded paths and settings (edit as needed)
    input_path = "../../../data/original/ground_truth_split/test_norm.json"
    output_dir = "../../../data/testing_gemini_mode/gemini_results_2.5"
    model_name = "gemini-2.5-flash"
    max_workers = 5

    # 3) Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 4) Initialize anonymizer
    anonymizer = GeminiAnonymizer(api_key, model_name)

    # 5) Run in parallel and collect results
    inp = Path(input_path)
    if inp.is_file() and inp.suffix.lower() == ".json":
        results = anonymizer.process_json_parallel(str(inp), max_workers=max_workers)
    elif inp.is_dir():
        results = anonymizer.process_directory_parallel(str(inp), max_workers=max_workers)
    else:
        logger.error(f"Invalid input path: {inp}")
        return 1

    # 6) Write a single combined JSON
    combined_file = Path(output_dir) / "combined_results_2.5.json"
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote combined results ({len(results)} entries) to {combined_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
