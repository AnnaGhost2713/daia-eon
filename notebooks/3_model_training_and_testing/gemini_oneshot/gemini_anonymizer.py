#!/usr/bin/env python3
"""
Gemini Email Anonymization Pipeline

A Python pipeline that uses the Google Gemini API to detect and anonymize
sensitive entities in German email texts using 21 custom labels, in parallel,
then writes all results into a single combined JSON file.
"""

import os
import sys
import json
import re
import getpass
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EntitySpan:
    """Represents an entity span with start, end positions and label."""
    start: int
    end: int
    label: str


class GeminiAnonymizer:
    """Main class for handling Gemini API-based email anonymization."""
    TARGET_LABELS = [
        'NACHNAME', 'VORNAME', 'STRASSE', 'POSTLEITZAHL', 'WOHNORT',
        'HAUSNUMMER', 'VERTRAGSNUMMER', 'DATUM', 'ZÄHLERNUMMER',
        'TELEFONNUMMER', 'GESENDET_MIT', 'ZAHLUNG', 'FIRMA', 'TITEL',
        'EMAIL', 'ZÄHLERSTAND', 'LINK', 'IBAN', 'BANK', 'BIC', 'FAX'
    ]

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize the Gemini anonymizer with API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _create_anonymization_prompt(self, text: str) -> str:
        labels_str = "', '".join(self.TARGET_LABELS)
        return f"""You are an expert text anonymization system. Your task is to identify and replace sensitive entities in German email text with anonymization placeholders.

IMPORTANT RULES:
1. Only replace entities that match these 21 labels: ['{labels_str}']
2. Replace each detected entity with <<LABEL>> (e.g., <<VORNAME>>, <<NACHNAME>>)
3. Preserve the original text structure, formatting, and whitespace exactly
4. Do not modify any text that doesn't contain sensitive entities
5. Return ONLY the anonymized text, no explanations or metadata

LABEL DEFINITIONS:
- NACHNAME: Last names/surnames
- VORNAME: First names/given names
- STRASSE: Street names
- POSTLEITZAHL: Postal codes
- WOHNORT: City/town names
- HAUSNUMMER: House numbers
- VERTRAGSNUMMER: Contract numbers, all other sensitive numbers that are not defined in other categories
- DATUM: Dates in any format
- ZÄHLERNUMMER: Meter numbers
- TELEFONNUMMER: Phone numbers
- GESENDET_MIT: "Sent with" messages
- ZAHLUNG: Payment information
- FIRMA: Company names
- TITEL: Titles (e.g., Dr., Dipl.)
- EMAIL: Email addresses
- ZÄHLERSTAND: Meter readings, could be in m3 or kWh or without units
- LINK: URLs and web links
- IBAN: Bank account numbers
- BANK: Bank names
- BIC: Bank identifier codes
- FAX: Fax numbers

TEXT TO ANONYMIZE:
{text}

ANONYMIZED TEXT:"""

    def call_gemini_api(self, text: str) -> str:
        """Call the Gemini API to anonymize the given text."""
        try:
            prompt = self._create_anonymization_prompt(text)
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return text

    def extract_spans(self, original: str, anonymized: str) -> List[Dict]:
        """
        Use difflib.SequenceMatcher to align original ↔ anonymized
        and extract all replace‐blocks that correspond to <<LABEL>>.
        """
        spans: List[Dict] = []
        sm = SequenceMatcher(None, original, anonymized)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace":
                segment = anonymized[j1:j2]
                m = re.fullmatch(r"<<([A-Z_]+)>>", segment)
                if m:
                    spans.append({
                        "start": i1,
                        "end":   i2,
                        "label": m.group(1)
                    })
        return spans

    def process_file(self, path: str) -> Dict:
        """Process a single .txt file and return its result dict."""
        logger.info(f"Processing file: {path}")
        original = Path(path).read_text(encoding='utf-8')
        anonymized = self.call_gemini_api(original)
        spans = self.extract_spans(original, anonymized)
        return {
            "file": Path(path).name,
            "anonymized_text": anonymized,
            "labels": spans,
            "success": True
        }

    def process_json_file(self, json_path: str) -> List[Dict]:
        """Process a JSON file of entries; return list of result dicts."""
        logger.info(f"Processing JSON file: {json_path}")
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))
        results: List[Dict] = []
        for entry in data:
            text = entry.get("text", "")
            fname = entry.get("file", "unknown.txt")
            anonymized = self.call_gemini_api(text)
            spans = self.extract_spans(text, anonymized)
            results.append({
                "file": fname,
                "anonymized_text": anonymized,
                "labels": spans,
                "success": True
            })
        return results

    def process_directory_parallel(
        self,
        dir_path: str,
        max_workers: int = 5
    ) -> List[Dict]:
        """Process all .txt files in a directory in parallel."""
        files = list(Path(dir_path).glob("*.txt"))
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, str(fp)): fp
                for fp in files
            }
            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error processing {fp.name}: {e}")
                    results.append({
                        "file": fp.name,
                        "anonymized_text": None,
                        "labels": [],
                        "success": False,
                        "error": str(e)
                    })
        return results

    def process_json_parallel(
        self,
        json_path: str,
        max_workers: int = 5
    ) -> List[Dict]:
        """Process JSON entries in parallel."""
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))
        entries = [e for e in data if isinstance(e, dict) and "text" in e and "file" in e]
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(self._process_entry, entry): entry
                for entry in entries
            }
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    fname = entry.get("file", "unknown.txt")
                    logger.error(f"Error on entry {fname}: {e}")
                    results.append({
                        "file": fname,
                        "anonymized_text": None,
                        "labels": [],
                        "success": False,
                        "error": str(e)
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
            "anonymized_text": anonymized,
            "labels": spans,
            "success": True
        }


def main():
    # 1) Load API key (prompt if needed)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Gemini API key: ")
        if not api_key:
            logger.error("No API key provided.")
            return 1

    # 2) Hard-coded paths and settings
    input_path = "../../../data/original/ground_truth_split/test_norm.json"
    output_dir = "../../../data/testing/gemini_results"
    model_name = "gemini-1.5-flash"
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
    combined_file = Path(output_dir) / "combined_results.json"
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote combined results ({len(results)} entries) to {combined_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())