#!/usr/bin/env python3
"""
Gemini Email PII Anonymization Pipeline

Uses the Google Gemini API to anonymize sensitive entities in German email texts
by replacing them with placeholders (<<LABEL_N>>) and returning the original
lengths. Processes a JSON list of entries and writes combined results to a single JSON output.
"""

import os
import sys
import json
import re
import time
import getpass
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai  # ensure the package is installed and configured

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === hardcoded paths & settings ===
INPUT_JSON = "../../../data/original/ground_truth_split/test_norm.json"
OUTPUT_DIR = "../../../data/testing_gemini_mode/gemini_results"
OUTPUT_FILE_NAME = "llm_span_results.json"
MODEL_NAME = "gemini-2.5-flash"
MAX_WORKERS = 5


class GeminiAnonymizer:
    TARGET_LABELS = [
        'NACHNAME', 'VORNAME', 'STRASSE', 'POSTLEITZAHL', 'WOHNORT',
        'HAUSNUMMER', 'VERTRAGSNUMMER', 'DATUM', 'ZÄHLERNUMMER',
        'TELEFONNUMMER', 'GESENDET_MIT', 'ZAHLUNG', 'FIRMA', 'TITEL',
        'EMAIL', 'ZÄHLERSTAND', 'LINK', 'IBAN', 'BANK', 'BIC', 'FAX'
    ]

    PLACEHOLDER_PATTERN = re.compile(r'<<([A-ZÄÖÜ]+)_(\d+)>>')

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0}
        )

    def _create_prompt(self, text: str) -> str:
        return f"""You are an expert PII anonymization system for German email text. Your task is to replace every sensitive entity from the following 21 label set with placeholders, and then output the original lengths of those entities.

LABELS (only these are allowed):
['NACHNAME', 'VORNAME', 'STRASSE', 'POSTLEITZAHL', 'WOHNORT',
 'HAUSNUMMER', 'VERTRAGSNUMMER', 'DATUM', 'ZÄHLERNUMMER',
 'TELEFONNUMMER', 'GESENDET_MIT', 'ZAHLUNG', 'FIRMA', 'TITEL',
 'EMAIL', 'ZÄHLERSTAND', 'LINK', 'IBAN', 'BANK', 'BIC', 'FAX']

RULES:
1. Replace each detected entity with a placeholder of the form <<LABEL_N>>, where LABEL is one of the above and N is a unique sequential number per label type (starting at 1). E.g., first first name becomes <<VORNAME_1>>, second <<VORNAME_2>>, etc.
2. Do NOT invent or use any labels outside the list above; if a span is ambiguous or does not clearly belong to one of these labels, leave it unchanged.
3. Preserve the rest of the text exactly, only substituting the sensitive spans with their placeholders.
4. After the anonymized text, output immediately (no explanation or markdown) a JSON object with a single key \"lengths\" mapping each placeholder to the character length of the original string it replaced.
5. Use only: (a) the anonymized text with placeholders, then (b) the JSON object. No extra commentary, headers, or formatting.

LABEL DEFINITIONS (for disambiguation):
- NACHNAME: Last names / surnames
- VORNAME: First names / given names
- STRASSE: Street names
- POSTLEITZAHL: Postal codes
- WOHNORT: City / town names
- HAUSNUMMER: House numbers
- VERTRAGSNUMMER: Contract numbers or other sensitive account identifiers not covered elsewhere
- DATUM: Dates in any format
- ZÄHLERNUMMER: Meter numbers
- TELEFONNUMMER: Phone numbers
- GESENDET_MIT: Phrases like \"Gesendet mit ...\" (sent-with messages)
- ZAHLUNG: Payment information or monetary amounts (e.g., \"Euro 103,22\")
- FIRMA: Company names
- TITEL: Honorifics / titles (e.g., Dr., Dipl.)
- EMAIL: Email addresses
- ZÄHLERSTAND: Meter readings (e.g., m3, kWh)
- LINK: URLs / web links (include surrounding angle brackets if present)
- IBAN: Bank account numbers (IBAN)
- BANK: Bank names
- BIC: Bank identifier codes
- FAX: Fax numbers

EXAMPLE:
Original:
\"Hallo John Doe, Ihre Vertragsnummer lautet 123456.\"

Correct output:
Hallo <<VORNAME_1>> <<NACHNAME_1>>, Ihre Vertragsnummer lautet <<VERTRAGSNUMMER_1>>.
{{
  \"lengths\": {{
    \"VORNAME_1\": 4,
    \"NACHNAME_1\": 3,
    \"VERTRAGSNUMMER_1\": 6
  }}
}}

<TEXT_BEGIN>
{text}
<TEXT_END>"""

    def _extract_trailing_json(self, response_text: str) -> Optional[Dict]:
        # Find the last valid JSON object (heuristic)
        idx = len(response_text) - 1
        while idx >= 0 and response_text[idx] != '}':
            idx -= 1
        if idx < 0:
            return None
        for start in range(idx, -1, -1):
            snippet = response_text[start:idx + 1]
            try:
                data = json.loads(snippet)
                return data
            except json.JSONDecodeError:
                continue
        return None

    def _parse_response(self, response_text: str) -> Dict:
        lengths_obj = self._extract_trailing_json(response_text)
        if not lengths_obj or "lengths" not in lengths_obj:
            logger.warning("Could not extract valid 'lengths' JSON from response. Full response:\n%s", response_text)
            lengths = {}
        else:
            lengths = lengths_obj["lengths"]

        if lengths_obj:
            try:
                json_str = json.dumps(lengths_obj, ensure_ascii=False)
                anonymized_text = response_text.rsplit(json_str, 1)[0].strip()
            except Exception:
                last_brace = response_text.rfind('{')
                anonymized_text = response_text[:last_brace].strip() if last_brace != -1 else response_text.strip()
        else:
            anonymized_text = response_text.strip()

        return {
            "anonymized_text": anonymized_text,
            "lengths": lengths
        }

    def _extract_placeholders(self, anonymized_text: str) -> List[Dict]:
        placeholders = []
        for m in self.PLACEHOLDER_PATTERN.finditer(anonymized_text):
            full = m.group(0)
            label = m.group(1)
            index = int(m.group(2))
            placeholders.append({
                "placeholder": full,
                "label": label,
                "index": index,
                "start": m.start(),
                "end": m.end()
            })
        return placeholders

    def call_gemini(self, text: str, max_retries: int = 3) -> Dict:
        prompt = self._create_prompt(text)
        backoff = 1.0
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                if not getattr(response, "text", None):
                    logger.warning("Empty response from Gemini API (attempt %d)", attempt)
                    raise ValueError("Empty response")
                parsed = self._parse_response(response.text)
                placeholders = self._extract_placeholders(parsed["anonymized_text"])
                return {
                    "anonymized_text": parsed["anonymized_text"],
                    "lengths": parsed["lengths"],
                    "placeholders": placeholders
                }
            except Exception as e:
                logger.error("Error calling Gemini API on attempt %d: %s", attempt, e)
                last_exc = e
                time.sleep(backoff)
                backoff *= 2
        logger.error("All retries failed for Gemini API call. Last error: %s", last_exc)
        return {
            "anonymized_text": "",
            "lengths": {},
            "placeholders": []
        }

    def process_json_parallel(self, json_path: str, max_workers: int = 5) -> List[Dict]:
        logger.info(f"Processing JSON file: {json_path}")
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))
        entries = [e for e in data if isinstance(e, dict) and "text" in e and "file" in e]
        results: List[Dict] = []

        def _worker(entry: Dict) -> Dict:
            fname = entry["file"]
            text = entry["text"]
            anonymization = self.call_gemini(text)
            return {
                "file": fname,
                "text_length": len(text),
                "anonymized_text": anonymization["anonymized_text"],
                "lengths": anonymization["lengths"],
                "placeholders": anonymization["placeholders"]
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {executor.submit(_worker, entry): entry for entry in entries}
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                fname = entry.get("file", "unknown.txt")
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error processing entry {fname}: {e}")
                    results.append({
                        "file": fname,
                        "text_length": len(entry.get("text", "")),
                        "anonymized_text": "",
                        "lengths": {},
                        "placeholders": []
                    })
        return results


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Gemini API key: ")
        if not api_key:
            logger.error("No API key provided.")
            sys.exit(1)

    # Ensure output directory exists
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    anonymizer = GeminiAnonymizer(api_key, model_name=MODEL_NAME)
    input_path = Path(INPUT_JSON)
    if not input_path.is_file():
        logger.error("Invalid input JSON path: %s", input_path)
        sys.exit(1)

    results = anonymizer.process_json_parallel(str(input_path), max_workers=MAX_WORKERS)

    combined_file = output_dir / OUTPUT_FILE_NAME
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote combined results ({len(results)} entries) to {combined_file}")


if __name__ == "__main__":
    main()