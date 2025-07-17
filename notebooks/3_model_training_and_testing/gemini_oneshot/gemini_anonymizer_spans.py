#!/usr/bin/env python3
"""
Gemini Email PII Detection Pipeline

A Python pipeline that uses the Google Gemini API to detect sensitive entities
in German email texts using 21 custom labels, in parallel, then writes all
results into a single combined JSON file with span information.
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

import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeminiDetector:
    """Main class for handling Gemini API-based PII detection."""
    TARGET_LABELS = [
        'NACHNAME', 'VORNAME', 'STRASSE', 'POSTLEITZAHL', 'WOHNORT',
        'HAUSNUMMER', 'VERTRAGSNUMMER', 'DATUM', 'ZÄHLERNUMMER',
        'TELEFONNUMMER', 'GESENDET_MIT', 'ZAHLUNG', 'FIRMA', 'TITEL',
        'EMAIL', 'ZÄHLERSTAND', 'LINK', 'IBAN', 'BANK', 'BIC', 'FAX'
    ]

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize the Gemini detector with API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0}
        )

    def _create_detection_prompt(self, text: str) -> str:
        """Create a prompt for PII detection that returns JSON spans."""
        labels_str = "', '".join(self.TARGET_LABELS)
        return f"""You are an expert PII detection system. Your task is to identify sensitive entities in German email text and return their exact positions.

IMPORTANT RULES:
1. Only detect entities that match these 21 labels: ['{labels_str}']
2. Return ONLY a JSON object with this exact structure: {{"entities":[{{"start":0,"end":5,"label":"VORNAME"}}, ...]}}
3. Use 0-based indexing where start is inclusive and end is exclusive
4. Include EVERY occurrence of each entity type separately (do NOT collapse duplicates)
5. Ensure all start/end positions are accurate for the exact text between <TEXT_BEGIN> and <TEXT_END>
6. If no entities found, return {{"entities":[]}}
7. Return ONLY the JSON object, no markdown, no explanations

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

<TEXT_BEGIN>
{text}
<TEXT_END>"""

    def call_gemini_api(self, text: str) -> List[Dict]:
        """Call the Gemini API to detect PII entities in the given text."""
        try:
            prompt = self._create_detection_prompt(text)
            response = self.model.generate_content(prompt)

            if not response.text:
                logger.warning("Empty response from Gemini API")
                return []

            # Extract JSON from response (robust parsing)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in Gemini response")
                return []

            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Validate and filter entities
            entities = data.get("entities", [])
            valid_entities = []
            text_length = len(text)

            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                start = entity.get("start")
                end = entity.get("end")
                label = entity.get("label")

                # Validate entity structure and bounds
                if (isinstance(start, int) and isinstance(end, int) and
                        isinstance(label, str) and label in self.TARGET_LABELS and
                        0 <= start < end <= text_length):
                    valid_entities.append({
                        "start": start,
                        "end": end,
                        "label": label
                    })
                else:
                    logger.warning(f"Invalid entity detected: {entity}")

            # Sort by start position
            valid_entities.sort(key=lambda x: x["start"])
            return valid_entities

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return []

    def process_file(self, path: str) -> Dict:
        """Process a single .txt file and return its result dict."""
        logger.info(f"Processing file: {path}")
        text = Path(path).read_text(encoding='utf-8')
        entities = self.call_gemini_api(text)
        return {
            "file": Path(path).name,
            "text_length": len(text),
            "entities": entities
        }

    def process_json_file(self, json_path: str) -> List[Dict]:
        """Process a JSON file of entries; return list of result dicts."""
        logger.info(f"Processing JSON file: {json_path}")
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))
        results: List[Dict] = []
        for entry in data:
            text = entry.get("text", "")
            fname = entry.get("file", "unknown.txt")
            entities = self.call_gemini_api(text)
            results.append({
                "file": fname,
                "text_length": len(text),
                "entities": entities
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
                        "text_length": 0,
                        "entities": []
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
                        "text_length": len(entry.get("text", "")),
                        "entities": []
                    })
        return results

    def _process_entry(self, entry: Dict) -> Dict:
        """Helper for JSON parallel processing."""
        fname = entry["file"]
        text = entry["text"]
        entities = self.call_gemini_api(text)
        return {
            "file": fname,
            "text_length": len(text),
            "entities": entities
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

    # 4) Initialize detector
    detector = GeminiDetector(api_key, model_name)

    # 5) Run in parallel and collect results
    inp = Path(input_path)
    if inp.is_file() and inp.suffix.lower() == ".json":
        results = detector.process_json_parallel(str(inp), max_workers=max_workers)
    elif inp.is_dir():
        results = detector.process_directory_parallel(str(inp), max_workers=max_workers)
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