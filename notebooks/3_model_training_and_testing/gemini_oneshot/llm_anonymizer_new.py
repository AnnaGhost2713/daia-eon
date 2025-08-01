#!/usr/bin/env python3
"""
Simplified Gemini Email Anonymization Pipeline

This version reads .txt files, sends them to Gemini for anonymization,
and saves the filename + anonymized text into a combined JSON file.

Output format:
[
  {
    "file": "example.txt",
    "anonymized_text": "<<VORNAME>> ... <<NACHNAME>> ..."
  },
  ...
]
"""

import os
import sys
import json
import getpass
import logging
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GeminiAnonymizer:
    TARGET_LABELS = [
        'NACHNAME', 'VORNAME', 'STRASSE', 'POSTLEITZAHL', 'WOHNORT',
        'HAUSNUMMER', 'VERTRAGSNUMMER', 'DATUM', 'ZÄHLERNUMMER',
        'TELEFONNUMMER', 'GESENDET_MIT', 'ZAHLUNG', 'FIRMA', 'TITEL',
        'EMAIL', 'ZÄHLERSTAND', 'LINK', 'IBAN', 'BANK', 'BIC', 'FAX'
    ]

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        try:
            self.model = genai.GenerativeModel(model_name, generation_config={"temperature": 0})
        except Exception:
            self.model = genai.GenerativeModel(model_name)

    def _create_anonymization_prompt(self, text: str) -> str:
        labels_str = "', '".join(self.TARGET_LABELS)
        return f"""You are an expert text anonymization system. Your task is to identify and replace sensitive entities in German email text with anonymization placeholders.

        IMPORTANT RULES:
        1. Only replace entities that match these 21 labels: ['{labels_str}']
        2. Replace *each* detected entity occurrence with <<LABEL_N>> (e.g., <<VORNAME_1>>, <<NACHNAME_1>>). If the same type appears multiple times, repeat the placeholder each time and increase the index.
        3. Preserve the original text structure, formatting, and whitespace exactly where possible.
        4. Do not modify any text that doesn't contain sensitive entities.
        5. Return ONLY the anonymized text, no explanations or metadata.

        LABEL DEFINITIONS:
        - NACHNAME: Last names/surnames
        - VORNAME: First names/given names
        - STRASSE: Street names
        - POSTLEITZAHL: Postal codes
        - WOHNORT: City/town names
        - HAUSNUMMER: House numbers
        - VERTRAGSNUMMER: Contract numbers, all other sensitive numbers that are not defined in other categories
        - DATUM: Dates in any format - also just as plain text, e.g. "Juni" or "August"
        - ZÄHLERNUMMER: Meter numbers, e.g., "1LOG0065054693"
        - TELEFONNUMMER: Phone numbers
        - GESENDET_MIT: "Sent with" messages
        - ZAHLUNG: Payment information, e.g., "110,0€" 
        - FIRMA: Company names
        - TITEL: Titles (e.g., Dr., Dipl.)
        - EMAIL: Email addresses
        - ZÄHLERSTAND: Meter readings, could be in m3 or kWh or without units
        - LINK: URLs and web links
        - IBAN: Bank account numbers
        - BANK: Bank names
        - BIC: Bank identifier codes
        - FAX: Fax numbers
        
        EXAMPLE:
        Input: "Hallo liebes Eon Team,\nes geht um die Vertragsnummer 406027919.\nBei der Einrichtung meines neuen Vertrages wurde leider die Überweisung als\nZahlungsart gewählt von dem jungen Kollegen an der Wohnungstür. Ich würde\nes gerne wieder per Lastschrift abbuchen lassen, um mir den Stress zu\nersparen.\nVerbraucherstelle ist weiterhin die Gertzgasse 2 in 17389 Anklam.\nGruß Berthold Huhn\n"
        Output: "Hallo liebes Eon Team,\nes geht um die Vertragsnummer <<VERTRAGSNUMMER_1>>.\nBei der Einrichtung meines neuen Vertrages wurde leider die Überweisung als\nZahlungsart gewählt von dem jungen Kollegen an der Wohnungstür. Ich würde\nes gerne wieder per Lastschrift abbuchen lassen, um mir den Stress zu\nersparen.\nVerbraucherstelle ist weiterhin die <<STRASSE_1>> <<HAUSNUMMER_1>> in <<POSTLEITZAHL_1>> <<WOHNORT_1>>.\nGruß <<VORNAME_1>> <<NACHNAME_1>>\n"

        TEXT TO ANONYMIZE:
        {text}

        ANONYMIZED TEXT:"""

    def call_gemini_api(self, text: str) -> str:
        try:
            prompt = self._create_anonymization_prompt(text)
            response = self.model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text is not None:
                return response.text.rstrip("\r")
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
        return text

    def process_file(self, path: str) -> Dict:
        logger.info(f"Processing file: {path}")
        original = Path(path).read_text(encoding='utf-8')
        anonymized = self.call_gemini_api(original)
        return {
            "file": Path(path).name,
            "anonymized_text": anonymized,
        }

    def process_directory_parallel(self, dir_path: str, max_workers: int = 5) -> List[Dict]:
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
                        "anonymized_text": None,
                        "error": str(e),
                    })
        return results

    def process_json_file_parallel(self, path: str, max_workers: int = 5) -> List[Dict]:
        logger.info(f"Processing JSON file: {path}")
        try:
            content = Path(path).read_text(encoding='utf-8')
            data = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to read or parse JSON file {path}: {e}")
            return []
        if not isinstance(data, list):
            logger.error(f"Expected a list in JSON file {path}, got {type(data)}")
            return []
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            for entry in data:
                file_name = entry.get("file", "<unknown>")
                original_text = entry.get("text", "")
                future = executor.submit(self.call_gemini_api, original_text)
                future_to_file[future] = file_name
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    anonymized = future.result()
                    results.append({
                        "file": file_name,
                        "anonymized_text": anonymized,
                    })
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    results.append({
                        "file": file_name,
                        "anonymized_text": None,
                        "error": str(e),
                    })
        return results


def main():
    print("Start Gemini Anonymization Process")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Gemini API key: ")
        if not api_key:
            logger.error("No API key provided.")
            return 1

    input_path = "../../../data/original/ground_truth_split/test_norm.json"
    output_dir = "../../../data/testing/gemini_results/anonymized_text_results"
    model_name = "gemini-1.5-flash"
    max_workers = 5

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    anonymizer = GeminiAnonymizer(api_key, model_name)
    if Path(input_path).is_file() and input_path.lower().endswith(".json"):
        results = anonymizer.process_json_file_parallel(input_path, max_workers=max_workers)
    elif Path(input_path).is_dir():
        results = anonymizer.process_directory_parallel(input_path, max_workers=max_workers)
    else:
        logger.error(f"Input path {input_path} is neither a directory nor a JSON file.")
        return 1

    combined_file = Path(output_dir) / "combined_results_1.5.json"
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote combined results ({len(results)} entries) to {combined_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())