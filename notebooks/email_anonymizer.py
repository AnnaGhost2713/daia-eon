
import os
import re
import pandas as pd
from pathlib import Path

# Mapping von Original-Spaltenkategorien auf eure neuen Gruppierungen
GROUPED_PLACEHOLDERS = {
    "NAME": ["VORNAME", "NACHNAME", "TITEL", "SKYPE"],
    "ADRESSE": ["STRASSE", "HAUSNUMMER", "POSTLEITZAHL", "WOHNORT"],
    "VERTRAG": ["VERTRAGSNUMMER", "KUNDENNUMMER", "ZUORDNUNGSNUMMER"],
    "ZAHLUNG": ["ZAHLUNG", "IBAN", "BIC", "FAX"],
    "TECHNISCHE_DATEN": ["ZÄHLERSTAND", "ZÄHLERNUMMER", "VERBRAUCH", "WLV"],
    "KONTAKT": ["TELEFONNUMMER", "EMAIL", "MAIL", "LINK", "GESENDET_MIT", "FIRMENDATEN"],
    "FIRMA": ["FIRMA"],
    "DATUM": ["DATUM"]
}

# Funktion zur Ermittlung des richtigen Platzhalters basierend auf Spaltennamen
def map_column_to_placeholder(col_name):
    upper_col = col_name.upper()
    for placeholder, keywords in GROUPED_PLACEHOLDERS.items():
        for keyword in keywords:
            if keyword in upper_col:
                return f"<{placeholder}>"
    return None  # Ignoriere Spalten, die nicht gemappt werden sollen

# Funktion zur Extraktion der zu ersetzenden Werte aus einer Zeile
def extract_replacements(row):
    replacements = []
    for col in row.index:
        if pd.isna(row[col]):
            continue
        placeholder = map_column_to_placeholder(col)
        if placeholder:
            value = str(row[col]).strip()
            if value:
                replacements.append((re.escape(value), placeholder))
    return replacements

# Hauptfunktion zur Anonymisierung
def generate_anonymization_script(df, email_folder_path, output_folder_path):
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    for index, row in df.iterrows():
        file_name = row["TextFile"]
        email_path = os.path.join(email_folder_path, file_name)

        try:
            with open(email_path, "r", encoding="utf-8") as file:
                email_text = file.read()
        except FileNotFoundError:
            print(f"Datei nicht gefunden: {file_name}")
            continue

        replacements = extract_replacements(row)

        for pattern, placeholder in replacements:
            email_text = re.sub(pattern, placeholder, email_text, flags=re.IGNORECASE)

        output_path = os.path.join(output_folder_path, f"{file_name}")
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(email_text)

    return f"Anonymisierte E-Mails gespeichert in: {output_folder_path}"
