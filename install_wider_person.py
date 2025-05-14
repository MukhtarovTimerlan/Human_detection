from pathlib import Path
import zipfile

zip_path = Path("WiderPerson.zip")
extract_dir = Path("datasets/WiderPerson")

extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)