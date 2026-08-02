import json, io, sys
from pathlib import Path

nb_path = Path(r"C:\Users\Asus\Downloads\WaveMamba_DG_end2end.ipynb")
out_path = Path(r"D:\AI-Projects\WaveMamba\run.py")

nb = json.loads(nb_path.read_text(encoding="utf-8"))

header = '''# Auto-generated from WaveMamba_DG_end2end.ipynb for headless CLI execution.
import os
ROOT = r"D:\\AI-Projects\\WaveMamba"
os.environ.setdefault("TMP", os.path.join(ROOT, "tmp"))
os.environ.setdefault("TEMP", os.path.join(ROOT, "tmp"))
os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(ROOT, "kagglehub_cache"))
os.environ.setdefault("KAGGLE_CONFIG_DIR", r"C:\\Users\\Asus\\.kaggle")
os.chdir(ROOT)
import matplotlib
matplotlib.use("Agg")
print(">>> run.py starting in", os.getcwd(), flush=True)

'''

parts = [header]
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    parts.append("\n# ==== cell ====\n" + src + "\n")

code = "".join(parts)

# Windows-safe: DataLoader workers must be 0 when run as a plain script (spawn).
code = code.replace("NUM_WORKERS = 2", "NUM_WORKERS = 0")

out_path.write_text(code, encoding="utf-8")
print("wrote", out_path, "chars:", len(code))
print("NUM_WORKERS patched:", "NUM_WORKERS = 0" in code)
