import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("stroke_prediction.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    source = cell["source"]
    first_line = source[0].strip() if source else ""
    print(f"Cell {idx} ({cell_type}): {first_line[:120]}")
