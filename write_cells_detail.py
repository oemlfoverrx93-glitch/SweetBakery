import json

with open("stroke_prediction.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("cells_detail.txt", "w", encoding="utf-8") as out:
    for idx, cell in enumerate(nb["cells"]):
        out.write(f"=== CELL {idx} ({cell['cell_type']}) ===\n")
        source = "".join(cell["source"])
        out.write(source)
        out.write("\n\n")
