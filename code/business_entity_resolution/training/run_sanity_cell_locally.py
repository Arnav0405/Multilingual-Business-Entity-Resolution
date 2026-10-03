"""Execute the notebook's sanity-check cell against the real local subset."""
import pathlib

import nbformat
import pandas as pd

nb = nbformat.read("entity_resolution_colab.ipynb", as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and not c.source.lstrip().startswith("!"):
        compile(c.source, f"cell{i}", "exec")
print("nb OK; all pure-python cells compile")

DATA = pathlib.Path("../../../data/entity_resolution_subset").resolve()
load_pairs = lambda split: pd.read_csv(DATA / f"{split}_pairs.tsv", sep="\t")

ns = {"pd": pd, "DATA": DATA, "load_pairs": load_pairs, "PHASE": "B", "TEXT_VARIANT": "name"}
cell = next(c.source for c in nb.cells if c.cell_type == "code" and "rows = " in c.source)
exec(cell, ns)
print("sanity cell executed green against the real subset")
