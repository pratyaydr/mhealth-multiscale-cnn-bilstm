"""Execute the evidence-only notebook in-process when local kernel sockets are unavailable.

This is a transparent fallback, not a test of Jupyter kernel transport. It runs
all code cells in order, stops on errors, and persists captured rich outputs.
"""

import os
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/reproducible_walkthrough.ipynb"
notebook = nbformat.read(path, as_version=4)
shell = InteractiveShell.instance()
os.chdir(root)
count = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    with capture_output() as captured:
        result = shell.run_cell(cell.source)
    result.raise_error()
    cell.execution_count = count
    cell.outputs = []
    if captured.stdout:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
    if captured.stderr:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(
            nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata)
        )
notebook.metadata["conversion_execution"] = {
    "method": "IPython in-process, sequential cells",
    "reason": "Local Jupyter ZeroMQ kernel socket creation is unavailable in conversion environment",
    "code_cells_executed": count,
}
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Executed {count} code cells with captured outputs: {path.name}")
