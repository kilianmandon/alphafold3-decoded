import re
import json
from pathlib import Path
import shutil
import subprocess

# ── helpers ────────────────────────────────────────────────────────────────────

def convert_py(src: Path, dst: Path):
    """Transform a .py solution file into a tutorial stub."""
    text = src.read_text(encoding="utf-8")

    # Pattern handles two formats:
    #
    # Format A (single-line opener):
    #   <indent>"""  TODO: anything  """
    #   <anything2>
    #   <indent>""" End of your code """
    #
    # Format B (multi-line, opener on its own line):
    #   <indent>"""
    #   <indent>TODO: anything
    #   <indent>"""
    #   <anything2>
    #   <indent>""" End of your code """
    #
    # We capture the indentation from the line that starts with """ (or TODO).
    pattern = re.compile(
        r'^(?P<indent>[ \t]*)"""[ \t]*\n?[ \t]*TODO:.*?"""[ \t]*\n(?P<anything2>.*?)(?P<close_indent>[ \t]*)"""[ \t]*End of your code[ \t]*"""',
        re.MULTILINE | re.DOTALL,
    )

    def replacement(m: re.Match) -> str:
        indent = m.group("indent")
        stub = f'\n{indent}# Replace \'pass\' with your code\n{indent}pass\n'
        full = m.group(0)
        # Everything up to and including the closing """ of the TODO docstring
        # i.e. up to the first """ ... TODO ... """  (the \n after it is consumed by the pattern)
        todo_block_end = re.search(
            r'"""[ \t]*\n?[ \t]*TODO:.*?"""',
            full,
            re.DOTALL,
        ).end()
        todo_block = full[:todo_block_end]
        return f'{todo_block}\n{stub}\n{indent}""" End of your code """'

    new_text = pattern.sub(replacement, text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_text, encoding="utf-8")
    print(f"  [py]  {src} → {dst}")


def clean_notebook(src: Path):
    nb = json.loads(src.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
    src.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

def convert_ipynb(src: Path, dst: Path, tutorials_root: Path):
    """Copy a notebook, clear outputs, and flip mode='write' → mode='read'."""
    clean_notebook(src)
    nb = json.loads(src.read_text(encoding="utf-8"))


    for cell in nb.get("cells", []):
        # Replace mode='write' -> mode='read' in source lines
        if cell.get("cell_type") == "code":
            cell["source"] = [
                line.replace("mode='write'", "mode='read'").replace('tests created', 'tests passed')
                for line in cell.get("source", [])
            ]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  [nb]  {src} → {dst}")

    notebook_name = dst.stem

    converted_path = tutorials_root / "converted_notebooks"
    converted_path.mkdir(exist_ok=True)

    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "script",
        str(dst),
        "--output", f"{notebook_name}_notebook",
        "--output-dir", str(converted_path)
    ], check=True)

    # Remove %magic  commands from converted script, e.g. get_ipython().*
    script_path = converted_path / f"{notebook_name}_notebook.py"
    script_text = script_path.read_text(encoding="utf-8")
    magic_pattern = re.compile(r'^\s*get_ipython\(\)\..*$')
    script_text = magic_pattern.sub(lambda m: f'# {m.group(0)}  # removed by prepare_tutorials.py', script_text)
    script_path.write_text(script_text, encoding="utf-8")



# ── main ───────────────────────────────────────────────────────────────────────

def main():
    solutions_root = Path("solutions")
    tutorials_root = Path("tutorials")

    if tutorials_root.exists():
        shutil.rmtree(tutorials_root)

    blacklist = "internal_tests.py"

    if not solutions_root.exists():
        print("Error: 'solutions/' directory not found. Run this script from the project root.")
        return

    py_files  = list(solutions_root.glob("**/*.py"))
    nb_files  = list(solutions_root.glob("**/*.ipynb"))

    print(f"Found {len(py_files)} .py files and {len(nb_files)} .ipynb files.\n")

    for src in py_files:
        rel = src.relative_to(solutions_root)

        if rel.name == blacklist:
            print(f"  [skipped] {rel}")
            continue

        dst = tutorials_root / rel
        convert_py(src, dst)

    for src in nb_files:
        rel = src.relative_to(solutions_root)
        dst = tutorials_root / rel
        convert_ipynb(src, dst, tutorials_root)

    print("Creating data/ symlink...")
    data_src = Path("../data")
    data_dst = tutorials_root / "data"
    if data_dst.exists():
        data_dst.unlink()
    data_dst.symlink_to(data_src, target_is_directory=True)

    print("\nDone!")


if __name__ == "__main__":
    if input("This will overwrite files in the 'tutorials/' directory. Do you want to proceed? (y/n) ").lower() == "y":
        main()
    else:
        print("Aborted.")