import os
from pathlib import Path
import re
import subprocess

def main():
    chapters = [
        'feature_extraction', 
        'input_embedding', 
        'evoformer',
        'diffusion',
        'training',
    ]

    for i, chapter in enumerate(chapters):
        print(f'({i+1}): {chapter}')
    
    chapter_idx = int(input('Enter the number of the chapter you want to generate test results for: ')) - 1
    chapter = chapters[chapter_idx]

    solutions_dir = Path('solutions')
    ipynb_path = solutions_dir / f'{chapter}/{chapter}.ipynb'

    print('Converting ipynb to script...')
    subprocess.run([
        'jupyter', 'nbconvert', 
        '--to', 'script', str(ipynb_path),
        '--output', f'{chapter}_notebook',
        '--output-dir', str(solutions_dir/"converted_notebooks")
        ] )

    # Remove %magic  commands from converted script
    script_path = solutions_dir / "converted_notebooks" / f"{chapter}_notebook.py"
    script_text = script_path.read_text(encoding="utf-8")
    magic_pattern = re.compile(r'get_ipython\(\)\..*$', re.MULTILINE)
    script_text = magic_pattern.sub(lambda m: f'# {m.group(0)}  # removed by prepare_tutorials.py', script_text)
    script_path.write_text(script_text, encoding="utf-8")

    print('Executing script to generate test results...')

    subprocess.run([
        "python", str(script_path.absolute())
    ], cwd=solutions_dir)

if __name__ == '__main__':  
    main()