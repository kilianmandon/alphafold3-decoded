import os
from pathlib import Path
import re
import subprocess
import sys

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

    if not (solutions_dir / 'converted_notebooks' / f'{chapter}_notebook.py').exists():
        print(f'Chapter {chapter} is not yet available.')
        return

    print('Executing script to generate test results...')

    subprocess.run([
        sys.executable, "-m", f"converted_notebooks.{chapter}_notebook"
    ], cwd=solutions_dir)

if __name__ == '__main__':  
    main()