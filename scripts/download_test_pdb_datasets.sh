atomworks ccd sync data/datasets/ccd_mirror
atomworks setup metadata data/datasets/pdb_metadata

python scripts/extract_pdb_ids_for_dataset.py

atomworks pdb sync data/datasets/pdb_mirror --pdb-ids-file data/datasets/pdb_metadata/pdb_ids.txt