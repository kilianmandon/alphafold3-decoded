import pandas as pd

def main():
    interfaces_df = pd.read_parquet('data/datasets/pdb_metadata/interfaces_df.parquet')
    interfaces_df[:1000].to_parquet('data/datasets/pdb_metadata/interfaces_df_top1000.parquet')

    pn_units_df = pd.read_parquet('data/datasets/pdb_metadata/pn_units_df.parquet')
    pn_units_df[:1000].to_parquet('data/datasets/pdb_metadata/pn_units_df_top1000.parquet')

    pdb_ids = sorted(list(set(interfaces_df[:1000]['pdb_id']) | set(pn_units_df[:1000]['pdb_id'])))
    with open('data/datasets/pdb_metadata/pdb_ids.txt', 'w') as f:
        f.write('\n'.join(pdb_ids))

if __name__=='__main__':
    main()