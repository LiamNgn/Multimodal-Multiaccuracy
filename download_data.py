import os
import pandas as pd
from nilearn import datasets
import numpy as np

def download_and_save_abide():
    # --- 1. Setup Paths ---
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"--- 1. Fetching FULL ABIDE Metadata ---")

    # Fetch all available data
    abide_data = datasets.fetch_abide_pcp(
        data_dir=raw_data_path,
        pipeline="cpac",
        quality_checked=False,
        n_subjects=None # No limit!
    )
    
    # --- 2. Process Metadata ---
    pheno_df = pd.DataFrame(abide_data.phenotypic)
    
    # Clean byte strings
    for col in pheno_df.select_dtypes([object]):
        pheno_df[col] = pheno_df[col].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        ).str.strip()
    
    pheno_df['nifti_path'] = abide_data.func_preproc
    
    # --- 3. INTELLIGENT FILTERING ---
    # We want sites with enough subjects to actually measure bias.
    # Let's keep sites with at least 40 subjects.
    site_counts = pheno_df['SITE_ID'].value_counts()
    valid_sites = site_counts[site_counts >= 40].index.tolist()
    
    print(f"\n--- Selecting Sites with N >= 40 ---")
    print(f"Found {len(valid_sites)} valid sites: {valid_sites}")
    
    final_df = pheno_df[pheno_df['SITE_ID'].isin(valid_sites)].copy()
    
    # Report Statistics
    print("\n--- Cohort Statistics ---")
    stats = final_df.groupby('SITE_ID')['DX_GROUP'].value_counts().unstack().fillna(0)
    stats.columns = ['Control', 'Autism']
    print(stats)
    print(f"\nTotal Subjects: {len(final_df)}")

    if len(final_df) == 0:
        print("❌ ERROR: No subjects found.")
        return

    # --- 4. Save Manifest ---
    save_path = os.path.join(processed_dir, "abide_sprint_dataset.csv")
    final_df.to_csv(save_path, index=False)
    
    print("-" * 30)
    print(f"✅ SUCCESS: Dataset updated with {len(valid_sites)} sites.")
    print(f"   Manifest saved to: {save_path}")
    print("   IMPORTANT: Now run 'python src/extract_features.py' to process the new scans.")
    print("-" * 30)

if __name__ == "__main__":
    download_and_save_abide()