import pandas as pd
import numpy as np
import os
from nilearn import datasets
# FIX: Updated import to silence deprecation warning
from nilearn.maskers import NiftiMapsMasker 
from nilearn.connectome import ConnectivityMeasure

def extract_features():
    # --- 1. Robust Path Setup ---
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    
    manifest_path = os.path.join(project_root, "data", "processed", "abide_sprint_dataset.csv")
    output_path = os.path.join(project_root, "data", "processed", "abide_features.csv")
    
    if not os.path.exists(manifest_path):
        print(f"❌ Error: Manifest not found at:\n   {manifest_path}")
        print("   Did you run download_data.py?")
        return
    
    df = pd.read_csv(manifest_path)
    print(f"--- Loading {len(df)} subjects for Feature Extraction ---")

    # --- 2. Load Brain Atlas (MSDL) ---
    print("Fetching MSDL Atlas...")
    msdl_data = datasets.fetch_atlas_msdl()
    
    masker = NiftiMapsMasker(
        maps_img=msdl_data.maps, 
        standardize=True, 
        memory='nilearn_cache', 
        verbose=0
    )

    # --- 3. Extract & Correlate ---
    print("Extracting signals...")
    correlation_measure = ConnectivityMeasure(kind='correlation')
    
    features = []
    subject_ids = []
    
    for i, row in df.iterrows():
        try:
            print(f"Processing {i+1}/{len(df)}: {row['SITE_ID']} - {row['SUB_ID']}")
            
            # The path in the CSV might be absolute; if it moved, we might need to adjust.
            # But usually, if you moved the whole data folder, the relative structure holds.
            # If this fails, we will need to re-point the paths.
            nifti_file = row['nifti_path']
            
            # Extract time series
            time_series = masker.fit_transform(nifti_file)
            
            # Compute correlation matrix
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            
            # Flatten upper triangle
            mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            features.append(correlation_matrix[mask])
            subject_ids.append(row['SUB_ID'])
            
        except Exception as e:
            print(f"⚠️ Failed on {row['SUB_ID']}: {e}")

    # --- 4. Save Results ---
    if not features:
        print("❌ No features extracted. Check paths.")
        return

    X_df = pd.DataFrame(features)
    X_df['SUB_ID'] = subject_ids
    
    # Merge with phenotype data
    final_df = pd.merge(df, X_df, on='SUB_ID')
    
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ SUCCESS: Extracted features saved to:\n   {output_path}")

if __name__ == "__main__":
    extract_features()