import numpy as np
import torch

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    probs: numpy array of probabilities (0.0 to 1.0)
    labels: numpy array of true labels (0 or 1)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find which predictions fall into this bin (e.g., 0.6 to 0.7)
        bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        bin_count = np.sum(bin_mask)
        
        if bin_count > 0:
            # Accuracy in this bin
            current_acc = np.mean(labels[bin_mask] == (probs[bin_mask] > 0.5))
            # Average confidence in this bin
            current_conf = np.mean(probs[bin_mask])
            
            # Weighted difference
            ece += (bin_count / len(probs)) * np.abs(current_acc - current_conf)
            
    return ece


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def compute_site_leakage(feature_extractor, data_loader, device):
    """
    Extracts embeddings z and tries to predict Site ID.
    Returns: Accuracy of the adversary (Chance level is ~5% for 20 sites).
    High accuracy (>20%) = BAD LEAKAGE.
    """
    feature_extractor.eval()
    all_z = []
    all_sites = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Unpack batch (assuming structure: x, y, site_id)
            x, y, site_ids = batch[0].to(device), batch[1], batch[2]
            
            # Get Latent Embeddings (z)
            # Note: We take the output of linear1 or linear2 before the GP
            # Assuming feature_extractor returns z
            z = feature_extractor(x) 
            
            all_z.append(z.cpu().numpy())
            all_sites.append(site_ids.numpy())
            
    X_latent = np.concatenate(all_z, axis=0)
    y_site = np.concatenate(all_sites, axis=0)
    
    # Train Adversary (Random Forest)
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    
    # 5-Fold Cross Validation to get robust leakage score
    scores = cross_val_score(clf, X_latent, y_site, cv=5)
    
    return np.mean(scores)