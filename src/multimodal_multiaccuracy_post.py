import torch
import gpytorch
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier

# --- 0. Reproducibility Engine ---
def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. Metric Helpers (To Match Previous Results) ---
def compute_ece(probs, y_true, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask] == (probs[mask] > 0.5))
            conf = np.mean(probs[mask])
            ece += np.abs(acc - conf) * np.mean(mask)
    return ece

def compute_weighted_bias(probs, y_true, sites):
    # sites should be integers here
    unique_sites = np.unique(sites)
    site_residuals = []
    site_counts = []
    for s in unique_sites:
        mask = (sites == s)
        n_g = np.sum(mask)
        if n_g > 0:
            # Bias = Abs(Mean Residual)
            bias_g = np.abs(np.mean(y_true[mask] - probs[mask]))
            site_residuals.append(bias_g)
            site_counts.append(n_g)
    total_N = np.sum(site_counts)
    return np.sum(np.array(site_residuals) * np.array(site_counts)) / total_N

# --- 2. DKL Architecture ---
class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 64))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(64, 2)) 

class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, var_dist, learn_inducing_locations=True)
        super(GPModel, self).__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

def train_base_model(X_tr, y_tr):
    data_dim = X_tr.shape[1]
    fe = FeatureExtractor(data_dim)
    # Init inducing points
    dummy = fe(X_tr[:50]).detach()
    gp = GPModel(dummy)
    like = gpytorch.likelihoods.GaussianLikelihood()
    
    opt = torch.optim.Adam(list(fe.parameters()) + list(gp.parameters()) + list(like.parameters()), lr=0.005)
    mll = gpytorch.mlls.VariationalELBO(like, gp, num_data=y_tr.size(0))
    
    fe.train(); gp.train(); like.train()
    for i in range(250):
        opt.zero_grad()
        loss = -mll(gp(fe(X_tr)), y_tr)
        loss.backward()
        opt.step()
    return fe, gp, like

# --- 3. The Scientific Execution Loop ---
def run_postproc_validation():
    seeds = [42, 100, 2023, 7, 999, 14,12,1412,1997,17,2,172,1702]
    
    history = {
        "Baseline": {"Acc": [], "Bias": [], "ECE": [], "Leakage": []},
        "PostProc": {"Acc": [], "Bias": [], "ECE": [], "Leakage": []}
    }
    
    # Load Data Once
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    data_path = os.path.join(project_root, "data", "processed", "abide_features.csv")
    if not os.path.exists(data_path): return
    df_full = pd.read_csv(data_path)
    
    # Preprocessing
    ignore = ['SUB_ID', 'SITE_ID', 'DX_GROUP', 'nifti_path', 'Unnamed: 0', 'func_preproc']
    X_raw = df_full.drop(columns=ignore, errors='ignore').select_dtypes(include=[np.number])
    X_all = StandardScaler().fit_transform(SimpleImputer(strategy='constant', fill_value=0).fit_transform(X_raw))
    y_all = (df_full['DX_GROUP'].values == 1).astype(float)
    
    # Site Encoders
    s_raw = df_full['SITE_ID'].values
    le = LabelEncoder()
    s_int = le.fit_transform(s_raw) # For stratification & metrics
    ohe = OneHotEncoder(sparse_output=False)
    s_ohe = ohe.fit_transform(s_raw.reshape(-1, 1)) # For boosting correction

    print(f"\n🧪 STARTING POST-PROCESSING AUDIT (N={len(seeds)})")
    print("="*60)

    for i, seed in enumerate(seeds):
        print(f"   ► Run {i+1}/{len(seeds)} (Seed {seed})...", end="", flush=True)
        set_global_seed(seed)
        
        # --- A. 3-Way Split (Model / Audit / Test) ---
        idx = np.arange(len(df_full))
        # 1. Hold out Test Set (20%)
        tr_audit_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=s_int, random_state=seed)
        
        # 2. Split remaining into Model Train (40%) and Fairness Audit (40%)
        # Crucial: The model NEVER sees the Audit set during training
        s_tr_audit = s_int[tr_audit_idx]
        model_idx, audit_idx = train_test_split(tr_audit_idx, test_size=0.5, stratify=s_tr_audit, random_state=seed)
        
        def get_tensors(indices):
            return (torch.tensor(X_all[indices]).float(), 
                    torch.tensor(y_all[indices]).float(), 
                    s_ohe[indices], # One-Hot for Corrector
                    s_int[indices]) # Int for Metrics
        
        X_m, y_m, s_ohe_m, s_int_m = get_tensors(model_idx)
        X_a, y_a, s_ohe_a, s_int_a = get_tensors(audit_idx)
        X_t, y_t, s_ohe_t, s_int_t = get_tensors(test_idx)
        
        # --- B. Train Base Model ---
        fe, gp, like = train_base_model(X_m, y_m)
        
        fe.eval(); gp.eval(); like.eval()
        with torch.no_grad():
            # Get Latent Embeddings (for Leakage Check)
            z_t = fe(X_t)
            
            # Get Raw Probabilities
            p_audit = like(gp(fe(X_a))).mean.numpy()
            p_test  = like(gp(z_t)).mean.numpy()

        # --- C. Baseline Metrics ---
        # 1. Leakage (Probe the Latent Space z)
        # Note: Post-processing CANNOT fix latent leakage, but we measure it anyway
        probe = RandomForestClassifier(n_estimators=50, max_depth=5).fit(z_t.numpy(), s_int_t)
        leakage = probe.score(z_t.numpy(), s_int_t)
        
        acc_base = accuracy_score(y_t.numpy(), (p_test > 0.5).astype(int))
        ece_base = compute_ece(p_test, y_t.numpy())
        bias_base = compute_weighted_bias(p_test, y_t.numpy(), s_int_t)
        
        # Log Baseline
        history["Baseline"]["Acc"].append(acc_base)
        history["Baseline"]["Bias"].append(bias_base)
        history["Baseline"]["ECE"].append(ece_base)
        history["Baseline"]["Leakage"].append(leakage)

        # --- D. Iterative Multiaccuracy Boosting (The Correction) ---
        # We perform boosting on Logits or Probs. Here we simply boost Probs additively (Kim et al. 2019 simplified)
        
        lr = 0.1
        current_p_audit = p_audit.copy()
        current_p_test = p_test.copy()
        
        for boost_round in range(15): # 15 rounds of correction
            residuals = y_a.numpy() - current_p_audit
            
            # Train Corrector on Audit Set Residuals
            # We use One-Hot sites as input -> Learns a constant shift per site
            corrector = KernelRidge(kernel='linear', alpha=1.0)
            corrector.fit(s_ohe_a, residuals)
            
            # Predict correction
            update_audit = corrector.predict(s_ohe_a)
            update_test  = corrector.predict(s_ohe_t)
            
            # Apply (Clamp to valid prob range 0-1)
            current_p_audit = np.clip(current_p_audit + (lr * update_audit), 0.01, 0.99)
            current_p_test  = np.clip(current_p_test + (lr * update_test), 0.01, 0.99)
            
        # --- E. Post-Processing Metrics ---
        acc_post = accuracy_score(y_t.numpy(), (current_p_test > 0.5).astype(int))
        ece_post = compute_ece(current_p_test, y_t.numpy())
        bias_post = compute_weighted_bias(current_p_test, y_t.numpy(), s_int_t)
        
        # Log PostProc
        history["PostProc"]["Acc"].append(acc_post)
        history["PostProc"]["Bias"].append(bias_post)
        history["PostProc"]["ECE"].append(ece_post)
        history["PostProc"]["Leakage"].append(leakage) # Leakage doesn't change in post-proc!
        
        print(" Done.")

    # --- FINAL REPORT ---
    print("\n" + "="*75)
    print("📢 FINAL POST-PROCESSING RESULTS (Mean ± Std Dev)")
    print("="*75)
    
    metrics = ["Acc", "Bias", "ECE", "Leakage"]
    print(f"{'METRIC':<10} | {'BASELINE':<28} | {'POST-PROC (OURS)':<28}")
    print("-" * 75)
    
    for m in metrics:
        b_mean = np.mean(history["Baseline"][m])
        b_std  = np.std(history["Baseline"][m])
        
        p_mean = np.mean(history["PostProc"][m])
        p_std  = np.std(history["PostProc"][m])
        
        b_str = f"{b_mean:.3f} ± {b_std:.3f}"
        p_str = f"{p_mean:.3f} ± {p_std:.3f}"
        
        sig = "(*)" if (m == "Bias" and b_mean - b_std > p_mean + p_std) else ""
        
        print(f"{m:<10} | {b_str:<28} | {p_str:<28} {sig}")
    print("="*75)

if __name__ == "__main__":
    run_postproc_validation()