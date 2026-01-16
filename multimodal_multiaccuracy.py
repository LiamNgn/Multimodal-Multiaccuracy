import torch
import gpytorch
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from utils import compute_ece  # Ensure utils.py exists or paste compute_ece here

# --- 0. Reproducibility Engine ---
def set_global_seed(seed):
    """Locks all sources of randomness for reproducible results."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"🔒 Global Seed set to {seed}")

# --- 1. Multimodal Feature Extractor ---
class MultimodalEncoder(torch.nn.Module):
    def __init__(self, fmri_dim, clinical_dim):
        super(MultimodalEncoder, self).__init__()
        # Path A: Brain Scan (High Dimensional)
        self.fmri_net = torch.nn.Sequential(
            torch.nn.Linear(fmri_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        # Path B: Clinical Data (Low Dimensional)
        self.clinical_net = torch.nn.Sequential(
            torch.nn.Linear(clinical_dim, 16),
            torch.nn.ReLU()
        )
        # Path C: Fusion
        self.fusion_net = torch.nn.Sequential(
            torch.nn.Linear(80, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x_fmri, x_clin):
        h_brain = self.fmri_net(x_fmri)
        h_clin = self.clinical_net(x_clin)
        combined = torch.cat([h_brain, h_clin], dim=1)
        z = self.fusion_net(combined)
        return z

# --- 2. The Gaussian Process ---
class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, z):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(z), self.covar_module(z))

# --- 3. The Multiaccuracy Penalty ---
def multiaccuracy_penalty(residuals, sensitive_attributes):
    s = sensitive_attributes.float().unsqueeze(1)
    dist = torch.cdist(s, s) 
    K_audit = torch.exp(-dist) 
    penalty = (residuals.unsqueeze(0) @ K_audit @ residuals.unsqueeze(1)) / (residuals.numel()**2)
    return penalty.squeeze()

# --- 4. Training Loop ---
def train_and_evaluate(data, use_penalty=False):
    (X_fmri_tr, X_clin_tr, y_tr, s_tr), (X_fmri_te, X_clin_te, y_te, s_te) = data
    
    encoder = MultimodalEncoder(fmri_dim=X_fmri_tr.shape[1], clinical_dim=X_clin_tr.shape[1])
    # Init GP with first 50 points as inducing
    dummy_z = encoder(X_fmri_tr[:50], X_clin_tr[:50]).detach()
    gp = GPModel(inducing_points=dummy_z)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': gp.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.005)

    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=y_tr.size(0))
    
    # Training
    encoder.train(); gp.train(); likelihood.train()
    
    for i in range(250): 
        optimizer.zero_grad()
        z = encoder(X_fmri_tr, X_clin_tr)
        output = gp(z)
        loss_main = -mll(output, y_tr)
        
        if use_penalty:
            residuals = y_tr - output.mean
            penalty_val = multiaccuracy_penalty(residuals, s_tr)
            loss = loss_main + (25.0 * penalty_val)
        else:
            loss = loss_main
            
        loss.backward()
        optimizer.step()
    
    # Evaluation
    encoder.eval(); gp.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        z_te = encoder(X_fmri_te, X_clin_te)
        preds = likelihood(gp(z_te)).mean
        
        probs_np = preds.numpy()
        y_te_np = y_te.numpy()
        s_te_np = s_te.numpy()
        pred_class = (probs_np > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_te_np, pred_class)
        b_acc = balanced_accuracy_score(y_te_np, pred_class)
        try: auroc = roc_auc_score(y_te_np, probs_np)
        except: auroc = 0.5 
        ece = compute_ece(probs_np, y_te_np)

        # Weighted Bias Calculation
        unique_sites = np.unique(s_te_np)
        site_residuals = []
        site_counts = []
        
        for site_code in unique_sites:
            mask = (s_te_np == site_code)
            n_g = np.sum(mask)
            if n_g > 0:
                bias_g = np.abs(np.mean(y_te_np[mask] - probs_np[mask]))
                site_residuals.append(bias_g)
                site_counts.append(n_g)
        
        total_N = np.sum(site_counts)
        weighted_bias = np.sum(np.array(site_residuals) * np.array(site_counts)) / total_N
        max_bias = np.max(site_residuals)

        # Leakage
        emb = z_te.numpy()
        probe = RandomForestClassifier(n_estimators=50, max_depth=5).fit(emb, s_te_np)
        leakage = probe.score(emb, s_te_np)
        
        return {
            "Accuracy": acc,
            "Balanced_Acc": b_acc,
            "AUROC": auroc,
            "ECE": ece,
            "Leakage": leakage,
            "Max_Bias": max_bias,
            "Weighted_Bias": weighted_bias
        }

# --- 5. The Scientific Execution Loop ---
def run_scientific_validation():
    # 5 Seeds for Robustness
    seeds = [11, 732, 12, 1412, 5512]
    
    # Storage for results
    history = {
        "Baseline": {"Acc": [], "Bias": [], "ECE": [], "Leakage": []},
        "Fairness": {"Acc": [], "Bias": [], "ECE": [], "Leakage": []}
    }
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    data_path = os.path.join(project_root, "data", "processed", "abide_features.csv")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    print(f"\n Starting loop (N={len(seeds)})")
    print("="*60)

    for i, seed in enumerate(seeds):
        print(f"   ► Run {i+1}/{len(seeds)} (Seed {seed})...", end="", flush=True)
        
        # 1. Lock Seed
        set_global_seed(seed)
        
        # 2. Reload & Re-split Data (Cross-Validation Style)
        # Note: We reload df inside loop or just split inside. 
        # Loading CSV is fast enough for ABIDE.
        df = pd.read_csv(data_path)
        
        ignore_cols = ['SUB_ID', 'SITE_ID', 'DX_GROUP', 'nifti_path', 'Unnamed: 0', 'func_preproc']
        clinical_cols = ['AGE_AT_SCAN', 'SEX', 'FIQ', 'VIQ', 'PIQ'] 
        
        X_fmri_raw = df.drop(columns=ignore_cols + clinical_cols, errors='ignore').select_dtypes(include=[np.number])
        X_fmri = StandardScaler().fit_transform(SimpleImputer(strategy='constant', fill_value=0).fit_transform(X_fmri_raw))
        
        clin_df = df[['AGE_AT_SCAN', 'SEX', 'FIQ']].copy()
        clin_imputer = SimpleImputer(strategy='mean')
        X_clin = StandardScaler().fit_transform(clin_imputer.fit_transform(clin_df))
        
        y = (df['DX_GROUP'].values == 1).astype(float)
        s = LabelEncoder().fit_transform(df['SITE_ID'])
        
        # Split using the CURRENT SEED
        inds = np.arange(len(df))
        train_idx, test_idx = train_test_split(inds, test_size=0.3, stratify=s, random_state=seed)
        
        def to_tens(idx):
            return (torch.tensor(X_fmri[idx]).float(), 
                    torch.tensor(X_clin[idx]).float(), 
                    torch.tensor(y[idx]).float(), 
                    torch.tensor(s[idx]).float())
        
        train_data = to_tens(train_idx)
        test_data = to_tens(test_idx)

        # 3. Train Both Models
        res_base = train_and_evaluate((train_data, test_data), use_penalty=False)
        res_fair = train_and_evaluate((train_data, test_data), use_penalty=True)
        
        # 4. Log
        history["Baseline"]["Acc"].append(res_base["Accuracy"])
        history["Baseline"]["Bias"].append(res_base["Weighted_Bias"])
        history["Baseline"]["ECE"].append(res_base["ECE"])
        history["Baseline"]["Leakage"].append(res_base["Leakage"])
        
        history["Fairness"]["Acc"].append(res_fair["Accuracy"])
        history["Fairness"]["Bias"].append(res_fair["Weighted_Bias"])
        history["Fairness"]["ECE"].append(res_fair["ECE"])
        history["Fairness"]["Leakage"].append(res_fair["Leakage"])
        
        print(" Done.")

    # --- FINAL REPORT GENERATION ---
    print("\n" + "="*75)
    print("FINAL PUBLICATION TABLE (Mean ± Std Dev)")
    print("="*75)
    
    metrics = ["Acc", "Bias", "ECE", "Leakage"]
    print(f"{'METRIC':<10} | {'BASELINE':<28} | {'FAIR (OURS)':<28}")
    print("-" * 75)
    
    summary = {}
    
    for m in metrics:
        b_mean = np.mean(history["Baseline"][m])
        b_std  = np.std(history["Baseline"][m])
        
        f_mean = np.mean(history["Fairness"][m])
        f_std  = np.std(history["Fairness"][m])
        
        # Format
        b_str = f"{b_mean:.3f} ± {b_std:.3f}"
        f_str = f"{f_mean:.3f} ± {f_std:.3f}"
        
        sig = ""
        # If bias reduced significantly
        if m == "Bias" and (b_mean - b_std > f_mean + f_std):
            sig = "(*)"
        
        print(f"{m:<10} | {b_str:<28} | {f_str:<28} {sig}")
        
        summary[m] = {"Baseline": f_str, "Fairness": f_str}

    print("="*75)
    
    # Return structure for Notebook
    return history

if __name__ == "__main__":
    run_scientific_validation()