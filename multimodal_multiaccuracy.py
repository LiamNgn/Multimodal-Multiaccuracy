import torch
import gpytorch
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from utils import compute_ece,compute_site_leakage
import random

def set_global_seed(seed=42):
    """
    Locks all sources of randomness for reproducible results.
    """
    # 1. Python standard library
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # 4. Force Deterministic Algorithms (Slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🔒 Global Seed set to {seed}. Results will be reproducible.")

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
        
        # Path B: Clinical Data (Low Dimensional: Age, Sex, IQ)
        self.clinical_net = torch.nn.Sequential(
            torch.nn.Linear(clinical_dim, 16),
            torch.nn.ReLU()
        )
        
        # Path C: Fusion Layer (Concatenate A + B)
        # 64 (brain) + 16 (clinical) = 80
        self.fusion_net = torch.nn.Sequential(
            torch.nn.Linear(80, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2) # Latent Z
        )

    def forward(self, x_fmri, x_clin):
        h_brain = self.fmri_net(x_fmri)
        h_clin = self.clinical_net(x_clin)
        # Concatenate
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

# --- 3. The Multiaccuracy Penalty (In-Processing) ---
def multiaccuracy_penalty(residuals, sensitive_attributes):
    """
    Calculates r^T * K_audit * r
    Ensures residuals are orthogonal to the sensitive subgroups.
    """
    # Create a Kernel Matrix for the Sensitive Attribute (Site)
    # If site is discrete, this acts like a group-match matrix
    s = sensitive_attributes.float().unsqueeze(1)
    dist = torch.cdist(s, s) # Distance matrix
    # RBF Kernel on Site: 1 if same site, <1 if different
    K_audit = torch.exp(-dist) 
    
    # The Penalty: Projections of residuals onto the audit kernel
    # Normalized by N^2 to keep it scale-invariant
    penalty = (residuals.unsqueeze(0) @ K_audit @ residuals.unsqueeze(1)) / (residuals.numel()**2)
    return penalty.squeeze()

# --- 4. Training Loop ---
def train_and_evaluate(data, use_penalty=False):
    (X_fmri_tr, X_clin_tr, y_tr, s_tr), (X_fmri_te, X_clin_te, y_te, s_te) = data
    
    # Initialize Models (Same as before)
    encoder = MultimodalEncoder(fmri_dim=X_fmri_tr.shape[1], clinical_dim=X_clin_tr.shape[1])
    dummy_z = encoder(X_fmri_tr[:50], X_clin_tr[:50]).detach()
    gp = GPModel(inducing_points=dummy_z)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': gp.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.005)

    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=y_tr.size(0))
    
    # --- TRAINING LOOP ---
    encoder.train(); gp.train(); likelihood.train()
    print(f"   Training {'[MULTIMODAL FAIR]' if use_penalty else '[MULTIMODAL BASELINE]'}...")
    
    for i in range(250): # Keeping your epoch count
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
    
    # --- SOPHISTICATED EVALUATION ---
    encoder.eval(); gp.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        z_te = encoder(X_fmri_te, X_clin_te)
        # Get probabilities (mean of the GP posterior passed through sigmoid/likelihood)
        preds = likelihood(gp(z_te)).mean
        
        # Convert to numpy for sklearn
        probs_np = preds.numpy()
        y_te_np = y_te.numpy()
        s_te_np = s_te.numpy()
        pred_class = (probs_np > 0.5).astype(int)

        # 1. Utility Metrics
        acc = accuracy_score(y_te_np, pred_class)
        b_acc = balanced_accuracy_score(y_te_np, pred_class)
        try:
            auroc = roc_auc_score(y_te_np, probs_np)
        except: auroc = 0.5 # Handle edge case
        ece = compute_ece(probs_np, y_te_np)

        # 2. Fairness Audit (Weighted Bias)
        unique_sites = np.unique(s_te_np)
        site_residuals = []
        site_counts = []
        
        for site_code in unique_sites:
            mask = (s_te_np == site_code)
            n_g = np.sum(mask)
            if n_g > 0:
                # Bias = Mean Signed Residual (Actual - Pred)
                # We take ABS value because we care about magnitude of error
                bias_g = np.abs(np.mean(y_te_np[mask] - probs_np[mask]))
                site_residuals.append(bias_g)
                site_counts.append(n_g)
        
        # Calculate Weighted Average Bias
        total_N = np.sum(site_counts)
        weighted_bias = np.sum(np.array(site_residuals) * np.array(site_counts)) / total_N
        max_bias = np.max(site_residuals)

        # 3. Leakage Probe
        emb = z_te.numpy()
        probe = RandomForestClassifier(n_estimators=50, max_depth=5).fit(emb, s_te_np)
        leakage = probe.score(emb, s_te_np)
        
        # Return Dictionary
        return {
            "Accuracy": acc,
            "Balanced_Acc": b_acc,
            "AUROC": auroc,
            "ECE": ece,
            "Leakage": leakage,
            "Max_Bias": max_bias,
            "Weighted_Bias": weighted_bias
        }

def run_multimodal_sprint():
    # Load Data
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    data_path = os.path.join(project_root, "data", "processed", "abide_features.csv")
    if not os.path.exists(data_path): return
    df = pd.read_csv(data_path)
    
    # --- PREPARE MULTIMODAL DATA ---
    print("--- 1. Fusing Brain + Clinical Data ---")
    
    # Modality A: fMRI Features
    ignore_cols = ['SUB_ID', 'SITE_ID', 'DX_GROUP', 'nifti_path', 'Unnamed: 0', 'func_preproc']
    # We remove the clinical cols from X_fmri specifically
    clinical_cols = ['AGE_AT_SCAN', 'SEX', 'FIQ', 'VIQ', 'PIQ'] 
    
    X_fmri_raw = df.drop(columns=ignore_cols + clinical_cols, errors='ignore').select_dtypes(include=[np.number])
    X_fmri = StandardScaler().fit_transform(SimpleImputer(strategy='constant', fill_value=0).fit_transform(X_fmri_raw))
    
    # Modality B: Clinical Features (Age, Sex, IQ)
    # Note: 'SEX' is usually 1=Male, 2=Female in ABIDE. 'FIQ' is Full IQ.
    # We must handle NaNs carefully here (IQ is often missing).
    clin_df = df[['AGE_AT_SCAN', 'SEX', 'FIQ']].copy() # Using FIQ (Full IQ)
    
    # Simple imputation for missing IQ (mean fill)
    clin_imputer = SimpleImputer(strategy='mean')
    X_clin = StandardScaler().fit_transform(clin_imputer.fit_transform(clin_df))
    
    # Targets
    y = (df['DX_GROUP'].values == 1).astype(float)
    s = LabelEncoder().fit_transform(df['SITE_ID'])
    
    # Split
    inds = np.arange(len(df))
    train_idx, test_idx = train_test_split(inds, test_size=0.3, stratify=s, random_state=42)
    
    # Package into Tensors
    def to_tens(idx):
        return (torch.tensor(X_fmri[idx]).float(), 
                torch.tensor(X_clin[idx]).float(), 
                torch.tensor(y[idx]).float(), 
                torch.tensor(s[idx]).float())
    
    train_data = to_tens(train_idx)
    test_data = to_tens(test_idx)
    
    print(f"Data Ready: {len(train_idx)} Train, {len(test_idx)} Test")
    print(f"Features: {X_fmri.shape[1]} fMRI + {X_clin.shape[1]} Clinical")
    # Run Experiments
    res_base = train_and_evaluate((train_data, test_data), use_penalty=False)
    res_fair = train_and_evaluate((train_data, test_data), use_penalty=True)

    print("\n" + "="*60)
    print(f"{'METRIC':<20} | {'BASELINE':<15} | {'FAIR (OURS)':<15}")
    print("="*60)
    print(f"{'Accuracy':<20} | {res_base['Accuracy']:.1%}          | {res_fair['Accuracy']:.1%}")
    print(f"{'Balanced Acc':<20} | {res_base['Balanced_Acc']:.1%}          | {res_fair['Balanced_Acc']:.1%}")
    print(f"{'AUROC':<20} | {res_base['AUROC']:.3f}           | {res_fair['AUROC']:.3f}")
    print("-" * 60)
    print(f"{'ECE (Uncertainty)':<20} | {res_base['ECE']:.3f}           | {res_fair['ECE']:.3f}  <-- Lower is better")
    print(f"{'Site Leakage':<20} | {res_base['Leakage']:.1%}          | {res_fair['Leakage']:.1%}  <-- Lower is better")
    print("-" * 60)
    print(f"{'Weighted Bias':<20} | {res_base['Weighted_Bias']:.3f}           | {res_fair['Weighted_Bias']:.3f}  <-- MAIN TARGET")
    print("="*60)
    
    if res_fair['Weighted_Bias'] < res_base['Weighted_Bias']:
        print("✅ SUCCESS: Multiaccuracy penalty reduced the Weighted Bias.")
    else:
        print("⚠️ WARNING: Penalty did not reduce bias. Try increasing lambda (25.0).")

    return {
        "Baseline": res_base,
        "Fairness": res_fair
    }
if __name__ == "__main__":
    run_multimodal_sprint()