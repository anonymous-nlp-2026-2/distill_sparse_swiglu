import re
import json
import os

LOG_DIR = "/root/runs/distill_sparse_swiglu"
OUT_PATH = "/root/distill_sparse_swiglu/artifacts/training_curves_data.json"

def parse_kl_log(filepath):
    """Parse KL-normalized Lagrangian log files."""
    pattern = re.compile(
        r'\[Step (\d+)/\d+\] loss=([\d.]+) kl_raw=([\d.]+) kl_norm=([\d.]+) '
        r'constraint=([\d.]+) sparsity=([\d.]+) lambda=([\d.]+) tau=([\d.]+) lr=([\d.e+-]+)'
    )
    data = {"steps": [], "loss": [], "kl_raw": [], "kl_norm": [], 
            "constraint": [], "sparsity": [], "lambda": [], "tau": [], "lr": []}
    
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data["steps"].append(int(m.group(1)))
                data["loss"].append(float(m.group(2)))
                data["kl_raw"].append(float(m.group(3)))
                data["kl_norm"].append(float(m.group(4)))
                data["constraint"].append(float(m.group(5)))
                data["sparsity"].append(float(m.group(6)))
                data["lambda"].append(float(m.group(7)))
                data["tau"].append(float(m.group(8)))
                data["lr"].append(float(m.group(9)))
    return data

def parse_bce_log(filepath):
    """Parse BCE log files (mvp_bce format)."""
    pattern = re.compile(
        r'\[Step (\d+)/\d+\] loss=([\d.]+) bce=([\d.]+) reg=([\d.]+) sparsity=([\d.]+) lr=([\d.e+-]+)'
    )
    data = {"steps": [], "loss": [], "bce": [], "reg": [], "sparsity": [], "lr": []}
    
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data["steps"].append(int(m.group(1)))
                data["loss"].append(float(m.group(2)))
                data["bce"].append(float(m.group(3)))
                data["reg"].append(float(m.group(4)))
                data["sparsity"].append(float(m.group(5)))
                data["lr"].append(float(m.group(6)))
    return data

def parse_bce_isocompute_log(filepath):
    """Parse BCE isocompute log (has elapsed time)."""
    pattern = re.compile(
        r'\[Step (\d+)/\d+\] loss=([\d.]+) bce=([\d.]+) reg=([\d.]+) sparsity=([\d.]+) lr=([\d.e+-]+)'
    )
    data = {"steps": [], "loss": [], "bce": [], "reg": [], "sparsity": [], "lr": []}
    
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data["steps"].append(int(m.group(1)))
                data["loss"].append(float(m.group(2)))
                data["bce"].append(float(m.group(3)))
                data["reg"].append(float(m.group(4)))
                data["sparsity"].append(float(m.group(5)))
                data["lr"].append(float(m.group(6)))
    return data

def parse_bce_staged_log(filepath):
    """Parse BCE comp staged log (has phase, comp loss)."""
    pattern = re.compile(
        r'\[Step (\d+)/\d+\] phase=(\w+) loss=([\d.]+) bce=([\d.]+) comp=([\d.]+) reg=([\d.]+) sparsity=([\d.]+) lr_pred=([\d.e+-]+)'
    )
    data = {"steps": [], "phase": [], "loss": [], "bce": [], "comp": [], 
            "reg": [], "sparsity": [], "lr_pred": []}
    
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data["steps"].append(int(m.group(1)))
                data["phase"].append(m.group(2))
                data["loss"].append(float(m.group(3)))
                data["bce"].append(float(m.group(4)))
                data["comp"].append(float(m.group(5)))
                data["reg"].append(float(m.group(6)))
                data["sparsity"].append(float(m.group(7)))
                data["lr_pred"].append(float(m.group(8)))
    return data

# Parse all runs
results = {}

# KL runs (3 seeds)
for seed in ["s42", "s123", "s456"]:
    path = os.path.join(LOG_DIR, f"mvp_kl_norm_v2_{seed}.log")
    if os.path.exists(path):
        data = parse_kl_log(path)
        results[f"kl_lagrangian_{seed}"] = data
        print(f"  kl_lagrangian_{seed}: {len(data['steps'])} steps")

# BCE runs (3 seeds)
for seed in ["s42", "s123", "s456"]:
    path = os.path.join(LOG_DIR, f"mvp_bce_{seed}.log")
    if os.path.exists(path):
        data = parse_bce_log(path)
        results[f"bce_vanilla_{seed}"] = data
        print(f"  bce_vanilla_{seed}: {len(data['steps'])} steps")

# BCE isocompute v2
path = os.path.join(LOG_DIR, "bce_isocompute_v2_s42.log")
if os.path.exists(path):
    data = parse_bce_isocompute_log(path)
    results["bce_isocompute_v2_s42"] = data
    print(f"  bce_isocompute_v2_s42: {len(data['steps'])} steps")

# BCE comp staged
path = os.path.join(LOG_DIR, "bce_comp_staged_s42.log")
if os.path.exists(path):
    data = parse_bce_staged_log(path)
    results["bce_comp_staged_s42"] = data
    print(f"  bce_comp_staged_s42: {len(data['steps'])} steps")

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT_PATH}")
print(f"Total runs: {len(results)}")

# Print summary stats
print("\n=== Summary ===")
for name, data in results.items():
    if "kl_raw" in data:
        final_kl = data["kl_raw"][-1] if data["kl_raw"] else None
        final_sp = data["sparsity"][-1] if data["sparsity"] else None
        final_lam = data["lambda"][-1] if data["lambda"] else None
        final_tau = data["tau"][-1] if data["tau"] else None
        print(f"  {name}: final kl_raw={final_kl}, sparsity={final_sp}, lambda={final_lam}, tau={final_tau}")
    elif "bce" in data:
        final_bce = data["bce"][-1] if data["bce"] else None
        final_sp = data["sparsity"][-1] if data["sparsity"] else None
        print(f"  {name}: final bce={final_bce}, sparsity={final_sp}")
