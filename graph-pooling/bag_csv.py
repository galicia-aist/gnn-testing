import os
import re
import pandas as pd

# Directories
results_dir = './results'
output_csv = './results/all_results.csv'
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Regex patterns
dataset_pattern = re.compile(r'^\s*d\s*:\s*(\w+)', re.MULTILINE)
model_pattern = re.compile(r'^\s*model\s*:\s*(\w+)', re.MULTILINE)
pool_pattern = re.compile(r'^\s*pool\s*:\s*(\w+)', re.MULTILINE)
bag_pattern = re.compile(r'^\s*bag_ratio\s*:\s*([\d\.]+)', re.MULTILINE)

# 🔹 NEW: single_layer pattern
single_layer_pattern = re.compile(r'^\s*single_layer\s*:\s*(True|False)', re.MULTILINE)

table_pattern = re.compile(
    r'^\s*(\d+)\s*\|\s*([\d\.]+|[A-Za-z]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)'
)

all_rows = []

for filename in os.listdir(results_dir):
    if not filename.endswith('.txt'):
        continue
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()

        # Extract dataset, model, pool, bag ratio
        dataset_match = dataset_pattern.search(content)
        model_match = model_pattern.search(content)
        pool_match = pool_pattern.search(content)
        bag_match = bag_pattern.search(content)
        single_layer_match = single_layer_pattern.search(content)

        if not (dataset_match and model_match and pool_match and bag_match and single_layer_match):
            continue

        dataset = dataset_match.group(1)
        model = model_match.group(1)
        pool = pool_match.group(1)
        bag_ratio = float(bag_match.group(1)) * 100  # convert to %
        single_layer = single_layer_match.group(1) == "True"  # bool

        # Extract table rows
        lines = content.splitlines()
        dash_count = 0
        epochs = []
        accuracies = []
        aucs = []
        times = []

        for line in lines:
            if line.strip().startswith('---'):
                dash_count += 1
                continue

            if dash_count == 2:
                match = table_pattern.match(line)
                if match:
                    epoch, loss, acc, auc, time_val = match.groups()
                    epochs.append(int(epoch))
                    accuracies.append(float(acc))
                    aucs.append(float(auc))
                    times.append(float(time_val))
            elif dash_count > 2:
                # Table ends at third ---
                break

        if not epochs:
            continue

        # Determine best accuracy and best AUC
        best_acc_val = max(accuracies)
        best_auc_val = max(aucs)

        # Store all rows
        for e, acc, auc, t in zip(epochs, accuracies, aucs, times):
            row = {
                'dataset': dataset,
                'model': model,
                'pool': pool,
                'bag_ratio': bag_ratio,
                'single_layer': single_layer,  # 🔹 NEW COLUMN
                'epoch': e,
                'accuracy': acc,
                'auc': auc,
                'time': t,
                'best_acc': 1 if acc == best_acc_val else 0,
                'best_auc': 1 if auc == best_auc_val else 0
            }
            all_rows.append(row)

# Save to CSV
df = pd.DataFrame(all_rows)
df.to_csv(output_csv, index=False)
print(f"✅ All results saved to {output_csv}")
