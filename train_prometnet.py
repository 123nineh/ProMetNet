from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, recall_score, \
    precision_score, accuracy_score, confusion_matrix

from ProMetNet import ProMetNet, PathwayGraph, ProMetNetExplainer, ImportanceNetwork
from util_for_examples import fit_data_matrix_to_network_input, generate_data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

random_seed = 2026
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

input_prot_data = pd.read_csv("./test_data/prot.tsv", sep="\t")
input_meta_data = pd.read_csv("./test_data/meta.tsv", sep="\t")
prot_group = pd.read_csv("./test_data/prot_group.tsv", sep="\t")
meta_group = pd.read_csv("./test_data/meta_group.tsv", sep="\t")

translation = pd.read_csv("./mapping/translations.tsv", sep="\t")
pathways = pd.read_csv("./mapping/pathways.tsv", sep="\t")

network = PathwayGraph(
    input_prot_data=input_prot_data,
    input_meta_data=input_meta_data,
    prot_group=prot_group,
    meta_group=meta_group,
    pathways=pathways,
    mapping=translation,
    input_data_column="Features",
    source_column="child",
    target_column="parent"
)
print(network.get_connectivity_matrices(1))
prometnet = ProMetNet(
    network=network,
    n_layers=4,
    dropout=0.1,
    validate=True,
    residual=False,
    device="cuda",
    activation='tanh',
    learning_rate=0.001,
    n_outputs=2
)
print(prometnet.layers)

batchsize = 12

# 1. Data Preparation
input_data, design_matrix = network.Integrator()
protein_matrix = fit_data_matrix_to_network_input(input_data, features=network.inputs)
X, y = generate_data(protein_matrix, design_matrix=design_matrix)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_seed)

num_epochs = 50
num_classes = len(np.unique(y))

# 2. Internal Validation
optimizer = prometnet.configure_optimizers()[0][0]
best_loss = float('inf')
patience_counter = 0

auc_data, prauc_data, f1_data, accuracy_data, precision_data, recall_data = [], [], [], [], [], []
results_data_inter = []
fpr_tpr_data, precision_recall_data = [], []
all_targets = []
all_probs = []

for epoch in range(num_epochs):
    prometnet.train()
    total_loss = 0.0
    total_accuracy = 0

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=prometnet.device),
        torch.tensor(y_train, dtype=torch.long, device=prometnet.device)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to("cuda", non_blocking=True)
        prometnet.to("cuda")
        outputs = prometnet(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += torch.sum(torch.argmax(outputs, axis=1) == targets) / len(targets)

        y_prob = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(y_prob.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())


    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_auc = roc_auc_score(all_targets, all_probs)
    print(f'Epoch {epoch}, Average Accuracy {avg_accuracy}, Average Auc {avg_auc}, Average Loss: {avg_loss}')

    # Validation set evaluation
    prometnet.eval()
    with torch.no_grad():
        outputs = prometnet(torch.tensor(X_test, dtype=torch.float32, device=prometnet.device))
        y_prob = outputs.cpu().numpy()[:, 1]
        y_pred = torch.argmax(outputs, axis=1)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr_tpr_data.append((fpr, tpr))

    auc_score = roc_auc_score(y_test, y_prob)
    auc_data.append(auc_score)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    precision_recall_data.append((precision, recall))

    prauc_score = auc(recall, precision)
    prauc_data.append(prauc_score)

    f1 = f1_score(y_test, y_pred)
    f1_data.append(f1)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_data.append(accuracy)
    pr = precision_score(y_test, y_pred)
    rcl = recall_score(y_test, y_pred)

    precision_data.append(pr)
    recall_data.append(rcl)

    conf_matrix = confusion_matrix(y_test, y_pred)

mean_auc = np.mean(auc_data)
std_auc = np.std(auc_data)
mean_prauc = np.mean(prauc_data)
std_prauc = np.std(prauc_data)
mean_f1 = np.mean(f1_data)
mean_accuracy = np.mean(accuracy_data)
mean_precision = np.mean(precision_data)
mean_recall = np.mean(recall_data)


results_data_inter.append(
    [mean_auc, std_auc, mean_prauc, std_prauc, mean_f1, mean_accuracy, mean_precision, mean_recall])

print(f'ProMetNet - Mean AUC (Internal): {mean_auc:.4f} ± {std_auc:.4f}')
print(f'ProMetNet - Mean PRAUC (Internal): {mean_prauc:.4f} ± {std_prauc:.4f}')
print(f'ProMetNet - Mean F1 (Internal): {mean_f1:.4f}')
print(f'ProMetNet - Mean Accuracy (Internal): {mean_accuracy:.4f}')
print(f'ProMetNet - Mean Precision (Internal): {mean_precision:.4f}')
print(f'ProMetNet - Mean Recall (Internal): {mean_recall:.4f}')

# save results
columns = [
    "mean_auc", "std_auc",
    "mean_prauc", "std_prauc",
    "mean_f1", "mean_accuracy",
    "mean_precision", "mean_recall"
]

df_results = pd.DataFrame(results_data_inter, columns=columns)
df_results.to_csv("./test_results_v2/results.csv", index=False)


# 3. prometnet explain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prometnet.to(device)
prometnet.eval()

explainer = ProMetNetExplainer(prometnet)
test_data = torch.Tensor(X)
background_data = torch.Tensor(X)

background_data = background_data.to(device)
test_data = test_data.to(device)

importance_df = explainer.explain(test_data, background_data)
importance_df.to_csv("./test_results_v2/importance_final.csv", index=False)

IG = ImportanceNetwork(importance_df, norm_method="DegNorm")

img = IG.plot_complete_sankey(
    show_top_n=5,
    node_cmap='Accent',
    edge_cmap='Accent'
)
# # Optional saving
img.write_html("./test_results_v2/sankey.html")

img.show()

subimg = IG.plot_subgraph_sankey(
    query_node='HMDB0000122',
    upstream=False,
    cmap='Accent'
)
subimg.write_html("./test_results_v2/subsankey.html")
subimg.show()
