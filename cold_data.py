import os
import torch
import pandas as pd
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

DATASET     = "dataset"              # 放 index_value.csv / node_link.csv 的目录
SAVE_PATH   = "./cold_data_splits"   # 保存 .pt 的目录
SEED        = 2026
N_SPLITS    = 5
COLD_START  = "both"                 # 选项：'both' / 'lncRNA' / 'miRNA'

os.makedirs(SAVE_PATH, exist_ok=True)
torch.manual_seed(SEED)

graph_table = pd.read_csv(f"{DATASET}/index_value.csv")
node_table  = pd.read_csv(f"{DATASET}/node_link.csv")    # 两列：node1(node: lnc)、node2(node: mi)

# 确保是 int 索引（若原表是名字，这里就需要先映射成索引）
node_table['node1'] = node_table['node1'].astype(int)
node_table['node2'] = node_table['node2'].astype(int)

num_nodes = int(max(node_table['node1'].max(), node_table['node2'].max()) + 1)

# 唯一的 lncRNA / miRNA 索引集合
lncRNAs = sorted(node_table['node1'].unique().tolist())
miRNAs  = sorted(node_table['node2'].unique().tolist())

# 5 折划分
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
miRNA_folds = list(kfold.split(miRNAs))  # 先把 miRNA 的 5 折准备好

for fold, (train_lnc_idx, val_lnc_idx) in enumerate(kfold.split(lncRNAs), start=1):
    train_lncRNAs = [lncRNAs[i] for i in train_lnc_idx]
    val_lncRNAs   = [lncRNAs[i] for i in val_lnc_idx]

    # 对应的 miRNA 折
    train_mi_idx, val_mi_idx = miRNA_folds[fold - 1]
    train_miRNAs = [miRNAs[i] for i in train_mi_idx]
    val_miRNAs   = [miRNAs[i] for i in val_mi_idx]

    # 冷启动
    if COLD_START == "both":
        train_set = node_table[node_table["node1"].isin(train_lncRNAs) & node_table["node2"].isin(train_miRNAs)]
        val_set   = node_table[node_table["node1"].isin(val_lncRNAs)   & node_table["node2"].isin(val_miRNAs)]
    elif COLD_START == "lncRNA":
        train_set = node_table[node_table["node1"].isin(train_lncRNAs)]
        val_set   = node_table[node_table["node1"].isin(val_lncRNAs)]
    elif COLD_START == "miRNA":
        train_set = node_table[node_table["node2"].isin(train_miRNAs)]
        val_set   = node_table[node_table["node2"].isin(val_miRNAs)]
    else:
        raise ValueError("Invalid COLD_START. Choose from 'both' / 'lncRNA' / 'miRNA'.")

    print(f"[Fold {fold}] train={len(train_set)}  val={len(val_set)}")

    # 构造训练边（无向）
    u_train = torch.tensor(train_set['node1'].to_numpy(), dtype=torch.long)
    v_train = torch.tensor(train_set['node2'].to_numpy(), dtype=torch.long)
    edge_index_train = torch.stack([torch.cat([u_train, v_train], dim=0),
                                    torch.cat([v_train, u_train], dim=0)], dim=0)
    # 训练监督正样本对
    edge_label_index_train = torch.stack([u_train, v_train], dim=0)
    edge_label_train = torch.ones(u_train.size(0), dtype=torch.float)

    u_val = torch.tensor(val_set['node1'].to_numpy(), dtype=torch.long)
    v_val = torch.tensor(val_set['node2'].to_numpy(), dtype=torch.long)
    pos_edge_label_index_val = torch.stack([u_val, v_val], dim=0)
    pos_edge_label_val = torch.ones(u_val.size(0), dtype=torch.float)

    neg_edge_index_val = negative_sampling(
        edge_index=edge_index_train,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_label_index_val.size(1),
        method='sparse'
    )
    neg_edge_label_val = torch.zeros(neg_edge_index_val.size(1), dtype=torch.long)

    # 拼接正/负监督对
    edge_label_index_val = torch.cat([pos_edge_label_index_val, neg_edge_index_val], dim=1)
    edge_label_val = torch.cat([pos_edge_label_val, neg_edge_label_val], dim=0)

    # 组装 Data 对象
    train_data = Data(
        x=None,                              # 这里不强制放 features
        edge_index=edge_index_train,
        edge_label=edge_label_train,         # 可选
        edge_label_index=edge_label_index_train
    )
    train_data.num_nodes = num_nodes

    val_data = Data(
        x=None,
        edge_index=edge_index_train,
        edge_label=edge_label_val,
        edge_label_index=edge_label_index_val
    )
    val_data.num_nodes = num_nodes

    # 保存到 .pt
    out_train = os.path.join(SAVE_PATH, f"{COLD_START}_cold_train_{fold}.pt")
    out_val   = os.path.join(SAVE_PATH, f"{COLD_START}_cold_val_{fold}.pt")
    torch.save(train_data.cpu(), out_train)
    torch.save(val_data.cpu(), out_val)
    print(f"  -> saved: {out_train}")
    print(f"  -> saved: {out_val}")




# import torch
# from sklearn.model_selection import KFold
# import pandas as pd
# import os
#
# # === 初始化统计变量 ===
# train_lnc_counts, val_lnc_counts = [], []
# train_mi_counts, val_mi_counts = [], []
# train_interactions, val_interactions = [], []
#
# def set_seed(seed):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.set_printoptions(precision=20)
#
# # === 设置参数 ===
# seed = 2025
# COLD_START = 'miRNA'  # 'lncRNA' / 'miRNA' / 'both'
# DATASET = "dataset"  # your dataset folder
# set_seed(seed)
#
# # === 加载数据 ===
# graph_table = pd.read_csv(f"{DATASET}/index_value.csv")
# node_table = pd.read_csv(f"{DATASET}/node_link.csv")
#
# lncRNAs = list(set(node_table["node1"]))
# miRNAs = list(set(node_table["node2"]))
#
# kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
# miRNA_folds = list(kfold.split(miRNAs))
#
# for fold, (train_lnc_idx, val_lnc_idx) in enumerate(kfold.split(lncRNAs), 1):
#     train_lncRNAs = [lncRNAs[i] for i in train_lnc_idx]
#     val_lncRNAs = [lncRNAs[i] for i in val_lnc_idx]
#     train_miRNAs = [miRNAs[i] for i in miRNA_folds[fold - 1][0]]
#     val_miRNAs = [miRNAs[i] for i in miRNA_folds[fold - 1][1]]
#
#     if COLD_START == "both":
#         train_set = node_table[node_table["node1"].isin(train_lncRNAs) & node_table["node2"].isin(train_miRNAs)]
#         val_set = node_table[node_table["node1"].isin(val_lncRNAs) & node_table["node2"].isin(val_miRNAs)]
#     elif COLD_START == "lncRNA":
#         train_set = node_table[node_table["node1"].isin(train_lncRNAs)]
#         val_set = node_table[node_table["node1"].isin(val_lncRNAs)]
#     elif COLD_START == "miRNA":
#         train_set = node_table[node_table["node2"].isin(train_miRNAs)]
#         val_set = node_table[node_table["node2"].isin(val_miRNAs)]
#     else:
#         raise ValueError("Invalid COLD_START type")
#
#     train_lnc_counts.append(len(set(train_set["node1"])))
#     val_lnc_counts.append(len(set(val_set["node1"])))
#     train_mi_counts.append(len(set(train_set["node2"])))
#     val_mi_counts.append(len(set(val_set["node2"])))
#     train_interactions.append(len(train_set))
#     val_interactions.append(len(val_set))
#
#     print(f"[Fold {fold}]")
#     print(f"  Train: lncRNAs={train_lnc_counts[-1]}, miRNAs={train_mi_counts[-1]}, interactions={train_interactions[-1]}")
#     print(f"  Val  : lncRNAs={val_lnc_counts[-1]}, miRNAs={val_mi_counts[-1]}, interactions={val_interactions[-1]}")
#
# # === 输出平均统计 ===
# print("\n====== 5-Fold Average Statistics ======")
# print(f"Train Avg. lncRNAs: {sum(train_lnc_counts) / 5:.1f}")
# print(f"Train Avg. miRNAs: {sum(train_mi_counts) / 5:.1f}")
# print(f"Train Avg. Interactions: {sum(train_interactions) / 5:.1f}")
# print(f"Val   Avg. lncRNAs: {sum(val_lnc_counts) / 5:.1f}")
# print(f"Val   Avg. miRNAs: {sum(val_mi_counts) / 5:.1f}")
# print(f"Val   Avg. Interactions: {sum(val_interactions) / 5:.1f}")
