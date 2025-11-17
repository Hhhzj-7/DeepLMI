import os
import random
import string

from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data

from dataset.process_rna import miRNA_dataset
from model import *
from utils import *

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_num_threads(1)

set_seed(seed=2)
EPOCH = 200
DATASET = "dataset"
COLD_START = "both"

SAVE_PATH = "cold_data_splits"


class CustomDataset(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.data3[idx]


lnc_seq = []
unique_lnc = []

with open(DATASET + '/lncrna.fasta', 'r') as fasta:
    lines = fasta.readlines()

for i in range(0, len(lines), 2):
    rna_name = lines[i].strip()[1:]
    rna_seq = lines[i + 1].strip().translate(str.maketrans('', '', string.punctuation))
    unique_lnc.append(rna_name)
    lnc_seq.append(rna_seq)

lnc_seq_mers_3 = []
for i in lnc_seq:
    lnc_seq_mers_3.append(k_mers(3, i))

lnc_seq_mers_5 = []
for i in lnc_seq:
    lnc_seq_mers_5.append(k_mers(9, i))

lnc_seq_mers_7 = []
for i in lnc_seq:
    lnc_seq_mers_7.append(k_mers(15, i))

all_mers_3 = lnc_seq_mers_3
all_name = unique_lnc
pretrain_model_3 = train_doc2vec(all_mers_3, all_name)
vectors_3 = get_vector(all_mers_3, unique_lnc, pretrain_model_3)

all_mers_5 = lnc_seq_mers_5
all_name = unique_lnc
pretrain_model_5 = train_doc2vec(all_mers_5, all_name)
vectors_5 = get_vector(all_mers_5, all_name, pretrain_model_5)

all_mers_7 = lnc_seq_mers_7
all_name = unique_lnc
pretrain_model_7 = train_doc2vec(all_mers_7, all_name)
vectors_7 = get_vector(all_mers_7, all_name, pretrain_model_7)

# Create index
graph_table = pd.read_csv(DATASET + "/index_value.csv")
graph_label = list(graph_table["rna"])

# get lncRNA name
lncRNA_name_list = []
with open(DATASET + '/lncrna.fasta', 'r') as fasta:
    lines = fasta.readlines()

for i in range(0, len(lines), 2):
    id = lines[i].strip()[1:]
    lncRNA_name_list.append(id)

# get miRNA name
miRNA_name_list = []
with open(DATASET + '/mirna.fasta', 'r') as fasta:
    lines = fasta.readlines()

for i in range(0, len(lines), 2):
    id = lines[i].strip()[1:] 
    miRNA_name_list.append(id)

graph_embedding_3 = np.zeros((len(lncRNA_name_list), 100))
graph_embedding_5 = np.zeros((len(graph_label), 100))
graph_embedding_7 = np.zeros((len(graph_label), 100))

for node, vec in vectors_3.items():
    position = lncRNA_name_list.index(node)
    graph_embedding_3[position] = vec
x_embedding_3 = torch.tensor(graph_embedding_3).float()

for node, vec in vectors_5.items():
    position = graph_label.index(node)
    graph_embedding_5[position] = vec
x_embedding_5 = torch.tensor(graph_embedding_5).float()

for node, vec in vectors_7.items():
    position = graph_label.index(node)
    graph_embedding_7[position] = vec
x_embedding_7 = torch.tensor(graph_embedding_7).float()

# Construct DGL Graph
node_table = pd.read_csv(DATASET + "/node_link.csv")


def train(miRNA_f, miRNA_name_list, lncRNA_f, lncRNA_name_list):

    rna_to_index = pd.Series(graph_table['index'].values, index=graph_table['rna']).to_dict()
    X = torch.zeros(len(miRNA_name_list) + len(lncRNA_name_list) + 2, 100).to(device)

    lnc_index = []
    mi_index = []
    
    for rna, feature in zip(miRNA_name_list, miRNA_f):
        index = rna_to_index.get(rna)  
        if index is not None:  
            X[index] = feature  
            mi_index.append(index)
    for rna, feature in zip(lncRNA_name_list, lncRNA_f):
        index = rna_to_index.get(rna)  
        if index is not None: 
            X[index] = feature 
            lnc_index.append(index)
    X[-1] = X[lnc_index].mean(dim=0, keepdim=True) # linc vir
    X[-2] = X[mi_index].mean(dim=0, keepdim=True) # mi vir
    vir_lnc_id = len(X)-1
    vir_mi_id = len(X)-2

    lnc_index = torch.tensor(lnc_index).cuda()
    mi_index = torch.tensor(mi_index).cuda()

    edge_lnc_to_virtual = torch.stack([
        lnc_index, 
        torch.full_like(lnc_index, vir_lnc_id)
    ], dim=0)
    edge_virtual_to_lnc = torch.stack([
        torch.full_like(lnc_index, vir_lnc_id),
        lnc_index
    ], dim=0)

    edge_mi_to_virtual = torch.stack([
        mi_index, 
        torch.full_like(mi_index, vir_mi_id)
    ], dim=0)
    edge_virtual_to_mi = torch.stack([
        torch.full_like(mi_index, vir_mi_id),
        mi_index
    ], dim=0)

    edge_index = torch.cat([
        train_data.edge_index, 
        edge_lnc_to_virtual, edge_virtual_to_lnc,
        edge_mi_to_virtual, edge_virtual_to_mi
    ], dim=1)

    z = model.encode(X, edge_index,lnc_index,mi_index)
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out, _ = model.decode(z, edge_label_index)
    loss = criterion(out, edge_label)

    return loss, lnc_index,mi_index

@torch.no_grad()
def test(data, miRNA_f, miRNA_name_list, lncRNA_f, lncRNA_name_list, lnc_index,mi_index):

    rna_to_index = pd.Series(graph_table['index'].values, index=graph_table['rna']).to_dict()
    X_t = torch.zeros(len(miRNA_name_list) + len(lncRNA_name_list), 100).to(device)  # 3个miRNA有重复

    for rna, feature in zip(miRNA_name_list, miRNA_f):
        index = rna_to_index.get(rna)
        if index is not None: 
            X_t[index] = feature  
    for rna, feature in zip(lncRNA_name_list, lncRNA_f):
        index = rna_to_index.get(rna)  
        if index is not None: 
            X_t[index] = feature  

    z = model.encode(X_t, data.edge_index,lnc_index,mi_index)

    out, emb = model.decode(z, data.edge_label_index)  
    out = out.view(-1).sigmoid()  
    y_true = data.edge_label.cpu().numpy()
    y_pred = out.cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_new = (y_pred >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_new)

    roc_auc = roc_auc_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    ndcg = NDCG(y_true, y_pred)
    precision = precision_score(y_true, y_pred_new)
    recall = recall_score(y_true, y_pred_new)

    return f1, roc_auc, avg_precision, ndcg, precision, recall, emb, y_pred_new


best_epoches = []
f1_scores = []
auc_scores = []
ap_scores = []
ndcg_scores = []
pre_scores = []
rec_scores = []


lncRNAs = list(set(node_table["node1"]))
miRNAs = list(set(node_table["node2"]))

for fold in range(1, 6):

    train_csv = f'{SAVE_PATH}/{COLD_START}_cold_train_{fold}.csv'
    val_csv = f'{SAVE_PATH}/{COLD_START}_cold_val_{fold}.csv'

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    u_train = torch.tensor(train_df['node1'].to_numpy(), dtype=torch.long)
    v_train = torch.tensor(train_df['node2'].to_numpy(), dtype=torch.long)

    edge_index_train = torch.stack([
        torch.cat([u_train, v_train], dim=0),
        torch.cat([v_train, u_train], dim=0)
    ], dim=0)

    edge_label_index_train = torch.stack([u_train, v_train], dim=0)
    edge_label_train = torch.ones(u_train.size(0), dtype=torch.float)

    u_val = torch.tensor(val_df['node1'].to_numpy(), dtype=torch.long)
    v_val = torch.tensor(val_df['node2'].to_numpy(), dtype=torch.long)
    edge_label_index_val = torch.stack([u_val, v_val], dim=0)
    edge_label_val = torch.tensor(val_df['label'].to_numpy(), dtype=torch.float)

    num_nodes = int(max(
        train_df['node1'].max(), train_df['node2'].max(),
        val_df['node1'].max(), val_df['node2'].max()
    ) + 1)

    # PyG Data
    train_data = Data(
        x=None,
        edge_index=edge_index_train,
        edge_label=edge_label_train,
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

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # model and optimizer
    model = GCNNet(100, 128, 64).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    miRNA_GCN = GCN_miRNA(100).to(device)
    miRNA_feature_dataset = miRNA_dataset()
    mirna_loader = DataLoader(
    miRNA_feature_dataset, batch_size=1, num_workers=0, drop_last=False, shuffle=False
    )

    lncRNA_dataset = CustomDataset(x_embedding_3.to(device), x_embedding_5.to(device), x_embedding_7.to(device))
    lncRNA_loader = DataLoader(
    lncRNA_dataset, batch_size=1, num_workers=0, drop_last=False, shuffle=False
    )

    model_k_f = SelfAttention(100).to(device)
    lncRNA_SA = SA_lncRNA(100).to(device)
    optimizer = torch.optim.Adam(params=list(model.parameters())+list(miRNA_GCN.parameters())+list(lncRNA_SA.parameters()), lr=0.0001)    

    results = []
    best_val_auc = 0

    for epoch in range(1, EPOCH + 1):
        miRNA_GCN.train()
        model.train()
        lncRNA_SA.train()
        optimizer.zero_grad()

        miRNA_f_list = []
        lncRNA_f_list = []

        for step, batch in enumerate(mirna_loader):
            miRAN_p = miRNA_GCN(batch.to(device))
            miRNA_f_list.append(miRAN_p)
        miRNA_f_list = torch.cat(miRNA_f_list, dim=0)

        for step, batch in enumerate(lncRNA_loader):
            lncRNA_p = lncRNA_SA(batch)
            lncRNA_f_list.append(lncRNA_p)
        lncRNA_f_list = torch.cat(lncRNA_f_list, dim=0)

        loss, lnc_index,mi_index = train(miRNA_f_list, miRNA_name_list, lncRNA_f_list, lncRNA_name_list)
        loss.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            miRNA_GCN.eval()
            model.eval()
            miRNA_f_list_t = []
            for step, batch_t in enumerate(mirna_loader):
                miRAN_p_t = miRNA_GCN(batch_t.to(device))
                miRNA_f_list_t.append(miRAN_p_t)
            miRNA_f_list_t = torch.cat(miRNA_f_list_t, dim=0)

            lncRNA_f_list_t = []
            for step, batch_t in enumerate(lncRNA_loader):
                lncRNA_p_t = lncRNA_SA(batch_t)
                lncRNA_f_list_t.append(lncRNA_p_t)
            lncRNA_f_list_t = torch.cat(lncRNA_f_list_t, dim=0)

            val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec, emb, label = test(val_data, miRNA_f_list_t,
                                                                                   miRNA_name_list, lncRNA_f_list_t,
                                                                                   lncRNA_name_list, lnc_index,mi_index)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path_e = f"save_cold/Emb_best_epoch{epoch}_fold_{fold}.pth"
            torch.save({
                'embeddings': emb.clone().detach(),
                'labels': val_data.edge_label.detach().cpu()
            }, save_path_e)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')
        results.append([epoch, val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec])

    best_result = max(results, key=lambda x: x[2])
    print(
    'Best result: Epoch: {}, F1: {:.3f}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f},Pre: {:.3f}, Rec: {:.3f}'.format(
        *best_result))
    best_epoches.append(best_result[0])
    f1_scores.append(best_result[1])
    auc_scores.append(best_result[2])
    ap_scores.append(best_result[3])
    ndcg_scores.append(best_result[4])
    pre_scores.append(best_result[5])
    rec_scores.append(best_result[6])


print("-----------------------Final Results-----------------------")
print('F1 scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(f1_scores), np.std(f1_scores)))
print('AUC scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(auc_scores), np.std(auc_scores)))
print('AP scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(ap_scores), np.std(ap_scores)))
print('NDCG scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(ndcg_scores), np.std(ndcg_scores)))
print('Pre scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(pre_scores), np.std(pre_scores)))
print('Rec scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(rec_scores), np.std(rec_scores)))
print("-----------------------------------------------------------")