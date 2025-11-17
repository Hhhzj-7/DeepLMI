import os
import random
import string
import csv
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from dataset.process_rna import miRNA_dataset
from model import *
from utils import *

OUT_DIR = "outputs/DeepSGLMI"
os.makedirs(OUT_DIR, exist_ok=True)

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


def run_once(seed):

    # device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    set_seed(seed)

    EPOCH = 120
    DATASET = "dataset_in"


    class CustomDataset(Dataset):
        def __init__(self, data1, data2, data3):
            self.data1 = data1
            self.data2 = data2
            self.data3 = data3

        def __len__(self):
            return len(self.data1)

        def __getitem__(self, idx):
            return self.data1[idx], self.data2[idx], self.data3[idx]


    def read_fasta(filepath):
        names, seqs = [], []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for i in range(0, len(lines), 2):
            name = lines[i].strip()[1:]
            seq = lines[i + 1].strip().translate(str.maketrans('', '', string.punctuation))  # 去除标点
            names.append(name)
            seqs.append(seq)
        return names, seqs


    def generate_kmers(seqs, k):
        return [k_mers(k, seq) for seq in seqs]


    def save_vectors(vectors, path):
        np.save(path, vectors)


    def load_vectors(path):
        return np.load(path, allow_pickle=True).item()


    def build_embedding_tensor(vectors, name_list, dim=100):
        tensor = np.zeros((len(name_list), dim), dtype=np.float32)
        for name in tqdm(vectors, desc="Embedding Match"):
            if name in name_list:
                idx = name_list.index(name)
                tensor[idx] = vectors[name]
            else:
                print(f"[Warning] name '{name}' not found in fasta list.")
        return torch.tensor(tensor).float()


    def get_or_load_embedding(seqs, names, k, save_dir="save_embedding", dim=100):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"lnc_k{k}.npy")
        if os.path.exists(save_path):
            print(f"Load saved embedding for k={k} from {save_path}")
            vectors = load_vectors(save_path)
        else:
            print(f"Training embedding for k={k}")
            kmers = generate_kmers(seqs, k)
            model = train_doc2vec(kmers, names)
            vectors = get_vector(kmers, names, model)
            save_vectors(vectors, save_path)
        return build_embedding_tensor(vectors, names, dim=dim)


    lncRNA_name_list, lnc_seqs = read_fasta(os.path.join(DATASET, "lncrna.fasta"))
    miRNA_name_list, _ = read_fasta(os.path.join(DATASET, "mirna.fasta"))

    x_embedding_3 = get_or_load_embedding(lnc_seqs, lncRNA_name_list, 3)
    x_embedding_5 = get_or_load_embedding(lnc_seqs, lncRNA_name_list, 9)
    x_embedding_7 = get_or_load_embedding(lnc_seqs, lncRNA_name_list, 15)


    graph_table = pd.read_csv(os.path.join(DATASET, "index_value.csv"))
    graph_label = list(graph_table["rna"])

    # Construct Graph
    train_table = pd.read_csv(DATASET + "/node_link_train.csv")
    testval_table = pd.read_csv(DATASET + "/node_link_testval.csv")

    val_set, test_set = train_test_split(testval_table, test_size=0.5, random_state=seed)


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
        X[-1] = X[lnc_index].mean(dim=0, keepdim=True)  # linc vir
        X[-2] = X[mi_index].mean(dim=0, keepdim=True)  # mi vir
        vir_lnc_id = len(X) - 1
        vir_mi_id = len(X) - 2

        lnc_index = torch.tensor(lnc_index).cuda()
        mi_index = torch.tensor(mi_index).cuda()

        edge_lnc_to_virtual = torch.stack([lnc_index,torch.full_like(lnc_index, vir_lnc_id)], dim=0)
        edge_virtual_to_lnc = torch.stack([torch.full_like(lnc_index, vir_lnc_id),lnc_index], dim=0)

        edge_mi_to_virtual = torch.stack([mi_index,torch.full_like(mi_index, vir_mi_id)], dim=0)
        edge_virtual_to_mi = torch.stack([torch.full_like(mi_index, vir_mi_id),mi_index], dim=0)

        edge_index = torch.cat([
            train_data.edge_index,
            edge_lnc_to_virtual, edge_virtual_to_lnc,
            edge_mi_to_virtual, edge_virtual_to_mi
        ], dim=1)

        z = model.encode(X, edge_index, lnc_index, mi_index)
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

        return loss, lnc_index, mi_index


    @torch.no_grad()
    def test(data, miRNA_f, miRNA_name_list, lncRNA_f, lncRNA_name_list, lnc_index, mi_index):
        rna_to_index = pd.Series(graph_table['index'].values, index=graph_table['rna']).to_dict()
        X_t = torch.zeros(len(miRNA_name_list) + len(lncRNA_name_list) + 2, 100).to(device)

        for rna, feature in zip(miRNA_name_list, miRNA_f):
            index = rna_to_index.get(rna)
            if index is not None:
                X_t[index] = feature

        for rna, feature in zip(lncRNA_name_list, lncRNA_f):
            index = rna_to_index.get(rna)
            if index is not None:
                X_t[index] = feature

        z = model.encode(X_t, data.edge_index, lnc_index, mi_index)

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

        return f1, roc_auc, avg_precision, ndcg, precision, recall, emb, y_pred_new, y_true, y_pred


    train_set = train_table.copy()

    u_train = train_set['node1'].astype(int).tolist()
    v_train = train_set['node2'].astype(int).tolist()

    u_val = val_set['node1'].astype(int).tolist()
    v_val = val_set['node2'].astype(int).tolist()

    u_test = test_set['node1'].astype(int).tolist()
    v_test = test_set['node2'].astype(int).tolist()


    def construct_edge_index(u_list, v_list):
        u_undir = [x for pair in zip(u_list, v_list) for x in pair]
        v_undir = [x for pair in zip(v_list, u_list) for x in pair]
        return torch.tensor(u_undir), torch.tensor(v_undir)


    u_train_tensor, v_train_tensor = construct_edge_index(u_train, v_train)
    u_val_tensor = torch.tensor(u_val)
    v_val_tensor = torch.tensor(v_val)
    u_test_tensor = torch.tensor(u_test)
    v_test_tensor = torch.tensor(v_test)

    edge_index = torch.stack([u_train_tensor, v_train_tensor], dim=0)
    edge_train_index = torch.stack([torch.tensor(u_train), torch.tensor(v_train)], dim=0)
    edge_val_index = torch.stack([u_val_tensor, v_val_tensor], dim=0)
    edge_test_index = torch.stack([u_test_tensor, v_test_tensor], dim=0)

    train_data = Data(x=graph_label, edge_index=edge_index,
                      edge_label=torch.ones(len(u_train)),
                      edge_label_index=edge_train_index).to(device)

    val_neg = negative_sampling(edge_index=edge_index, num_nodes=len(graph_label),
                                num_neg_samples=edge_val_index.size(1), method='sparse')
    val_edge_label_index = torch.cat([edge_val_index, val_neg], dim=1)
    val_edge_label = torch.cat([
        torch.ones(edge_val_index.size(1)),
        torch.zeros(val_neg.size(1))
    ], dim=0)
    val_data = Data(x=graph_label, edge_index=edge_index,
                    edge_label_index=val_edge_label_index,
                    edge_label=val_edge_label).to(device)

    test_neg = negative_sampling(edge_index=edge_index, num_nodes=len(graph_label),
                                 num_neg_samples=edge_test_index.size(1), method='sparse')
    test_edge_label_index = torch.cat([edge_test_index, test_neg], dim=1)
    test_edge_label = torch.cat([
        torch.ones(edge_test_index.size(1)),
        torch.zeros(test_neg.size(1))
    ], dim=0)
    test_data = Data(x=graph_label, edge_index=edge_index,
                     edge_label_index=test_edge_label_index,
                     edge_label=test_edge_label).to(device)

    train_data.num_nodes = len(graph_label)
    val_data.num_nodes = len(graph_label)
    test_data.num_nodes = len(graph_label)

    # model and optimizer
    model = GCNNet(100, 128, 64).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    miRNA_GCN = GCN_miRNA(100).to(device)
    miRNA_feature_dataset = miRNA_dataset(is_merged=True)
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

    # Train and Validation
    results = []
    best_test_result = []
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

        loss, lnc_index, mi_index = train(miRNA_f_list, miRNA_name_list, lncRNA_f_list, lncRNA_name_list)
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

            val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec, emb, label, y_true_val, y_score_val = test(val_data, miRNA_f_list_t,
                                                                                   miRNA_name_list, lncRNA_f_list_t,
                                                                                   lncRNA_name_list, lnc_index,
                                                                                   mi_index)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc

            test_f1, test_auc, test_ap, test_ndcg, test_pre, test_rec, _, _, y_true_test, y_score_test = test(
                test_data, miRNA_f_list_t, miRNA_name_list, lncRNA_f_list_t, lncRNA_name_list, lnc_index, mi_index
            )
            best_test_result = [test_f1, test_auc, test_ap, test_ndcg, test_pre, test_rec]
            print(f'Test: {test_auc:.4f}')

        results.append([epoch, val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec])

    best_result = max(results, key=lambda x: x[2])
    print("Best result on validation set:")
    print(
        'Epoch: {}, F1: {:.3f}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f}, Pre: {:.3f}, Rec: {:.3f}'.format(*best_result)
    )
    print("Corresponding test result:")
    print(
        'Test F1: {:.3f}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f}, Pre: {:.3f}, Rec: {:.3f}'.format(*best_test_result)
    )

    return best_result, best_test_result


all_val_metrics = []
all_test_metrics = []

for seed in [1, 2, 3, 4, 5]:
    set_seed(seed)

    val_result, test_result = run_once(seed)

    all_val_metrics.append(val_result)
    all_test_metrics.append(test_result)


val_results = np.array(all_val_metrics)
test_results = np.array(all_test_metrics)

val_mean = val_results.mean(axis=0)
val_std = val_results.std(axis=0)
test_mean = test_results.mean(axis=0)
test_std = test_results.std(axis=0)

metric_names = ['F1', 'AUC', 'AP', 'NDCG', 'Precision', 'Recall']
for i, name in enumerate(metric_names):
    # print(f"Val {name}: {val_mean[i]:.3f} ± {val_std[i]:.3f}")
    print(f"Test {name}: {test_mean[i]:.3f} ± {test_std[i]:.3f}")
