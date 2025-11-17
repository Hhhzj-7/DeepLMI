# DeepLMI

DeepLMI: Deep Feature Mining with a Globally Enhanced Graph Convolutional Network for Robust lncRNA–miRNA Interaction Prediction

---

## Directory Structure

```
DeepLMI/
├─ cold_data_splits/          # Cold-start fold splits (lncRNA/miRNA/both)
├─ dataset/                   # Cross-validation dataset
│  ├─ lncrna.fasta, mirna.fasta, mirna_id.fasta
│  ├─ mirna_lncrna_interaction.csv
│  ├─ node_link.csv, index_value.csv
│  ├─ mirna_emb/, mirna_contact/, mirna_f/
│  ├─ merged_mirna*, process_rna.py
├─ dataset_in/             # Processed independent dataset
├─ save/                      # Standard 5-fold CV results
├─ save_cold/                 # Cold-start results
├─ save_embedding/            
├─ main.py                    # Standard cross-validation
├─ main_cold_start.py         # Cold-start experiments
├─ main_independent.py              # Independent test
├─ model.py                   # DeepLMI model architecture
├─ utils.py                   # Data loading and metrics
├─ requirements.txt
└─ README.md
```

---

## Environment

Create a new conda environment:

```bash
conda create -n deeplmi python=3.10
conda activate deeplmi
pip install -r requirements.txt
```

---

## Datasets

* **Cross-validation dataset** → `dataset/`
* **Cold-start splits** → `cold_data_splits/`
* **Independent dataset** → `dataset_in/`

---

## Experiments

Three main experimental setups are implemented:

### 1. Standard Cross-Validation

Run:

```bash
python main.py
```

* Outputs to `save/`

### 2. Blind Tests

Run:

```bash
python main_cold_start.py
```

* Modes: blind lncRNA / blind miRNA / blind both
* Outputs to `save_cold/`

### 3. Independent Test

Run:

```bash
python main_independent.py
```

* Outputs to `save/`

---

## Contact

If you have any issues or questions about this paper or need assistance with reproducing the results, please contact me.

Zhijian Huang

School of Computer Science and Engineering,

Central South University

Email: zhijianhuang@csu.edu.cn
