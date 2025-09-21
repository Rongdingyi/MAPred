# MAPred: Multi-modal Approach for Protein EC Number Prediction

MAPred is a deep learning-based tool for predicting protein enzyme classification (EC Number), combining protein sequence and structural information to achieve high-precision enzyme function prediction.



## 🛠️ Requirements

```bash
pip install torch torchvision
pip install transformers
pip install fair-esm
pip install scikit-learn
pip install pandas numpy
pip install tqdm
```

## 📊 Data Preparation

### 1. Data Format
Training data should be in CSV format with the following columns:
- `Entry`: Protein ID
- `EC number`: EC numbers (multiple EC numbers separated by semicolons)
- `Sequence`: Protein sequence

### 2. Preprocessing Steps

```bash
# 1. Prepare protein features
python prepare.py

# 2. This will generate:
# - ESM-1b sequence features (./data/esm/)
# - ProstT5 structural features (./data/3di/)
# - Distance mapping files
```

## 🚀 Usage


### 1. Train Fusion Model

```bash
# Multi-GPU training
bash scripts/train_cl.sh

# Or single GPU training
python train_cl_fuse.py \
    --model_name fuse_model \
    --model_dir ./results/fuse \
    --batch_size 36 \
    --epoch 2000
```

### 2. Train Classification Model

```bash
# Use provided script
bash scripts/train.sh

# Or run directly
python train_classification.py \
    --training_data split100_train_split_0 \
    --valid_data split100_test_split_0_curate \
    --model_name my_model \
    --epoch 400 \
    --device cuda:0
```


### 3. Model Testing

```bash
# Test classification model
python test_classification.py \
    --model_path ./data/model/model.pth \
    --test_data new \
    --device cuda:0

# Test fusion model
python infer_fuse.py
```

## 🔧 Configuration

Main hyperparameters:
- `learning_rate`: Learning rate (default: 1e-3)
- `epoch`: Number of training epochs (default: 2000)
- `batch_size`: Batch size (default: 36)
- `hidden_dim`: Hidden layer dimension (default: 512)
- `dropout`: Dropout ratio (default: 0.1)

## 📄 Output Files

- `./results/`: Model checkpoints
- `./data/model/`: Trained models
- `./results/metric_result/`: Evaluation result CSV files
- `./data/distance_map/`: Distance mapping cache

## 📚 Citation

If you use MAPred in your research, please cite our work:

```bibtex
@article{10.1093/bib/bbaf476,
    author = {Rong, Dingyi and Zhong, Bozitao and Zheng, Wenzhuo and Hong, Liang and Liu, Ning},
    title = {Autoregressive enzyme function prediction with multi-scale multi-modality fusion},
    journal = {Briefings in Bioinformatics},
    volume = {26},
    number = {5},
    pages = {bbaf476},
    year = {2025},
    month = {09},
    issn = {1477-4054},
    doi = {10.1093/bib/bbaf476},
    url = {https://doi.org/10.1093/bib/bbaf476},
    eprint = {https://academic.oup.com/bib/article-pdf/26/5/bbaf476/64273824/bbaf476.pdf},
}

```


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

