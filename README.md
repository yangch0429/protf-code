# ProtoTF

Source code for the paper submission.

## Requirements

```
torch>=1.12
numpy
pandas
scipy
scikit-learn
tsaug
```

Install dependencies:

```bash
pip install torch numpy pandas scipy scikit-learn tsaug
```

## Data

- **UEA datasets**: Download from [timeseriesclassification.com](http://www.timeseriesclassification.com/dataset.php) and place under `data/Multivariate2018_arff/`.
- **UCR datasets**: Download from [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and place under `data/UCRArchive_2018/`.

## Usage

```bash
# UEA multivariate dataset
python main.py \
  --dataset BasicMotions \
  --archive UEA \
  --data_path /path/to/Multivariate2018_arff \
  --label_noise_type 0 \
  --label_noise_rate 0.3

# UCR univariate dataset
python main.py \
  --dataset ArrowHead \
  --archive UCR \
  --data_path /path/to/UCRArchive_2018 \
  --label_noise_type 0 \
  --label_noise_rate 0.2
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `BasicMotions` | Dataset name |
| `--archive` | `UEA` | Dataset archive (`UEA` or `UCR`) |
| `--data_path` | `../data/Multivariate2018_arff` | Path to dataset root |
| `--label_noise_type` | `0` | `0`=symmetric, `1`=asymmetric, `-1`=instance-dependent |
| `--label_noise_rate` | `0.3` | Noise rate (0–1) |
| `--ssl_epochs` | `100` | Pretraining epochs |
| `--epochs` | `100` | Main training epochs |
| `--hidden_dim` | `128` | Encoder hidden dimension |
| `--min_weight` | `0.5` | Minimum soft weight for borderline samples |

## Files

- `main.py` — Training pipeline
- `model.py` — Model architecture (`ProtoTFModel`, dual-view encoders)
- `utils.py` — Data loading, noise generation, evaluation utilities
