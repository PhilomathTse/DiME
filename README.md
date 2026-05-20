# DiME

This repository contains the implementation of **DiME** for multimodal stance detection.

The project explores multimodal fusion and mixture-of-experts based modeling for stance detection. It includes baseline Transformer models, interaction MoE modules, dataset loaders, training scripts, and shared utility functions.

## Repository Structure

```text
DiME/
├── fastmoe/                    # FastMoE related implementation or dependency code
├── scripts/
│   └── train_scripts/
│       ├── baseline/
│       │   └── transformer/    # Training scripts for baseline Transformer models
│       └── imoe/
│           └── transformer/    # Training scripts for IMoE / DiME models
├── src/
│   ├── baseline/               # Baseline training code
│   ├── common/                 # Shared datasets, modules, fusion models, and utilities
│   └── imoe/                   # Interaction MoE / DiME model implementation
├── output.py                   # Output or evaluation related script
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Environment Setup

Create a Python environment:

```bash
conda create -n dime python=3.10 -y
conda activate dime
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the local `fastmoe` package is required, install it with:

```bash
cd fastmoe
pip install -e .
cd ..
```

## Dataset

The dataset is not included in this repository.

Please place the dataset files under:

```text
Dataset/
```

The `Dataset/` directory is ignored by Git and will not be uploaded to GitHub.

## Training

### DiME Transformer

```bash
bash scripts/train_scripts/imoe/transformer/run_mmcsd.sh
```

or:

```bash
bash scripts/train_scripts/imoe/transformer/run_mmcsd_2.sh
```

Before running experiments, please check and update the dataset paths, output paths, and GPU settings in the shell scripts.

## Main Files

- `src/baseline/train_transformer.py`: training entry for the baseline Transformer model.
- `src/imoe/train_transformer.py`: training entry for the IMoE / DiME model.
- `src/imoe/InteractionMoE.py`: implementation of the interaction mixture-of-experts module.
- `src/common/datasets/`: dataset loading and preprocessing modules.
- `src/common/modules/`: shared neural network modules.
- `src/common/fusion_models/transformer.py`: Transformer-based multimodal fusion model.
- `src/common/utils.py`: common utility functions.

## Notes

- Runtime cache files such as `__pycache__/` are ignored.
- Dataset files should be stored locally under `Dataset/`.
- Training outputs, logs, and checkpoints should be configured according to the experimental environment.
- The scripts are designed for command-line execution in a Linux or server environment.

## License

This repository is for academic research purposes.
