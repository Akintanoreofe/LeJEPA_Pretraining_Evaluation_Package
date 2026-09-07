# LeJEPA Pretraining and Evaluation Package

A modular PyTorch suite for backbone pretraining, linear-probe classification and moisture-regression evaluation, and data visualization. See the top-level README for full documentation.

---

##  Repository Structure

```text
lejepa_workspace/
├── lejepa_Core/
│   ├── __init__.py                # Package exports
│   ├── Backbone_pretrain.py       # Backbone loading & pretraining routines
│   ├── Evaluate_classification.py # Linear-probe classification (80/20 split)
│   ├── Evaluate_moisture.py       # Linear-probe moisture regression (80/20 split)
│   └── visualizations.py         # PCA plots, confusion matrices, & residuals
├── notebooks/                     # Interactive demonstration notebooks
├── tests/                         # Weight-loading round-trip tests
├── pyproject.toml / setup.py      # Package configuration (lejepa_Core v0.1.0)
└── README.md                      # Documentation