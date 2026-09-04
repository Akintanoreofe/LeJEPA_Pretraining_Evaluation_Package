# LeJEPA Pretraining and Evaluation Package

A modular PyTorch suite for backbone pretraining, leak-free classification probing, moisture regression evaluation, and data visualization.

---

##  Repository Structure

```text
lejepa_workspace/
├── lejepa_Core/
│   ├── __init__.py                # Package exports
│   ├── Backbone_pretrain.py       # Backbone loading & pretraining routines
│   ├── Evaluate_classification.py # Classification probes with 3-way splits
│   ├── Evaluate_moisture.py       # Moisture regression with group-aware splits
│   └── visualizations.py         # PCA plots, confusion matrices, & residuals
├── notebooks/                     # Interactive demonstration notebooks
├── setup.py                       # Package configuration (lejepa v0.1.0)
└── README.md                      # Documentation