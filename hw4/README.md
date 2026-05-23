# PRML 2026 - Assignment 4: Transformer Ablation

This assignment reproduces the core ideas of *Attention Is All You Need* with
a small CPU-friendly experiment. The task is synthetic pointer retrieval: the
first token tells the model which later position to read, and the model must
classify the digit stored at that position.

## Files

```text
hw4/
  main.py              # experiment code
  generate_report.py   # PDF report generator
  figures/             # generated plots
  results/             # generated metrics
  report4.pdf          # final report
  作业.png             # original assignment screenshot
```

## Run

```powershell
cd hw4
python main.py
python generate_report.py
```

Required Python packages:

```powershell
pip install numpy pandas matplotlib torch reportlab
```

## Experiment

Compared variants:

- Standard Transformer with sinusoidal positional encoding
- Learned absolute positional encoding
- Simple scalar absolute positional encoding
- No positional encoding
- Shared K/V attention projection
- No residual connections
- Positional CNN baseline
- Adaptive sinusoidal positional encoding improvement

The generated report discusses positional encoding, Q/K/V separation,
residual connections, CNN replacement, and the proposed adaptive positional
encoding improvement.
