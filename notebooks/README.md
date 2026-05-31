# 🏀 March Madness Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Cookiecutter](https://img.shields.io/badge/CCDS-Project%20Template-328F97?logo=cookiecutter)
![License](https://img.shields.io/badge/License-MIT-green)

> **Final project for the BC Applied Analytics Program** — a structured machine learning pipeline for analyzing and predicting NCAA March Madness outcomes.

---

## 📖 Overview

This project applies data analytics and machine learning techniques to NCAA March Madness basketball data. It follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) project template for a clean, reproducible, and modular structure — from raw data ingestion all the way to trained model predictions and visualizations.

---

## 🗂️ Project Structure

```
├── LICENSE
├── Makefile                    <- Convenience commands (e.g. `make data`, `make train`)
├── README.md
├── data
│   ├── external                <- Data from third-party sources
│   ├── interim                 <- Intermediate, transformed data
│   ├── processed               <- Final datasets ready for modeling
│   └── raw                     <- Original, immutable data dump
│
├── docs                        <- MkDocs project documentation
│
├── models                      <- Trained models, predictions, and summaries
│
├── notebooks                   <- Jupyter notebooks for exploration and analysis
│                                  (named: `1.0-initials-description`)
│
├── pyproject.toml              <- Project metadata and tool configuration
├── requirements.txt            <- Python dependencies
├── setup.cfg                   <- Flake8 configuration
├── reports
│   └── figures                 <- Generated charts and visualizations
│
└── MarchMadness/               <- Main Python package
    ├── __init__.py
    ├── config.py               <- Path configuration and environment setup
    ├── dataset.py              <- Data loading and preprocessing scripts
    ├── features.py             <- Feature engineering pipeline
    ├── plots.py                <- Visualization generation
    └── modeling/
        ├── __init__.py
        ├── train.py            <- Model training
        └── predict.py          <- Model inference / predictions
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- `pip` or a virtual environment manager (e.g. `venv`, `conda`)

### 1. Clone the repository

```bash
git clone https://github.com/colin-truran/BC-AppliedAnalyticsProject.git
cd BC-AppliedAnalyticsProject
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set up environment variables

Copy `.env.example` to `.env` and fill in any required values:

```bash
cp .env.example .env
```

---

## 🚀 Usage

The project uses a step-by-step pipeline. Each step can be run individually or orchestrated via `make`.

### Process raw data

```bash
python -m MarchMadness.dataset
```

### Generate features

```bash
python -m MarchMadness.features
```

### Train the model

```bash
python -m MarchMadness.modeling.train
```

### Run predictions

```bash
python -m MarchMadness.modeling.predict
```

### Generate visualizations

```bash
python -m MarchMadness.plots
```

### Or use Makefile shortcuts

```bash
make data       # Process raw data
make features   # Generate features
make train      # Train model
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Typer** | CLI interface for pipeline scripts |
| **Loguru** | Structured logging |
| **tqdm** | Progress bars |
| **python-dotenv** | Environment variable management |
| **Jupyter** | Exploratory data analysis |
| **MkDocs** | Project documentation |
| **Flake8** | Code linting |

---

## 📓 Notebooks

Jupyter notebooks live in the `notebooks/` directory and follow the naming convention:

```
<step>.<version>-<initials>-<description>.ipynb
```

For example: `1.0-ct-initial-data-exploration.ipynb`

---

## 📊 Reports & Figures

Generated reports (HTML, PDF) and figures are saved to the `reports/` directory. Figures specifically go into `reports/figures/`.

---

## 🤝 Contributing

This is an academic project. If you'd like to suggest improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push the branch (`git push origin feature/my-improvement`)
5. Open a Pull Request

---

## 📄 License

This project is open-source. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Colin Truran**  
BC Applied Analytics Program  
[GitHub](https://github.com/colin-truran)
