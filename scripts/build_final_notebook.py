"""Build the consolidated MDLC notebook for the MarchMadness project.

Run with:
    python scripts/build_final_notebook.py

This produces notebooks/MarchMadness_Final.ipynb — a single end-to-end
notebook organized by the Model Development Life Cycle (MDLC). It does
NOT re-do every weekly assignment; it only includes the final pipeline.
"""

from pathlib import Path

import nbformat as nbf

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_DIR / "notebooks" / "MarchMadness_Final.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


cells = []

# ---------------------------------------------------------------------------
# 0. Title / Overview
# ---------------------------------------------------------------------------
cells.append(md(
    "# MarchMadness Tournament Game Predictor — End-to-End Project\n"
    "\n"
    "**Goal:** Given two NCAA Men's Basketball tournament teams, predict the "
    "probability that **Team 1 (the lower-numbered seed) wins** the game.\n"
    "\n"
    "This notebook is the consolidated final deliverable. It walks an end "
    "user (instructor, peer, or future me) through every major step of the "
    "**Model Development Life Cycle (MDLC)** in one place:\n"
    "\n"
    "1. Setup\n"
    "2. Data Ingestion\n"
    "3. Data Cleaning\n"
    "4. Feature Engineering\n"
    "5. Exploratory Data Analysis (EDA)\n"
    "6. Preprocessing Validation\n"
    "7. Train / Validation / Test Split\n"
    "8. Baseline Model — Logistic Regression\n"
    "9. Final Model — Random Forest\n"
    "10. Model Evaluation\n"
    "11. Model Explainability\n"
    "12. Bias Detection & Mitigation\n"
    "13. Model Deployment (serialize)\n"
    "14. Monitoring (data drift + concept drift)\n"
    "15. Production Inference Example\n"
    "\n"
    "Run the cells top-to-bottom. Every cell is self-contained relative to "
    "the cells above it; nothing depends on a separate week's notebook."
))

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup\n\nImport libraries and define project paths."))

cells.append(code(
    "import warnings\n"
    "from pathlib import Path\n"
    "import pickle\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.impute import SimpleImputer\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.inspection import permutation_importance\n"
    "from sklearn.metrics import (\n"
    "    accuracy_score, precision_score, recall_score,\n"
    "    f1_score, roc_auc_score, confusion_matrix,\n"
    ")\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "# Locate the project root whether the notebook is run from notebooks/ or repo root.\n"
    "PROJECT_DIR = Path.cwd().resolve()\n"
    "if not (PROJECT_DIR / 'data').exists():\n"
    "    PROJECT_DIR = PROJECT_DIR.parent\n"
    "\n"
    "RAW_DIR        = PROJECT_DIR / 'data' / 'raw'\n"
    "INTERIM_DIR    = PROJECT_DIR / 'data' / 'interim'\n"
    "PROCESSED_DIR  = PROJECT_DIR / 'data' / 'processed'\n"
    "EXTERNAL_DIR   = PROJECT_DIR / 'data' / 'external'\n"
    "MODELS_DIR     = PROJECT_DIR / 'models'\n"
    "\n"
    "for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "RANDOM_STATE = 226\n"
    "print('Project root:', PROJECT_DIR)"
))

# ---------------------------------------------------------------------------
# 2. Data Ingestion
# ---------------------------------------------------------------------------
cells.append(md(
    "## 2. Data Ingestion\n"
    "\n"
    "Three raw inputs sit in `data/raw/`:\n"
    "\n"
    "| File | Purpose |\n"
    "|------|---------|\n"
    "| `DEV _ March Madness.csv` | Per-team season profile (KenPom + meta) |\n"
    "| `MTeamSpellings.csv` | Maps team-name spellings → numeric `TeamID` |\n"
    "| `MarchMadnessGameStats2003-2024.csv` | Historical tournament game results |\n"
))

cells.append(code(
    "dev_seasons   = pd.read_csv(RAW_DIR / 'DEV _ March Madness.csv')\n"
    "team_spellings = pd.read_csv(RAW_DIR / 'MTeamSpellings.csv')\n"
    "game_stats     = pd.read_csv(RAW_DIR / 'MarchMadnessGameStats2003-2024.csv')\n"
    "\n"
    "print(f'dev_seasons    : {dev_seasons.shape}')\n"
    "print(f'team_spellings : {team_spellings.shape}')\n"
    "print(f'game_stats     : {game_stats.shape}')\n"
    "dev_seasons.head(2)"
))

# ---------------------------------------------------------------------------
# 3. Data Cleaning
# ---------------------------------------------------------------------------
cells.append(md(
    "## 3. Data Cleaning\n"
    "\n"
    "Two outputs:\n"
    "\n"
    "- **`team_season`** — one row per (Season, TeamID) with their season profile + how far they advanced in the tournament.\n"
    "- **`tourney_games`** — one row per tournament game between two March-Madness teams, with each side's full season profile attached.\n"
    "\n"
    "We then drop columns where >30% of values are missing, since they are unreliable predictors."
))

cells.append(code(
    "# 3.1 — attach numeric TeamID to every season profile via the spelling map\n"
    "team_spellings['nameKey'] = team_spellings['TeamNameSpelling'].astype(str).str.strip().str.lower()\n"
    "dev_seasons['nameKey']    = dev_seasons['Mapped ESPN Team Name'].astype(str).str.strip().str.lower()\n"
    "\n"
    "dev_seasons = dev_seasons.merge(\n"
    "    team_spellings[['TeamID', 'nameKey']].drop_duplicates(),\n"
    "    on='nameKey', how='left',\n"
    ")\n"
    "\n"
    "team_season = (\n"
    "    dev_seasons.dropna(subset=['TeamID'])\n"
    "               .drop_duplicates(['Season', 'TeamID'])\n"
    "               .drop(columns=['nameKey'])\n"
    "               .copy()\n"
    ")\n"
    "print('team_season raw:', team_season.shape)"
))

cells.append(code(
    "# 3.2 — LEAKAGE GUARD.  The DEV file mixes pre-tournament KenPom snapshots\n"
    "# with end-of-season metrics that include the tournament we are trying to\n"
    "# predict (e.g. AdjOE, Tournament Winner?, sorting indices).  Training on\n"
    "# those would let the model peek at the answer.  We therefore keep ONLY\n"
    "# columns that are knowable BEFORE the tournament tips off:\n"
    "#   - identifiers (Season, TeamID, team name)\n"
    "#   - the official seed and conference\n"
    "#   - every column whose name starts with 'Pre-Tournament.'\n"
    "# We also keep 'Post-Season Tournament' temporarily so we can filter to\n"
    "# the 64 March-Madness teams, then drop it before modelling.\n"
    "SAFE_KEEP = {\n"
    "    'Season', 'TeamID', 'Mapped ESPN Team Name',\n"
    "    'Short Conference Name', 'Seed', 'Post-Season Tournament',\n"
    "}\n"
    "safe_cols = [c for c in team_season.columns\n"
    "             if c in SAFE_KEEP or c.startswith('Pre-Tournament.')]\n"
    "team_season = team_season[safe_cols].copy()\n"
    "\n"
    "# Parse Seed to a clean integer 1..16 (raw values are strings like '1','16').\n"
    "team_season['Seed'] = pd.to_numeric(team_season['Seed'], errors='coerce')\n"
    "\n"
    "print('team_season cleaned:', team_season.shape)\n"
    "print('kept cols sample:', safe_cols[:8], '...')"
))

cells.append(code(
    "# 3.3 — keep only games where BOTH teams qualified for March Madness, build a\n"
    "#        symmetric (team1, team2) representation, and define the binary target y.\n"
    "mm_teams = (\n"
    "    team_season.loc[team_season['Post-Season Tournament'].eq('March Madness'),\n"
    "                    ['Season', 'TeamID']]\n"
    "    .drop_duplicates()\n"
    ")\n"
    "\n"
    "games = game_stats[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'WScore', 'LScore']].copy()\n"
    "games = games.merge(mm_teams.rename(columns={'TeamID': 'WTeamID'}),\n"
    "                    on=['Season', 'WTeamID'], how='inner')\n"
    "games = games.merge(mm_teams.rename(columns={'TeamID': 'LTeamID'}),\n"
    "                    on=['Season', 'LTeamID'], how='inner')\n"
    "\n"
    "# Convention: team1 = lower TeamID, team2 = higher TeamID.\n"
    "games['team1Id']      = games[['WTeamID', 'LTeamID']].min(axis=1)\n"
    "games['team2Id']      = games[['WTeamID', 'LTeamID']].max(axis=1)\n"
    "games['winnerTeamId'] = games['WTeamID']\n"
    "games['y']            = (games['team1Id'] == games['WTeamID']).astype(int)  # 1 if team1 won\n"
    "\n"
    "# Attach human-readable team names\n"
    "names = (team_season[['Season', 'TeamID', 'Mapped ESPN Team Name']]\n"
    "         .drop_duplicates(['Season', 'TeamID'])\n"
    "         .rename(columns={'Mapped ESPN Team Name': 'teamName'}))\n"
    "games = games.merge(names.rename(columns={'TeamID': 'team1Id', 'teamName': 'team1Name'}),\n"
    "                    on=['Season', 'team1Id'], how='left')\n"
    "games = games.merge(names.rename(columns={'TeamID': 'team2Id', 'teamName': 'team2Name'}),\n"
    "                    on=['Season', 'team2Id'], how='left')\n"
    "\n"
    "# Attach each team's full season profile\n"
    "t1 = team_season.rename(columns={'TeamID': 'team1Id'}).copy()\n"
    "t1 = t1.rename(columns={c: f'team1_{c}' for c in t1.columns if c not in ['Season', 'team1Id']})\n"
    "t2 = team_season.rename(columns={'TeamID': 'team2Id'}).copy()\n"
    "t2 = t2.rename(columns={c: f'team2_{c}' for c in t2.columns if c not in ['Season', 'team2Id']})\n"
    "\n"
    "tourney_games = games.merge(t1, on=['Season', 'team1Id'], how='left') \\\n"
    "                     .merge(t2, on=['Season', 'team2Id'], how='left')\n"
    "\n"
    "# Drop columns with >30% missing values (unreliable signal)\n"
    "KEY_COLS = {\n"
    "    'Season', 'DayNum', 'team1Id', 'team2Id', 'winnerTeamId', 'y',\n"
    "    'WTeamID', 'LTeamID', 'WScore', 'LScore', 'team1Name', 'team2Name',\n"
    "    'team1_Post-Season Tournament', 'team2_Post-Season Tournament',\n"
    "}\n"
    "missing_frac = tourney_games.isna().mean()\n"
    "drop_high_na = [c for c, frac in missing_frac.items() if frac > 0.30 and c not in KEY_COLS]\n"
    "tourney_games = tourney_games.drop(columns=drop_high_na)\n"
    "\n"
    "print(f'Dropped {len(drop_high_na)} columns with >30% missing values')\n"
    "print('tourney_games:', tourney_games.shape)"
))

# ---------------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------------
cells.append(md(
    "## 4. Feature Engineering\n"
    "\n"
    "The model only needs to know *how the two teams differ*, so we transform "
    "every paired `team1_X` / `team2_X` column into a single `X_diff` column. "
    "We also add five **shooting-efficiency rates** (FG%, 3PT%, FT%, eFG%, "
    "AST/TO ratio) computed from the raw box-score totals."
))

cells.append(code(
    "# 4.1 — symmetric difference features for every paired column\n"
    "team1_cols = [c for c in tourney_games.columns if c.startswith('team1_')]\n"
    "team2_cols = [c for c in tourney_games.columns if c.startswith('team2_')]\n"
    "\n"
    "shared_stats = sorted(\n"
    "    {c.replace('team1_', '') for c in team1_cols}\n"
    "    & {c.replace('team2_', '') for c in team2_cols}\n"
    ")\n"
    "\n"
    "for stat in shared_stats:\n"
    "    a = tourney_games[f'team1_{stat}']\n"
    "    b = tourney_games[f'team2_{stat}']\n"
    "    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):\n"
    "        tourney_games[f'{stat}_diff'] = a - b\n"
    "\n"
    "diff_cols = [c for c in tourney_games.columns if c.endswith('_diff')]\n"
    "print(f'Created {len(diff_cols)} differential features')"
))

cells.append(code(
    "# 4.2 — shooting-efficiency rates from the raw box-score CSV\n"
    "winner_box = game_stats[['Season', 'WTeamID', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',\n"
    "                         'WFTM', 'WFTA', 'WAst', 'WTO']].copy()\n"
    "winner_box.columns = ['Season', 'TeamID', 'FGM', 'FGA', 'FGM3', 'FGA3',\n"
    "                      'FTM', 'FTA', 'AST', 'TO']\n"
    "loser_box = game_stats[['Season', 'LTeamID', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',\n"
    "                        'LFTM', 'LFTA', 'LAst', 'LTO']].copy()\n"
    "loser_box.columns = winner_box.columns\n"
    "\n"
    "team_totals = pd.concat([winner_box, loser_box], ignore_index=True)\n"
    "team_totals = team_totals.groupby(['Season', 'TeamID'], as_index=False).sum()\n"
    "\n"
    "def safe_pct(num, den):\n"
    "    return np.where(den == 0, 0.0, (num / den) * 100)\n"
    "\n"
    "team_totals['fgPct']      = safe_pct(team_totals['FGM'],  team_totals['FGA']).round(2)\n"
    "team_totals['threePtPct'] = safe_pct(team_totals['FGM3'], team_totals['FGA3']).round(2)\n"
    "team_totals['ftPct']      = safe_pct(team_totals['FTM'],  team_totals['FTA']).round(2)\n"
    "team_totals['eFgPct']     = safe_pct(team_totals['FGM'] + 0.5 * team_totals['FGM3'],\n"
    "                                     team_totals['FGA']).round(2)\n"
    "team_totals['astTo']      = np.where(team_totals['TO'] == 0, 0.0,\n"
    "                                     team_totals['AST'] / team_totals['TO']).round(2)\n"
    "\n"
    "RATE_COLS = ['fgPct', 'threePtPct', 'ftPct', 'eFgPct', 'astTo']\n"
    "team_rates = team_totals[['Season', 'TeamID'] + RATE_COLS]\n"
    "\n"
    "# Attach team1 and team2 efficiency rates and create their differences\n"
    "rates_t1 = team_rates.rename(columns={'TeamID': 'team1Id',\n"
    "    **{c: f'team{c[0].upper()}{c[1:]}' for c in RATE_COLS}})\n"
    "rates_t2 = team_rates.rename(columns={'TeamID': 'team2Id',\n"
    "    **{c: f'opp{c[0].upper()}{c[1:]}' for c in RATE_COLS}})\n"
    "\n"
    "tourney_games = tourney_games.merge(rates_t1, on=['Season', 'team1Id'], how='left')\n"
    "tourney_games = tourney_games.merge(rates_t2, on=['Season', 'team2Id'], how='left')\n"
    "\n"
    "for c in RATE_COLS:\n"
    "    Cap = c[0].upper() + c[1:]\n"
    "    tourney_games[f'{c}Diff'] = (tourney_games[f'team{Cap}'] - tourney_games[f'opp{Cap}']).round(2)\n"
    "\n"
    "# Persist the final modeling-ready dataset.\n"
    "matchup = tourney_games.copy()\n"
    "matchup_path = PROCESSED_DIR / 'matchupDiff_week5_features.csv'\n"
    "matchup.to_csv(matchup_path, index=False)\n"
    "print(f'Saved modeling dataset to {matchup_path.name}: {matchup.shape}')"
))

# ---------------------------------------------------------------------------
# 5. EDA
# ---------------------------------------------------------------------------
cells.append(md(
    "## 5. Exploratory Data Analysis\n"
    "\n"
    "A handful of plots to sanity-check the engineered differentials. We "
    "expect any feature that meaningfully predicts winning to have a clearly "
    "different distribution between wins (`y=1`) and losses (`y=0`)."
))

cells.append(code(
    "# 5.1 — which differentials separate winners from losers the most?\n"
    "diff_cols_all = [c for c in matchup.columns if c.endswith('_diff') or c.endswith('Diff')]\n"
    "means_by_outcome = matchup.groupby('y')[diff_cols_all].mean()\n"
    "mean_gap = (means_by_outcome.loc[1] - means_by_outcome.loc[0]).sort_values()\n"
    "top_gap = mean_gap.abs().sort_values(ascending=False).head(15).index\n"
    "\n"
    "plt.figure(figsize=(8, 5))\n"
    "plt.barh(top_gap, mean_gap[top_gap])\n"
    "plt.axvline(0, linestyle='--', color='black')\n"
    "plt.title('Top 15 differentials separating winners (y=1) from losers (y=0)')\n"
    "plt.xlabel('Mean(y=1) − Mean(y=0)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(code(
    "# 5.2 — win rate vs seed differential\n"
    "if 'Seed_diff' in matchup.columns:\n"
    "    win_rate_by_seed = matchup.groupby('Seed_diff')['y'].mean()\n"
    "    plt.figure(figsize=(8, 4))\n"
    "    plt.plot(win_rate_by_seed.index, win_rate_by_seed.values, marker='o')\n"
    "    plt.axhline(0.5, linestyle='--', color='gray')\n"
    "    plt.title('Win rate of team1 vs Seed differential')\n"
    "    plt.xlabel('team1 seed − team2 seed (negative = team1 is the better seed)')\n"
    "    plt.ylabel('P(team1 wins)')\n"
    "    plt.tight_layout()\n"
    "    plt.show()"
))

# ---------------------------------------------------------------------------
# 6. Preprocessing Validation
# ---------------------------------------------------------------------------
cells.append(md(
    "## 6. Preprocessing Validation\n"
    "\n"
    "Before training, sanity-check the dataset for: shape, missing values, "
    "duplicate rows, and obviously skewed features. Failures here would mean "
    "we need to revisit cleaning."
))

cells.append(code(
    "print('Dataset shape         :', matchup.shape)\n"
    "print('Duplicate rows        :', matchup.duplicated().sum())\n"
    "print('Class balance (y mean):', round(matchup['y'].mean(), 3))\n"
    "\n"
    "missing_summary = matchup.isna().mean().sort_values(ascending=False)\n"
    "missing_summary = missing_summary[missing_summary > 0]\n"
    "print(f'Columns with any missing values: {len(missing_summary)}')\n"
    "missing_summary.head(10)"
))

# ---------------------------------------------------------------------------
# 7. Train / Val / Test split
# ---------------------------------------------------------------------------
cells.append(md(
    "## 7. Train / Validation / Test Split\n"
    "\n"
    "Tournaments are time-ordered, so we split **by season** to avoid leaking "
    "future information into training:\n"
    "\n"
    "- **Train**: seasons ≤ 2021\n"
    "- **Validation**: 2022\n"
    "- **Test**: 2023 – 2025\n"
    "\n"
    "We also drop columns that would obviously leak the label (final scores, "
    "winner ID, team names, raw team IDs)."
))

cells.append(code(
    "matchup['y'] = matchup['y'].astype(int)\n"
    "\n"
    "LEAK_COLS = [\n"
    "    'WScore', 'LScore', 'WTeamID', 'LTeamID', 'winnerTeamId',\n"
    "    'team1Name', 'team2Name', 'Unnamed: 0',\n"
    "]\n"
    "ID_COLS   = ['team1Id', 'team2Id']\n"
    "DROP_COLS = ['Season', 'y'] + LEAK_COLS + ID_COLS\n"
    "\n"
    "train_df = matchup[matchup['Season'] <= 2021].copy()\n"
    "val_df   = matchup[matchup['Season'] == 2022].copy()\n"
    "test_df  = matchup[matchup['Season'].between(2023, 2025)].copy()\n"
    "\n"
    "def build_xy(split_df, feature_cols=None):\n"
    "    X = split_df.drop(columns=DROP_COLS, errors='ignore').select_dtypes(include=[np.number])\n"
    "    if feature_cols is None:\n"
    "        feature_cols = X.columns.tolist()\n"
    "    else:\n"
    "        X = X.reindex(columns=feature_cols)\n"
    "    y = split_df['y'].astype(int)\n"
    "    return X, y, feature_cols\n"
    "\n"
    "X_train, y_train, FEATURE_COLS = build_xy(train_df)\n"
    "X_val,   y_val,   _            = build_xy(val_df,  FEATURE_COLS)\n"
    "X_test,  y_test,  _            = build_xy(test_df, FEATURE_COLS)\n"
    "\n"
    "print(f'Train: {X_train.shape}  pos rate {y_train.mean():.3f}')\n"
    "print(f'Val  : {X_val.shape}  pos rate {y_val.mean():.3f}')\n"
    "print(f'Test : {X_test.shape}  pos rate {y_test.mean():.3f}')\n"
    "print(f'Number of model features: {len(FEATURE_COLS)}')"
))

cells.append(code(
    "# Reusable scoring helper\n"
    "def score(y_true, proba, threshold=0.5):\n"
    "    pred = (proba >= threshold).astype(int)\n"
    "    return {\n"
    "        'accuracy' : accuracy_score(y_true, pred),\n"
    "        'precision': precision_score(y_true, pred, zero_division=0),\n"
    "        'recall'   : recall_score(y_true, pred, zero_division=0),\n"
    "        'f1'       : f1_score(y_true, pred, zero_division=0),\n"
    "        'rocAuc'   : roc_auc_score(y_true, proba),\n"
    "    }\n"
    "\n"
    "def report(name, pipe, threshold=0.5):\n"
    "    rows = []\n"
    "    for split_name, X, y in [('train', X_train, y_train),\n"
    "                             ('val',   X_val,   y_val),\n"
    "                             ('test',  X_test,  y_test)]:\n"
    "        proba = pipe.predict_proba(X)[:, 1]\n"
    "        rows.append({'model': name, 'split': split_name, **score(y, proba, threshold)})\n"
    "    return pd.DataFrame(rows)"
))

# ---------------------------------------------------------------------------
# 8. Baseline model
# ---------------------------------------------------------------------------
cells.append(md(
    "## 8. Baseline Model — Logistic Regression\n"
    "\n"
    "A simple, interpretable baseline. If a much-more-complex model can't beat "
    "this, the complexity isn't worth it."
))

cells.append(code(
    "logreg_pipe = Pipeline([\n"
    "    ('imputer', SimpleImputer(strategy='median')),\n"
    "    ('scaler',  StandardScaler()),\n"
    "    ('model',   LogisticRegression(\n"
    "        solver='saga', penalty='l2', C=1.0,\n"
    "        max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1,\n"
    "    )),\n"
    "])\n"
    "logreg_pipe.fit(X_train, y_train)\n"
    "logreg_results = report('logreg_baseline', logreg_pipe)\n"
    "logreg_results.round(3)"
))

# ---------------------------------------------------------------------------
# 9. Final model
# ---------------------------------------------------------------------------
cells.append(md(
    "## 9. Final Model — Random Forest\n"
    "\n"
    "After comparing logistic regression, random forests, and gradient-boosted "
    "models across many hyperparameter configurations during model "
    "selection, a tuned **Random Forest** was chosen for production: it "
    "handles missing values, mixes well with our wide feature set, and "
    "matches the performance of the boosted models without an extra "
    "library dependency.\n"
    "\n"
    "Below we just refit the chosen architecture (no big grid search) so "
    "this notebook stays fast to run end-to-end."
))

cells.append(code(
    "rf_pipe = Pipeline([\n"
    "    ('imputer', SimpleImputer(strategy='median')),\n"
    "    ('model',   RandomForestClassifier(\n"
    "        n_estimators=200, max_depth=None, min_samples_split=2,\n"
    "        max_features='sqrt', class_weight=None,\n"
    "        random_state=RANDOM_STATE, n_jobs=-1,\n"
    "    )),\n"
    "])\n"
    "rf_pipe.fit(X_train, y_train)\n"
    "rf_results = report('random_forest', rf_pipe)\n"
    "rf_results.round(3)"
))

# ---------------------------------------------------------------------------
# 10. Evaluation
# ---------------------------------------------------------------------------
cells.append(md(
    "## 10. Model Evaluation\n"
    "\n"
    "Side-by-side comparison and the **confusion matrix on the held-out test "
    "set** for the chosen final model."
))

cells.append(code(
    "comparison = pd.concat([logreg_results, rf_results], ignore_index=True)\n"
    "display(comparison.pivot(index='model', columns='split',\n"
    "                         values=['accuracy', 'precision', 'recall', 'f1', 'rocAuc'])\n"
    "        .round(3))"
))

cells.append(code(
    "test_proba = rf_pipe.predict_proba(X_test)[:, 1]\n"
    "test_pred  = (test_proba >= 0.5).astype(int)\n"
    "cm = pd.DataFrame(\n"
    "    confusion_matrix(y_test, test_pred),\n"
    "    index=['Actual 0 (team1 lost)', 'Actual 1 (team1 won)'],\n"
    "    columns=['Pred 0', 'Pred 1'],\n"
    ")\n"
    "print('Confusion matrix on test set (Random Forest, threshold=0.5):')\n"
    "cm"
))

# ---------------------------------------------------------------------------
# 11. Explainability
# ---------------------------------------------------------------------------
cells.append(md(
    "## 11. Model Explainability\n"
    "\n"
    "Two complementary views:\n"
    "\n"
    "1. **Built-in feature importance** — how much each feature is used in the trees.\n"
    "2. **Permutation importance on the test set** — how much accuracy drops "
    "   when we shuffle each feature. This is more honest because it scores "
    "   importance against held-out data."
))

cells.append(code(
    "rf_model = rf_pipe.named_steps['model']\n"
    "builtin_importance = (\n"
    "    pd.DataFrame({'feature': FEATURE_COLS, 'importance': rf_model.feature_importances_})\n"
    "    .sort_values('importance', ascending=False)\n"
    "    .head(15)\n"
    "    .reset_index(drop=True)\n"
    ")\n"
    "builtin_importance.round(4)"
))

cells.append(code(
    "# Permutation importance is expensive; restrict to the top-30 built-in\n"
    "# features and reuse them for the audit. This keeps the notebook fast\n"
    "# while still giving an honest, held-out view of feature value.\n"
    "top_features = builtin_importance['feature'].head(30).tolist()\n"
    "feat_idx = [FEATURE_COLS.index(f) for f in top_features]\n"
    "X_test_top = X_test.iloc[:, feat_idx]\n"
    "\n"
    "# We score against a fitted model that only sees those columns, so\n"
    "# refit a small surrogate to honestly attribute permutation effects.\n"
    "surrogate = Pipeline([\n"
    "    ('imputer', SimpleImputer(strategy='median')),\n"
    "    ('model',   RandomForestClassifier(\n"
    "        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),\n"
    "])\n"
    "surrogate.fit(X_train.iloc[:, feat_idx], y_train)\n"
    "\n"
    "perm = permutation_importance(\n"
    "    surrogate, X_test_top, y_test,\n"
    "    n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1,\n"
    ")\n"
    "perm_df = (\n"
    "    pd.DataFrame({'feature': top_features,\n"
    "                  'permImportance': perm.importances_mean})\n"
    "    .sort_values('permImportance', ascending=False)\n"
    "    .head(15)\n"
    "    .reset_index(drop=True)\n"
    ")\n"
    "perm_df.round(4)"
))

# ---------------------------------------------------------------------------
# 12. Bias detection & mitigation
# ---------------------------------------------------------------------------
cells.append(md(
    "## 12. Bias Detection & Mitigation\n"
    "\n"
    "March Madness has very few historical upsets, so a model can look "
    "accurate by simply learning *“the better seed always wins.”* That is a "
    "**representation bias**.\n"
    "\n"
    "We measure it by bucketing test games into **chalk** (team1 is favored "
    "by ≥4 seeds), **toss-up** (within 3 seeds), and **upset** (team1 is the "
    "underdog by ≥4 seeds), and reporting accuracy in each bucket. We then "
    "retrain with `class_weight='balanced'` to give upsets more weight and "
    "compare the two."
))

cells.append(code(
    "BUCKET_ORDER = ['chalk', 'tossup', 'upset']\n"
    "\n"
    "def bias_audit(pipe, model_name):\n"
    "    proba = pipe.predict_proba(X_test)[:, 1]\n"
    "    pred  = (proba >= 0.5).astype(int)\n"
    "\n"
    "    seed_diff = (test_df['Seed_diff'].values\n"
    "                 if 'Seed_diff' in test_df.columns\n"
    "                 else np.zeros(len(test_df)))\n"
    "    bucket = np.where(seed_diff <= -4, 'chalk',\n"
    "             np.where(seed_diff >=  4, 'upset', 'tossup'))\n"
    "\n"
    "    audit_df = pd.DataFrame({'bucket': bucket,\n"
    "                             'y':      y_test.values,\n"
    "                             'pred':   pred})\n"
    "    rows = []\n"
    "    for b, sub in audit_df.groupby('bucket'):\n"
    "        rows.append({\n"
    "            'model'   : model_name,\n"
    "            'bucket'  : b,\n"
    "            'nGames'  : int(len(sub)),\n"
    "            'baseRate': round(sub['y'].mean(), 3),\n"
    "            'accuracy': round((sub['pred'] == sub['y']).mean(), 3),\n"
    "        })\n"
    "    out = pd.DataFrame(rows).set_index('bucket')\n"
    "    return out.reindex([b for b in BUCKET_ORDER if b in out.index]).reset_index()\n"
    "\n"
    "before_df = bias_audit(rf_pipe, 'rf_default')\n"
    "before_df"
))

cells.append(code(
    "# Mitigation: retrain with class_weight='balanced' so upsets carry more weight\n"
    "rf_balanced_pipe = Pipeline([\n"
    "    ('imputer', SimpleImputer(strategy='median')),\n"
    "    ('model',   RandomForestClassifier(\n"
    "        n_estimators=200, max_depth=None, min_samples_split=2,\n"
    "        max_features='sqrt', class_weight='balanced',\n"
    "        random_state=RANDOM_STATE, n_jobs=-1,\n"
    "    )),\n"
    "])\n"
    "rf_balanced_pipe.fit(X_train, y_train)\n"
    "after_df = bias_audit(rf_balanced_pipe, 'rf_balanced')\n"
    "\n"
    "bias_compare = pd.concat([before_df, after_df], ignore_index=True)\n"
    "bias_compare"
))

# ---------------------------------------------------------------------------
# 13. Deployment
# ---------------------------------------------------------------------------
cells.append(md(
    "## 13. Model Deployment\n"
    "\n"
    "Persist the chosen model **as a single payload**: the fitted pipeline, "
    "the feature column order it expects, the decision threshold, and a "
    "version tag. Anyone downstream can `pickle.load(...)` and immediately "
    "score new matchups without reading any other code."
))

cells.append(code(
    "deployed_model = {\n"
    "    'model'       : rf_pipe,\n"
    "    'modelFamily' : 'randomForest',\n"
    "    'featureCols' : FEATURE_COLS,\n"
    "    'threshold'   : 0.5,\n"
    "    'version'     : 'final-v1',\n"
    "}\n"
    "deployed_path = MODELS_DIR / 'final_deployed_model.pkl'\n"
    "with open(deployed_path, 'wb') as f:\n"
    "    pickle.dump(deployed_model, f)\n"
    "\n"
    "with open(deployed_path, 'rb') as f:\n"
    "    reloaded = pickle.load(f)\n"
    "print(f'Saved {deployed_path.name}; reloaded family = {reloaded[\"modelFamily\"]} '\n"
    "      f'with {len(reloaded[\"featureCols\"])} features.')"
))

# ---------------------------------------------------------------------------
# 14. Monitoring
# ---------------------------------------------------------------------------
cells.append(md(
    "## 14. Monitoring — Data Drift & Concept Drift\n"
    "\n"
    "After deployment we compare two windows:\n"
    "\n"
    "- **Baseline** = older seasons the model trained on (≤ 2017).\n"
    "- **Recent**   = newer seasons (≥ 2018).\n"
    "\n"
    "We measure:\n"
    "\n"
    "- **Data drift** via the **Population Stability Index (PSI)** for each top feature. A common rule of thumb is `PSI > 0.20` ⇒ meaningful shift.\n"
    "- **Concept drift** via the change in accuracy. A drop > 5 percentage points warrants retraining.\n"
    "- **Monitoring plan** — written to `data/processed/` so other teams can act on it."
))

cells.append(code(
    "baseline_df = matchup[matchup['Season'] <= 2017].copy()\n"
    "recent_df   = matchup[matchup['Season'] >= 2018].copy()\n"
    "\n"
    "def psi(base_series, recent_series, bins=10):\n"
    "    base, recent = base_series.dropna(), recent_series.dropna()\n"
    "    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))\n"
    "    if len(edges) < 3:\n"
    "        return 0.0\n"
    "    base_ct,   _ = np.histogram(base,   bins=edges)\n"
    "    recent_ct, _ = np.histogram(recent, bins=edges)\n"
    "    base_pct   = np.clip(base_ct   / base_ct.sum(),   1e-6, None)\n"
    "    recent_pct = np.clip(recent_ct / recent_ct.sum(), 1e-6, None)\n"
    "    return float(np.sum((recent_pct - base_pct) * np.log(recent_pct / base_pct)))\n"
    "\n"
    "monitor_features = builtin_importance['feature'].head(10).tolist()\n"
    "drift_df = pd.DataFrame([\n"
    "    {'feature': c, 'psi': round(psi(baseline_df[c], recent_df[c]), 4)}\n"
    "    for c in monitor_features\n"
    "]).sort_values('psi', ascending=False)\n"
    "drift_df"
))

cells.append(code(
    "model = reloaded['model']\n"
    "thr   = reloaded['threshold']\n"
    "feats = reloaded['featureCols']\n"
    "\n"
    "base_acc = ((model.predict_proba(baseline_df[feats])[:, 1] >= thr).astype(int)\n"
    "            == baseline_df['y'].astype(int)).mean()\n"
    "recent_acc = ((model.predict_proba(recent_df[feats])[:, 1] >= thr).astype(int)\n"
    "               == recent_df['y'].astype(int)).mean()\n"
    "\n"
    "print(f'Baseline accuracy: {base_acc:.3f}')\n"
    "print(f'Recent  accuracy : {recent_acc:.3f}')\n"
    "print(f'Accuracy drop    : {base_acc - recent_acc:+.3f}')\n"
    "print(f'Max feature PSI  : {drift_df[\"psi\"].max():.3f}  '\n"
    "      f'({\"⚠ drift\" if drift_df[\"psi\"].max() > 0.20 else \"ok\"})')"
))

cells.append(code(
    "monitoring_plan = pd.DataFrame([\n"
    "    {'metric': 'topFeaturePsi',          'threshold': '0.20',          'cadence': 'weekly',  'owner': 'dataScience'},\n"
    "    {'metric': 'modelAccuracy',          'threshold': 'drop > 0.05',   'cadence': 'weekly',  'owner': 'dataScience'},\n"
    "    {'metric': 'predictionPositiveRate', 'threshold': 'delta > 0.10',  'cadence': 'weekly',  'owner': 'analytics'},\n"
    "    {'metric': 'retrainDecision',        'threshold': '2 alerts in a row', 'cadence': 'monthly', 'owner': 'team'},\n"
    "])\n"
    "plan_path = PROCESSED_DIR / 'final_monitoring_plan.csv'\n"
    "monitoring_plan.to_csv(plan_path, index=False)\n"
    "monitoring_plan"
))

# ---------------------------------------------------------------------------
# 15. Inference example
# ---------------------------------------------------------------------------
cells.append(md(
    "## 15. Production Inference Example\n"
    "\n"
    "How an end-user (e.g., the instructor) actually *uses* the deployed "
    "model. We load the pickle file, take three real matchups from the "
    "test set, and print the predicted win probability.\n"
    "\n"
    "Loading and prediction are intentionally trivial — that is the point of "
    "shipping the column order inside the payload."
))

cells.append(code(
    "# 15.1 — load the deployed payload from disk\n"
    "with open(MODELS_DIR / 'final_deployed_model.pkl', 'rb') as f:\n"
    "    payload = pickle.load(f)\n"
    "\n"
    "model_obj = payload['model']\n"
    "feat_cols = payload['featureCols']\n"
    "thr       = payload['threshold']\n"
    "\n"
    "# 15.2 — pick three real matchups from the held-out test set\n"
    "sample = test_df.sample(3, random_state=RANDOM_STATE).reset_index(drop=True)\n"
    "X_sample = sample.reindex(columns=feat_cols)\n"
    "\n"
    "team1_win_prob = model_obj.predict_proba(X_sample)[:, 1]\n"
    "predictions = pd.DataFrame({\n"
    "    'season'        : sample['Season'].astype(int).values,\n"
    "    'team1'         : sample['team1Name'].values,\n"
    "    'team2'         : sample['team2Name'].values,\n"
    "    'team1WinProb'  : np.round(team1_win_prob, 3),\n"
    "    'predictedWinner': np.where(team1_win_prob >= thr,\n"
    "                                sample['team1Name'].values,\n"
    "                                sample['team2Name'].values),\n"
    "    'actualWinner'  : np.where(sample['y'] == 1,\n"
    "                               sample['team1Name'].values,\n"
    "                               sample['team2Name'].values),\n"
    "})\n"
    "predictions"
))

cells.append(md(
    "---\n"
    "\n"
    "### Project Summary\n"
    "\n"
    "We took raw NCAA tournament data, engineered a clean per-matchup "
    "differential dataset, trained a logistic-regression baseline and a "
    "random-forest final model, audited the model for seed-related bias, "
    "exported a pickle-deployed payload, set up a drift-monitoring plan, "
    "and demonstrated end-user inference. The artifacts of every step "
    "(processed CSVs, the final model `.pkl`, the monitoring plan) are all "
    "saved under `data/processed/` and `models/`."
))

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
OUT_PATH.write_text(nbf.writes(nb))
print(f"Wrote {OUT_PATH}  ({len(cells)} cells)")
