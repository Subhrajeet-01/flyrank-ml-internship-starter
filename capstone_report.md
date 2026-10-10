# Google Search Ranking & Discoverability Capstone
## Lane 2 — Refresh / Content Opportunity Scoring

This project builds and evaluates a page-level score to help a content team decide which pages deserve a closer human review. It uses the FlyRank internship warehouse, Google Search Console (GSC) performance fields, DuckDB for warehouse-side aggregation, and scikit-learn for modelling.

> **Important:** This is a prioritisation experiment, not proof that a page is decaying, that refreshing content will recover clicks, or that the model explains Google's ranking algorithm.

## Project links

- **Research paper:** `https://YOUR_GITHUB_USERNAME.github.io/YOUR_REPOSITORY/`
- **Notebook:** [`work/notebooks/capstone.ipynb`](work/notebooks/capstone.ipynb)
- **Dataset provider / credit:** [FlyRank](https://flyrank.ai)

Replace the research-paper URL above after GitHub Pages is enabled. The capstone submission also requires `submission/paper_url.txt` to contain the direct public URL on one line.

## 1. Problem statement

Content teams may have too many pages to review manually. The project asks:

> At a monthly cutoff, can recent page-level performance and metadata help rank pages by the chance that their next-30-day clicks will decline at least 25% more than the client's overall click trend?

The score is intended to help decide **what to review first**. It is not an automatic instruction to rewrite or delete a page.

## 2. What the notebook does

The notebook is organised as a top-to-bottom workflow:

1. **Set up the environment.** Installs DuckDB, pandas, PyArrow, Hugging Face Hub access, and scikit-learn. It reads the gated warehouse using a Hugging Face token supplied through Colab Secrets or a password prompt.
2. **Inspect the source schema.** Checks the daily content-performance fact table and content dimension table before analysis.
3. **Profile data coverage.** Summarises content status, GSC availability, client coverage, and concentration of clicks across clients.
4. **Create a compact local cache.** Uses DuckDB to select only the required GSC fields and date range, then writes a compressed Parquet file. This avoids loading the full warehouse into pandas.
5. **Build monthly page panels.** Aggregates the previous 7, 30, 60, and 90 days for each page at six monthly cutoffs from January through June 2026. It joins safe content metadata and applies eligibility filters.
6. **Define the future outcome.** Creates a relative-decline label from the following 30 days. Future-window values are used only for the label, not as model features.
7. **Train and compare models.** Evaluates simple momentum and spike baselines, logistic regression, and histogram gradient boosting.
8. **Run robustness checks.** Checks forward-in-time performance, held-out-client performance, per-client gains, a sign-flipped baseline, permutation importance, feature-group ablations, and actual decline rates by score decile.
9. **Score the latest eligible pages.** Fits a lean logistic model on all labelled monthly panels, scores the July 1, 2026 scoring panel, creates reason codes, and assigns human-review action tiers.
10. **Run final checks.** Confirms that expected caches and output columns exist and reminds the user not to publish private ranked exports.

## 3. Data source and scope

The notebook reads the gated dataset:

- `fact_content_daily_performance` — daily client/page performance.
- `dim_content` — page/content metadata.

Important performance fields include `report_date`, `client_hash_id`, `content_hash_id`, `gsc_clicks`, `gsc_impressions`, and `gsc_avg_position`. Metadata used by the model includes fields such as `content_type`, `main_intent`, `word_count`, `search_volume`, `backlinks`, `competition`, `keyword_token_count`, and content age.

The notebook caches GSC rows from **September 1, 2025 through June 30, 2026**. It uses monthly cutoffs on **January 1, February 1, March 1, April 1, May 1, and June 1, 2026**. The cache shown in the successful run contained **26,585,916 daily rows** across **307,186 distinct content IDs**. These counts describe the cached performance rows, not the final model sample.

The notebook's final July scoring panel contained **17,413 eligible pages across 33 clients**.

### Eligibility and exclusions

The panel builder:
- requires client history and GSC activity around the cutoff and future-label window;
- includes published content and excludes content marked as deleted;
- requires at least 10 clicks and 100 impressions in the preceding 30-day feature window;
- derives predictors from data available before the cutoff;
- keeps hashed IDs in local intermediate data only.

The warehouse is gated. A public repository should contain the notebook and public paper, **not raw warehouse exports, cache Parquet files, credentials, or ranked output containing hashed identifiers**.

## 4. Label definition

For each page and monthly cutoff:

- `c30` = page clicks in the 30 days before the cutoff.
- `y_clicks` = page clicks in the 30 days after the cutoff.
- `client_ratio` = the client's future 30-day clicks divided by that client's preceding 30-day clicks.
- `ratio` = the page's future 30-day clicks divided by its preceding 30-day clicks.

The target `y_rel` is positive when:

`(page future / page previous) / (client future / client previous) <= 0.75`

In plain language, the page's click trend must be at least 25% worse than the client's overall click trend. This relative target is intended to distinguish page-level underperformance from a broad change affecting the whole client.

The notebook also calculates `y_raw`, an absolute page decline of at least 25%, for diagnostic comparison. The model target is **`y_rel`**.

## 5. Feature engineering

All model predictors are derived from the feature window available at the cutoff. The notebook excludes future outcome columns and target-construction fields from the feature list.

Examples:

| Feature | Plain-language meaning |
|---|---|
| `recent` | Last-week click pace compared with the 30-day pace |
| `cv90` | Click variability over the recent 90-day window |
| `spike` | Recent 30-day click level compared with the 90-day average |
| `mom` | Click momentum versus the previous 30 days |
| `impr_mom` | Change in impressions versus the previous 30 days |
| `ctr30` / `ctr_chg` | Recent click-through rate and its change |
| `pos30` / `pos_chg` | Impression-weighted average position and its change |
| `peak_share` | Share of 30-day clicks coming from the single highest-click day |
| `act_frac30` | Number of days with activity in the 30-day window, divided by 30 |
| `share_of_client` | Page's share of its client's recent clicks |
| `word_count`, `search_volume`, `backlinks` | Content metadata available in the dimension table |
| `age_days` | Page age at the cutoff |

Missing numeric values are imputed in the logistic-regression pipeline. Numeric features are standardised before logistic regression. The histogram gradient-boosting model uses the feature matrix with configured categorical feature indices.

## 6. Validation design

The main evaluation is time-forward:

- **Training:** January–March 2026 cutoffs.
- **Validation:** April 2026 cutoff.
- **Test:** May and June 2026 cutoffs.

The model is fit on training data for validation. For the final test, it is refit on train plus validation (January–April) and evaluated on May–June. Test data is not used to tune the model's hyperparameters in the notebook.

Additional checks include:
- **Held-out-client check:** GroupKFold splits by client, trains using cutoffs through March, and evaluates held-out clients on May–June.
- **Per-client comparison:** Compares model and momentum-baseline PR-AUC for clients with enough test rows and both target classes.
- **Bootstrap interval:** Resamples clients to estimate uncertainty in the difference in test PR-AUC.
- **Sign-flipped baseline:** Tests the opposite direction of the client-adjusted momentum heuristic.
- **Permutation importance:** Measures the reduction in PR-AUC when a feature is permuted.
- **Ablation tests:** Removes groups of features to check whether the test score changes.
- **Score deciles:** Compares observed positive-label rates across ten score groups.

These checks improve the credibility of the evaluation, but they do not remove all uncertainty. Monthly panels may contain repeated observations of the same page, clients differ in data coverage, and a limited number of time cutoffs constrains generalisation claims.

## 7. Metrics

- **PR-AUC / Average Precision:** Main metric because it evaluates ranking quality for a positive class whose prevalence is below 50%. Its baseline depends on the positive-class rate.
- **ROC-AUC:** Measures ranking discrimination across thresholds; it can look optimistic when positives are not common.
- **Precision at 10%:** The positive-label rate among the top 10% highest-scored rows. In the notebook report this is averaged across test cutoffs.
- **Base rate:** The share of positive labels in the evaluated sample.
- **Client-bootstrap 95% interval:** An uncertainty interval for the difference between model and baseline PR-AUC, resampling clients rather than treating all page rows as independent.

## 8. Results from the saved successful notebook run

These are the values printed in the provided cleaned notebook's saved outputs. Re-run the notebook before publication if you change the code or data.

### Forward test: May–June 2026

| Model / baseline | PR-AUC | ROC-AUC | Precision at 10% | Positive-label rate |
|---|---:|---:|---:|---:|
| Client-adjusted momentum baseline | 0.387 | 0.494 | 0.494 | 0.362 |
| Spike baseline | 0.374 | 0.514 | 0.321 | 0.362 |
| Logistic regression | 0.502 | 0.634 | 0.610 | 0.362 |
| Histogram gradient boosting | 0.490 | 0.622 | 0.580 | 0.362 |
| Sign-flipped momentum baseline | 0.374 | 0.506 | 0.386 | 0.362 |

The **full logistic model** had the highest PR-AUC in the reported test table (0.502). The later scoring cell also compared a lean logistic feature set and reported test PR-AUC of **0.506**, slightly above the full logistic model's 0.502. These are distinct feature sets and should not be described as the same model.

### Robustness checks

- Histogram gradient boosting exceeded the client-adjusted momentum baseline in all five saved held-out-client folds.
- In the per-client test comparison, the model beat the momentum baseline for **87% of 15 clients with sufficient data**, with a median PR-AUC gain of **0.092**.
- The saved client-bootstrap estimate for gradient boosting minus the sign-flipped momentum baseline was **+0.114 PR-AUC**, with a 95% interval of **[0.081, 0.145]**.
- Permutation importance ranked recent click pace (`recent`), click variability (`cv90`), spike size (`spike`), and impression momentum (`impr_mom`) among the strongest signals.
- Removing momentum/spike-related features reduced gradient-boosting test PR-AUC from **0.490 to 0.420**. Removing client-context or content-metadata groups did not improve the score in this run.
- The highest gradient-boosting score decile had an observed positive-label rate of **0.597**, compared with the test-set rate of **0.362**. The lowest decile's observed rate was **0.190**.

The test results are encouraging for ranking pages for review, but they are not evidence that the score will improve business outcomes. PR-AUC differences are modest, metrics depend on the sampled clients and cutoffs, and the score should be treated as decision support.

## 9. Final scoring and action tiers

The final scoring cell fits a **lean logistic-regression model** on all labelled panels and scores the July 1, 2026 panel. It reported:

- **17,413 pages**
- **33 clients**
- Lean logistic test PR-AUC: **0.506**
- Full logistic test PR-AUC (comparison): **0.502**

It then ranks pages by predicted risk percentile and combines that percentile with each page's relative click volume within its client.

| Action tier | Rule in plain language | Count |
|---|---|---:|
| 1 — Review now | Top 10% risk and relatively meaningful traffic | 600 |
| 2 — Watch closely | Top 10% risk, lower relative traffic | 1,142 |
| 3 — Monitor | 70th–90th risk percentiles | 3,482 |
| 4 — Protect | Bottom 30% risk and relatively meaningful traffic | 2,798 |
| 5 — No action | Remaining pages | 9,391 |
| **Total** | | **17,413** |

The notebook creates short reason codes from the lean logistic model's standardised feature contributions. They indicate which measured features contributed to a high score; they are not causal explanations.

The detailed ranked export is written locally to `/content/cache/ranked_actions_PRIVATE.csv`. It contains hashed IDs and must **not** be committed or published. The notebook prints a small public-safe preview with generic page/client labels.

## 10. How to run the notebook

The notebook is designed for Google Colab.

1. Open `work/notebooks/capstone.ipynb` in Colab.
2. Ensure you have access to the gated `FlyRank/internship-warehouse` dataset.
3. In Colab, open **Secrets** (key icon) and add a secret named `HF_TOKEN`, or enter the token when prompted.
4. Run all cells from top to bottom.
5. Wait for DuckDB to query the remote Parquet files and create `/content/cache/daily_gsc.parquet`.
6. Review the printed validation, test, robustness, scoring, and final self-check outputs.
7. Update the paper only with metrics from the latest successful run.

### Dependencies

The setup cell installs:

- `duckdb`
- `pandas`
- `pyarrow`
- `huggingface_hub`
- `scikit-learn`

The notebook also uses the Google Colab `userdata` interface and is therefore not a plain local Python script without adaptation.

## 11. Suggested repository layout

```text
YOUR_REPOSITORY/
├── README.md
├── docs/
│   └── index.html                 # GitHub Pages research paper
├── work/
│   └── notebooks/
│       └── capstone.ipynb         # Final reproducible notebook
└── submission/
    └── paper_url.txt              # One line: public paper URL
```

Do not commit `/content/cache/`, `daily_gsc.parquet`, `panel.parquet`, `ranked_actions_PRIVATE.csv`, tokens, or any raw warehouse exports.

## 12. Publish the paper with GitHub Pages

1. Put the final paper HTML at `docs/index.html`.
2. Push `README.md`, the notebook, `docs/index.html`, and `submission/paper_url.txt` to the default branch.
3. In the GitHub repository, open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select the default branch (usually `main`) and folder `/docs`, then click **Save**.
6. Wait for the Pages deployment to finish. The project-site URL normally follows this pattern: `https://YOUR_GITHUB_USERNAME.github.io/YOUR_REPOSITORY/`.
7. Open the URL in a private/incognito window to verify that the page is public and that charts render.
8. Copy the exact deployed URL into `submission/paper_url.txt` as a single line, then commit and push that file.

If the repository is a fork of the internship starter, prefer adding these files to that existing fork rather than creating a disconnected repository, unless the assignment explicitly asks for a separate repository.

## 13. Privacy and responsible interpretation

Never commit or publish:
- Hugging Face access tokens or other credentials;
- raw warehouse files or local Parquet caches;
- hashed client/content identifiers;
- `ranked_actions_PRIVATE.csv` or any export that exposes hashed IDs;
- private queries, client domains, or URLs.

Use only generic labels in public examples. The target is a relative performance pattern, not a direct measurement of content quality. A high score means “prioritise for investigation,” not “this page is definitely decaying.” A content refresh may or may not help; this notebook does not estimate causal refresh impact.

## 14. Reproducibility and acknowledgements

- Notebook: [`work/notebooks/capstone.ipynb`](work/notebooks/capstone.ipynb)
- Research paper: `docs/index.html` (served by GitHub Pages)
- Data credit: **Built on the FlyRank ML Internship dataset.** [FlyRank](https://flyrank.ai)

For reproducibility, keep the notebook source and public paper in version control, record material changes to the label/features/splits, and refresh reported metrics whenever the analysis is rerun.

---

*Prepared for the FlyRank ML Internship capstone — Lane 2: Refresh / Content Opportunity Scoring.*
