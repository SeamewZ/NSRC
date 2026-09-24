# Neural Structured Residual Correction (NSRC)

<p align="center">
  <strong>Event-addressed residual memory for multi-horizon financial forecasting</strong><br>
  A causal correction layer that turns mature event responses into reusable forecast trajectories.
</p>

<p align="center">
  <a href="paper/ICASSP-neurosymbolic/build/main.pdf"><img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-b31b1b"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776ab">
  <img alt="License MIT" src="https://img.shields.io/badge/Code%20license-MIT-2ea44f">
</p>

NSRC augments a numerical forecaster with a memory of **completed, event-conditioned forecast errors**. Each mature error trajectory is addressed by event category, direction, and description; related trajectories are retrieved and calibrated using only feedback available before the current forecast. A validation-frozen router decides whether a cohort should use a correction path or retain a forecasting anchor.

The release contains the manuscript and compiled PDF, experiment and baseline scripts, the reported aggregate results, statistical and stability summaries, and a selected cryptocurrency input snapshot.

## Method at a glance

~~~mermaid
flowchart LR
    H[Historical price windows] --> F[Frozen numerical forecaster]
    F --> P[Base forecast]
    Y[Observed outcomes after horizon matures] --> R[Residual trajectory]
    P --> R
    R --> G{Maturity gate}
    G --> M[Event-addressed memory]
    E[Current event: category, direction, description] --> Q[Addressed query]
    Q --> M
    M --> T[Similarity-weighted retrieval<br/>volatility alignment and shrinkage]
    T --> C[Gain from previously matured feedback]
    P --> V[Validation-frozen cohort route]
    C --> V
    V --> O[Forecast path]
~~~

An event’s own future target cannot enter its memory read or calibration. The stored residual becomes eligible only after its complete forecast horizon has matured. Multiple events at the same origin contribute equally. The router is selected on the pre-test validation interval and held fixed during testing.

## Results

The main table reports origin-price-normalized errors, pooled over all 24 forecast steps. MSE is scaled by $10^3$, MAE by $10^2$, and MAPE is in percent.

| Cohort | Assets | Method | MSE × 10³ | MAE × 10² | MAPE (%) |
|:--|:--|:--|--:|--:|--:|
| Celebrity events | BTC / ETH / SOL / DOGE | Persistence | 0.77052 | 1.69004 | 1.66709 |
|  |  | DLinear | 0.91826 | 1.90757 | 1.87894 |
|  |  | **NSRC** | **0.77448** | **1.68974** | **1.66632** |
| News extension | BTC / ETH / SOL / DOGE | Persistence | 0.29355 | 1.18851 | 1.17952 |
|  |  | DLinear | 0.41033 | 1.43537 | 1.42205 |
|  |  | **NSRC** | **0.29310** | **1.18509** | **1.17633** |
| Macro crypto | BTC / ETH / SOL / DOGE | Persistence | 0.83228 | 2.07600 | 2.08960 |
|  |  | DLinear | 0.90287 | 2.19579 | 2.21258 |
|  |  | **NSRC** | **0.83228** | **2.07600** | **2.08960** |
| Macro equities | SPX / NDX / INDU | Persistence | 0.12917 | 0.83780 | 0.84204 |
|  |  | DLinear | 0.20139 | 1.14871 | 1.15083 |
|  |  | **NSRC** | **0.12917** | **0.83780** | **0.84204** |

Across the four cohorts, equal-cohort averaging gives reductions of **16.60% MSE, 13.44% MAE, and 13.36% MAPE** relative to fixed DLinear. The routed system improves all 12 reported DLinear metric values and leads the learned baselines in 11 of 12 cohort–metric cells. Validation selects a 90/10 persistence/uniform-memory blend for Celebrity and News, and Persistence for both macro cohorts; the macro gains over DLinear therefore come from the selected anchor. The ablation table separately measures the unrouted semantic-memory branch and its components.

Paired calendar-block bootstrap comparisons against DLinear remain significant after Holm correction in 8 of 12 cohort–metric tests. Macro-crypto mean gains are positive but uncertain under the reported intervals. Horizon and temporal stability summaries are included with the result files.

**Evaluation scope.** The test periods also informed method exploration, so the manuscript reports retrospective within-period results. The reported intervals should be read in that context, not as a prospective untouched holdout.

## Baselines and ablations

The paper compares Persistence, DLinear, GPT4TS, PatchTST, iTransformer, TimesNet, Time-LLM, CAMEF, and Time-MoE. Price-only baselines use price-history inputs; CAMEF uses its event-fusion design. The repository contains the experiment adapters and final comparison table, while upstream model implementations and large model checkpoints remain external dependencies.

The ablation results distinguish the deployed routed system from the fixed, unrouted semantic branch. They cover description, event category and direction, semantic weighting, delayed-feedback calibration, online updates, recency, volatility scaling, cross-asset sharing, support shrinkage, and shuffled keys. The fallback did not activate in the reported evaluation and is documented as such.

## Repository map

| Path | Contents |
|:--|:--|
| <code>paper/ICASSP-neurosymbolic/</code> | Final manuscript source, figures, tables, bibliography, and compiled PDF |
| <code>reproduction/</code> | NSRC online update, route selection, baseline adapters, ablation, and statistical-analysis code |
| <code>celebcrypto/</code> | Dataset parsing, schemas, preprocessing, and shared utilities |
| <code>scripts/</code> | Dataset snapshot inspection and standardized bundle preparation |
| <code>data/celebcrypto/</code> | Selected OHLCV, event/news, and social-post input snapshot |
| <code>results/main_comparison.csv</code> | Full aggregate main table across all methods and cohorts |
| <code>results/component_ablations.csv</code> | Component ablations across all cohorts |
| <code>results/validation_route_selection.csv</code> | Validation-only route choices and scores |
| <code>results/stability/</code> | Paired significance, horizon, period, volatility, and seed summaries |
| <code>results/baselines/time_moe/</code> | Detailed Time-MoE horizon and paired-comparison summaries |

## Quick start

Use Python 3.10 or later. The following installs the research dependencies, runs the lightweight protocol tests, and summarizes the included data snapshot:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/summarize_bundle.py --bundle-root data/celebcrypto
~~~

The paper source is in <code>paper/ICASSP-neurosymbolic/main.tex</code>; the compiled manuscript is <code>paper/ICASSP-neurosymbolic/build/main.pdf</code>.

## Reproducing the reported experiments

The committed CSV and PDF files are the final reported outputs. The experiment code is provided for inspection and reruns, but an exact end-to-end rerun also requires the original prepared per-origin windows, baseline predictions/checkpoints, and upstream baseline repositories. Those large run-time artifacts are not included in this release. The bundled raw snapshot supports data inspection and preparation; it does not by itself reconstruct every frozen input and annotation used for the reported tables. See [the reproduction notes](docs/REPRODUCTION.md) for input layout, commands, and scope.

For model comparisons, the adapter scripts expect the corresponding upstream implementations and their published configuration. Set them up locally as described in the reproduction notes; this repository does not modify baseline methods by adding NSRC components.

## Citation and license

If you use this work, cite the manuscript using [CITATION.cff](CITATION.cff). The MIT license applies to project software. Bundled market, event, and social-media data retain their original provenance and any applicable source terms; the software license does not relicense third-party data.
