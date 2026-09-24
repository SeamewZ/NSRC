# Reproduction notes

This release separates the **frozen manuscript outputs** from the **code and inputs needed to regenerate them**. The paper, aggregate comparison table, ablation table, and statistical summaries are included. Exact end-to-end regeneration requires several original run-time artifacts that are not committed here: prepared per-origin windows and event labels, baseline prediction arrays and provenance, and upstream model checkouts or checkpoints.

## Included artifacts

- Final manuscript source and compiled PDF: <code>paper/ICASSP-neurosymbolic/</code>
- All-method, four-cohort aggregate results: <code>results/main_comparison.csv</code>
- Component ablations: <code>results/component_ablations.csv</code>
- Validation route choices and route-level summaries: <code>results/validation_route_selection.csv</code> and <code>results/stability/</code>
- Primary paired significance and stability results: <code>results/stability/</code>
- Detailed Time-MoE horizon and pairwise summaries: <code>results/baselines/time_moe/</code>

The Time-MoE pairwise bootstrap file is a separate detailed comparator analysis. The paper’s primary significance table is the NSRC-versus-DLinear analysis in <code>results/stability/significance.csv</code>.

## Data snapshot and reconstruction

The selected snapshot under <code>data/celebcrypto/</code> contains OHLCV, event/news records, core celebrity posts, and asset-related posts. Inspect its coverage with:

~~~bash
python scripts/summarize_bundle.py --bundle-root data/celebcrypto
~~~

The crypto preparation adapter writes a coverage and provenance audit alongside its windows:

~~~bash
python reproduction/prepare.py \
  --bundle data/celebcrypto \
  --out data/prepared/celebrity \
  --protocol paper
~~~

Preparation stops with an error if the snapshot does not cover the requested period. When it succeeds, the generated audit describes which recorded label and description fields were available. This archive reconstruction is not guaranteed to be byte-identical to the original experimental inputs or annotations.

Macro-event adaptation accepts a prepared price-window directory and a timestamped event JSON file:

~~~bash
python reproduction/prepare_macro.py \
  --data data/prepared/macro_prices \
  --events data/source/macro_events.json \
  --out data/prepared/macro_crypto
~~~

The macro price/event source files used for the reported runs are not bundled; obtain and document the source data before running this step.

## Expected inputs for the online evaluator

The online scripts in <code>reproduction/</code> use the run layout from the experiment. For each cohort, the prepared data directory contains:

- <code>index.csv</code>: one row per forecast origin, including asset, origin timestamp, target end, split, event count, and memory-eligibility flag;
- <code>windows.npz</code>: normalized historical inputs and 24-step targets;
- <code>features.npz</code>: input-window features, including volatility scale;
- <code>events.json</code>: event records with category, direction, and description for each origin;
- <code>manifest.json</code>: split boundaries and protocol metadata.

Price-baseline predictions and provenance are stored under <code>baselines/&lt;cohort&gt;/&lt;method&gt;_s&lt;seed&gt;/</code>. The main online evaluator reads the experiment root configured in <code>reproduction/full_online.py</code>; the route evaluator uses the same data root and writes its selected/evaluated outputs under <code>runs/nsrc_safe_route_20260922/</code>. To run from a different location, update those root constants or recreate the documented layout.

Example baseline training command after preparing those inputs and installing the corresponding upstream repository under <code>vendor/</code>:

~~~bash
python reproduction/train.py \
  --data runs/full_evaluation_20260921/data/celebrity \
  --model DLinear \
  --vendor vendor \
  --out runs/full_evaluation_20260921/baselines/celebrity/DLinear_s2026 \
  --seed 2026
~~~

The supported price-only model names are DLinear, GPT4TS, PatchTST, iTransformer, and TimesNet. The adapter records the upstream commit and run configuration in its provenance file. Time-LLM and CAMEF have dedicated adapters in <code>reproduction/full_timellm.py</code> and <code>reproduction/full_camef.py</code>; Time-MoE has <code>reproduction/time_moe_baseline.py</code>. These methods require their upstream packages and, where applicable, model downloads.

The online correction, semantic ablations, validation routing, and final summaries are implemented in <code>full_online.py</code>, <code>semantic_online.py</code>, <code>adaptive_route.py</code>, and the reporting scripts. They enforce mature-history access in the evaluator; the included verification outputs record the causal checks for the reported runs.

## Statistical protocol

The route is selected separately for each cohort from the 30 calendar days before its frozen test interval. Validation windows crossing the test boundary are excluded. Selection averages validation MAE over the recorded DLinear seeds and uses MSE as a tie-break; the selected route is fixed before test evaluation.

Primary significance results compare the deployed NSRC route with fixed DLinear using paired, calendar-day circular block bootstrap, followed by Holm correction within the 12 cohort–metric tests. The stability files stratify the same comparison by horizon, test period, and pre-test volatility. The paper also reports 3-day and 14-day block-length sensitivity.

Because the test periods informed method exploration, these outputs are retrospective within-period results. The causal timing checks address future-target access in the prediction procedure; they do not turn the periods into untouched prospective holdouts.

## Baseline implementations

The repository includes adapters and recorded aggregate results, not copies of the upstream baseline repositories. Install the official implementation for each baseline and preserve its published design and configuration. The paper’s main table records the evaluated methods and their reported scores. Do not interpret the summary CSVs as a substitute for upstream checkpoints or per-origin predictions.
