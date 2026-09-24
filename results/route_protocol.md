# Validation-frozen route and causal evaluation

## Route selection

Each cohort uses the 30 calendar days immediately before its frozen test interval for route selection. Validation origins are excluded when their complete target window crosses the test boundary. Candidate scores are averaged equally over DLinear seeds 2026, 2027, and 2028. Validation MAE is primary; MSE breaks a tie. The selected route is fixed before test evaluation, and no test target is used to choose it.

The candidate set includes Persistence, the frozen DLinear forecast, semantic and uniform-relevance residual paths, and predeclared blends of Persistence with each correction path at weights 0.10, 0.25, 0.50, and 0.75. The recorded choices are a 90/10 Persistence/uniform-residual blend for Celebrity and News, and Persistence for Macro Crypto and Macro Equities.

## Causal update rules

At an event origin, memory contains only historical forecast outcomes whose full target horizon ended before that origin. The current event may be used for retrieval only after its publication and before the forecast origin. Calibration uses only previously matured examples. Origins without an eligible event retain the underlying numerical forecast.

## Included result artifacts

- <code>results/validation_route_selection.csv</code>: selected candidate and validation scores by cohort.
- <code>results/causal_checks.json</code>: aggregate future-target intervention, quiet-origin identity, and maturity checks.
- <code>results/main_comparison.csv</code> and <code>results/component_ablations.csv</code>: reported aggregate route and component metrics.
- <code>results/stability/</code>: paired significance and stability analyses for the reported route.

The paper reports retrospective within-period results because the test periods also informed method exploration. Causal timing checks verify the prediction-time access rules; they do not remove adaptive model-selection exposure.
