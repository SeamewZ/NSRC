"""Independent verification of reported measurements and causal traces."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from reproduction.full_report import ROOT,R
import sys
partial="--allow-pending" in sys.argv;pending=[];n_metric_checks=0
checks=[];metrics=pd.read_csv(R/"metrics.csv");counts={}
for flag in sorted((ROOT/"evaluation").glob("*/*/COMPLETE")):
 out=flag.parent;ds=out.parent.name;base,seed=out.name.rsplit("_s",1);seed=int(seed);a=np.load(out/"predictions.npz");idx=pd.read_csv(out/"test_index.csv");trace=pd.read_csv(out/"update_trace.csv");meta=json.loads((out/"verification.json").read_text());valid=trace.latest_memory_target.notna()
 if base in ["TimesNet","Time-LLM"] and not meta["base_provenance"].get("causal_inference"):
  if partial:pending.append(dict(dataset=ds,method=base,seed=seed,reason="awaiting causal inference"));continue
  raise AssertionError("Unverified batch-dependent baseline inference")
 assert (pd.to_datetime(trace.loc[valid,"latest_memory_target"],utc=True)<pd.to_datetime(trace.loc[valid,"origin"],utc=True)).all()
 assert meta["future_target_intervention_invariant"]
 for variant in a.files:
  if variant=="y":continue
  method=("Persistence" if variant=="no_correction" else variant) if base=="Persistence" else (base if variant=="no_correction" else base+"+"+variant)
  assert np.isfinite(a[variant]).all();none=idx.events.to_numpy()==0;assert np.array_equal(a[variant][none],a["no_correction"][none])
  for subset,keep in [("all",np.ones(len(idx),bool)),("event",idx.events.to_numpy()>0)]:
   truth=a["y"][keep];error=a[variant][keep]-truth;actual=np.asarray([np.mean(error**2),np.mean(np.abs(error)),100*np.mean(np.abs(error)/np.maximum(np.abs(truth),1e-8))]);row=metrics[(metrics.dataset==ds)&(metrics.method==method)&(metrics.seed==seed)&(metrics.asset=="Pooled")&(metrics.subset==subset)&(metrics.horizon=="avg24")]
   if len(row)==0 and partial:
    pending.append(dict(dataset=ds,method=method,seed=seed,subset=subset));continue
   assert len(row)==1;n_metric_checks+=1;np.testing.assert_allclose(actual,row[["mse","mae","mape"]].iloc[0].to_numpy(float),rtol=1e-10,atol=1e-14)
 checks.append(dict(dataset=ds,backbone=base,seed=seed,n_predictions=len(idx),n_event_origins=int((idx.events>0).sum()),variants=len(a.files)-1,prediction_sha256=hashlib.sha256((out/"predictions.npz").read_bytes()).hexdigest()))
ci=pd.read_csv(R/"significance.csv");assert (ci.ci_low<=ci.ci_high).all();assert ci.p_two_sided.between(0,1,inclusive="both").all();assert ci.p_two_sided.min()>=1/50000
manifest={str(p.relative_to(Path.cwd())) if p.is_absolute() else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(Path("reproduction").glob("full*.py"))+[ROOT/"PROTOCOL.md",ROOT/"ADAPTERS.md"]}
coverage=pd.read_csv(R/"coverage.csv");missing=coverage[coverage.n_seeds==0].fillna("").to_dict("records")
report=dict(partial=partial,n_metric_checks=n_metric_checks,not_yet_reported_rows=pending,n_evaluated_cells=len(checks),checks=checks,checked_pooled_24step_metrics=True,all_memory_outcomes_strictly_prior=True,no_event_identity=True,all_future_target_interventions_passed=True,missing_baselines=missing,source_hashes=manifest,status="partial_baseline_coverage" if missing else "complete_measurement_coverage",interpretation="Exploratory reused periods; causal timing checks do not erase adaptive model-selection exposure")
(ROOT/("verification_partial.json" if partial else "verification.json")).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ["checks","source_hashes"]},indent=2),flush=True)
audit_records={}
for audit_model in ["DLinear","GPT4TS","PatchTST","iTransformer","CAMEF"]:
 audit_path=ROOT/"batch_audits"/(audit_model+".json")
 if audit_path.exists():audit_records[audit_model]=json.loads(audit_path.read_text());assert audit_records[audit_model]["passed"]
 elif not partial:raise AssertionError("Missing batch independence audit: "+audit_model)
report["batch_independence_audits"]=audit_records
(ROOT/("verification_partial.json" if partial else "verification.json")).write_text(json.dumps(report,indent=2))
if partial:raise SystemExit(0)
# The independent numerical-reference checks must also have completed successfully.
bootstrap_checks=json.loads((ROOT/"bootstrap_checks.json").read_text())
assert len(bootstrap_checks)==12 and all(r["passed"] for r in bootstrap_checks)
reference_checks=json.loads((ROOT/"numerical_checks.json").read_text())
assert all(r["all_event_publications_before_origins"] and r["profile_matches_reference"] for r in reference_checks)
# Independently verify every exported actual-price curve against its prediction arrays.
curve_rows=0
for path in sorted((R/"curves").glob("*/*.csv.gz")):
 ds=path.parent.name;method,seed=path.name.removesuffix(".csv.gz").rsplit("_s",1)
 base="Persistence" if method=="ARM90" else method.removesuffix("+ARM90")
 folder=ROOT/"evaluation"/ds/(base+"_s"+seed)
 a=np.load(folder/"predictions.npz");idx=pd.read_csv(folder/"test_index.csv");c=pd.read_csv(path);row=c.row.to_numpy(int);h=c.horizon.to_numpy(int)-1;scale=idx.origin_close.to_numpy()[row]
 assert (idx.iloc[row].asset.to_numpy()==c.asset.to_numpy()).all()
 assert (idx.iloc[row].timestamp.to_numpy()==c.origin.to_numpy()).all()
 for column,key in [("observed","y"),("baseline","no_correction"),("corrected","ARM90")]:
  np.testing.assert_allclose(c[column].to_numpy(),a[key][row,h]*scale,rtol=1e-12,atol=1e-10)
 curve_rows+=len(c)
report["verified_exported_price_curve_rows"]=curve_rows
(ROOT/"verification.json").write_text(json.dumps(report,indent=2))
subprocess.run([".venv/bin/python","-u","-m","reproduction.full_tables"],check=True)
subprocess.run([".venv/bin/python","-u","-m","reproduction.full_interpret"],check=True)
import zipfile
with zipfile.ZipFile(ROOT/"review_bundle.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
 for p in sorted(R.rglob("*")):
  if p.is_file():archive.write(p,str(p.relative_to(ROOT)))
 for p in [ROOT/"PROTOCOL.md",ROOT/"ADAPTERS.md",ROOT/"RESULT_GUIDE.md",ROOT/"METHOD_AND_ABLATIONS.md",ROOT/"verification.json",ROOT/"numerical_checks.json",ROOT/"bootstrap_checks.json",ROOT/"TEST_BLOCKED.json"]:archive.write(p,str(p.relative_to(ROOT)))
 for p in Path("reproduction").glob("full*.py"):archive.write(p,str(p))
 for p in (ROOT/"batch_audits").glob("*.json"):archive.write(p,str(p.relative_to(ROOT)))
print("Verified review_bundle.zip generated",flush=True)
