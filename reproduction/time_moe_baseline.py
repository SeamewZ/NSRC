"""Official Time-MoE zero-shot price-only baseline on frozen CelebCrypto windows."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Maple728/TimeMoE-50M")
    ap.add_argument("--context-length", type=int, default=96)
    ap.add_argument("--prediction-length", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    args=ap.parse_args()
    data=Path(args.data); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/"COMPLETE").exists():
        print("already complete", out, flush=True); return
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, torch_dtype="auto", trust_remote_code=True
    )
    model.eval()
    arr=np.load(data/"windows.npz")
    x=arr["x"].astype(np.float32)
    close=x[:, -args.context_length:, 3]
    preds=[]
    with torch.no_grad():
        for lo in range(0, len(close), args.batch_size):
            raw=torch.from_numpy(close[lo:lo+args.batch_size]).to(device)
            mu=raw.mean(dim=-1, keepdim=True)
            sd=raw.std(dim=-1, keepdim=True).clamp_min(1e-6)
            seq=((raw-mu)/sd).unsqueeze(-1).to(model.dtype)
            made=0; chunks=[]
            while made < args.prediction_length:
                outputs=model(input_ids=seq, use_cache=False,
                              max_horizon_length=args.prediction_length-made,
                              return_dict=True)
                chunk=outputs.logits[:, -1, :]
                n=min(int(chunk.shape[-1]), args.prediction_length-made)
                chunk=chunk[:, :n]
                chunks.append(chunk)
                seq=torch.cat([seq, chunk.unsqueeze(-1)], dim=1)
                made += n
            p=torch.cat(chunks, dim=1)*sd+mu
            preds.append(p.float().cpu().numpy())
            if lo % (args.batch_size*10)==0: print("rows",lo,flush=True)
    pred=np.concatenate(preds,axis=0)
    np.save(out/"predictions.npy", pred)
    meta=dict(method="Time-MoE",model=args.model,data=str(data.resolve()),
              model_device=device,context_length=args.context_length,
              prediction_length=args.prediction_length,batch_size=args.batch_size,
              model_input="close-only per-window normalized sequence; official Time-MoE horizon-head autoregressive loop",
              event_inputs=False,
              index_sha256=hashlib.sha256((data/"index.csv").read_bytes()).hexdigest(),
              transformers_version=__import__("transformers").__version__,
              torch_version=torch.__version__)
    (out/"provenance.json").write_text(json.dumps(meta,indent=2))
    (out/"COMPLETE").write_text("Official Time-MoE zero-shot inference complete\n")
    print("COMPLETE",out,flush=True)

if __name__=="__main__": main()
