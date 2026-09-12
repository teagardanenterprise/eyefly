"""MaleCNS v1.0 -> signed, type-level weight matrix for the looming circuit."""
import pandas as pd, numpy as np, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
D, OUT = ROOT/"data", ROOT/"src"
MIN_W = 3          # drop single/double-synapse noise (median edge weight is 1)

TYPES = (["LC4","LPLC2","LC11","LPLC1","LPLC4"]
         + [f"T4{x}" for x in "abcd"] + [f"T5{x}" for x in "abcd"]
         + ["HSE","HSN","HSS","VS"]
         + ["DNp01","DNp02","DNp03","DNp04","DNp06","DNp11"])

SIGN = {"gaba":-1.0, "glutamate":-1.0,        # glutamate is usually inhibitory in fly
        "acetylcholine":1.0, "dopamine":1.0,
        "serotonin":1.0, "octopamine":1.0}

print("loading annotations...")
ann = pd.read_feather(D/"body-annotations-male-cns-v1.0-minconf-0.5.feather",
                      columns=["bodyId","type","instance","somaSide","class","superclass"])
ann = ann[ann["type"].isin(TYPES)].copy()
print(f"  {len(ann)} neurons across {ann['type'].nunique()} types")
print(ann["type"].value_counts().to_string())

print("\nloading neurotransmitters...")
nt = pd.read_feather(D/"body-neurotransmitters-male-cns-v1.0.feather",
                     columns=["body","consensus_nt"])
ann = ann.merge(nt, left_on="bodyId", right_on="body", how="left")
ann["sign"] = ann["consensus_nt"].str.lower().map(SIGN)
_miss = ann["sign"].isna()
if _miss.any():
    print(f"WARNING: {_miss.sum()} neurons lack a mapped NT "
          f"({ann.loc[_miss,'type'].value_counts().to_dict()}) -> defaulting to +1")
    ann["sign"] = ann["sign"].fillna(1.0)
else:
    print(f"  all {len(ann)} neurons have a mapped NT; "
          f"signs present: {sorted(ann['sign'].unique())}")
print(ann.groupby("type")["consensus_nt"].agg(lambda s: s.mode().iat[0]
                                              if len(s.mode()) else "?").to_string())

ids = set(ann["bodyId"].astype(np.int64))
print(f"\nloading 151M edges (slow, ~30s)...")
w = pd.read_feather(D/"connectome-weights-male-cns-v1.0-minconf-0.5.feather")
w = w[(w["weight"] >= MIN_W)
      & w["body_pre"].isin(ids) & w["body_post"].isin(ids)]
print(f"  {len(w)} edges within subgraph (weight >= {MIN_W})")

m = ann.set_index("bodyId")
w["type_pre"]  = w["body_pre"].map(m["type"])
w["type_post"] = w["body_post"].map(m["type"])
w["signed"]    = w["weight"] * w["body_pre"].map(m["sign"])

agg = (w.groupby(["type_pre","type_post"])
        .agg(synapses=("weight","sum"), signed=("signed","sum"),
             n_edges=("weight","size")).reset_index()
        .sort_values("synapses", ascending=False))

OUT.mkdir(exist_ok=True)
matrix = {f"{r.type_pre}->{r.type_post}":
          {"synapses": int(r.synapses), "signed": float(r.signed),
           "n_edges": int(r.n_edges)} for r in agg.itertuples()}
counts = ann["type"].value_counts().to_dict()
signs  = ann.groupby("type")["sign"].median().to_dict()

(OUT/"circuit_weights.json").write_text(json.dumps(
    {"source": "MaleCNS v1.0 (CC-BY 4.0), HHMI Janelia / Google Research / "
               "Cambridge / MRC LMB",
     "min_synapse_weight": MIN_W,
     "neuron_counts": {k:int(v) for k,v in counts.items()},
     "type_sign": {k:float(v) for k,v in signs.items()},
     "edges": matrix}, indent=2))

print("\n--- KEY EDGES ---")
for e in ["LC4->DNp01","LPLC2->DNp01","LPLC2->LC4","LC11->DNp01",
          "T4a->LPLC2","T5a->LPLC2","T4a->LC4","T5a->LC4"]:
    a,b = e.split("->")
    row = agg[(agg.type_pre==a)&(agg.type_post==b)]
    print(f"  {e:18s} {int(row.synapses.iat[0]) if len(row) else 0:>7} synapses")

print("\n--- TOP 25 EDGES ---")
print(agg.head(25).to_string(index=False))
print(f"\nwrote {OUT/'circuit_weights.json'}")

ann[["bodyId","type","instance","somaSide"]].to_csv(OUT/"subgraph_bodies.csv", index=False)
print(f"wrote {OUT/'subgraph_bodies.csv'} ({len(ann)} bodies, for skeleton fetch)")
