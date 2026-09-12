import pandas as pd, re, pathlib
D = pathlib.Path(__file__).resolve().parent.parent / "data"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("="*70, "\nANNOTATIONS\n", "="*70)
ann = pd.read_feather(D/"body-annotations-male-cns-v1.0-minconf-0.5.feather")
print("rows:", len(ann))
print("cols:", ann.columns.tolist())
print(ann.head(3).to_string())

cands = [c for c in ann.columns if "type" in c.lower() or "instance" in c.lower()]
print("\ntype-ish cols:", cands)

PAT = r'^(LC\d|LPLC|LPC|LLPC|DNp|T4|T5|HS[EN]?|VS)'
for c in cands:
    s = ann[c].dropna().astype(str)
    hits = sorted({x for x in s.unique() if re.match(PAT, x)})
    if hits:
        print(f"\n--- {c}: {len(hits)} matches ---")
        print(hits)

print("\nTOP TYPES BY COUNT")
for c in cands[:2]:
    print(f"\n[{c}]"); print(ann[c].value_counts().head(15))

print("\n" + "="*70, "\nNEUROTRANSMITTERS\n", "="*70)
nt = pd.read_feather(D/"body-neurotransmitters-male-cns-v1.0.feather")
print("rows:", len(nt)); print("cols:", nt.columns.tolist())
print(nt.head(3).to_string())

print("\n" + "="*70, "\nWEIGHTS\n", "="*70)
w = pd.read_feather(D/"connectome-weights-male-cns-v1.0-minconf-0.5.feather")
print("rows:", len(w)); print("cols:", w.columns.tolist())
print(w.head(3).to_string())
print("\nnumeric summary:")
print(w.select_dtypes("number").describe().to_string())
