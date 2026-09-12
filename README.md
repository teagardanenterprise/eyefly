# Fruitfly CCTV

Security-camera alert logic built from the *Drosophila* looming-escape circuit
(LC4 + LPLC2 -> Giant Fiber), using the MaleCNS v1.0 connectome.

Runs locally. No GPU, no training data, no cloud.

Full data-flow documentation (every pipeline stage, its inputs and outputs):
`docs/pipeline.html` — open it in a browser.

## Data
MaleCNS v1.0 (2026) — HHMI Janelia FlyEM, Google Research, University of
Cambridge, MRC LMB. Licensed CC-BY 4.0.
Not included in this repo; see scripts/download_data.sh

## Setup
    conda create -n fruitflycctv python=3.11 -y
    conda activate fruitflycctv
    pip install -r requirements.txt

## Data source

Connectome weights in `src/circuit_weights.json` are derived from the
**MaleCNS v1.0** connectome (released 3 Sept 2026), licensed **CC-BY 4.0**.

Produced by HHMI Janelia Research Campus (FlyEM), Google Research,
the University of Cambridge, and the MRC Laboratory of Molecular Biology.

- Downloads: https://male-cns.janelia.org/download/
- Files used (see `scripts/download_data.sh`):
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather`
  - `body-neurotransmitters-male-cns-v1.0.feather`
  - `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

Reproduce with `python scripts/extract_subgraph.py`.

Extracted subgraph: 14,301 neurons, 23 cell types, 107,544 edges (weight >= 3).

### Key weights

| connection | synapses | model weight |
|---|---|---|
| LC4 -> DNp01 | 6362 | 0.567 |
| LPLC2 -> DNp01 | 4858 | 0.433 |
| LPLC2 -> LC4 | 1780 | 0.159 |

### Model

Response functions follow Ache et al. (2019), *Neural Basis for Looming Size and
Velocity Encoding in the Drosophila Giant Fiber Escape Pathway*, Current Biology.
Weights are derived from connectome synapse counts, not fitted. Thresholds and
habituation constants are our own.

No FlyWire data is used — FlyWire is CC BY-NC and incompatible with commercial use.
