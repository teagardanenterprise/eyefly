"""Dump per-frame traces so we can see what the front end is actually producing."""
import cv2, sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path("src").resolve()))
from motion import Retina
from circuit import LoomingCircuit

def trace(path, every=15):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    r, c = Retina(), LoomingCircuit()
    th, fl, gf = [], [], []
    n = 0
    print(f"\n=== {pathlib.Path(path).name} ===")
    print(f"{'t':>6} {'theta':>7} {'dtheta':>8} {'flow':>6} {'LC4':>5} {'LPLC2':>6} {'DNp01':>6}")
    prev_th = 0.0
    while True:
        ok, f = cap.read()
        if not ok: break
        t = n/fps; n += 1
        m = r(f, t); s = c.step(t, m["theta"], m["area"], m["flow"])
        th.append(m["theta"]); fl.append(m["flow"]); gf.append(s["DNp01"])
        if n % every == 0:
            print(f"{t:6.2f} {m['theta']:7.1f} {m['theta']-prev_th:8.1f} "
                  f"{m['flow']:6.2f} {s['LC4']:5.2f} {s['LPLC2']:6.2f} {s['DNp01']:6.2f}")
        prev_th = m["theta"]
    cap.release()
    th, fl, gf = np.array(th), np.array(fl), np.array(gf)
    print(f"\n  theta : min={th.min():.1f} med={np.median(th):.1f} "
          f"p90={np.percentile(th,90):.1f} max={th.max():.1f}  zero-frac={np.mean(th==0):.2f}")
    print(f"  flow  : med={np.median(fl):.2f} p90={np.percentile(fl,90):.2f} max={fl.max():.2f}")
    print(f"  DNp01 : med={np.median(gf):.2f} p90={np.percentile(gf,90):.2f} max={gf.max():.2f}")

for p in sys.argv[1:]: trace(p)
