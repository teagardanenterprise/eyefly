"""Print per-frame front-end values from the webcam so we can see what's wrong."""
import cv2, time, sys, pathlib
sys.path.insert(0, str(pathlib.Path("src").resolve()))
from motion import Retina
from circuit import LoomingCircuit

cap = cv2.VideoCapture(0)
r, c = Retina(), LoomingCircuit()
t0 = time.time(); n = 0
print(f"{'t':>6} {'theta':>7} {'area':>6} {'flow':>6} {'LC4':>5} {'LPLC2':>6} {'DNp01':>6} {'gain':>5}")
while True:
    ok, f = cap.read()
    if not ok: break
    t = time.time() - t0; n += 1
    m = r(f, t); s = c.step(t, m["theta"], m["area"], m["flow"])
    if n % 5 == 0:
        print(f"{t:6.1f} {m['theta']:7.1f} {m['area']:6.3f} {m['flow']:6.2f} "
              f"{s['LC4']:5.2f} {s['LPLC2']:6.2f} {s['DNp01']:6.2f} {s['gain']:5.2f}")
    cv2.imshow("mask", cv2.resize(m["mask"], (300,300), interpolation=cv2.INTER_NEAREST))
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
