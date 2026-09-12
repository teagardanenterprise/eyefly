"""Live webcam demo: fly escape circuit as a motion alarm."""
import cv2, time, numpy as np, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from circuit import LoomingCircuit
from motion import Retina

def main(src=0, log=None):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): sys.exit(f"cannot open {src}")
    retina, circuit = Retina(), LoomingCircuit(habituation=False)
    t0 = time.time(); events = []; hold_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break
        t = time.time() - t0 if src == 0 else cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        m = retina(frame, t)
        theta, area, mask = m['theta'], m['area'], m['mask']
        s = circuit.step(t, theta, area, m['flow'])

        if s["event"] == "ALARM": hold_until = t + 1.5
        if s["event"] == "ALARM":
            events.append({"t": round(t,2), "theta": round(theta,1)})
            print(f"\n  *** ALARM  t={t:.2f}s  θ={theta:.0f}°")

        vis = cv2.resize(frame, (640, 360))
        for k, (name, col) in enumerate([("LC4",(255,180,0)), ("LPLC2",(0,200,255)),
                                         ("DNp01",(0,0,255))]):
            v = s[name]; y = 30 + k*28
            cv2.rectangle(vis, (10,y-14), (10+int(v*260), y+4), col, -1)
            cv2.putText(vis, f"{name} {v:.2f}", (280,y), 0, 0.6, col, 2)
        cv2.putText(vis, f"theta {theta:5.1f}  gain {s['gain']:.2f}  supp {s['suppress']:.2f}",
                    (10,340), 0, 0.6, (200,200,200), 1)
        show_alarm = t < hold_until
        if show_alarm:
            cv2.putText(vis, "ALARM", (430,45), 0, 1.4, (0,0,255), 4)
            cv2.rectangle(vis, (2,2), (638,358), (0,0,255), 6)
        elif s["event"]:
            cv2.putText(vis, s["event"], (460,40), 0, 1.0, (0,200,255), 3)
        cv2.imshow("fly", vis)
        cv2.imshow("retina", cv2.resize(mask, (240,240), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()
    print(f"\n{len(events)} alarms")
    if log: pathlib.Path(log).write_text(json.dumps(events, indent=2))

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else 0
    main(int(a) if str(a).isdigit() else a,
         log=sys.argv[2] if len(sys.argv) > 2 else None)
