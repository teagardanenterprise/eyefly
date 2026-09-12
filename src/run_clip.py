"""Run the circuit over a video file. Reports alarms with timestamps."""
import cv2, sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from circuit import LoomingCircuit
from motion import Retina

def run(path, show=False, habituation=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): sys.exit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    r, c = Retina(), LoomingCircuit(habituation=habituation)
    events, peak, n, flows = [], 0.0, 0, []

    while True:
        ok, frame = cap.read()
        if not ok: break
        t = n/fps; n += 1
        m = r(frame, t)
        s = c.step(t, m["theta"], m["area"], m["flow"])
        flows.append(m["flow"]); peak = max(peak, s["DNp01"])
        if s["event"] == "ALARM":
            events.append({"t": round(t,2), "theta": round(m["theta"],1),
                           "flow": round(m["flow"],2)})
        if show:
            vis = cv2.resize(frame, (640,360))
            for k,(nm,col) in enumerate([("LC4",(255,180,0)),("LPLC2",(0,200,255)),
                                         ("DNp01",(0,0,255))]):
                v=s[nm]; y=30+k*28
                cv2.rectangle(vis,(10,y-14),(10+int(v*260),y+4),col,-1)
                cv2.putText(vis,f"{nm} {v:.2f}",(280,y),0,0.6,col,2)
            cv2.putText(vis,f"flow {m['flow']:.2f} supp {s['suppress']:.2f}",
                        (10,340),0,0.6,(200,200,200),1)
            if s["event"]: cv2.putText(vis,s["event"],(460,40),0,1.1,(0,0,255),3)
            cv2.imshow("clip", vis)
            if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()
    mean_flow = sum(flows)/max(len(flows),1)
    print(f"{pathlib.Path(path).name:40s} {n:5d}f  peak={peak:.2f}  "
          f"meanflow={mean_flow:.2f}  ALARMS={len(events)}")
    for e in events[:10]: print(f"     {e}")
    return events

if __name__ == "__main__":
    show = "--show" in sys.argv
    for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
        run(p, show=show)
