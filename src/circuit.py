"""Drosophila looming-escape circuit (Ache et al. 2019, Current Biology).

GF response = w1 * highpass(angular velocity)  [LC4]
            + w2 * gaussian(angular size)      [LPLC2, peak ~40 deg]
            + cross term                        [LPLC2 -> LC4]
Weights derived from MaleCNS v1.0 connectome (CC-BY 4.0).
"""
import json, math, pathlib

W = json.loads((pathlib.Path(__file__).parent/"circuit_weights.json").read_text())
def _syn(a, b): return W["edges"].get(f"{a}->{b}", {}).get("synapses", 0)


class LoomingCircuit:
    # --- published physiology (Ache 2019) ---
    SIZE_PEAK_DEG  = 40.0    # LPLC2 angular-size tuning peak
    SIZE_WIDTH_DEG = 18.0    # Gaussian sigma            [empirical]
    VEL_TAU        = 0.10    # LC4 high-pass tau, s      [empirical]
    VEL_NORM       = 90.0    # deg/s -> unit response    [empirical]
    SUPRALINEAR    = 1.4     # summation exponent        [empirical]

    # --- alerting ---
    ALARM_THRESH   = 0.80
    NOTICE_THRESH  = 0.30
    REFRACTORY_S   = 3.0
    MIN_ALARM_DEG  = 0.0    # size gate: velocity alone must not alarm
    FLOW_SUPPRESS  = 2.5     # wide-field flow (px/frame) that fully gates DNp01

    # --- habituation: PER SECOND, not per frame [empirical] ---
    HAB_DECAY_PER_S   = 0.15   # gain lost per second of sustained full response
    HAB_RECOVER_PER_S = 0.60
    HAB_FLOOR         = 0.45

    def __init__(self, habituation=True):
        lc4, lplc2 = _syn("LC4","DNp01"), _syn("LPLC2","DNp01")
        tot = (lc4 + lplc2) or 1
        self.w_vel   = lc4/tot                  # 0.567 from connectome
        self.w_size  = lplc2/tot                # 0.433
        self.w_cross = _syn("LPLC2","LC4")/tot  # 0.159
        self.habituation = habituation
        self.reset()

    def reset(self):
        self._vel_hp = 0.0; self._prev_theta = None; self._t = None
        self.gain = 1.0; self._last_alarm = -1e9; self.state = {}

    def lc4(self, dtheta_dt, dt):
        """Leaky high-pass on expansion velocity."""
        a = self.VEL_TAU/(self.VEL_TAU + dt)
        self._vel_hp = a*self._vel_hp + (1.0-a)*dtheta_dt
        return max(0.0, min(1.0, self._vel_hp/self.VEL_NORM))

    def lplc2(self, theta_deg):
        """Non-monotonic Gaussian tuning on subtended angular size."""
        z = (theta_deg - self.SIZE_PEAK_DEG)/self.SIZE_WIDTH_DEG
        return math.exp(-0.5*z*z)

    def step(self, t, theta_deg, blob_area_frac=0.0, flow=0.0):
        if self._t is None:
            self._t, self._prev_theta = t, theta_deg
            return self._emit(0.0, 0.0, 0.0, None)
        dt = max(t - self._t, 1e-3)
        dtheta = (theta_deg - self._prev_theta)/dt
        self._t, self._prev_theta = t, theta_deg

        a_lc4   = self.lc4(max(dtheta, 0.0), dt)   # only expansion
        a_lplc2 = self.lplc2(theta_deg)

        drive = (self.w_vel*a_lc4 + self.w_size*a_lplc2
                 + self.w_cross*a_lplc2*a_lc4)
        gf_raw = min(1.0, drive**(1.0/self.SUPRALINEAR)) if drive > 0 else 0.0
        # HS/VS wide-field suppression: self-motion is not a threat
        supp = 1.0/(1.0 + (flow/self.FLOW_SUPPRESS)**2)
        gf = gf_raw * self.gain * supp

        if self.habituation:
            self.gain += (-self.HAB_DECAY_PER_S*gf_raw
                          + self.HAB_RECOVER_PER_S*(1.0-self.gain)) * dt
            self.gain = max(self.HAB_FLOOR, min(1.0, self.gain))

        event = None
        if (gf >= self.ALARM_THRESH and theta_deg >= self.MIN_ALARM_DEG
                and (t - self._last_alarm) > self.REFRACTORY_S):
            event, self._last_alarm = "ALARM", t
        elif gf >= self.NOTICE_THRESH:
            event = "NOTICE"
        return self._emit(a_lc4, a_lplc2, gf, event, supp)

    def _emit(self, lc4, lplc2, gf, event, supp=1.0):
        self.state = {"LC4": round(lc4,3), "LPLC2": round(lplc2,3),
                      "DNp01": round(gf,3), "gain": round(self.gain,3), "suppress": round(supp,3),
                      "event": event}
        return self.state


def _approach(c, label, t0=0.0, dur=1.5, th0=5.0, th1=70.0, fps=20):
    print(f"\n{label}")
    n = int(dur*fps); peak = 0.0
    for i in range(n+1):
        t = t0 + i/fps
        th = th0 + (th1-th0)*(i/n)**2
        s = c.step(t, th); peak = max(peak, s["DNp01"])
        if i % 2 == 0:
            print(f"  t={t:5.2f} θ={th:5.1f}° "
                  f"{'#'*int(s['DNp01']*40):<40} g={s['gain']:.2f} {s['event'] or ''}")
    return peak


if __name__ == "__main__":
    c = LoomingCircuit()
    print(f"connectome weights: LC4(vel)={c.w_vel:.3f}  "
          f"LPLC2(size)={c.w_size:.3f}  cross={c.w_cross:.3f}")
    print(f"  LC4->DNp01   {_syn('LC4','DNp01')} synapses")
    print(f"  LPLC2->DNp01 {_syn('LPLC2','DNp01')} synapses")

    _approach(c, "APPROACH (should ALARM near 40 deg):")

    print("\n\nHABITUATION — same stimulus repeated 5x:")
    c.reset(); t = 0.0
    for k in range(5):
        p = _approach(c, f"  -- repeat {k+1} --", t0=t)
        print(f"     peak={p:.2f} gain={c.gain:.2f}")
        t += 4.0
        for _ in range(20): c.step(t-2.0+_*0.05, 2.0)   # quiet gap
