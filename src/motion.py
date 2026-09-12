"""Retina + motion front end.

Outputs subtended angle of the dominant moving object, plus a wide-field
optic-flow estimate (the HS/VS "am I moving?" signal) used to suppress
self-motion.
"""
import cv2, numpy as np


class Retina:
    def __init__(self, grid=48, fov_deg=60.0, decay=0.6, warmup_s=1.0):
        self.grid, self.fov_deg, self.decay = grid, fov_deg, decay
        self.warmup_s = warmup_s
        self.prev = None; self.prev_small = None
        self.accum = None; self.t0 = None
        self.prev_cx = None; self.prev_cy = None

    def __call__(self, frame, t):
        if self.t0 is None: self.t0 = t
        warm = (t - self.t0) < self.warmup_s

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        gf = cv2.GaussianBlur(small, (3,3), 0).astype(np.float32)
        gf = (gf - gf.mean())/(gf.std() + 1e-3)

        if self.prev is None:
            self.prev, self.prev_small = gf, small
            self.accum = np.zeros_like(gf)
            return dict(theta=0.0, area=0.0, flow=0.0, warm=True,
                        mask=np.zeros_like(small))

        # --- wide-field optic flow (HS/VS analogue) ---
        fl = cv2.calcOpticalFlowFarneback(self.prev_small, small, None,
                                          0.5, 2, 9, 2, 5, 1.1, 0)
        self.prev_small = small
        mu = fl.reshape(-1,2).mean(axis=0)              # global translation
        resid = fl - mu                                  # after cancellation
        flow_mag = float(np.hypot(*mu))                  # self-motion strength

        d = np.abs(gf - self.prev); self.prev = gf
        self.accum = self.decay*self.accum + (1-self.decay)*d

        # weight motion energy by residual (non-global) flow
        rmag = np.hypot(resid[...,0], resid[...,1])
        rmag = rmag/(rmag.max() + 1e-6)
        energy = self.accum * (0.3 + 0.7*rmag)

        med = float(np.median(energy)); mad = float(np.median(np.abs(energy-med)))
        thr = med + 6.0*(mad + 1e-4)
        # absolute floor: a quiet scene must report NOTHING
        peak = float(energy.max())
        if peak < 0.35 or peak < 4.0*(med + 1e-4):
            thr = np.inf
        mask = (energy > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8))

        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1 or warm:
            return dict(theta=0.0, area=0.0, flow=flow_mag, warm=warm,
                        mask=mask*255)
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        cx = stats[i, cv2.CC_STAT_LEFT] + w/2.0
        cy = stats[i, cv2.CC_STAT_TOP]  + h/2.0
        drift = 0.0
        if self.prev_cx is not None:
            drift = float(np.hypot(cx-self.prev_cx, cy-self.prev_cy))/self.grid
        self.prev_cx, self.prev_cy = cx, cy

        # min(w,h) not max(w,h): a lateral sweep smears wide but stays thin.
        # drift penalty: looming expands about a fixed point, passing translates.
        theta = (min(w, h)/self.grid) * self.fov_deg * max(0.0, 1.0 - 8.0*drift)
        return dict(theta=theta, area=area/self.grid**2, flow=flow_mag,
                    drift=drift, warm=False,
                    mask=(lab == i).astype(np.uint8)*255)
