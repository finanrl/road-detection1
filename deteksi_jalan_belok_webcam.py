"""
Deteksi Jalur Jalan Lurus & Belok - Webcam / Video / Foto
=========================================================
Alur:
  Input -> Grayscale -> Gaussian Blur -> Canny -> ROI -> HoughLinesP
  -> klasifikasi kiri/kanan (gating) -> fit kurva berbobot -> EMA
  -> arah belok (hysteresis) & offset
  -> finalize_lanes(): koordinat FINAL dengan fallback
     (deteksi / turunan / estimasi / hilang)
  -> output JSON Lines (--koordinat)

Cara pakai:
  python deteksi_jalan_belok_webcam.py --source 0
  python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --koordinat koordinat.jsonl
  python deteksi_jalan_belok_webcam.py --source jalan.jpg --stages

Tombol: q = keluar | d = debug | s = snapshot | p = hasil per tahap
"""

import argparse
import copy       # deepcopy state (hasil per tahap pada video)
import json       # menulis koordinat ke file JSON Lines
import os         # membuat folder penyimpanan
import threading  # membaca kamera di thread terpisah (anti-delay)
import time       # pengatur kecepatan video & hitung FPS
from collections import deque  # riwayat deteksi untuk fallback

import cv2
import numpy as np

# ============================ PARAMETER ============================
PROCESS_WIDTH = 640

# --- Grayscale -> Gaussian Blur -> Canny
BLUR_KSIZE = 5                     # harus ganjil (3, 5, 7, ...)
CANNY_LOW, CANNY_HIGH = 50, 150

# --- ROI (titik poligon dalam proporsi lebar/tinggi: [x, y])
ROI_PROP = np.float32([[0.04, 0.92], [0.28, 0.60], [0.64, 0.60], [0.78, 0.92]])
Y_TOP, Y_BOTTOM = 0.60, 0.92
Y_REF = 0.88                       # tinggi acuan untuk cek lompatan & offset

# --- HoughLinesP
HOUGH_THRESH = 25
HOUGH_MIN_LEN = 30
HOUGH_MAX_GAP = 40
MIN_ABS_SLOPE = 0.35
BRIGHT_DELTA = 35                  # 0 = brightness filter dimatikan
BRIGHT_FRACTION = 0.4
BRIGHT_DILATE = 7

# --- Gating & fit
GATE_TOL = 0.07
MIN_LANE_GAP = 0.12
MAX_LANE_GAP = 0.0                 # 0 = nonaktif
Y_HORIZON = 0.48                   # tinggi cakrawala (proporsi tinggi gambar)
MIN_SAMPLES = 8
MIN_SPAN = 0.08
SHORT_SPAN = 0.18
SLOPE_CHECK = True
CURVE_MIN_SPAN = 0.10
NOISE_FRAC = 0.006
CURVE_PRIOR_FRAC = 0.030

# --- Smoothing filter (EMA)
ALPHA = 0.30
MAX_JUMP = 0.08
RELOCK_AFTER = 6
MAX_MISSING = 15

# --- Arah belok
BEND_ON, BEND_OFF = 0.018, 0.010   # hysteresis: ON > OFF
BEND_EMA = 0.15
CURVE_SIGN = 1                     # ubah ke -1 jika arah kiri/kanan terbalik
LANE_WIDTH_M = 3.5
OFFSET_OK_M = 0.8

# --- Sisi yang dideteksi: "dua", "kiri", atau "kanan"
SISI = "dua"

# --- Output koordinat & fallback
N_TITIK = 6                        # jumlah titik per garis (dari Y_TOP ke Y_BOTTOM)
HIST_LEN = 10                      # jumlah deteksi valid terakhir untuk rata-rata
HOLD_MAX_FRAMES = 30               # frame maksimum memakai estimasi sebelum HILANG
DECAY = 0.90                       # pelemahan tren saat estimasi (0-1)
LEBAR_PROP_DEFAULT = 0.45          # tebakan awal lebar jalur (proporsi lebar gambar di Y_REF)
LEBAR_ALPHA = 0.10                 # kecepatan kalibrasi lebar jalur (EMA)
Y_LEVELS = np.linspace(Y_TOP, Y_BOTTOM, N_TITIK)
# ===================================================================


# ------------------------------------------------------------------ model garis
def eval_curve(theta, dom, t):
    """x = A t^2 + B t + C, dipakai di dalam rentang data (dom). Di luar rentang,
    garis diperpanjang LURUS mengikuti garis singgung."""
    A, B, C = theta
    t_lo, t_hi = dom
    t = np.asarray(t, dtype=np.float64)
    tc = np.clip(t, t_lo, t_hi)
    x = A * tc ** 2 + B * tc + C
    x = x + (2 * A * t_lo + B) * np.minimum(t - t_lo, 0.0)
    x = x + (2 * A * t_hi + B) * np.maximum(t - t_hi, 0.0)
    return x


def to_t(y, h):
    """Ubah koordinat y (pixel) menjadi t ternormalisasi (0 di dasar ROI)."""
    return np.asarray(y, dtype=np.float64) / h - Y_BOTTOM


def _arah_ok(fit, side):
    """Tolak garis yang arah miringnya salah.
    Garis KANAN jalur: dx/dt > 0. Garis KIRI jalur: dx/dt < 0."""
    if fit is None or side is None or not SLOPE_CHECK:
        return fit
    th, dom = fit
    t_mid = 0.5 * (dom[0] + dom[1])
    dxdt = 2 * th[0] * t_mid + th[1]
    if side == "R" and dxdt <= 0:
        return None
    if side == "L" and dxdt >= 0:
        return None
    return fit


def fit_lane(xs, ys, w, h, side=None, prior=None):
    """Fit x = A t^2 + B t + C berbobot ridge + buang pencilan.
    Mengembalikan (theta, domain) atau None kalau data tidak cukup."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < MIN_SAMPLES:
        return None
    t = to_t(ys, h)
    span = t.max() - t.min()
    if span < MIN_SPAN:
        return None
    use_curve = span >= CURVE_MIN_SPAN

    # Data pendek + sudah ada garis lama: kemiringan & lengkung dikunci dari
    # garis lama, hanya posisi (C) yang dicari.
    if prior is not None and span < SHORT_SPAN:
        A0, B0 = float(prior[0]), float(prior[1])
        c = float(np.median(xs - (A0 * t ** 2 + B0 * t)))   # median: kebal pencilan
        fit = (np.array([A0, B0, c]), (float(t.min()), float(t.max())))
        return _arah_ok(fit, side)

    t_top = Y_TOP - Y_BOTTOM
    sigma_a = CURVE_PRIOR_FRAC * w / (t_top ** 2)
    ridge = (NOISE_FRAC * w / sigma_a) ** 2

    keep = np.ones(len(xs), dtype=bool)
    best = None
    for _ in range(3):                       # maksimal 3 putaran buang outlier
        tt, xx = t[keep], xs[keep]
        if len(xx) < MIN_SAMPLES:
            break
        if use_curve:
            phi = np.stack([tt ** 2, tt, np.ones_like(tt)], axis=1)
            g = phi.T @ phi + np.diag([ridge, 1e-9, 1e-9])   # ridge hanya pada A
            th = np.linalg.solve(g, phi.T @ xx)
        else:
            phi = np.stack([tt, np.ones_like(tt)], axis=1)
            b, c = np.linalg.lstsq(phi, xx, rcond=None)[0]
            th = np.array([0.0, b, c])                       # garis lurus: A = 0
        best = (th, (float(tt.min()), float(tt.max())))
        res = xs - (th[0] * t ** 2 + th[1] * t + th[2])
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        new_keep = np.abs(res) < max(0.012 * w, 2.5 * mad)
        if new_keep.sum() < MIN_SAMPLES or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return _arah_ok(best, side)


class LaneTrack:
    """Satu sisi garis + smoothing EMA + tolak lompatan + persistensi."""

    def __init__(self):
        self.theta = None      # parameter [A, B, C]
        self.dom = None        # rentang t yang punya data
        self.missing = 0       # frame berturut-turut gagal
        self.rejects = 0       # penolakan lompatan berturut-turut
        self.fresh = False     # True jika frame INI diterima dari deteksi

    def x_at(self, y, h):
        """Posisi x garis pada ketinggian y."""
        if self.theta is None:
            return None
        return float(eval_curve(self.theta, self.dom, to_t(y, h)))

    def update(self, fit, w, h):
        """Gabungkan hasil fitting frame baru dengan garis lama."""
        self.fresh = False
        if fit is None:
            self._miss()
            return
        theta, dom = fit
        dom = np.array(dom)
        if self.theta is None:
            self.theta, self.dom, self.missing, self.rejects = theta, dom, 0, 0
            self.fresh = True
            return
        new_x = float(eval_curve(theta, dom, to_t(Y_REF * h, h)))
        jump = abs(new_x - self.x_at(Y_REF * h, h))
        if jump > MAX_JUMP * w:
            self.rejects += 1
            if self.rejects >= RELOCK_AFTER:          # konsisten pindah -> terima
                self.theta, self.dom, self.missing, self.rejects = theta, dom, 0, 0
                self.fresh = True
            else:
                self._miss()
            return
        # EMA: campur hasil baru (ALPHA) dengan hasil lama (1-ALPHA)
        self.theta = ALPHA * theta + (1 - ALPHA) * self.theta
        self.dom = ALPHA * dom + (1 - ALPHA) * np.array(self.dom)
        self.missing, self.rejects = 0, 0
        self.fresh = True

    def _miss(self):
        """Catat kegagalan; hapus garis jika gagal terlalu lama."""
        self.missing += 1
        if self.missing > MAX_MISSING:
            self.theta, self.dom, self.rejects = None, None, 0


class LaneState:
    """Seluruh kondisi deteksi yang dibawa dari frame ke frame."""

    def __init__(self):
        self.left = LaneTrack()
        self.right = LaneTrack()
        self.bend = 0.0
        self.arah = "Lurus"
        self.debug = False
        # riwayat & status untuk fallback
        self.kiri_hist = deque(maxlen=HIST_LEN)
        self.kanan_hist = deque(maxlen=HIST_LEN)
        self.kiri_hold = 0             # frame sejak deteksi kiri terakhir
        self.kanan_hold = 0            # frame sejak deteksi kanan terakhir
        self.lebar_prop = LEBAR_PROP_DEFAULT
        self.lebar_terkalibrasi = False
        self.frame_idx = 0


# ------------------------------------------------------------------ tahap pipeline
def preprocess(frame):
    """Grayscale -> Gaussian Blur -> Canny. Mengembalikan ketiga hasilnya."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return gray, blur, edges


_ROI_CACHE = {}
_BRIGHT_KERNEL = np.ones((BRIGHT_DILATE, BRIGHT_DILATE), np.uint8)


def get_roi(w, h):
    """Hitung polygon & mask ROI sekali saja per resolusi."""
    key = (w, h)
    if key not in _ROI_CACHE:
        poly = (ROI_PROP * np.float32([w, h])).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        _ROI_CACHE.clear()
        _ROI_CACHE[key] = (poly, mask)
    return _ROI_CACHE[key]


def apply_roi(edges, mask):
    """Pertahankan edge hanya di dalam area ROI."""
    return cv2.bitwise_and(edges, mask)


def sample_segment(x1, y1, x2, y2, step=6.0):
    """Ambil titik-titik merata sepanjang segmen."""
    n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
    return np.linspace(x1, x2, n), np.linspace(y1, y2, n)


def collect_segments(lines, w, h, left, right, bright=None):
    """Pisahkan segmen Hough ke kiri/kanan (dengan gating)."""
    L, R = ([], []), ([], [])
    if lines is None:
        return L, R
    tol = GATE_TOL * w

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue

        slope = dy / dx
        if abs(slope) < MIN_ABS_SLOPE or abs(slope) > 12:
            continue

        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ends = ((x1, y1), (x2, y2))
        sx, sy = sample_segment(x1, y1, x2, y2)

        if bright is not None:
            ix = np.clip(sx.astype(int), 0, w - 1)
            iy = np.clip(sy.astype(int), 0, h - 1)
            if bright[iy, ix].mean() < BRIGHT_FRACTION:
                continue

        def near(track):
            """True jika kedua ujung segmen dekat dengan garis track."""
            return all(abs(x - track.x_at(y, h)) < tol for x, y in ends)

        if left.theta is not None or right.theta is not None:
            side = None
            if left.theta is not None and slope < 0 and near(left):
                side = "L"
            elif right.theta is not None and slope > 0 and near(right):
                side = "R"
            elif left.theta is None and slope < 0 and xm < 0.62 * w:
                side = "L"
            elif right.theta is None and slope > 0 and xm > 0.38 * w:
                side = "R"

            if side == "R" and left.theta is not None and \
                    xm < left.x_at(ym, h) + MIN_LANE_GAP * w:
                side = None

            if side == "L" and right.theta is not None and \
                    xm > right.x_at(ym, h) - MIN_LANE_GAP * w:
                side = None

            if MAX_LANE_GAP > 0 and side == "R" and left.theta is not None and \
                    xm > left.x_at(ym, h) + MAX_LANE_GAP * w * \
                    max(ym / h - Y_HORIZON, 0.01) / (Y_REF - Y_HORIZON):
                side = None
        else:
            side = (
                "L" if (slope < 0 and xm < 0.62 * w)
                else ("R" if (slope > 0 and xm > 0.38 * w) else None)
            )

        if (side == "R" and SISI == "kiri") or (side == "L" and SISI == "kanan"):
            side = None

        if side:
            tgt = L if side == "L" else R
            tgt[0].extend(sx.tolist())
            tgt[1].extend(sy.tolist())

    return L, R


def bend_fraction(state, w):
    """Mengukur kelengkungan garis."""
    t_top = Y_TOP - Y_BOTTOM
    vals, wts = [], []
    for tr in (state.left, state.right):
        if tr.theta is not None:
            span = float(tr.dom[1] - tr.dom[0])
            vals.append(tr.theta[0] * t_top ** 2 / w)
            wts.append(span ** 2)
    if not vals:
        return None
    b = float(np.average(vals, weights=wts))
    if len(vals) == 1:
        b *= 0.6
    return b * CURVE_SIGN


def update_direction(state, w):
    """Tentukan Lurus / Belok KIRI / Belok KANAN dengan hysteresis."""
    b = bend_fraction(state, w)
    if b is None:
        return
    state.bend = (1 - BEND_EMA) * state.bend + BEND_EMA * b
    if state.arah == "Lurus":
        if abs(state.bend) > BEND_ON:
            state.arah = "Belok KANAN" if state.bend > 0 else "Belok KIRI"
    else:
        if abs(state.bend) < BEND_OFF:
            state.arah = "Lurus"
        else:
            state.arah = "Belok KANAN" if state.bend > 0 else "Belok KIRI"


def curve_points(track, h, w):
    """Titik-titik kurva (40 titik dari Y_TOP ke Y_BOTTOM) siap digambar."""
    y = np.linspace(Y_TOP * h, Y_BOTTOM * h, 40)
    x = eval_curve(track.theta, track.dom, to_t(y, h))
    return np.stack([x, y], axis=1).astype(np.int32)


def draw_panel(img, state, offset_m, both, any_line):
    """Panel informasi (arah, kelengkungan, posisi, status) di pojok kiri atas."""
    h, w = img.shape[:2]
    s = w / 640.0

    if any_line:
        kat = abs(state.bend)
        kat_txt = (
            "-"
            if state.arah == "Lurus"
            else ("ringan" if kat < 0.05 else
                  "sedang" if kat < 0.10 else "tajam")
        )

        sisi = "kanan" if offset_m > 0 else "kiri"
        baik = abs(offset_m) < OFFSET_OK_M

        lines = [
            (f"Arah: {state.arah}", (0, 255, 255), 0.6),
            (f"Kelengkungan: {kat_txt}", (255, 255, 255), 0.42),
            (
                f"Posisi: {abs(offset_m):.2f} m di {sisi} tengah"
                if both
                else "Posisi: - (baru satu garis)",
                (255, 255, 255),
                0.42
            ),
            (
                "Status: Baik" if baik else "Status: Perhatikan posisi!",
                (0, 255, 0) if baik else (0, 165, 255),
                0.45
            ),
        ] if both else [
            (f"Arah: {state.arah}", (0, 255, 255), 0.6),
            (f"Kelengkungan: {kat_txt}", (255, 255, 255), 0.42),
            (
                ("Baru satu garis terdeteksi", (0, 165, 255), 0.45)
                if SISI == "dua"
                else (f"Mode: hanya garis {SISI}", (255, 255, 255), 0.45)
            ),
        ]
    else:
        lines = [("Jalur tidak terdeteksi", (0, 0, 255), 0.6)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [
        cv2.getTextSize(t, font, sc * s, 2 if sc > 0.5 else 1)[0]
        for t, _, sc in lines
    ]

    pw = int(max(sz[0] for sz in sizes) + 24 * s)
    line_h = [int(sz[1] + 9 * s) for sz in sizes]
    ph = int(sum(line_h) + 10 * s)

    roi = img[0:ph, 0:pw]
    img[0:ph, 0:pw] = cv2.addWeighted(roi, 0.35, np.full_like(roi, 30), 0.65, 0)
    cv2.rectangle(img, (0, 0), (pw, ph), (255, 255, 255), 1)

    y = 0
    for (txt, col, sc), lh in zip(lines, line_h):
        y += lh
        cv2.putText(
            img, txt, (int(12 * s), int(y)), font, sc * s, col,
            2 if sc > 0.5 else 1, cv2.LINE_AA
        )


def detect_lanes(frame, state):
    """Tahap deteksi jalur (mentah, dari gambar)."""
    h, w = frame.shape[:2]

    gray, blur, edges = preprocess(frame)

    poly, roi_mask = get_roi(w, h)
    roi_edges = apply_roi(edges, roi_mask)

    bright = None
    if BRIGHT_DELTA > 0:
        med = float(np.median(blur[roi_mask > 0]))
        bright = cv2.dilate(
            (blur > med + BRIGHT_DELTA).astype(np.uint8),
            _BRIGHT_KERNEL
        ) > 0
        roi_edges = roi_edges * bright.astype(np.uint8)

    lines = cv2.HoughLinesP(
        roi_edges, 1, np.pi / 180, HOUGH_THRESH,
        minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP
    )

    Lp, Rp = collect_segments(lines, w, h, state.left, state.right, bright)

    state.left.update(fit_lane(Lp[0], Lp[1], w, h, "L", state.left.theta), w, h)
    state.right.update(fit_lane(Rp[0], Rp[1], w, h, "R", state.right.theta), w, h)

    update_direction(state, w)

    return lines, roi_edges, poly


# ================================================================
#   KOORDINAT OUTPUT + FALLBACK
# ================================================================
def xs_di_level(track, h):
    """Posisi x garis di setiap level output (array sepanjang N_TITIK)."""
    return eval_curve(track.theta, track.dom, to_t(Y_LEVELS * h, h))


def lebar_jalur(y_px, h, w, lebar_prop):
    """Lebar jalur (pixel) pada ketinggian y_px, mengikuti perspektif:
    makin dekat ke kamera makin lebar; di cakrawala mendekati 0."""
    denom = max(Y_REF - Y_HORIZON, 0.01)
    frac = (np.asarray(y_px, dtype=np.float64) / h - Y_HORIZON) / denom
    return lebar_prop * w * np.maximum(frac, 0.02)


def kalibrasi_lebar(state, w, h):
    """Ukur lebar jalur dari dua garis yang terdeteksi bersamaan, lalu haluskan (EMA)."""
    xl = state.left.x_at(Y_REF * h, h)
    xr = state.right.x_at(Y_REF * h, h)
    if xl is None or xr is None:
        return
    prop = (xr - xl) / w
    if not (0.15 < prop < 0.9):
        return
    if state.lebar_terkalibrasi:
        state.lebar_prop = (1 - LEBAR_ALPHA) * state.lebar_prop + LEBAR_ALPHA * prop
    else:
        state.lebar_prop = prop
        state.lebar_terkalibrasi = True


def prediksi_riwayat(hist, n):
    """Estimasi posisi garis saat tidak ada deteksi:
    rata-rata riwayat (data baru lebih berat) + tren yang makin lama makin diredam.
    n = jumlah frame sejak deteksi terakhir."""
    arr = np.asarray(hist, dtype=np.float64)          # bentuk (K, N_TITIK)
    k = len(arr)
    bobot = np.arange(1, k + 1, dtype=np.float64)
    rata = np.average(arr, axis=0, weights=bobot)
    if k < 2:
        return rata
    tren = (arr[-1] - arr[0]) / (k - 1)
    langkah = sum(DECAY ** j for j in range(1, n + 1))
    return rata + tren * langkah


def finalize_lanes(state, w, h):
    """Tentukan koordinat FINAL kiri & kanan untuk frame ini.
      1. kedua garis terdeteksi              -> 'deteksi'
      2. hanya satu garis terdeteksi         -> 'turunan' (garis lain = garis itu + lebar)
      3. tidak ada deteksi                   -> 'estimasi' (rata-rata riwayat + tren)
      4. riwayat habis (> HOLD_MAX_FRAMES)   -> 'hilang'
    """
    state.frame_idx += 1
    lv = Y_LEVELS * h
    fresh_l, fresh_r = state.left.fresh, state.right.fresh

    # 1) riwayat deteksi valid & penghitung 'hold'
    if fresh_l:
        state.kiri_hist.append(xs_di_level(state.left, h))
        state.kiri_hold = 0
    else:
        state.kiri_hold += 1
    if fresh_r:
        state.kanan_hist.append(xs_di_level(state.right, h))
        state.kanan_hold = 0
    else:
        state.kanan_hold += 1

    # 2) kalibrasi lebar jalur bila kedua garis terdeteksi bersamaan
    if fresh_l and fresh_r:
        kalibrasi_lebar(state, w, h)
    lebar = lebar_jalur(lv, h, w, state.lebar_prop)

    # 3) koordinat dasar dari deteksi langsung
    xl = xr = None
    st_l = st_r = "hilang"
    if fresh_l:
        xl, st_l = xs_di_level(state.left, h), "deteksi"
    if fresh_r:
        xr, st_r = xs_di_level(state.right, h), "deteksi"

    # 4) tidak ada deteksi sama sekali -> rata-rata riwayat + tren
    if xl is None and xr is None:
        if state.kiri_hist and state.kiri_hold <= HOLD_MAX_FRAMES:
            xl = prediksi_riwayat(state.kiri_hist, state.kiri_hold)
            st_l = "estimasi"
        if state.kanan_hist and state.kanan_hold <= HOLD_MAX_FRAMES:
            xr = prediksi_riwayat(state.kanan_hist, state.kanan_hold)
            st_r = "estimasi"

    # 5) satu sisi kosong -> turunkan dari sisi lain + lebar jalur
    if xl is None and xr is not None:
        xl = xr - lebar
        st_l = "estimasi" if st_r == "estimasi" else "turunan"
    elif xr is None and xl is not None:
        xr = xl + lebar
        st_r = "estimasi" if st_l == "estimasi" else "turunan"

    # 6) status & keyakinan
    if st_l == "hilang" and st_r == "hilang":
        status, keyakinan = "hilang", 0.0
    elif st_l == "deteksi" and st_r == "deteksi":
        status, keyakinan = "normal", 1.0
    else:
        status = "perkiraan"
        if "estimasi" in (st_l, st_r):
            hold = max(state.kiri_hold, state.kanan_hold)
            keyakinan = 0.5 * max(0.0, 1.0 - hold / HOLD_MAX_FRAMES)
        else:
            keyakinan = 0.7

    # 7) tengah jalur & offset (dihitung di Y_REF)
    tengah = offset_m = None
    if xl is not None and xr is not None:
        tengah = (xl + xr) / 2.0
        xl_r = float(np.interp(Y_REF, Y_LEVELS, xl))
        xr_r = float(np.interp(Y_REF, Y_LEVELS, xr))
        if xr_r - xl_r > 1:
            offset_m = (w / 2.0 - (xl_r + xr_r) / 2.0) * LANE_WIDTH_M / (xr_r - xl_r)

    def bentuk(xs):
        """Ubah array x menjadi daftar titik: x, y (pixel) dan x_m (meter, + = kanan)."""
        if xs is None:
            return None
        return [
            {"x": int(round(x)), "y": int(round(y)),
             "x_m": round(float((x - w / 2.0) * LANE_WIDTH_M / lb), 3)}
            for x, y, lb in zip(xs, lv, lebar)
        ]

    return {
        "frame": state.frame_idx,
        "status": status,                          # normal / perkiraan / hilang
        "keyakinan": round(float(keyakinan), 2),   # 0.0 - 1.0
        "kalibrasi_lebar": state.lebar_terkalibrasi,
        "arah": state.arah,
        "offset_m": None if offset_m is None else round(float(offset_m), 3),
        "kiri": {"status": st_l, "titik": bentuk(xl)},
        "kanan": {"status": st_r, "titik": bentuk(xr)},
        "tengah": bentuk(tengah),
    }


def kurva_dari_hasil(info, h):
    """Ubah titik-titik hasil final menjadi kurva halus (40 titik) untuk digambar."""
    if info["titik"] is None:
        return None
    xs_lv = np.array([p["x"] for p in info["titik"]], dtype=np.float64)
    ys = np.linspace(Y_TOP * h, Y_BOTTOM * h, 40)
    xs = np.interp(ys, Y_LEVELS * h, xs_lv)
    return np.stack([xs, ys], axis=1).astype(np.int32)


class PenulisKoordinat:
    """Menulis koordinat tiap frame ke file JSON Lines (1 baris = 1 frame)."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def tulis(self, hasil):
        self.f.write(json.dumps(hasil, ensure_ascii=False) + "\n")
        self.f.flush()

    def tutup(self):
        self.f.close()


def gambar_hasil_akhir(img, hasil):
    """Tambahan visual: titik-titik untuk sisi yang diperkirakan (bukan deteksi
    langsung) dan teks status jalur di pojok kiri bawah."""
    warna = {"kiri": (0, 0, 255), "kanan": (0, 255, 255)}
    for sisi in ("kiri", "kanan"):
        info = hasil[sisi]
        if info["titik"] is None or info["status"] == "deteksi":
            continue
        for p in info["titik"]:
            cv2.circle(img, (p["x"], p["y"]), 4, warna[sisi], -1)

    teks, col = {
        "normal": ("Jalur: NORMAL", (0, 255, 0)),
        "perkiraan": ("Jalur: PERKIRAAN (sebagian dari riwayat)", (0, 165, 255)),
        "hilang": ("Jalur: HILANG - kendaraan harus melambat", (0, 0, 255)),
    }[hasil["status"]]
    teks += f"  (keyakinan {hasil['keyakinan']:.2f})"
    cv2.putText(img, teks, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, col, 2, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ gambar
def render_frame(frame, state, lines=None, roi_edges=None, poly=None, hasil=None):
    """Gambar garis jalur, area hijau, ROI, panel info di atas frame.
    Jika 'hasil' diberikan (dari finalize_lanes), garis & area hijau mengikuti
    koordinat FINAL (termasuk estimasi), bukan garis mentah."""
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)

    if hasil is not None:
        lp = kurva_dari_hasil(hasil["kiri"], h)
        rp = kurva_dari_hasil(hasil["kanan"], h)
    else:
        lp = curve_points(state.left, h, w) if state.left.theta is not None else None
        rp = curve_points(state.right, h, w) if state.right.theta is not None else None

    th = max(3, int(round(w / 90)))
    th_kiri = th if (hasil is None or hasil["kiri"]["status"] == "deteksi") else max(2, th // 2)
    th_kanan = th if (hasil is None or hasil["kanan"]["status"] == "deteksi") else max(2, th // 2)

    if lp is not None:
        cv2.polylines(overlay, [lp], False, (0, 0, 255), th_kiri)      # kiri = merah
    if rp is not None:
        cv2.polylines(overlay, [rp], False, (0, 255, 255), th_kanan)   # kanan = kuning

    offset_m, both = 0.0, lp is not None and rp is not None

    if both:
        cv2.fillPoly(overlay, [np.vstack([lp, rp[::-1]])], (0, 255, 0))
        if hasil is not None:
            if hasil["offset_m"] is not None:
                offset_m = hasil["offset_m"]
        else:
            xl = state.left.x_at(Y_REF * h, h)
            xr = state.right.x_at(Y_REF * h, h)
            if xr - xl > 1:
                offset_m = (w / 2.0 - (xl + xr) / 2.0) * LANE_WIDTH_M / (xr - xl)

    out = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

    if poly is None:
        poly, _ = get_roi(w, h)

    cv2.polylines(out, [poly], True, (255, 0, 0), 1)     # ROI = biru

    if state.debug and roi_edges is not None:
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 1)

        ih, iw = h // 3, w // 3
        inset = cv2.cvtColor(cv2.resize(roi_edges, (iw, ih)), cv2.COLOR_GRAY2BGR)
        out[h - ih:h, w - iw:w] = inset
        cv2.rectangle(out, (w - iw, h - ih), (w - 1, h - 1), (255, 255, 255), 1)

    draw_panel(out, state, offset_m, both, lp is not None or rp is not None)

    return out


def process_frame(frame, state):
    """Deteksi + gambar untuk satu frame (tanpa output koordinat)."""
    lines, roi_edges, poly = detect_lanes(frame, state)
    return render_frame(frame, state, lines, roi_edges, poly)


# =====================================================================
#            HASIL PER TAHAP (TAMPIL + SIMPAN)
# =====================================================================
def _bgr(img):
    """Pastikan gambar 3 kanal (BGR)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _label(img, text):
    """Tulis judul tahap di pojok kiri atas gambar (pada bar gelap)."""
    out = _bgr(img)
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_stages(frame, state):
    """Jalankan pipeline SATU frame sambil mengumpulkan gambar tiap tahap.
    Mengembalikan list [(judul, gambar_BGR), ...]. state ikut ter-update."""
    h, w = frame.shape[:2]
    stages = [("01 Original (resize)", frame.copy())]

    gray, blur, edges = preprocess(frame)
    stages.append(("02 Grayscale", gray))
    stages.append(("03 Gaussian Blur", blur))
    stages.append(("04 Canny Edge", edges))

    poly, roi_mask = get_roi(w, h)
    roi_vis = frame.copy()
    tint = np.zeros_like(frame)
    cv2.fillPoly(tint, [poly], (0, 255, 0))
    roi_vis = cv2.addWeighted(roi_vis, 1.0, tint, 0.35, 0)
    cv2.polylines(roi_vis, [poly], True, (255, 0, 0), 2)
    stages.append(("05 ROI (area jalan)", roi_vis))

    roi_edges = apply_roi(edges, roi_mask)
    stages.append(("06 Edge di dalam ROI", roi_edges))

    bright = None
    if BRIGHT_DELTA > 0:
        med = float(np.median(blur[roi_mask > 0]))
        bright = cv2.dilate(
            (blur > med + BRIGHT_DELTA).astype(np.uint8),
            np.ones((BRIGHT_DILATE, BRIGHT_DILATE), np.uint8)
        ) > 0
        stages.append(("07 Mask Brightness", bright.astype(np.uint8) * 255))
        roi_edges = roi_edges * bright.astype(np.uint8)
        stages.append(("08 Edge setelah Brightness Filter", roi_edges))

    lines = cv2.HoughLinesP(
        roi_edges, 1, np.pi / 180, HOUGH_THRESH,
        minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP
    )
    hough_vis = frame.copy()
    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(hough_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    stages.append((f"09 HoughLinesP ({n_lines} segmen)", hough_vis))

    Lp, Rp = collect_segments(lines, w, h, state.left, state.right, bright)
    seg_vis = frame.copy()
    for x, y in zip(*Lp):
        cv2.circle(seg_vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    for x, y in zip(*Rp):
        cv2.circle(seg_vis, (int(x), int(y)), 2, (0, 255, 255), -1)
    stages.append(("10 Klasifikasi Kiri(merah)/Kanan(kuning)", seg_vis))

    state.left.update(fit_lane(Lp[0], Lp[1], w, h, "L", state.left.theta), w, h)
    state.right.update(fit_lane(Rp[0], Rp[1], w, h, "R", state.right.theta), w, h)
    update_direction(state, w)

    fit_vis = frame.copy()
    th = max(2, int(round(w / 160)))
    if state.left.theta is not None:
        cv2.polylines(fit_vis, [curve_points(state.left, h, w)], False, (0, 0, 255), th)
    if state.right.theta is not None:
        cv2.polylines(fit_vis, [curve_points(state.right, h, w)], False, (0, 255, 255), th)
    stages.append(("11 Fitting kurva + EMA", fit_vis))

    final = render_frame(frame, state, lines, roi_edges, poly)
    stages.append((f"12 Hasil Akhir: {state.arah}", final))

    return stages


def make_montage(stages, cols=3, tile_w=360):
    """Gabungkan semua tahap jadi satu gambar grid (cols kolom)."""
    tiles = []
    for title, img in stages:
        img = _bgr(img)
        r = tile_w / img.shape[1]
        img = cv2.resize(img, (tile_w, int(img.shape[0] * r)))
        tiles.append(_label(img, title))
    th = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
               for t in row]
        while len(row) < cols:
            row.append(np.zeros((th, tile_w, 3), np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def save_stages(stages, out_dir="hasil_tahap", prefix=""):
    """Simpan tiap tahap sebagai file .jpg + satu gambar rangkuman."""
    os.makedirs(out_dir, exist_ok=True)
    for title, img in stages:
        nama = title.split("(")[0].split(":")[0].strip()
        nama = nama.lower().replace(" ", "_").replace("/", "_")
        path = os.path.join(out_dir, f"{prefix}{nama}.jpg")
        cv2.imwrite(path, _bgr(img))
    mont = os.path.join(out_dir, f"{prefix}00_semua_tahap.jpg")
    cv2.imwrite(mont, make_montage(stages))
    print(f"[OK] {len(stages)} tahap tersimpan di folder: {out_dir}/")
    print(f"[OK] Rangkuman semua tahap: {mont}")


def _in_colab():
    """True jika program berjalan di Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def show_stages(stages, per_stage=False):
    """Tampilkan hasil per tahap (Colab / matplotlib / jendela OpenCV)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if per_stage:
        if _in_colab():
            from google.colab.patches import cv2_imshow
            for title, img in stages:
                print(title)
                cv2_imshow(_bgr(img))
            return
        if plt is not None:
            for title, img in stages:
                plt.figure(figsize=(8, 5))
                plt.imshow(cv2.cvtColor(_bgr(img), cv2.COLOR_BGR2RGB))
                plt.title(title)
                plt.axis("off")
            plt.show()
            return
        for title, img in stages:
            cv2.imshow(title, _bgr(img))
        print("Tekan tombol apa saja pada jendela untuk menutup.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if plt is not None:
        cols = 3
        rows = (len(stages) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (title, img) in zip(axes, stages):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=9)
        plt.tight_layout()
        plt.show()
    else:
        mont = make_montage(stages)
        cv2.imshow("Semua tahap", mont)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def proses_per_tahap(frame, out_dir="hasil_tahap", warmup=12,
                     tampil=True, per_stage=False, simpan=True, sisi=None):
    """FUNGSI UTAMA untuk Colab / notebook."""
    if sisi:
        set_sisi(sisi)
    h0, w0 = frame.shape[:2]
    if w0 > PROCESS_WIDTH:
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * PROCESS_WIDTH / w0)))
    state = LaneState()
    for _ in range(warmup):
        detect_lanes(frame, state)
    stages = build_stages(frame, state)
    if simpan:
        save_stages(stages, out_dir)
    if tampil:
        show_stages(stages, per_stage=per_stage)
    return stages


# ------------------------------------------------------------------ input foto
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def show_still_result(frame, save_path=None,
                      window="Deteksi Jalur Jalan Lurus dan Belok"):
    """Proses satu foto (dengan warmup) lalu tampilkan di jendela OpenCV."""
    state = LaneState()

    for _ in range(12):
        out = process_frame(frame, state)

    if save_path:
        cv2.imwrite(save_path, out)
        print("Tersimpan ke:", save_path)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, out)

    print("Jendela hasil terbuka. Tekan 'd' untuk debug, "
          "tombol apa saja lain (atau tutup jendelanya) untuk lanjut.")

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord("d"):
            state.debug = not state.debug
            out = process_frame(frame, state)
            cv2.imshow(window, out)
            continue

        if key != 255:
            break

        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break


def run_on_image(path, args):
    """Jalankan deteksi pada sebuah foto."""
    frame = cv2.imread(path)

    if frame is None:
        print("Tidak bisa membuka foto:", path)
        return

    h0, w0 = frame.shape[:2]
    if w0 > args.width:
        frame = cv2.resize(frame, (args.width, int(h0 * args.width / w0)))

    if args.stages:
        proses_per_tahap(frame, out_dir=args.stages_dir)
        if args.save:
            state = LaneState()
            for _ in range(12):
                out = process_frame(frame, state)
            cv2.imwrite(args.save, out)
            print("Tersimpan ke:", args.save)
        return

    show_still_result(frame, args.save or None)
    cv2.destroyAllWindows()


def set_sisi(nilai):
    """Ubah mode sisi garis: 'dua', 'kiri', atau 'kanan'."""
    global SISI
    SISI = nilai


# ------------------------------------------------------------------ pembaca kamera
class FrameGrabber:
    """Membaca webcam di thread terpisah dan hanya menyimpan frame TERBARU
    (mencegah delay akibat frame menumpuk di buffer)."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.fid = 0
        self.ok = True
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self.lock:
                self.frame = f
                self.fid += 1

    def read(self):
        """Mengembalikan (ok, frame_terbaru, nomor_frame)."""
        with self.lock:
            return self.ok, self.frame, self.fid

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source", default="0",
        help="0 = webcam, path video (mis. video_jalan.mp4), atau path FOTO (mis. jalan.jpg)"
    )
    ap.add_argument(
        "--width", type=int, default=PROCESS_WIDTH,
        help="lebar frame proses (default 640)"
    )
    ap.add_argument(
        "--save", default="",
        help="simpan hasil: video (.mp4) untuk video/webcam, gambar (.jpg/.png) untuk foto"
    )
    ap.add_argument(
        "--skip", type=int, default=1,
        help="proses 1 dari tiap N frame (pakai jika laptop berat). Default 1"
    )
    ap.add_argument(
        "--stages", action="store_true",
        help="(foto) tampilkan & simpan hasil tiap tahap proses"
    )
    ap.add_argument(
        "--stages-frame", type=int, default=0,
        help="(video) ambil hasil per tahap pada frame ke-N (0 = tidak otomatis)"
    )
    ap.add_argument(
        "--stages-dir", default="",
        help="folder hasil per tahap (default: hasil_tahap/<nama_sumber>)"
    )
    ap.add_argument(
        "--lebar-max", type=float, default=None,
        help="lebar jalur maksimum (0-1, proporsi lebar gambar). Default 0 = nonaktif"
    )
    ap.add_argument(
        "--horizon", type=float, default=None,
        help="tinggi cakrawala/titik hilang jalan (0-1 dari atas gambar, default 0.48)"
    )
    ap.add_argument(
        "--sisi", choices=["dua", "kiri", "kanan"], default="dua",
        help="garis yang dideteksi: dua (default), kiri, atau kanan"
    )
    ap.add_argument(
        "--no-sync", action="store_true",
        help="(file video) proses SEMUA frame tanpa menyesuaikan kecepatan asli video"
    )
    ap.add_argument(
        "--koordinat", default="",
        help="simpan koordinat jalur tiap frame ke file .jsonl (mis. koordinat.jsonl). "
             "Khusus webcam/video"
    )

    args, _ = ap.parse_known_args()

    set_sisi(args.sisi)
    if args.lebar_max is not None:
        globals()["MAX_LANE_GAP"] = args.lebar_max
    if args.horizon is not None:
        globals()["Y_HORIZON"] = args.horizon

    if not args.stages_dir:
        nama = os.path.splitext(os.path.basename(args.source))[0]
        if args.source.isdigit():
            nama = f"webcam_{args.source}"
        args.stages_dir = os.path.join("hasil_tahap", nama)

    if args.source.lower().endswith(IMAGE_EXT):
        run_on_image(args.source, args)
        return

    is_cam = args.source.isdigit()
    src = int(args.source) if is_cam else args.source

    cap = (
        cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if is_cam
        else cv2.VideoCapture(src)
    )
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print("Tidak bisa membuka sumber:", src)
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if is_cam:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
        cap.set(cv2.CAP_PROP_FPS, 30)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps_src <= 1 or fps_src > 240:
        fps_src = 30

    sync = (not is_cam) and (not args.save) and (not args.no_sync)

    grabber = FrameGrabber(cap) if is_cam else None

    # file koordinat dibuka setelah sumber berhasil dibuka
    penulis = PenulisKoordinat(args.koordinat) if args.koordinat else None

    WIN = "Deteksi Jalur Jalan Lurus dan Belok"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    state, i = LaneState(), 0
    lines = roi_edges = poly = None
    hasil = None
    writer = None

    last_id = -1
    pos = 0                      # jumlah frame sumber yang sudah dibaca
    src_idx = 0                  # nomor frame sumber untuk output koordinat
    t_start = time.perf_counter()
    t_prev, fps_show = t_start, 0.0

    while True:

        if is_cam:
            ok, frame, fid = grabber.read()
            if not ok:
                break
            if frame is None or fid == last_id:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.002)
                continue
            last_id = fid
            src_idx = fid
        else:
            if sync:
                due = int((time.perf_counter() - t_start) * fps_src)
                while pos < due:
                    if not cap.grab():
                        break
                    pos += 1
            ok, frame = cap.read()
            if not ok:
                break
            pos += 1
            src_idx = pos

        h0, w0 = frame.shape[:2]
        if w0 > args.width:
            frame = cv2.resize(frame, (args.width, int(h0 * args.width / w0)))

        ambil_tahap = (args.stages_frame > 0 and i == args.stages_frame)
        if ambil_tahap:
            stages = build_stages(frame.copy(), copy.deepcopy(state))
            save_stages(stages, args.stages_dir, prefix=f"frame{i:05d}_")
            show_stages(stages)
            t_start = time.perf_counter() - pos / fps_src

        h, w = frame.shape[:2]
        if args.skip <= 1 or i % args.skip == 0 or poly is None:
            lines, roi_edges, poly = detect_lanes(frame, state)
            hasil = finalize_lanes(state, w, h)
            hasil["sumber_frame"] = src_idx
            waktu = (time.perf_counter() - t_start) if is_cam else (src_idx / fps_src)
            hasil["waktu_detik"] = round(waktu, 3)
            if penulis:
                penulis.tulis(hasil)

        out = render_frame(frame, state, lines, roi_edges, poly, hasil)
        if hasil is not None:
            out = gambar_hasil_akhir(out, hasil)

        i += 1

        if args.save:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                writer = cv2.VideoWriter(
                    args.save,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (out.shape[1], out.shape[0])
                )
            writer.write(out)      # disimpan SEBELUM teks FPS ditambahkan

        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps_show = (1.0 / dt) if fps_show == 0 else (0.9 * fps_show + 0.1 / dt)
        cv2.putText(
            out, f"FPS {fps_show:.0f}", (out.shape[1] - 90, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

        if sync:
            tunggu = t_start + pos / fps_src - time.perf_counter()
            if tunggu > 0:
                time.sleep(tunggu)

        cv2.imshow(WIN, out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("d"):
            state.debug = not state.debug

        if key == ord("p"):
            print("Mengambil hasil per tahap pada frame", i)
            stages = build_stages(frame.copy(), copy.deepcopy(state))
            save_stages(stages, args.stages_dir, prefix=f"frame{i:05d}_")
            show_stages(stages)
            t_start = time.perf_counter() - pos / fps_src

        if key == ord("s"):
            snap_name = f"snapshot_{i:05d}.jpg"
            print("Snapshot diambil, memproses...")
            show_still_result(frame.copy(), snap_name, window="Snapshot - Deteksi Jalur")
            cv2.destroyWindow("Snapshot - Deteksi Jalur")
            t_start = time.perf_counter() - pos / fps_src

    if grabber:
        grabber.stop()

    cap.release()

    if writer:
        writer.release()

    if penulis:
        penulis.tutup()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()