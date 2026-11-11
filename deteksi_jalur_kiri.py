#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deteksi Jalur pada Video Jalan (dashcam)
========================================
Membagi jalan di depan kendaraan menjadi jalur-jalur, lalu menggambarnya dengan warna
berbeda dan menampilkan titik koordinat x,y di setiap garis.

ROI         : 40% bagian bawah frame, selebar frame. Hanya membatasi area pencarian garis.
              Garis ROI disembunyikan secara default (tampilkan dengan --tampil-roi atau tombol r).

Jumlah jalur (otomatis, atau paksa dengan --jalur 2 / --jalur 3):
  2 jalur : | jalur kiri | jalur kanan |
  3 jalur : | bahu | jalur kiri | jalur kanan |
  Jalur kanan tambahan (3 jalur) dikenali bila di kanan garis tepi kanan masih ada aspal.

Garis yang dilacak:
  tepi kiri    (biru)    : tepi kiri jalur kiri (2 jalur) / tepi luar bahu (3 jalur)
  garis tengah (kuning)  : pemisah jalur kiri dan jalur kanan (2 jalur) /
                           pemisah bahu dan jalur kiri (3 jalur)
  tepi kanan   (magenta) : tepi kanan jalur kanan (2 jalur) /
                           pemisah jalur kiri dan jalur kanan (3 jalur)

Warna area:
  2 jalur : hijau = jalur kiri, merah = jalur kanan
  3 jalur : kuning = bahu, hijau = jalur kiri, merah = jalur kanan
  Area di kanan garis tepi kanan hanya diwarnai pada permukaan aspal, bukan rumput.

Alur pemrosesan per frame:
  Frame -> ROI -> Grayscale -> Gaussian Blur -> Canny (hanya pada area marka)
        -> filter marka putih/kuning (kontras dua sisi) -> HoughLinesP
        -> segmen dikelompokkan menjadi garis marka
        -> peran garis: tepi kiri, garis tengah, tepi kanan (sesuai posisi terhadap tengah)
        -> fit kurva (parabola) -> uji kewajaran -> EMA -> Kalman per titik
        -> fallback: lebar jalur dari sisi lain, prediksi Kalman, lalu dinyatakan hilang
        -> rasio aspal di kanan garis tepi kanan -> jumlah jalur (voting + histeresis)
        -> gambar area, garis, label, titik x,y, dan legenda

Cara pakai:
  python deteksi_jalur_kiri.py
  python deteksi_jalur_kiri.py --source video_jalan.mp4 --save hasil.mp4 --tanpa-tampil
  python deteksi_jalur_kiri.py --source video_jalan.mp4 --mulai 40 --durasi 12 --save pesisir.mp4
  python deteksi_jalur_kiri.py --source video_jalan.mp4 --jalur 2 --koordinat koordinat.jsonl
  python deteksi_jalur_kiri.py --source video_jalan.mp4 --tampil-roi
  python deteksi_jalur_kiri.py --source 0                     (webcam)
  python deteksi_jalur_kiri.py --source foto.jpg --save hasil.jpg

Tombol saat jendela tampil:
  q = keluar           | d = mode debug (titik kandidat marka)
  c = titik x,y on/off | r = garis ROI on/off
  a = jumlah jalur otomatis | 2 = paksa 2 jalur | 3 = paksa 3 jalur

Kebutuhan: Python 3, opencv-python, numpy.
"""

import argparse
import json
import time
from collections import deque

import cv2
import numpy as np

# ============================================================
#   PARAMETER
# ============================================================
LEBAR_PROSES = 640                   # frame diperkecil ke lebar ini sebelum diproses

# --- ROI: 40% bagian bawah frame, selebar frame
ROI_TINGGI = 0.40
ROI_ATAS = 1.0 - ROI_TINGGI          # 0.60 = batas atas ROI (proporsi tinggi frame)

# --- Sumbu acuan dan titik garis
Y_BAWAH = 0.88                       # garis digambar sampai di sini (kap mobil menutupi di bawahnya)
Y_REF = 0.80                         # garis acuan: pemisah kiri/kanan dan lebar jalur dihitung di sini
Y_HORIZON = 0.48                     # perkiraan cakrawala, untuk perspektif lebar jalur
N_TITIK = 9                          # jumlah titik per garis (dari ROI_ATAS ke Y_BAWAH)
Y_LEVELS = np.linspace(ROI_ATAS, Y_BAWAH, N_TITIK)
SKALA_T = 0.30                       # t = (y - Y_REF) / (SKALA_T * tinggi), dipakai saat fitting

# --- Pra-pemrosesan
BLUR_KSIZE = 5
CANNY_LOW, CANNY_HIGH = 50, 150

# --- Filter marka jalan
#     Marka = garis yang LEBIH TERANG dari KEDUA sisinya (pada jarak MARKA_JARAK_SISI).
#     Tepi antara area terang dan gelap (tanah, kerikil, rumput) hanya terang di satu sisi.
MARKA_JARAK_SISI = 12                # jarak sisi pembanding (pixel pada lebar 640)
MARKA_KONTRAS = 22                   # selisih kecerahan minimum terhadap kedua sisi
MARKA_PUTIH_MIN_V = 110              # kecerahan minimum marka putih
MARKA_KUNING_H = (15, 35)            # rona kuning (skala OpenCV 0-180)
MARKA_KUNING_S = 90
MARKA_KUNING_V = 120
MARKA_DILATE = 7

# --- HoughLinesP
HOUGH_THRESH = 20
HOUGH_MIN_LEN = 15
HOUGH_MAX_GAP = 30
SLOPE_MIN = 0.35                     # |dy/dx| minimum: garis hampir mendatar dibuang
SLOPE_MAX = 8.0                      # |dy/dx| maksimum: garis hampir tegak (tiang) dibuang

# --- Pengelompokan segmen menjadi garis marka
KELOMPOK_TOL = 0.035                 # segmen berdekatan (proporsi lebar) dianggap satu garis
KELOMPOK_MIN_SPAN = 0.04             # rentang vertikal minimum garis (garis putus-putus ikut terbaca)
KELOMPOK_SPAN_LUAR = 0.10            # rentang minimum garis luar (tepi kanan): harus garis menerus
KELOMPOK_TOL_KEMIRING = 0.40         # selisih kemiringan (dx/dy) segmen dengan garis yang dimasuki
GATE_TOL = 0.06                      # kandidat dicocokkan ke garis lama bila selisihnya di bawah ini
JARAK_MIN_GARIS_LUAR = 0.06          # tepi kanan harus berjarak minimal ini dari garis tengah (proporsi lebar)
JARAK_PINDAH = 0.10                  # garis yang dilacak pindah ke garis kuat yang lebih dekat ke tengah

# --- Fit kurva & penghalusan
FIT_MIN_TITIK = 8
FIT_MIN_SPAN = 0.10                  # rentang vertikal minimum (proporsi tinggi) untuk memakai parabola
CURVE_MAX = 0.15                     # kelengkungan maksimum (proporsi lebar)
EMA_ALPHA = 0.15                     # EMA parameter kurva
WAJAR_TOL = 0.04                     # kurva baru harus dekat kurva lama (proporsi lebar)
RELOCK_FRAMES = 8                    # garis baru yang konsisten selama sekian frame menggantikan yang lama
MAX_HILANG = 40                      # frame tanpa deteksi sebelum garis dinyatakan hilang

# --- Kalman filter (per titik level y)
KF_Q_POS = 0.0008
KF_Q_VEL = 0.0001
KF_VEL_DECAY = 0.90
KF_R_DETEKSI = 0.006                 # derau ukur saat garis terdeteksi langsung (proporsi lebar)
KF_R_TURUNAN = 0.012                 # derau ukur saat garis diturunkan dari sisi lain
KF_GATE = 3.0

# --- Lebar jalur (dipelajari dari frame yang kedua garisnya terlihat, pada mode 2 jalur)
LEBAR_DEFAULT = 0.40                 # tebakan awal (proporsi lebar gambar pada garis acuan)
LEBAR_ALPHA = 0.05

# --- Jumlah jalur
VOTE_FRAMES = 120                    # frame (~2 detik di 60 fps) untuk voting 2 / 3 jalur
VOTE_NAIK = 0.70                     # pindah ke 3 jalur bila >= 70% frame terakhir mendukung
VOTE_TURUN = 0.30                    # kembali ke 2 jalur bila <= 30% (histeresis)
ASPAL_KANAN_MIN = 0.50               # bukti 3 jalur bila rasio aspal di kanan tepi kanan >= nilai ini
ASPAL_V_MAKS = 165                   # aspal lebih gelap dari kerikil terang

# --- Warna (format BGR)
#     2 jalur: hijau = jalur kiri, merah = jalur kanan
WARNA_JALUR_KIRI = (0, 200, 0)
WARNA_JALUR_KANAN = (0, 0, 255)
#     3 jalur: kuning = bahu, hijau = jalur kiri, merah = jalur kanan
WARNA_BAHU = (0, 255, 255)
#     Garis (sama untuk kedua mode)
WARNA_TEPI_KIRI = (255, 0, 0)        # biru
WARNA_TENGAH = (0, 255, 255)         # kuning
WARNA_TEPI_KANAN = (255, 0, 255)     # magenta
WARNA_ROI = (255, 255, 0)            # cyan: garis ROI (bila ditampilkan)
WARNA_KANDIDAT = (160, 160, 160)     # abu-abu: titik kandidat (mode debug)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
JUDUL = "Deteksi Jalur"


# ============================================================
#   MODEL GARIS: kurva x = A t^2 + B t + C
# ============================================================
def t_dari_y(y, h):
    """Ubah koordinat y (pixel) menjadi t yang ternormalisasi terhadap garis acuan."""
    return (np.asarray(y, dtype=np.float64) - Y_REF * h) / (SKALA_T * h)


def eval_kurva(theta, dom, t):
    """Nilai x kurva di t. Di luar rentang data (dom), garis diperpanjang lurus
    mengikuti garis singgung di ujung rentang."""
    A, B, C = theta
    lo, hi = dom
    tc = np.clip(t, lo, hi)
    x = A * tc * tc + B * tc + C
    x = x + (2 * A * lo + B) * np.minimum(t - lo, 0.0)
    x = x + (2 * A * hi + B) * np.maximum(t - hi, 0.0)
    return x


def _lstsq(t, x, bobot, kuadrat):
    """Kuadrat terkecil berbobot. Bila tidak kuadrat, A dipaksa 0 (garis lurus)."""
    sw = np.sqrt(bobot)
    if kuadrat:
        phi = np.stack([t * t, t, np.ones_like(t)], axis=1)
    else:
        phi = np.stack([np.zeros_like(t), t, np.ones_like(t)], axis=1)
    return np.linalg.lstsq(phi * sw[:, None], x * sw, rcond=None)[0]


def fit_kurva(ys, xs, bobot, w, h):
    """Fit x = A t^2 + B t + C ke titik-titik marka. Pencilan dibuang dengan MAD.
    Mengembalikan (theta, dom) atau None bila titik tidak cukup."""
    if len(xs) < FIT_MIN_TITIK:
        return None
    t = t_dari_y(ys, h)
    kuadrat = (ys.max() - ys.min()) / h >= FIT_MIN_SPAN
    keep = np.ones(len(xs), dtype=bool)
    th = None
    for _ in range(3):
        tt, xx, bb = t[keep], xs[keep], bobot[keep]
        if len(tt) < FIT_MIN_TITIK:
            return None
        th = _lstsq(tt, xx, bb, kuadrat)
        tm = max(abs(tt.min()), abs(tt.max()))
        if kuadrat and abs(th[0]) * tm * tm > CURVE_MAX * w:   # kelengkungan tidak masuk akal
            kuadrat = False
            th = _lstsq(tt, xx, bb, False)
        res = xs - (th[0] * t * t + th[1] * t + th[2])
        med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med))
        baru = np.abs(res - med) < max(0.02 * w, 2.5 * mad)
        if baru.sum() < FIT_MIN_TITIK or np.array_equal(baru, keep):
            break
        keep = baru
    dom = np.array([t[keep].min(), t[keep].max()])
    return th, dom


# ============================================================
#   KALMAN FILTER UNTUK TITIK-TITIK GARIS
# ============================================================
class KalmanTitik:
    """Kalman filter model kecepatan konstan untuk semua titik (level y) sekaligus.
    Tiap titik punya state [posisi, kecepatan]. Bila tidak ada ukuran (z = None),
    filter hanya memprediksi posisi berikutnya (dipakai saat garis sementara tertutup)."""

    def __init__(self):
        self.aktif = False

    def reset(self):
        self.aktif = False

    def langkah(self, z, r_std, w):
        if not self.aktif:
            if z is None:
                return None
            n = len(z)
            self.x = np.asarray(z, dtype=np.float64).copy()
            self.v = np.zeros(n)
            self.p00 = np.asarray(r_std, dtype=np.float64) ** 2
            self.p01 = np.zeros(n)
            self.p11 = np.full(n, (5.0 * KF_Q_VEL * w) ** 2)
            self.aktif = True
            return self.x.copy()

        # prediksi
        qx = (KF_Q_POS * w) ** 2
        qv = (KF_Q_VEL * w) ** 2
        d = KF_VEL_DECAY
        x = self.x + self.v
        v = d * self.v
        p00 = self.p00 + 2 * self.p01 + self.p11 + qx
        p01 = d * (self.p01 + self.p11)
        p11 = d * d * self.p11 + qv

        # koreksi dengan ukuran (penjaga loncatan: ukuran aneh -> derau diperbesar)
        if z is not None:
            r = np.asarray(r_std, dtype=np.float64) ** 2
            y = np.asarray(z, dtype=np.float64) - x
            s0 = p00 + r
            infl = np.maximum(1.0, y * y / (KF_GATE ** 2 * s0))
            s = p00 + r * infl
            k0 = p00 / s
            k1 = p01 / s
            x = x + k0 * y
            v = v + k1 * y
            p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01

        self.x, self.v = x, v
        self.p00, self.p01, self.p11 = p00, p01, p11
        return self.x.copy()


def langkah_kf(garis, z, st, w, n):
    """Jalankan Kalman untuk satu garis. Mengembalikan dict {x, status}.
    z = None -> prediksi saja (status 'estimasi'); bila Kalman belum aktif -> 'hilang'."""
    if z is None:
        x = garis.kf.langkah(None, None, w)
        return {"x": x, "status": "estimasi"} if x is not None else {"x": None, "status": "hilang"}
    r = np.full(n, (KF_R_TURUNAN if st == "turunan" else KF_R_DETEKSI) * w)
    return {"x": garis.kf.langkah(z, r, w), "status": st}


# ============================================================
#   GARIS YANG DILACAK ANTAR-FRAME
# ============================================================
class Garis:
    """Satu garis (tepi kiri, garis tengah, atau tepi kanan) yang dilacak antar-frame.
    Berisi kurva aktif, penghitung frame tanpa deteksi, dan Kalman filter untuk titik-titiknya."""

    def __init__(self):
        self.kf = KalmanTitik()
        self.reset()

    def reset(self):
        self.theta = None           # parameter kurva [A, B, C]
        self.dom = None             # rentang t yang punya data
        self.hilang = 0             # frame berturut-turut tanpa deteksi
        self.tolak = 0              # penolakan berturut-turut oleh uji kewajaran
        self.segar = False          # True bila frame ini diterima dari deteksi
        self.kf.reset()

    def x_pada(self, y, h):
        """Posisi x garis pada ketinggian y (boleh berupa daftar / array)."""
        return eval_kurva(self.theta, self.dom, t_dari_y(y, h))

    def x_ref(self, h):
        return float(self.x_pada([Y_REF * h], h)[0])

    def _wajar(self, theta, dom, w, h):
        """Kurva baru harus cocok dengan kurva lama di seluruh area ROI."""
        t = t_dari_y(Y_LEVELS * h, h)
        beda = eval_kurva(theta, dom, t) - eval_kurva(self.theta, self.dom, t)
        return float(np.mean(np.abs(beda))) < WAJAR_TOL * w

    def update(self, fit, w, h):
        """Gabungkan hasil fit frame ini dengan garis lama."""
        self.segar = False
        if fit is None:
            self._hilang()
            return
        theta, dom = fit
        if self.theta is None:
            self._ganti(theta, dom)
            return
        if not self._wajar(theta, dom, w, h):
            # dianggap pencilan sesaat; diterima hanya bila konsisten selama RELOCK_FRAMES
            self.tolak += 1
            if self.tolak >= RELOCK_FRAMES:
                self._ganti(theta, dom)
            else:
                self._hilang()
            return
        self.theta = EMA_ALPHA * theta + (1 - EMA_ALPHA) * self.theta
        self.dom = EMA_ALPHA * dom + (1 - EMA_ALPHA) * self.dom
        self.hilang = 0
        self.tolak = 0
        self.segar = True

    def _ganti(self, theta, dom):
        self.theta = np.asarray(theta, dtype=np.float64)
        self.dom = np.asarray(dom, dtype=np.float64)
        self.hilang = 0
        self.tolak = 0
        self.segar = True
        self.kf.reset()

    def _hilang(self):
        self.hilang += 1
        if self.hilang > MAX_HILANG:
            self.reset()


# ============================================================
#   DETEKSI MARKA & KANDIDAT GARIS
# ============================================================
KERNEL_DILATE = np.ones((MARKA_DILATE, MARKA_DILATE), np.uint8)


def mask_marka(roi, w):
    """Mask piksel marka jalan (putih atau kuning) pada area ROI. Hasil 0 / 255.
    Kontras dua sisi: kecerahan pusat dikurangi kecerahan sisi yang LEBIH terang
    (kiri atau kanan), sehingga hanya garis sempit yang lolos."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    pusat = cv2.blur(vv.astype(np.float32), (5, 5))
    d = max(4, int(round(MARKA_JARAK_SISI * w / 640.0)))
    sisi_terang = np.maximum(np.roll(pusat, d, axis=1), np.roll(pusat, -d, axis=1))
    kontras = pusat - sisi_terang

    warna_putih = (ss <= 60) | ((hh <= 30) & (ss <= 130))     # putih / krem, bukan hijau rumput
    putih = (kontras >= MARKA_KONTRAS) & (vv >= MARKA_PUTIH_MIN_V) & warna_putih
    kuning = ((hh >= MARKA_KUNING_H[0]) & (hh <= MARKA_KUNING_H[1]) & (ss >= MARKA_KUNING_S)
              & (vv >= MARKA_KUNING_V) & (kontras >= 0.6 * MARKA_KONTRAS))
    m = np.where(putih | kuning, 255, 0).astype(np.uint8)
    return cv2.dilate(m, KERNEL_DILATE)


def deteksi_kandidat(roi, y0, w, h):
    """Cari garis marka di dalam ROI. Mengembalikan daftar kandidat, masing-masing berisi
    posisi di garis acuan (xref), rentang vertikal (span), dan titik-titik untuk fitting.
    Koordinat sudah dalam frame penuh."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    edges = cv2.bitwise_and(edges, mask_marka(roi, w))       # Canny hanya pada area marka

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    if lines is None:
        return []

    yref = Y_REF * h
    segmen = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(np.float64):
        y1 += y0
        y2 += y0
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 or dy == 0:
            continue
        if not (SLOPE_MIN <= abs(dy / dx) <= SLOPE_MAX):
            continue
        xref = x1 + (yref - y1) * dx / dy
        if not (0 <= xref < w):
            continue
        kmiring = dx / dy                     # garis kiri condong ke tengah saat naik: dx/dy < 0
        if (xref < w / 2 and kmiring >= 0) or (xref >= w / 2 and kmiring <= 0):
            continue                          # arah salah: bukan garis jalur
        panjang = float(np.hypot(dx, dy))
        n = max(2, int(panjang / 4))
        segmen.append((xref, panjang, kmiring,
                       np.linspace(x1, x2, n), np.linspace(y1, y2, n)))

    # kelompokkan segmen secara berurutan: segmen masuk ke garis terdekat bila posisinya
    # (di titik tengah segmen) dan kemiringannya cocok. Cara ini mengikuti garis melengkung.
    segmen.sort(key=lambda s: -s[1])                       # segmen terpanjang dulu sebagai awal
    grup = []                                              # tiap grup: segmen + titik (x, y, dx/dy)
    for s in segmen:
        xref_s, panjang_s, k_s, sx, sy = s
        xm = 0.5 * (sx[0] + sx[-1])
        ym = 0.5 * (sy[0] + sy[-1])
        terbaik, jarak_terbaik = None, None
        for g in grup:
            # garis lokal: dari anggota grup yang paling dekat secara vertikal
            tx, ty, tk = min(g["titik"], key=lambda t: abs(t[1] - ym))
            jarak = abs(xm - (tx + tk * (ym - ty)))
            if jarak < KELOMPOK_TOL * w and abs(k_s - tk) < KELOMPOK_TOL_KEMIRING:
                if jarak_terbaik is None or jarak < jarak_terbaik:
                    terbaik, jarak_terbaik = g, jarak
        if terbaik is None:
            grup.append({"anggota": [s], "titik": [(xm, ym, k_s)]})
        else:
            terbaik["anggota"].append(s)
            terbaik["titik"].append((xm, ym, k_s))

    kandidat = []
    for g in grup:
        anggota = g["anggota"]
        ys = np.concatenate([a[4] for a in anggota])
        span = (ys.max() - ys.min()) / h
        if span < KELOMPOK_MIN_SPAN:          # potongan terlalu pendek: derau
            continue
        total = sum(a[1] for a in anggota)
        xs = np.concatenate([a[3] for a in anggota])
        bobot = np.concatenate([np.full(len(a[3]), a[1] / len(a[3])) for a in anggota])
        xref = float(np.average([a[0] for a in anggota], weights=[a[1] for a in anggota]))
        kandidat.append({"xref": xref, "panjang": total, "span": span,
                         "xs": xs, "ys": ys, "bobot": bobot})
    return kandidat


def pilih_kandidat(daftar, garis, w, h, dari_tengah):
    """Pilih kandidat untuk satu garis. daftar sudah terurut dari yang paling dekat ke tengah.
    - Bila garis belum ada: ambil kandidat paling dekat ke tengah.
    - Bila garis sudah ada dan ada garis KUAT (menerus) yang jauh lebih dekat ke tengah
      daripada garis yang dilacak, pindah ke garis itu. Ini mencegah penguncian ke tepi jalan.
    - Selain itu: kandidat terdekat ke prediksi garis, asal masih dalam GATE_TOL.
    dari_tengah = True bila garis berada di sisi kanan (yang lebih dekat tengah = xref lebih kecil)."""
    if not daftar:
        return None
    if garis.theta is None:
        return daftar[0]
    pred = garis.x_ref(h)
    jarak_pindah = JARAK_PINDAH * w
    kuat = [c for c in daftar if c["span"] >= KELOMPOK_SPAN_LUAR]
    if dari_tengah:
        dekat = [c for c in kuat if c["xref"] < pred - jarak_pindah]
    else:
        dekat = [c for c in kuat if c["xref"] > pred + jarak_pindah]
    if dekat:
        return dekat[0]
    terdekat = min(daftar, key=lambda c: abs(c["xref"] - pred))
    return terdekat if abs(terdekat["xref"] - pred) <= GATE_TOL * w else None


def fit_kandidat(c, w, h):
    if c is None:
        return None
    return fit_kurva(c["ys"], c["xs"], c["bobot"], w, h)


# ============================================================
#   KEADAAN PELACAK, JUMLAH JALUR, LEBAR JALUR
# ============================================================
class Pelacak:
    """Seluruh keadaan yang dibawa dari frame ke frame, dan pengaturan tampilan."""

    def __init__(self, jalur="auto"):
        self.tepi_kiri = Garis()
        self.tengah = Garis()
        self.tepi_kanan = Garis()
        self.lebar_prop = LEBAR_DEFAULT
        self.lebar_n = 0
        self.jalur = jalur                  # "auto", "2", atau "3"
        self.vote = deque(maxlen=VOTE_FRAMES)
        self.mode = 2
        self.frame = 0
        self.tampil_koordinat = True        # titik x,y di gambar (tombol c)
        self.tampil_roi = False             # garis ROI di gambar (tombol r); tersembunyi secara default


def tentukan_mode(p, bukti_3_jalur):
    """Jumlah jalur: 3 bila bukti 3 jalur konsisten selama beberapa frame, selain itu 2.
    Voting dengan histeresis (VOTE_NAIK / VOTE_TURUN) agar mode tidak berkedip.
    --jalur 2 atau 3 memaksa mode."""
    if p.jalur in ("2", "3"):
        p.mode = int(p.jalur)
        return p.mode
    p.vote.append(1 if bukti_3_jalur else 0)
    frac = sum(p.vote) / len(p.vote)
    if p.mode == 2 and frac >= VOTE_NAIK:
        p.mode = 3
    elif p.mode == 3 and frac <= VOTE_TURUN:
        p.mode = 2
    return p.mode


def lebar_px(y, h, w, p):
    """Lebar jalur (pixel) pada ketinggian y. Makin jauh (ke atas) makin sempit (perspektif)."""
    persp = np.maximum(0.05, (np.asarray(y, dtype=np.float64) / h - Y_HORIZON) / (Y_REF - Y_HORIZON))
    return p.lebar_prop * w * persp


def pelajari_lebar(p, w, h):
    """Perbarui lebar jalur dari jarak tepi kiri dan garis tengah di garis acuan."""
    yr = [Y_REF * h]
    xk = p.tepi_kiri.x_pada(yr, h)[0]
    xt = p.tengah.x_pada(yr, h)[0]
    prop = (xt - xk) / w
    if 0.12 < prop < 0.80:
        p.lebar_n += 1
        a = max(LEBAR_ALPHA, 1.0 / p.lebar_n)
        p.lebar_prop = (1 - a) * p.lebar_prop + a * prop


def hitung_koordinat(p, w, h):
    """Koordinat final tiap garis untuk frame ini.
      - tepi kiri dan garis tengah saling mengisi bila salah satunya tidak terlihat
        (garis yang hilang = garis lain +/- lebar jalur, status 'turunan')
      - tepi kanan hanya dari deteksi, lalu diprediksi Kalman sebentar bila hilang
    Setiap garis dihaluskan dengan Kalman filter."""
    lv = Y_LEVELS * h
    n = len(lv)
    lebar = lebar_px(lv, h, w, p)
    kiri, tengah, luar = p.tepi_kiri, p.tengah, p.tepi_kanan

    if kiri.segar:
        z_k, st_k = kiri.x_pada(lv, h), "deteksi"
    elif tengah.segar:
        z_k, st_k = tengah.x_pada(lv, h) - lebar, "turunan"
    else:
        z_k, st_k = None, "estimasi"

    if tengah.segar:
        z_t, st_t = tengah.x_pada(lv, h), "deteksi"
    elif kiri.segar:
        z_t, st_t = kiri.x_pada(lv, h) + lebar, "turunan"
    else:
        z_t, st_t = None, "estimasi"

    z_r = luar.x_pada(lv, h) if luar.segar else None

    garis = {
        "tepi_kiri": langkah_kf(kiri, z_k, st_k, w, n),
        "tengah": langkah_kf(tengah, z_t, st_t, w, n),
        "tepi_kanan": langkah_kf(luar, z_r, "deteksi", w, n),
    }

    st_inti = {garis["tepi_kiri"]["status"], garis["tengah"]["status"]}
    if st_inti == {"deteksi"}:
        status = "normal"
    elif st_inti <= {"hilang"}:
        status = "hilang"
    else:
        status = "perkiraan"
    return {"garis": garis, "lebar": lebar, "y": lv, "status": status}


# ============================================================
#   ASPAL DI KANAN GARIS TEPI KANAN (PENENTU 2 / 3 JALUR)
# ============================================================
def mask_aspal_frame(frame):
    """Piksel permukaan aspal: saturasi rendah dan kecerahan sedang. Rumput (saturasi tinggi)
    dan langit tidak lolos; kerikil yang sangat terang juga tidak lolos."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, ss, vv = cv2.split(hsv)
    return (ss <= 70) & (vv >= 60) & (vv <= ASPAL_V_MAKS)


def area_kanan_garis(lv, luar_x, h, w):
    """Mask area di kanan garis tepi kanan, sampai tepi frame, di dalam ROI."""
    yi = np.round(lv).astype(np.int32)
    xi = np.clip(np.round(luar_x), -w, 2 * w).astype(np.int32)
    region = np.zeros((h, w), np.uint8)
    poly = np.vstack([np.stack([xi, yi], axis=1),
                      np.stack([np.full(len(lv), w - 1, dtype=np.int32), yi], axis=1)[::-1]])
    cv2.fillPoly(region, [poly], 1)
    return region > 0


def rasio_aspal_kanan(frame, luar_x, lv):
    """Bagian area di kanan garis tepi kanan (sampai tepi frame, di dalam ROI) yang berupa
    aspal, 0-1. Jalan 3 jalur: di kanan garis magenta masih ada jalur aspal.
    Jalan 2 jalur: di kanan garis tepi kanan biasanya rumput atau tanah."""
    if luar_x is None:
        return 0.0
    h, w = frame.shape[:2]
    region = area_kanan_garis(lv, luar_x, h, w)
    total = int(region.sum())
    if total < 50:
        return 0.0
    return float((mask_aspal_frame(frame) & region).sum()) / total


# ============================================================
#   ANALISIS SATU FRAME
# ============================================================
def analisis(frame, p, debug=False):
    """Satu frame: deteksi garis, pelacakan, koordinat final, lalu jumlah jalur."""
    h, w = frame.shape[:2]
    p.frame += 1
    y0 = int(round(ROI_ATAS * h))
    kand = deteksi_kandidat(frame[y0:h], y0, w, h)

    kiri_list = sorted([c for c in kand if c["xref"] < w / 2], key=lambda c: -c["xref"])
    kanan_list = sorted([c for c in kand if c["xref"] >= w / 2], key=lambda c: c["xref"])

    c_kiri = pilih_kandidat(kiri_list, p.tepi_kiri, w, h, dari_tengah=False)
    c_tengah = pilih_kandidat(kanan_list, p.tengah, w, h, dari_tengah=True)

    # tepi kanan harus menerus dan berjarak dari garis tengah
    jarak = JARAK_MIN_GARIS_LUAR * w
    kandidat_luar = [c for c in kanan_list if c_tengah is not None
                     and c["xref"] > c_tengah["xref"] + jarak and c["span"] >= KELOMPOK_SPAN_LUAR]
    c_luar = pilih_kandidat(kandidat_luar, p.tepi_kanan, w, h, dari_tengah=True)

    for garis, c in ((p.tepi_kiri, c_kiri), (p.tengah, c_tengah), (p.tepi_kanan, c_luar)):
        garis.update(fit_kandidat(c, w, h), w, h)

    if p.tepi_kiri.segar and p.tengah.segar and p.mode == 2:
        pelajari_lebar(p, w, h)

    hasil = hitung_koordinat(p, w, h)

    # jumlah jalur: 3 bila di kanan garis tepi kanan masih ada aspal
    rasio = rasio_aspal_kanan(frame, hasil["garis"]["tepi_kanan"]["x"], hasil["y"])
    tentukan_mode(p, rasio >= ASPAL_KANAN_MIN)

    hasil["mode"] = p.mode
    hasil["rasio_aspal"] = rasio
    hasil["frame"] = p.frame
    if debug:
        hasil["kandidat"] = [(c["xref"], Y_REF * h) for c in kand]
    return hasil


# ============================================================
#   GAMBAR
# ============================================================
def tulis_teks(img, teks, org, fs, warna, tebal=2):
    """Teks dengan garis tepi hitam agar mudah dibaca di atas gambar apa pun."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, teks, org, font, fs, (0, 0, 0), tebal + 2, cv2.LINE_AA)
    cv2.putText(img, teks, org, font, fs, warna, tebal, cv2.LINE_AA)


def tulis_label(img, teks, x, y, s):
    """Nama bagian jalan: teks putih di tengah (x, y), tetap di dalam gambar."""
    h, w = img.shape[:2]
    fs = 0.5 * s
    (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    x = int(np.clip(x - tw / 2, 4, w - tw - 4))
    y = int(np.clip(y + th / 2, th + 4, h - 4))
    tulis_teks(img, teks, (x, y), fs, (255, 255, 255))


def gambar_titik(img, daftar, lv, s):
    """Titik koordinat setiap garis di setiap level, dengan teks 'x,y' (piksel frame yang diproses).
    daftar = [(x_array, warna, sisi)]; sisi -1 = teks di kiri titik, +1 = teks di kanan titik."""
    h, w = img.shape[:2]
    fs = 0.34 * s
    for x_arr, warna, sisi in daftar:
        if x_arr is None:
            continue
        for xv, yv in zip(x_arr, lv):
            px, py = int(round(xv)), int(round(yv))
            if not (0 <= px < w and 0 <= py < h):
                continue
            cv2.circle(img, (px, py), max(2, int(round(3 * s))), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), max(1, int(round(2 * s))), warna, -1, cv2.LINE_AA)
            teks = f"{px},{py}"
            (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            tx = px - tw - int(6 * s) if sisi < 0 else px + int(6 * s)
            tx = int(np.clip(tx, 2, w - tw - 2))
            ty = int(np.clip(py + th // 2, th + 2, h - 2))
            tulis_teks(img, teks, (tx, ty), fs, (255, 255, 255), 1)


def gambar_legenda(img, mode, s):
    """Keterangan warna di pojok kiri atas, sesuai jumlah jalur."""
    if mode == 3:
        items = [("Bahu jalan (kuning)", WARNA_BAHU),
                 ("Jalur kiri (hijau)", WARNA_JALUR_KIRI),
                 ("Jalur kanan (merah)", WARNA_JALUR_KANAN),
                 ("Tepi bahu (biru)", WARNA_TEPI_KIRI),
                 ("Garis kuning: bahu | jalur kiri", WARNA_TENGAH),
                 ("Garis magenta: jalur kiri | kanan", WARNA_TEPI_KANAN)]
    else:
        items = [("Jalur kiri (hijau)", WARNA_JALUR_KIRI),
                 ("Tepi jalur kiri (biru)", WARNA_TEPI_KIRI),
                 ("Garis tengah (kuning)", WARNA_TENGAH),
                 ("Jalur kanan (merah)", WARNA_JALUR_KANAN),
                 ("Tepi kanan (magenta)", WARNA_TEPI_KANAN)]
    kotak = max(7, int(round(10 * s)))
    x = int(8 * s)
    for k, (teks, warna) in enumerate(items):
        y = int((48 + 18 * k) * s)
        cv2.rectangle(img, (x, y), (x + kotak, y + kotak), warna, -1)
        cv2.rectangle(img, (x, y), (x + kotak, y + kotak), (255, 255, 255), 1)
        tulis_teks(img, teks, (x + kotak + int(6 * s), y + kotak), 0.42 * s, (255, 255, 255), 1)


def gambar(frame, hasil, p, debug=False):
    """Gambar area, garis, label, titik x,y, garis ROI (bila aktif), dan legenda.
    Area di kanan garis tepi kanan hanya diwarnai pada permukaan aspal, bukan rumput."""
    h, w = frame.shape[:2]
    s = w / 640.0
    out = frame.copy()
    lv = hasil["y"]
    yi = np.round(lv).astype(np.int32)
    mode = hasil["mode"]
    G = {k: v["x"] for k, v in hasil["garis"].items()}
    lebar = hasil["lebar"]
    kiri, tengah, luar = G["tepi_kiri"], G["tengah"], G["tepi_kanan"]
    batas_kanan = luar if luar is not None else (tengah + lebar if tengah is not None else None)
    kk = N_TITIK - 1
    k2 = max(0, kk - 2)
    yl = 0.94 * h

    def pts(x):
        xi = np.clip(np.round(x), -w, 2 * w).astype(np.int32)
        return np.stack([xi, yi], axis=1)

    # 1) area berwarna
    ov = np.zeros_like(out)
    if mode == 3:
        if kiri is not None and tengah is not None:                 # bahu
            cv2.fillPoly(ov, [np.vstack([pts(kiri), pts(tengah)[::-1]])], WARNA_BAHU)
        if tengah is not None:                                      # jalur kiri
            cv2.fillPoly(ov, [np.vstack([pts(tengah), pts(batas_kanan)[::-1]])], WARNA_JALUR_KIRI)
    else:
        if kiri is not None and tengah is not None:                 # jalur kiri
            cv2.fillPoly(ov, [np.vstack([pts(kiri), pts(tengah)[::-1]])], WARNA_JALUR_KIRI)
        if tengah is not None:                                      # jalur kanan
            cv2.fillPoly(ov, [np.vstack([pts(tengah), pts(batas_kanan)[::-1]])], WARNA_JALUR_KANAN)
    if luar is not None:                                            # jalur kanan di kanan tepi kanan
        ov[area_kanan_garis(lv, luar, h, w) & mask_aspal_frame(frame)] = WARNA_JALUR_KANAN
    m = ov.any(axis=2)
    out[m] = (0.62 * out[m] + 0.38 * ov[m]).astype(np.uint8)

    # 2) garis
    tebal = max(2, int(round(w / 200)))
    for x, warna, t in ((kiri, WARNA_TEPI_KIRI, tebal + 1),
                        (tengah, WARNA_TENGAH, tebal + 1),
                        (luar, WARNA_TEPI_KANAN, tebal)):
        if x is not None:
            cv2.polylines(out, [pts(x)], False, warna, t, cv2.LINE_AA)

    # 3) garis ROI (hanya bila ditampilkan)
    if p.tampil_roi:
        y0 = int(round(ROI_ATAS * h))
        cv2.rectangle(out, (0, y0), (w - 1, h - 1), WARNA_ROI, 1, cv2.LINE_AA)
        persen = 100.0 * (h - y0) / h
        teks = f"ROI 40% bawah: y={y0}..{h} ({h - y0} dari {h} px = {persen:.0f}%)"
        tulis_teks(out, teks, (int(6 * s), h - int(6 * s)), 0.36 * s, WARNA_ROI, 1)

    # 4) nama bagian jalan (tidak menumpuk titik x,y)
    if mode == 3:
        if kiri is not None and tengah is not None:
            tulis_label(out, "BAHU", (kiri[kk] + tengah[kk]) / 2, yl, s)
        if tengah is not None:
            tulis_label(out, "JALUR KIRI", (tengah[k2] + batas_kanan[k2]) / 2, lv[k2], s)
        if luar is not None:
            tulis_label(out, "JALUR KANAN", (luar[kk] + w - 1) / 2, yl, s)
    else:
        if kiri is not None and tengah is not None:
            tulis_label(out, "JALUR KIRI", (kiri[k2] + tengah[k2]) / 2, lv[k2], s)
        if tengah is not None:
            tulis_label(out, "JALUR KANAN", (batas_kanan[kk] + w - 1) / 2, yl, s)

    # 5) titik koordinat x,y
    if p.tampil_koordinat:
        gambar_titik(out, [(kiri, WARNA_TEPI_KIRI, -1),
                           (tengah, WARNA_TENGAH, 1),
                           (luar, WARNA_TEPI_KANAN, 1)], lv, s)

    # 6) mode debug: titik kandidat garis marka
    if debug and "kandidat" in hasil:
        for x, y in hasil["kandidat"]:
            cv2.circle(out, (int(x), int(y)), max(2, int(round(3 * s))), WARNA_KANDIDAT, -1)

    # 7) status dan legenda
    sumber = "otomatis" if p.jalur == "auto" else "manual"
    tulis_teks(out, f"Jalur: {mode} ({sumber})  |  Status: {hasil['status']}",
               (int(8 * s), int(22 * s)), 0.5 * s, (255, 255, 255))
    gambar_legenda(out, mode, s)
    return out


# ============================================================
#   KOORDINAT (JSON)
# ============================================================
def ke_json(hasil, waktu):
    """Hasil satu frame sebagai objek JSON. Titik dalam pixel frame yang diproses."""
    garis_json = {}
    for nama, info in hasil["garis"].items():
        if info["x"] is None:
            garis_json[nama] = None
            continue
        garis_json[nama] = {
            "status": info["status"],
            "titik": [{"x": int(round(x)), "y": int(round(y))}
                      for x, y in zip(info["x"], hasil["y"])],
        }
    return {"frame": hasil["frame"], "waktu": round(waktu, 3), "jumlah_jalur": hasil["mode"],
            "rasio_aspal_kanan": round(hasil["rasio_aspal"], 3),
            "status": hasil["status"], "garis": garis_json}


# ============================================================
#   MASUKAN: FOTO, VIDEO, DAN WEBCAM
# ============================================================
def ubah_lebar(frame, lebar):
    h0, w0 = frame.shape[:2]
    if w0 == lebar:
        return frame
    return cv2.resize(frame, (lebar, int(round(h0 * lebar / w0))), interpolation=cv2.INTER_AREA)


def proses_foto(path, args, p):
    """Foto tunggal: diproses berulang agar pelacak stabil, lalu ditampilkan / disimpan."""
    frame = cv2.imread(path)
    if frame is None:
        print("Tidak bisa membuka foto:", path)
        return
    frame = ubah_lebar(frame, args.lebar)
    for _ in range(15):
        hasil = analisis(frame, p, args.debug)
    out = gambar(frame, hasil, p, args.debug)
    data = ke_json(hasil, 0.0)
    print(json.dumps(data, ensure_ascii=False))
    if args.koordinat:
        with open(args.koordinat, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    if args.save:
        cv2.imwrite(args.save, out)
        print("Tersimpan:", args.save)
    if not args.tanpa_tampil:
        cv2.namedWindow(JUDUL, cv2.WINDOW_NORMAL)
        cv2.imshow(JUDUL, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def proses_video(args, p):
    """Video file atau webcam. Berhenti bila video habis, tombol q, atau durasi tercapai."""
    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        print("Tidak bisa membuka sumber:", args.source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps <= 240.0):
        fps = 30.0

    # --mulai: lompat ke posisi, dan proses 2 detik sebelumnya dulu (tanpa disimpan)
    # agar pelacak sudah stabil saat bagian yang diminta mulai tampil
    mulai_pem, n_pemanasan = 0.0, 0
    if args.mulai > 0 and not is_cam:
        mulai_pem = max(0.0, args.mulai - 2.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, mulai_pem * 1000.0)
        n_pemanasan = int(round((args.mulai - mulai_pem) * fps))

    batas_waktu = args.mulai + args.durasi if args.durasi > 0 else None
    koor = open(args.koordinat, "w", encoding="utf-8") if args.koordinat else None
    writer = None
    if not args.tanpa_tampil:
        cv2.namedWindow(JUDUL, cv2.WINDOW_NORMAL)

    idx = 0
    tersimpan = 0
    n_mode = {2: 0, 3: 0}
    n_status = {"normal": 0, "perkiraan": 0, "hilang": 0}
    t_mulai = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        waktu = mulai_pem + idx / fps
        idx += 1
        if batas_waktu is not None and waktu >= batas_waktu:
            break

        frame = ubah_lebar(frame, args.lebar)
        hasil = analisis(frame, p, args.debug)
        if idx <= n_pemanasan:
            continue                                  # masa pemanasan: tidak disimpan / ditampilkan

        out = gambar(frame, hasil, p, args.debug)
        n_mode[hasil["mode"]] += 1
        n_status[hasil["status"]] += 1
        tersimpan += 1

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)
        if koor is not None:
            koor.write(json.dumps(ke_json(hasil, waktu), ensure_ascii=False) + "\n")

        if not args.tanpa_tampil:
            cv2.imshow(JUDUL, out)
            if not is_cam:                             # video file: ikuti kecepatan asli
                tunggu = (idx - n_pemanasan) / fps - (time.perf_counter() - t_mulai)
                if tunggu > 0:
                    time.sleep(tunggu)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                args.debug = not args.debug
            if key == ord("c"):
                p.tampil_koordinat = not p.tampil_koordinat
            if key == ord("r"):
                p.tampil_roi = not p.tampil_roi
            if key == ord("a"):
                p.jalur = "auto"
            if key == ord("2"):
                p.jalur = "2"
            if key == ord("3"):
                p.jalur = "3"

    cap.release()
    if writer is not None:
        writer.release()
    if koor is not None:
        koor.close()
    if not args.tanpa_tampil:
        cv2.destroyAllWindows()

    print(f"Frame diproses: {tersimpan} | jalur 2: {n_mode[2]} | jalur 3: {n_mode[3]} | "
          f"status normal/perkiraan/hilang: {n_status['normal']}/{n_status['perkiraan']}/{n_status['hilang']}")
    if args.save and tersimpan:
        print("Video hasil tersimpan:", args.save)
    if args.koordinat:
        print("Koordinat tersimpan:", args.koordinat)


def main():
    ap = argparse.ArgumentParser(
        description="Deteksi jalur dari video jalan (2 jalur: kiri-kanan; 3 jalur: bahu-kiri-kanan).")
    ap.add_argument("--source", default="video_jalan.mp4",
                    help="0 = webcam, path video (mis. video_jalan.mp4), atau path foto")
    ap.add_argument("--jalur", choices=["auto", "2", "3"], default="auto",
                    help="jumlah jalur: auto (otomatis), 2, atau 3")
    ap.add_argument("--lebar", type=int, default=LEBAR_PROSES,
                    help="lebar frame yang diproses (default 640)")
    ap.add_argument("--mulai", type=float, default=0.0,
                    help="(video) mulai dari detik ke-N")
    ap.add_argument("--durasi", type=float, default=0.0,
                    help="(video) proses N detik saja (0 = sampai habis)")
    ap.add_argument("--save", default="",
                    help="simpan hasil: file .mp4 (video) atau .jpg/.png (foto)")
    ap.add_argument("--koordinat", default="",
                    help="simpan koordinat tiap frame ke file .jsonl")
    ap.add_argument("--tanpa-tampil", action="store_true",
                    help="jangan buka jendela (untuk proses tanpa layar)")
    ap.add_argument("--debug", action="store_true",
                    help="tampilkan titik kandidat garis marka")
    ap.add_argument("--tanpa-koordinat", action="store_true",
                    help="sembunyikan titik koordinat x,y (tombol c = tampil/sembunyi)")
    ap.add_argument("--tampil-roi", action="store_true",
                    help="tampilkan garis ROI (tombol r = tampil/sembunyi; default tersembunyi)")
    args = ap.parse_args()

    p = Pelacak(jalur=args.jalur)
    p.tampil_koordinat = not args.tanpa_koordinat
    p.tampil_roi = args.tampil_roi
    if args.source.lower().endswith(IMAGE_EXT):
        proses_foto(args.source, args, p)
    else:
        proses_video(args, p)


if __name__ == "__main__":
    main()