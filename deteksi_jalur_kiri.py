#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deteksi Jalur pada Video Jalan (dashcam)
========================================
Membagi jalan di depan kendaraan menjadi jalur-jalur, lalu menggambarnya dengan warna
berbeda dan menampilkan titik koordinat x,y di setiap garis.

ROI         : 40% bagian bawah frame, selebar frame. Hanya membatasi area pencarian garis.
              Garis ROI disembunyikan secara default (tampilkan dengan --tampil-roi atau tombol r).

Jumlah jalur (otomatis, atau paksa dengan --jalur 2 / --jalur 3). Pada rentang jalan bahu yang
ditandai dengan tombol b (disimpan di <nama video>_jalur.json), jumlah jalur selalu 3:
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
        -> marka tidak ada: pakai histori (posisi terakhir yang diterima) selama garis pernah
           terlihat; bila video dari awal tanpa marka, jalan dibagi dua dari tepi aspal kiri dan
           kanan (bagian kiri = jalur kiri, bagian kanan = jalur kanan)
        -> fallback: lebar jalur dari sisi lain, lalu dinyatakan hilang
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
  python deteksi_jalur_kiri.py --source 0 --horizon 0.58 --kalibrasi   (webcam, kalibrasi sekali)
  python deteksi_jalur_kiri.py --source 0 --horizon 0.58               (webcam, memakai kalibrasi tersimpan)
  python deteksi_jalur_kiri.py --source 0 --pasang                    (bantu pasang kamera di mobil)
  python deteksi_jalur_kiri.py --source 0 --cermin                    (gambar dibalik kiri-kanan)
  python deteksi_jalur_kiri.py --source 0 --balik                     (gambar dibalik atas-bawah)

Tombol saat jendela tampil:
  q = keluar           | d = mode debug (titik kandidat marka)
  c = titik x,y on/off | r = garis ROI on/off
  a = jumlah jalur otomatis | 2 = paksa 2 jalur | 3 = paksa 3 jalur
  b = mulai / akhiri rentang jalan bahu (3 jalur) di posisi video ini; disimpan saat keluar
  v = rekam webcam on/off (atau klik tombol REC di kanan atas; hanya webcam)

Kebutuhan: Python 3, opencv-python, numpy.
"""

import argparse
import json
import os
import signal
import time
import warnings
from collections import Counter, deque

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
MARKA_KUNING_V = 120                 # kecerahan minimum marka kuning
MARKA_KUNING_SELISIH = 70            # marka kuning: selisih kanal terbesar (merah/hijau dibanding biru)
MARKA_KUNING_HIJAU_B = 40            # marka kuning: hijau harus lebih besar dari biru sebesar ini
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
HISTORI_DETIK = 2.0                  # marka hilang: posisi terakhir (histori) dipakai paling lama sekian detik

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
ASPAL_KANAN_MIN = 0.35               # bukti 3 jalur bila rasio aspal di kanan tepi kanan >= nilai ini
ASPAL_V_MAKS = 120                   # aspal gelap; bahu tanah/kerikil yang terang tidak ikut terhitung
ASPAL_SELISIH_MAKS = 35              # aspal abu-abu: selisih kanal kecil (rumput berwarna hijau tidak lolos)
ROAD_PATCH = (0.80, 0.88, 0.42, 0.58)  # area jalan tepat di depan mobil (baris atas-bawah, kolom kiri-kanan)
ROAD_EMA = 0.05                      # kecepatan penyesuaian kecerahan jalan antar frame
ASPAL_TOLERANSI = 40                 # tepi aspal: kecerahan piksel boleh berbeda dari jalan ini sebesar ini
ASPAL_SELISIH_TEPI = 50              # jalan tanpa marka: selisih kanal lebih longgar (jalan jauh lebih kebiruan)
TEPI_MIN_TITIK = 5                   # jalan tanpa marka: minimal baris tinggi yang punya tepi (sisanya dibuang MAD)
TEPI_HIST = 5                        # jalan tanpa marka: median posisi tepi dari sekian frame terakhir (meredam loncatan)
TEPI_PUSAT_LEBAR_MIN = 0.12          # jalan tanpa marka: lebar jalan minimal (proporsi lebar frame) per level; jalan jauh lebih sempit
# Rentang waktu jalan bahu (3 jalur) disimpan per video di berkas "<nama video>_jalur.json".
# Dibuat dengan menekan b saat menonton video. Di luar rentang itu, jumlah jalur otomatis.
JALUR_FILE_SUFFIX = "_jalur.json"
REKAM_FOLDER = "rekaman"             # folder hasil rekaman webcam (tombol v atau tombol REC)

KALIBRASI_WEBCAM_FILE = "kalibrasi_webcam.json"   # kalibrasi untuk webcam (--kalibrasi)

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


def fit_kurva(ys, xs, bobot, w, h, min_titik=FIT_MIN_TITIK, izinkan_kuadrat=True):
    """Fit x = A t^2 + B t + C ke titik-titik. Pencilan dibuang dengan MAD.
    izinkan_kuadrat=False memaksa garis lurus (dipakai untuk tepi aspal, agar tidak melengkung oleh derau).
    Mengembalikan (theta, dom) atau None bila titik tidak cukup."""
    if len(xs) < min_titik:
        return None
    t = t_dari_y(ys, h)
    kuadrat = izinkan_kuadrat and (ys.max() - ys.min()) / h >= FIT_MIN_SPAN
    keep = np.ones(len(xs), dtype=bool)
    th = None
    for _ in range(3):
        tt, xx, bb = t[keep], xs[keep], bobot[keep]
        if len(tt) < min_titik:
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
        if baru.sum() < min_titik or np.array_equal(baru, keep):
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
    r = np.full(n, (KF_R_TURUNAN if st in ("turunan", "histori") else KF_R_DETEKSI) * w)
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
        self.sumber = "marka"       # "marka" atau "tepi_aspal" (jalan tanpa marka)
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

    def update(self, fit, w, h, longgar=False):
        """Gabungkan hasil fit frame ini dengan garis lama. longgar=True (tepi aspal tanpa marka):
        tiap frame diukur ulang dari jalan, jadi uji kewajaran dilewati dan hanya dihaluskan (EMA)."""
        self.segar = False
        if fit is None:
            self._hilang()
            return
        theta, dom = fit
        if self.theta is None:
            self._ganti(theta, dom)
            return
        if longgar:
            self.theta = EMA_ALPHA * theta + (1 - EMA_ALPHA) * self.theta
            self.dom = EMA_ALPHA * dom + (1 - EMA_ALPHA) * self.dom
            self.hilang = 0
            self.tolak = 0
            self.segar = True
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
        """Marka tidak terdeteksi: penghitung naik. Garis dari marka TIDAK direset, sehingga posisi
        terakhir (histori) tetap dipakai sampai marka kembali. Garis dari tepi aspal direset setelah
        MAX_HILANG frame (karena jalan tanpa marka akan diukur ulang dari awal)."""
        self.hilang += 1
        if self.hilang > MAX_HILANG and self.sumber != "marka":
            self.reset()


# ============================================================
#   DETEKSI MARKA & KANDIDAT GARIS
# ============================================================
KERNEL_DILATE = np.ones((MARKA_DILATE, MARKA_DILATE), np.uint8)


def _kanal(bgr):
    """Kanal biru, hijau, dan merah sebagai float32 (tanpa konversi warna)."""
    return (bgr[..., 0].astype(np.float32),
            bgr[..., 1].astype(np.float32),
            bgr[..., 2].astype(np.float32))


def mask_marka(roi, w):
    """Mask piksel marka jalan (putih/krem atau kuning) pada area ROI, hasil 0 / 255.
    Dihitung langsung dari kanal BGR:
      - kecerahan = kanal terang (maksimum dari biru, hijau, merah);
      - putih/krem: selisih kanal kecil, atau merah >= hijau >= biru. Rumput (hijau) tidak lolos;
      - kuning: merah >= hijau dan hijau jauh di atas biru;
      - kontras dua sisi: garis harus lebih terang dari KEDUA sisinya (pada jarak MARKA_JARAK_SISI)."""
    b, g, r = _kanal(roi)
    terang = np.maximum(np.maximum(r, g), b)
    selisih = terang - np.minimum(np.minimum(r, g), b)
    pusat = cv2.blur(terang, (5, 5))
    d = max(4, int(round(MARKA_JARAK_SISI * w / 640.0)))
    sisi_terang = np.maximum(np.roll(pusat, d, axis=1), np.roll(pusat, -d, axis=1))
    kontras = pusat - sisi_terang

    krem = (r >= g - 8) & (g >= b - 8)
    putih = ((kontras >= MARKA_KONTRAS) & (terang >= MARKA_PUTIH_MIN_V)
             & ((selisih <= 50) | (krem & (selisih <= 100))))
    kuning = ((r >= g) & (g >= b + MARKA_KUNING_HIJAU_B) & (selisih >= MARKA_KUNING_SELISIH)
              & (terang >= MARKA_KUNING_V) & (kontras >= 0.6 * MARKA_KONTRAS))
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
    if "fit" in c:
        return c["fit"]
    return fit_kurva(c["ys"], c["xs"], c["bobot"], w, h, min_titik=c.get("min_titik", FIT_MIN_TITIK),
                     izinkan_kuadrat=not c.get("virtual", False))


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
        self.rentang_bahu = []              # [(awal, akhir)] detik, jalan bahu: 3 jalur
        self.awal_rentang = None            # awal rentang yang sedang dibuka dengan tombol b
        self.rentang_berubah = False        # True bila rentang berubah dan perlu disimpan
        self.waktu = None                   # waktu frame saat ini (detik); None untuk foto / webcam
        self.dalam_rentang_sebelumnya = False
        self.dalam_rentang = False          # frame ini berada di rentang bahu
        self.road_v = None                  # kecerahan jalan di depan mobil (disesuaikan per frame)
        self.fps = 30.0                     # fps video (untuk batas histori dalam detik)
        self.hist_tepi = deque(maxlen=TEPI_HIST)  # riwayat (tepi kiri, tepi kanan) per baris tinggi
        self.kalibrasi = None               # titik kalibrasi per video (klik manual), None = otomatis


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


def _pilih_z(garis, lv, h, z_turunan, batas_histori):
    """Sumber posisi garis di frame ini, urutan prioritas:
      1. deteksi (marka, atau tepi aspal bila jalan tanpa marka)
      2. histori: posisi terakhir yang diterima, selama marka hilang paling lama batas_histori frame
      3. turunan dari garis lain +/- lebar jalur (histori terlalu lama, atau garis belum pernah terlihat)
      4. estimasi: tidak ada sumber lain -> hanya prediksi Kalman"""
    if garis.segar:
        st = {"marka": "deteksi", "tepi_aspal": "tepi_aspal", "kalibrasi": "kalibrasi"}.get(garis.sumber, "deteksi")
        return garis.x_pada(lv, h), st
    if garis.theta is not None and garis.hilang <= batas_histori:
        return garis.x_pada(lv, h), "histori"
    if z_turunan is not None:
        return z_turunan, "turunan"
    return None, "estimasi"


def _status_umum(st_inti):
    """Status keseluruhan dari status tepi kiri dan garis tengah."""
    if st_inti == {"kalibrasi"}:
        return "kalibrasi"
    if st_inti == {"deteksi"}:
        return "normal"
    if st_inti <= {"deteksi", "tepi_aspal"}:
        return "tanpa_marka"
    if st_inti <= {"hilang"}:
        return "hilang"
    return "perkiraan"


def hitung_koordinat(p, w, h):
    """Koordinat final tiap garis untuk frame ini (lihat _pilih_z untuk urutan sumbernya).
    Setiap garis dihaluskan dengan Kalman filter."""
    lv = Y_LEVELS * h
    n = len(lv)
    lebar = lebar_px(lv, h, w, p)
    kiri, tengah, luar = p.tepi_kiri, p.tengah, p.tepi_kanan

    batas = int(round(HISTORI_DETIK * p.fps))
    z_k, st_k = _pilih_z(kiri, lv, h, tengah.x_pada(lv, h) - lebar if tengah.segar else None, batas)
    z_t, st_t = _pilih_z(tengah, lv, h, kiri.x_pada(lv, h) + lebar if kiri.segar else None, batas)
    z_r, st_r = _pilih_z(luar, lv, h, tengah.x_pada(lv, h) + lebar if tengah.segar else None, batas)

    garis = {
        "tepi_kiri": langkah_kf(kiri, z_k, st_k, w, n),
        "tengah": langkah_kf(tengah, z_t, st_t, w, n),
        "tepi_kanan": langkah_kf(luar, z_r, st_r, w, n),
    }
    st_inti = {garis["tepi_kiri"]["status"], garis["tengah"]["status"]}
    return {"garis": garis, "lebar": lebar, "y": lv, "status": _status_umum(st_inti)}


# ============================================================
#   ASPAL DI KANAN GARIS TEPI KANAN (PENENTU 2 / 3 JALUR)
# ============================================================
def mask_aspal_frame(frame):
    """Piksel permukaan aspal: abu-abu gelap (selisih kanal kecil, kecerahan sedang).
    Rumput (berwarna hijau), langit, dan kerikil yang terang tidak lolos."""
    b, g, r = _kanal(frame)
    terang = np.maximum(np.maximum(r, g), b)
    selisih = terang - np.minimum(np.minimum(r, g), b)
    return (selisih <= ASPAL_SELISIH_MAKS) & (terang >= 60) & (terang <= ASPAL_V_MAKS)


def area_kanan_garis(lv, luar_x, h, w):
    """Mask area di kanan garis tepi kanan, sampai tepi frame, di dalam ROI."""
    yi = np.round(lv).astype(np.int32)
    xi = np.clip(np.round(luar_x), -w, 2 * w).astype(np.int32)
    region = np.zeros((h, w), np.uint8)
    poly = np.vstack([np.stack([xi, yi], axis=1),
                      np.stack([np.full(len(lv), w - 1, dtype=np.int32), yi], axis=1)[::-1]])
    cv2.fillPoly(region, [poly], 1)
    return region > 0


def kecerahan_jalan(frame):
    """Kecerahan median jalan tepat di depan mobil (kanal terang, 0-255)."""
    h, w = frame.shape[:2]
    a, b_, c, d_ = ROAD_PATCH
    patch = frame[int(a * h):int(b_ * h), int(c * w):int(d_ * w)]
    return float(np.median(patch.max(axis=2)))


def _scan_aspal(baris, x0, arah):
    """Dari x0 ke arah (-1 kiri, +1 kanan), cari piksel aspal terakhir sebelum bertemu
    bukan-aspal (toleransi celah kecil). Mengembalikan x, atau None bila tidak ada aspal."""
    x = x0
    terakhir = None
    salah = 0
    while 0 <= x < len(baris):
        if baris[x]:
            terakhir = x
            salah = 0
        else:
            salah += 1
            if salah > 4:
                break
        x += arah
    return terakhir


def tepi_per_level(frame, lv, w, road_v):
    """Jalan tanpa marka: dari pusat frame, cari tepi aspal kiri dan kanan di tiap baris tinggi lv.
    Mengembalikan (xl, xr) sepanjang N_TITIK; NaN bila baris itu tidak punya tepi yang masuk akal."""
    b, g, r = _kanal(frame)
    terang = np.maximum(np.maximum(r, g), b)
    selisih = terang - np.minimum(np.minimum(r, g), b)
    # jalan = abu-abu netral (aspal dan garis cat di atasnya); rumput (berwarna) bukan jalan
    aspal = (selisih <= ASPAL_SELISIH_TEPI) & (terang >= 60) & (terang <= road_v + ASPAL_TOLERANSI + 60)
    xl = np.full(len(lv), np.nan)
    xr = np.full(len(lv), np.nan)
    xc = w // 2
    for k, y in enumerate(lv):
        yi = int(np.clip(round(y), 0, frame.shape[0] - 1))
        kiri = _scan_aspal(aspal[yi], xc, -1)
        kanan = _scan_aspal(aspal[yi], xc, +1)
        if kiri is None or kanan is None or kanan - kiri < TEPI_PUSAT_LEBAR_MIN * w:
            continue
        xl[k] = kiri
        xr[k] = kanan
    return xl, xr


def fit_model_jalan(lv, xl, xr, h, w):
    """Model jalan tanpa marka dengan perspektif kamera:
      - pusat jalan berupa garis lurus: c(t) = c0 + c1 t
      - lebar jalan mengecil ke cakrawala: W(t) = W0 * p(t), dengan p(t) = 1 + k t (linear)
      - tepi kiri = c - W/2, tepi kanan = c + W/2, sehingga ketiganya garis lurus yang sejajar perspektif.
    xl / xr: tepi kiri dan kanan per baris tinggi lv (NaN bila tidak ada). Pencilan dibuang (MAD).
    Mengembalikan (fit_kiri, fit_tengah, fit_kanan) dengan format (theta, dom), atau None."""
    k = SKALA_T / (Y_REF - Y_HORIZON)
    t = (lv - Y_REF * h) / (SKALA_T * h)
    p = 1.0 + k * t
    rows, obs = [], []
    for i in range(len(lv)):
        if not np.isnan(xl[i]):
            rows.append([1.0, t[i], -p[i] / 2.0])
            obs.append(xl[i])
        if not np.isnan(xr[i]):
            rows.append([1.0, t[i], p[i] / 2.0])
            obs.append(xr[i])
    if len(obs) < 2 * TEPI_MIN_TITIK:
        return None
    A = np.array(rows)
    b = np.array(obs)
    keep = np.ones(len(b), dtype=bool)
    th = None
    for _ in range(3):
        th = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
        res = b - A @ th
        med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med))
        keep = np.abs(res - med) < max(0.02 * w, 2.5 * mad)
        if keep.sum() < 4:
            return None
    c0, c1, W0 = th
    if not (0.15 * w < W0 < 0.95 * w) or not (0.25 * w < c0 < 0.75 * w):
        return None                          # hasil tidak masuk akal (mis. jalan tidak terlihat)
    dom = np.array([t[0], t[-1]])
    kiri = (np.array([0.0, c1 - W0 * k / 2.0, c0 - W0 / 2.0]), dom)
    tengah = (np.array([0.0, c1, c0]), dom)
    kanan = (np.array([0.0, c1 + W0 * k / 2.0, c0 + W0 / 2.0]), dom)
    return kiri, tengah, kanan


def kandidat_dari_titik(ys, xs, h):
    """Kandidat garis dari titik-titik (tanpa marka). Ditandai 'virtual' agar sumbernya tercatat."""
    return {"xref": float(np.interp(Y_REF * h, ys, xs)),
            "span": float((ys.max() - ys.min()) / h),
            "xs": xs, "ys": ys, "bobot": np.ones(len(xs)), "virtual": True, "min_titik": TEPI_MIN_TITIK}


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
def perbarui_dari_marka(frame, p, w, h):
    """Perbarui garis dari marka (atau tepi aspal bila jalan tanpa marka).
    Mengembalikan kandidat marka (dipakai untuk mode debug)."""
    y0 = int(round(ROI_ATAS * h))
    kand = deteksi_kandidat(frame[y0:h], y0, w, h)

    kiri_list = sorted([c for c in kand if c["xref"] < w / 2], key=lambda c: -c["xref"])
    kanan_list = sorted([c for c in kand if c["xref"] >= w / 2], key=lambda c: c["xref"])

    # kandidat dari marka
    c_kiri = pilih_kandidat(kiri_list, p.tepi_kiri, w, h, dari_tengah=False)
    c_tengah = pilih_kandidat(kanan_list, p.tengah, w, h, dari_tengah=True)
    jarak = JARAK_MIN_GARIS_LUAR * w
    kandidat_luar = [c for c in kanan_list if c_tengah is not None
                     and c["xref"] > c_tengah["xref"] + jarak and c["span"] >= KELOMPOK_SPAN_LUAR]
    c_luar = pilih_kandidat(kandidat_luar, p.tepi_kanan, w, h, dari_tengah=True)

    # jalan tanpa marka sejak awal (garis belum pernah terlihat): bagi jalan dari tepi aspal
    # marka tidak ada: histori dipakai selama HISTORI_DETIK; setelah itu (atau bila belum pernah
    # ada marka) tepi jalan diukur dari aspal pada frame ini
    batas = int(round(HISTORI_DETIK * p.fps))

    def perlu_tepi_aspal(garis, c):
        if c is not None:
            return False
        return (garis.theta is None or garis.sumber == "tepi_aspal"
                or garis.hilang > batas)

    kiri_belum = perlu_tepi_aspal(p.tepi_kiri, c_kiri)
    tengah_belum = perlu_tepi_aspal(p.tengah, c_tengah)
    kanan_belum = perlu_tepi_aspal(p.tepi_kanan, c_luar)
    if kiri_belum or tengah_belum or kanan_belum:
        rv = kecerahan_jalan(frame)
        p.road_v = rv if p.road_v is None else (1 - ROAD_EMA) * p.road_v + ROAD_EMA * rv
        lv_ = Y_LEVELS * h
        xl, xr = tepi_per_level(frame, lv_, w, p.road_v)
        p.hist_tepi.append((xl, xr))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med_l = np.nanmedian(np.array([a_ for a_, _ in p.hist_tepi]), axis=0)
            med_r = np.nanmedian(np.array([b_ for _, b_ in p.hist_tepi]), axis=0)
        ada = ~np.isnan(med_l) & ~np.isnan(med_r)
        if ada.sum() >= TEPI_MIN_TITIK:
            fits = fit_model_jalan(lv_, med_l, med_r, h, w)
            if fits is not None:
                f_kiri, f_tengah, f_kanan = fits
                ref = {"xref": w / 2.0, "span": 1.0, "xs": np.array([0.0]), "ys": np.array([0.0]),
                       "bobot": np.ones(1), "virtual": True}
                if kiri_belum:
                    c_kiri = dict(ref, fit=f_kiri)
                if kanan_belum:
                    c_luar = dict(ref, fit=f_kanan)
                if tengah_belum:
                    c_tengah = dict(ref, fit=f_tengah)
    elif p.hist_tepi:
        p.hist_tepi.clear()                 # marka sudah ada lagi: riwayat tepi aspal tidak dipakai

    for garis, c in ((p.tepi_kiri, c_kiri), (p.tengah, c_tengah), (p.tepi_kanan, c_luar)):
        virtual = c is not None and c.get("virtual", False)
        if virtual and garis.sumber == "marka" and garis.theta is not None and garis.hilang > batas:
            garis.reset()                   # histori sudah habis: mulai ulang dari tepi aspal
        garis.update(fit_kandidat(c, w, h), w, h, longgar=virtual)
        garis.sumber = "tepi_aspal" if virtual else "marka"

    if p.tepi_kiri.segar and p.tengah.segar and p.mode == 2 \
            and p.tepi_kiri.sumber == "marka" and p.tengah.sumber == "marka":
        pelajari_lebar(p, w, h)

    return kand


def analisis(frame, p, debug=False):
    """Satu frame: garis (dari kalibrasi bila ada, bila tidak dari marka), koordinat final,
    lalu jumlah jalur."""
    h, w = frame.shape[:2]
    p.frame += 1
    if p.kalibrasi is not None:
        pakai_kalibrasi(p, w, h)
        kand = []
    else:
        kand = perbarui_dari_marka(frame, p, w, h)

    hasil = hitung_koordinat(p, w, h)

    # jumlah jalur: 3 bila di kanan garis tepi kanan (dari marka atau kalibrasi) masih ada aspal
    if p.tepi_kanan.segar and p.tepi_kanan.sumber in ("marka", "kalibrasi"):
        rasio = rasio_aspal_kanan(frame, hasil["garis"]["tepi_kanan"]["x"], hasil["y"])
    else:
        rasio = 0.0
    dalam_rentang = p.waktu is not None and (
        any(a <= p.waktu < b for a, b in p.rentang_bahu)
        or (p.awal_rentang is not None and p.waktu >= p.awal_rentang))
    p.dalam_rentang = dalam_rentang
    if p.jalur == "auto" and dalam_rentang:
        p.mode = 3                          # jalan bahu (rentang waktu): 3 jalur
        p.vote.append(1)
    elif p.jalur == "auto" and p.kalibrasi is not None:
        p.mode = 2                          # dengan kalibrasi: 2 jalur, kecuali rentang bahu atau --jalur 3
        p.vote.clear()
    else:
        if p.jalur == "auto" and p.dalam_rentang_sebelumnya:
            p.vote.clear()                  # keluar dari rentang: buang bukti lama agar cepat kembali
            p.mode = 2
        tentukan_mode(p, rasio >= ASPAL_KANAN_MIN)
    p.dalam_rentang_sebelumnya = dalam_rentang

    hasil["mode"] = p.mode
    hasil["dalam_rentang"] = dalam_rentang
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
    Pada 3 jalur, area di kanan garis tepi kanan diwarnai hanya pada permukaan aspal."""
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
    if mode == 3 and luar is not None:                              # jalur kanan (3 jalur): di kanan
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
            tulis_label(out, "JALUR KANAN", (luar[k2] + w - 1) / 2, lv[k2], s)
    else:
        if kiri is not None and tengah is not None:
            tulis_label(out, "JALUR KIRI", (kiri[k2] + tengah[k2]) / 2, lv[k2], s)
        if tengah is not None:
            tulis_label(out, "JALUR KANAN", (tengah[k2] + batas_kanan[k2]) / 2, lv[k2], s)

    # 5) titik koordinat x,y
    if p.tampil_koordinat:
        gambar_titik(out, [(kiri, WARNA_TEPI_KIRI, -1),
                           (tengah, WARNA_TENGAH, -1),
                           (luar, WARNA_TEPI_KANAN, 1)], lv, s)

    # 6) mode debug: titik kandidat garis marka
    if debug and "kandidat" in hasil:
        for x, y in hasil["kandidat"]:
            cv2.circle(out, (int(x), int(y)), max(2, int(round(3 * s))), WARNA_KANDIDAT, -1)

    # 7) status dan legenda
    sumber = "otomatis" if p.jalur == "auto" else "manual"
    rentang = "  |  rentang bahu" if hasil.get("dalam_rentang") else ""
    tulis_teks(out, f"Jalur: {mode} ({sumber})  |  Status: {hasil['status']}{rentang}",
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
#   REKAM WEBCAM
# ============================================================
class Perekam:
    """Merekam tampilan webcam (area jalur, garis, titik x,y; tanpa tombol) ke berkas MP4.
    Dimulai / dihentikan dengan tombol v atau dengan mengklik tombol REC di pojok kanan atas."""

    def __init__(self, folder):
        self.folder = folder
        self.pesan = ""                 # teks singkat di bawah tombol REC
        self.pesan_sampai = 0.0
        self.aktif = False              # True = sedang merekam
        self.writer = None
        self.berkas = ""
        self.fps = 30.0
        self.frame_n = 0
        self.t_mulai = 0.0
        self.rect = None                # area tombol pada frame tampilan (x1, y1, x2, y2)
        self.ukuran = None              # ukuran frame tampilan (w, h), untuk mengubah koordinat klik

    def ubah(self):
        """Mulai bila berhenti, berhenti bila sedang merekam."""
        if self.aktif:
            self.hentikan()
        else:
            self.aktif = True           # berkas dibuka saat frame pertama datang (ukuran frame diketahui)

    def tulis(self, frame, fps_perkiraan):
        if not self.aktif:
            return
        if self.writer is None:
            os.makedirs(self.folder, exist_ok=True)
            self.berkas = os.path.join(self.folder, "rekam_" + time.strftime("%Y%m%d_%H%M%S") + ".mp4")
            self.fps = min(60.0, max(1.0, fps_perkiraan))
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(self.berkas, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not writer.isOpened():
                print("Tidak bisa membuat berkas rekaman:", self.berkas)
                self.aktif = False
                return
            self.writer = writer
            self.frame_n = 0
            self.t_mulai = time.time()
            print("Mulai merekam:", self.berkas)
        self.writer.write(frame)
        self.frame_n += 1

    def hentikan(self):
        if self.writer is not None:
            self.writer.release()
            print(f"Rekaman selesai: {self.berkas} ({self.frame_n} frame, "
                  f"{self.frame_n / self.fps:.1f} detik)")
            self.pesan = "Tersimpan: " + os.path.join(os.path.basename(self.folder), os.path.basename(self.berkas))
            self.pesan_sampai = time.time() + 8
        self.writer = None
        self.aktif = False

    def durasi(self):
        return time.time() - self.t_mulai if self.writer is not None else 0.0


def _ke_frame(x, y, ukuran):
    """Ubah koordinat klik di jendela menjadi koordinat frame (jendela bisa diubah ukurannya)."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect(JUDUL)
    except cv2.error:
        return x, y
    if ww <= 0 or wh <= 0:
        return x, y
    return x * ukuran[0] / ww, y * ukuran[1] / wh


def gambar_tombol_rekam(img, rekam):
    """Tombol REC di pojok kanan atas. Merah = siap merekam. Kotak putih = sedang merekam (klik atau v untuk berhenti).
    Tombol hanya digambar di tampilan, tidak ikut direkam."""
    h, w = img.shape[:2]
    s = w / 640.0
    bw, bh = int(190 * s), int(30 * s)
    x1 = w - bw - int(8 * s)
    y1 = int(8 * s)
    x2, y2 = x1 + bw, y1 + bh
    rekam.rect = (x1, y1, x2, y2)
    rekam.ukuran = (w, h)

    warna_latar = (0, 0, 180) if rekam.aktif else (30, 30, 30)
    cv2.rectangle(img, (x1, y1), (x2, y2), warna_latar, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cx, cy, r = x1 + int(16 * s), y1 + bh // 2, max(3, int(6 * s))
    if rekam.aktif:
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), (255, 255, 255), -1)
        m, d = divmod(int(rekam.durasi()), 60)
        teks = f"STOP  {m:02d}:{d:02d}"
    else:
        cv2.circle(img, (cx, cy), r, (0, 0, 255), -1, cv2.LINE_AA)
        teks = "REC  (tekan v)"
    tulis_teks(img, teks, (x1 + int(30 * s), y1 + int(21 * s)), 0.5 * s, (255, 255, 255), 1)
    if rekam.pesan and time.time() < rekam.pesan_sampai:
        tulis_teks(img, rekam.pesan, (x1, y2 + int(16 * s)), 0.42 * s, (255, 255, 255), 1)


def saat_klik_rekam(event, x, y, flags, rekam):
    """Callback mouse: klik di dalam tombol REC untuk mulai / berhenti merekam."""
    if event != cv2.EVENT_LBUTTONDOWN or rekam.rect is None:
        return
    x, y = _ke_frame(x, y, rekam.ukuran)
    x1, y1, x2, y2 = rekam.rect
    if x1 <= x <= x2 and y1 <= y <= y2:
        rekam.ubah()


# ============================================================
#   MASUKAN: FOTO, VIDEO, DAN WEBCAM
# ============================================================
def tandai_rentang(p, waktu):
    """Tombol b: mulai rentang jalan bahu pada posisi video ini, atau akhiri rentang yang sedang dibuka."""
    if p.awal_rentang is None:
        p.awal_rentang = waktu
    else:
        if waktu > p.awal_rentang:
            p.rentang_bahu.append((p.awal_rentang, waktu))
        p.awal_rentang = None
    p.rentang_berubah = True


def jalur_file_default(source):
    """Berkas rentang untuk video, mis. video_jalan.mp4 -> video_jalan_jalur.json. None untuk webcam / foto."""
    if source.isdigit() or source.lower().endswith(IMAGE_EXT):
        return None
    return os.path.splitext(source)[0] + JALUR_FILE_SUFFIX


def info_video(source, cap):
    """Sidik jari video: ukuran berkas (byte) dan durasi (detik). Dipakai untuk memastikan berkas
    rentang memang milik video ini, bukan video lain yang kebetulan bernama sama."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    durasi = round(frames / fps, 1) if fps > 0 else 0.0
    ukuran = os.path.getsize(source) if os.path.isfile(source) else 0
    return {"ukuran_byte": ukuran, "durasi_detik": durasi}


def baca_rentang(path, info):
    """Membaca rentang jalan bahu dari berkas JSON. Berkas tidak ada -> daftar kosong.
    Bila sidik jari di berkas tidak sama dengan video ini, berkas diabaikan dengan peringatan."""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    catatan = data.get("info")
    if catatan is not None and info is not None:
        cocok = (catatan.get("ukuran_byte") == info["ukuran_byte"]
                 and abs(catatan.get("durasi_detik", -99) - info["durasi_detik"]) <= 1.0)
        if not cocok:
            print(f"PERINGATAN: {path} dibuat untuk video lain (ukuran/durasi berbeda). "
                  f"Rentang diabaikan. Buat berkas baru dengan tombol b atau hapus berkas ini.")
            return []
    return [(float(a), float(b)) for a, b in data.get("jalur3", [])]


def simpan_rentang(path, rentang, info):
    """Menyimpan rentang jalan bahu beserta sidik jari video ke berkas JSON."""
    if not path:
        return
    data = {"keterangan": "rentang waktu (detik) jalan bahu = 3 jalur. Dibuat dengan tombol b saat "
                          "menonton video. Di luar rentang ini, jumlah jalur otomatis.",
            "info": info,
            "jalur3": [[round(a, 2), round(b, 2)] for a, b in rentang]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Rentang bahu tersimpan:", path)


def ubah_lebar(frame, lebar):
    h0, w0 = frame.shape[:2]
    if w0 == lebar:
        return frame
    return cv2.resize(frame, (lebar, int(round(h0 * lebar / w0))), interpolation=cv2.INTER_AREA)


KALIBRASI_GARIS = ("tepi_kiri", "tengah", "tepi_kanan")
KALIBRASI_LABEL = {"tepi_kiri": "tepi kiri (biru)", "tengah": "garis tengah (kuning)",
                   "tepi_kanan": "tepi kanan (magenta)"}
KALIBRASI_WARNA = {"tepi_kiri": (255, 0, 0), "tengah": (0, 255, 255), "tepi_kanan": (255, 0, 255)}  # BGR


def kalibrasi_file_default(source):
    """Berkas kalibrasi: video_jalan2.mp4 -> video_jalan2_kalibrasi.json; webcam -> kalibrasi_webcam.json.
    None untuk foto."""
    if source.isdigit():
        return KALIBRASI_WEBCAM_FILE
    if source.lower().endswith(IMAGE_EXT):
        return None
    return os.path.splitext(source)[0] + "_kalibrasi.json"


def _x_di_baris(pts, yrel):
    """x garis lurus lewat dua titik (x, y) pada tinggi yrel (proporsi tinggi)."""
    (x1, y1), (x2, y2) = pts
    return x1 + (yrel - y1) * (x2 - x1) / (y2 - y1)


def kalibrasi_dari_klik(klik, w, h):
    """klik: 6 titik piksel frame yang diproses, berurutan: 2 titik tepi kiri, 2 garis tengah,
    2 tepi kanan. Disimpan sebagai proporsi lebar/tinggi frame, jadi tetap berlaku bila ukuran berubah."""
    garis = {}
    for i, nama in enumerate(KALIBRASI_GARIS):
        (x1, y1), (x2, y2) = klik[2 * i], klik[2 * i + 1]
        if abs(y1 - y2) < 0.05 * h:
            raise ValueError(f"{KALIBRASI_LABEL[nama]}: dua titik terlalu sejajar, pilih titik yang lebih berjauhan vertikal")
        garis[nama] = [[x1 / w, y1 / h], [x2 / w, y2 / h]]
    xs = {nama: _x_di_baris(garis[nama], Y_REF) for nama in KALIBRASI_GARIS}
    if not (xs["tepi_kiri"] < xs["tengah"] < xs["tepi_kanan"]):
        raise ValueError("urutan salah: tepi kiri harus di kiri garis tengah, dan garis tengah di kiri tepi kanan")
    return {"garis": garis}


def kalibrasi_interaktif(frame):
    """Klik 2 titik pada tiap garis (tepi kiri, garis tengah, tepi kanan), boleh di mana saja selama
    garis itu terlihat jelas. U = ulang titik terakhir, Enter / S = simpan, Esc = batal.
    Mengembalikan data kalibrasi, atau None bila dibatalkan."""
    h, w = frame.shape[:2]
    ya = int(round(ROI_ATAS * h))
    klik = []

    def saat_klik(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(klik) < 6:
            klik.append((x, y))

    cv2.namedWindow(JUDUL, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(JUDUL, saat_klik)
    pesan = "Klik 2 titik di tiap garis, pilih titik yang berjauhan secara vertikal."
    while True:
        img = frame.copy()
        cv2.line(img, (0, ya), (w - 1, ya), (255, 255, 255), 1, cv2.LINE_AA)   # batas atas ROI
        for i, nama in enumerate(KALIBRASI_GARIS):
            pts = klik[2 * i:2 * i + 2]
            for x, y in pts:
                cv2.circle(img, (x, y), 4, KALIBRASI_WARNA[nama], -1, cv2.LINE_AA)
            if len(pts) == 2 and abs(pts[1][1] - pts[0][1]) >= 3:
                (x1, y1), (x2, y2) = pts
                xb = x1 + (h - 1 - y1) * (x2 - x1) / (y2 - y1)
                xb = float(np.clip(xb, -w, 2 * w))
                cv2.line(img, (x1, y1), (int(round(xb)), h - 1), KALIBRASI_WARNA[nama], 1, cv2.LINE_AA)
        if len(klik) < 6:
            nama = KALIBRASI_GARIS[len(klik) // 2]
            teks = f"Klik {len(klik) + 1}/6: {KALIBRASI_LABEL[nama]}, titik {len(klik) % 2 + 1}"
        else:
            teks = "Enter = simpan   |   U = ulang titik terakhir   |   Esc = batal"
        tulis_teks(img, teks, (8, 22), 0.5, (255, 255, 255))
        if pesan:
            tulis_teks(img, pesan, (8, 44), 0.45, (0, 0, 255))
        cv2.imshow(JUDUL, img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(JUDUL)
            return None
        if key in (ord("u"), ord("U")) and klik:
            klik.pop()
            pesan = ""
        if key in (13, 10, ord("s")) and len(klik) == 6:
            try:
                data = kalibrasi_dari_klik(klik, w, h)
            except ValueError as e:
                pesan = str(e)
                continue
            cv2.destroyWindow(JUDUL)
            return data
        if cv2.getWindowProperty(JUDUL, cv2.WND_PROP_VISIBLE) < 1:
            return None


def garis_dari_kalibrasi(kal, w, h):
    """Dua titik tiap garis menjadi garis lurus (theta, dom) dalam sistem t, untuk rentang ROI."""
    dom = np.array([t_dari_y(ROI_ATAS * h, h), t_dari_y(h - 1, h)])
    hasil = {}
    for nama in KALIBRASI_GARIS:
        (xa, ya_), (xb, yb_) = kal["garis"][nama]
        x1, y1 = xa * w, ya_ * h
        x2, y2 = xb * w, yb_ * h
        t1 = t_dari_y(y1, h)
        t2 = t_dari_y(y2, h)
        B = (x1 - x2) / (t1 - t2)
        C = x1 - B * t1
        hasil[nama] = (np.array([0.0, B, C]), dom.copy())
    return hasil


def pakai_kalibrasi(p, w, h):
    """Garis mengikuti kalibrasi (tetap); marka tidak mengubah garis, hanya dipakai untuk jumlah jalur."""
    gl = garis_dari_kalibrasi(p.kalibrasi, w, h)
    for garis, nama in ((p.tepi_kiri, "tepi_kiri"), (p.tengah, "tengah"), (p.tepi_kanan, "tepi_kanan")):
        theta, dom = gl[nama]
        garis.theta = theta.copy()
        garis.dom = dom.copy()
        garis.segar = True
        garis.sumber = "kalibrasi"
        garis.hilang = 0
        garis.tolak = 0
    xk = p.tepi_kiri.x_pada([Y_REF * h], h)[0]
    xt = p.tengah.x_pada([Y_REF * h], h)[0]
    p.lebar_prop = (xt - xk) / w


def simpan_kalibrasi(path, data):
    """Menyimpan titik kalibrasi beserta sidik jari video."""
    isi = {"keterangan": "kalibrasi jalur (klik manual). Tiap garis: dua titik [x/lebar, y/tinggi].",
           "info": data.get("info"), "garis": data["garis"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(isi, f, ensure_ascii=False, indent=2)
    print("Kalibrasi tersimpan:", path)


def baca_kalibrasi(path, info):
    """Membaca kalibrasi. Berkas tidak ada atau milik video lain -> None (dengan peringatan)."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    catatan = data.get("info")
    if catatan is not None and info is not None:
        cocok = (catatan.get("ukuran_byte") == info["ukuran_byte"]
                 and abs(catatan.get("durasi_detik", -99) - info["durasi_detik"]) <= 1.0)
        if not cocok:
            print(f"PERINGATAN: {path} dibuat untuk video lain. Kalibrasi diabaikan. "
                  f"Jalankan dengan --kalibrasi untuk membuat yang baru.")
            return None
    return {"garis": data["garis"]}


def ambil_frame(source, detik):
    """Satu frame dari video pada detik tertentu (untuk kalibrasi)."""
    cap = cv2.VideoCapture(source)
    if detik > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, detik * 1000.0)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def proses_foto(path, args, p):
    """Foto tunggal: diproses berulang agar pelacak stabil, lalu ditampilkan / disimpan."""
    frame = cv2.imread(path)
    if frame is None:
        print("Tidak bisa membuka foto:", path)
        return
    frame = ubah_lebar(frame, args.lebar)
    frame = balik_frame(frame, args.cermin, args.balik)
    p.waktu = None
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
    p.fps = fps
    info = None if is_cam else info_video(args.source, cap)
    if args.jalur_file and not is_cam:
        p.rentang_bahu = baca_rentang(args.jalur_file, info)
    kal_file = kalibrasi_file_default(args.source)
    if args.kalibrasi:
        if args.tanpa_tampil:
            print("Kalibrasi memerlukan jendela tampilan (jangan pakai --tanpa-tampil).")
            cap.release()
            return
        if is_cam:
            ok0, frame0 = cap.read()         # webcam: pakai frame yang sedang tampil
            frame0 = frame0 if ok0 else None
        else:
            frame0 = ambil_frame(args.source, args.mulai)
        data = kalibrasi_interaktif(ubah_lebar(balik_frame(frame0, args.cermin, args.balik), args.lebar)) if frame0 is not None else None
        if data is None:
            print("Kalibrasi dibatalkan. Garis memakai mode otomatis.")
        else:
            data["info"] = info
            simpan_kalibrasi(kal_file, data)
            p.kalibrasi = data
    elif kal_file and not args.tanpa_kalibrasi:
        p.kalibrasi = baca_kalibrasi(kal_file, info)
    if p.kalibrasi is not None:
        print("Memakai kalibrasi jalur (garis tetap).")

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
    perekam = Perekam(os.path.join(os.path.dirname(os.path.abspath(__file__)), REKAM_FOLDER)) if is_cam else None      # tombol rekam: webcam saja
    if not args.tanpa_tampil:
        cv2.namedWindow(JUDUL, cv2.WINDOW_NORMAL)
        if perekam is not None:
            cv2.setMouseCallback(JUDUL, saat_klik_rekam, perekam)

    idx = 0
    berhenti = {"ya": False}             # diisi bila Ctrl+C ditekan
    def _ctrl_c(sig, bingkai):
        berhenti["ya"] = True
    signal.signal(signal.SIGINT, _ctrl_c)
    tersimpan = 0
    n_mode = {2: 0, 3: 0}
    n_status = Counter()
    t_mulai = time.perf_counter()
    t_frame = None
    fps_ukur = None                     # fps yang diukur dari laju proses (dipakai untuk berkas rekaman)

    while True:
        ok, frame = cap.read()
        if not ok or berhenti["ya"]:
            break
        now = time.perf_counter()
        if t_frame is not None and now > t_frame:
            sesaat = 1.0 / (now - t_frame)
            fps_ukur = sesaat if fps_ukur is None else 0.9 * fps_ukur + 0.1 * sesaat
        t_frame = now

        waktu = mulai_pem + idx / fps
        idx += 1
        if batas_waktu is not None and waktu >= batas_waktu:
            break

        frame = ubah_lebar(frame, args.lebar)
        frame = balik_frame(frame, args.cermin, args.balik)
        p.waktu = waktu if not is_cam else None
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
        if perekam is not None:
            perekam.tulis(out, fps_ukur or fps)      # sebelum tombol digambar, agar tombol tidak ikut terekam
        if koor is not None:
            koor.write(json.dumps(ke_json(hasil, waktu), ensure_ascii=False) + "\n")

        if not args.tanpa_tampil:
            if perekam is not None:
                gambar_tombol_rekam(out, perekam)
            cv2.imshow(JUDUL, out)
            if not is_cam:                             # video file: ikuti kecepatan asli
                tunggu = (idx - n_pemanasan) / fps - (time.perf_counter() - t_mulai)
                if tunggu > 0:
                    time.sleep(tunggu)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(JUDUL, cv2.WND_PROP_VISIBLE) < 1:
                break                            # jendela ditutup dengan tombol X: rekaman tetap disimpan
            if key == ord("q"):
                break
            if key == ord("d"):
                args.debug = not args.debug
            if key == ord("c"):
                p.tampil_koordinat = not p.tampil_koordinat
            if key == ord("r"):
                p.tampil_roi = not p.tampil_roi
            if key == ord("b") and not is_cam:
                tandai_rentang(p, waktu)
            if key == ord("v") and perekam is not None:
                perekam.ubah()
            if key == ord("a"):
                p.jalur = "auto"
            if key == ord("2"):
                p.jalur = "2"
            if key == ord("3"):
                p.jalur = "3"

    if p.awal_rentang is not None:                   # rentang yang belum ditutup: tutup di akhir
        tandai_rentang(p, mulai_pem + idx / fps)
    if p.rentang_berubah and args.jalur_file and not is_cam:
        simpan_rentang(args.jalur_file, p.rentang_bahu, info)

    if perekam is not None:
        perekam.hentikan()                           # rekaman yang masih berjalan disimpan dengan benar
    cap.release()
    if writer is not None:
        writer.release()
    if koor is not None:
        koor.close()
    if not args.tanpa_tampil:
        cv2.destroyAllWindows()

    print(f"Frame diproses: {tersimpan} | jalur 2: {n_mode[2]} | jalur 3: {n_mode[3]} | "
          f"status: {dict(n_status)}")
    if args.save and tersimpan:
        print("Video hasil tersimpan:", args.save)
    if args.koordinat:
        print("Koordinat tersimpan:", args.koordinat)


def balik_frame(frame, cermin, balik):
    """cermin: balik kiri-kanan (gambar seperti cermin). balik: balik atas-bawah (kamera terpasang terbalik).
    Keduanya bersamaan = diputar 180 derajat."""
    if cermin:
        frame = cv2.flip(frame, 1)
    if balik:
        frame = cv2.flip(frame, 0)
    return frame


def gambar_panduan(frame):
    """Gambar bantu pemasangan: kotak ROI (40% bawah), garis cakrawala, garis acuan, dan garis tengah."""
    h, w = frame.shape[:2]
    s = w / 640.0
    y0 = int(round(ROI_ATAS * h))
    yh = int(round(Y_HORIZON * h))
    yr = int(round(Y_REF * h))
    out = frame.copy()
    kotak = out.copy()
    cv2.rectangle(kotak, (0, y0), (w - 1, h - 1), WARNA_ROI, -1)
    out = cv2.addWeighted(kotak, 0.15, out, 0.85, 0)
    cv2.rectangle(out, (0, y0), (w - 1, h - 1), WARNA_ROI, 2, cv2.LINE_AA)
    cv2.line(out, (0, yh), (w - 1, yh), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(out, (0, yr), (w - 1, yr), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(out, (w // 2, y0), (w // 2, h - 1), (0, 255, 255), 1, cv2.LINE_AA)
    fs = 0.42 * s
    tulis_teks(out, "cakrawala (putih)", (int(8 * s), yh - int(6 * s)), fs, (255, 255, 255), 1)
    tulis_teks(out, "acuan (hijau): marka kiri-kanan di sini", (int(8 * s), yr - int(6 * s)), fs, (0, 255, 0), 1)
    tulis_teks(out, "tengah mobil (kuning)", (w // 2 + int(6 * s), h - int(10 * s)), fs, (0, 255, 255), 1)
    tulis_teks(out, "Jalan di dalam kotak cyan. q = selesai", (int(8 * s), int(22 * s)), 0.5 * s, (255, 255, 255))
    return out


def mode_pasang(args):
    """Bantu memasang kamera: tampilkan kotak ROI, cakrawala, dan garis acuan tanpa analisis. q = selesai."""
    cap = cv2.VideoCapture(int(args.source))
    if not cap.isOpened():
        print("Tidak bisa membuka kamera:", args.source)
        return
    cv2.namedWindow(JUDUL, cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = balik_frame(frame, args.cermin, args.balik)
        cv2.imshow(JUDUL, gambar_panduan(ubah_lebar(frame, args.lebar)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(JUDUL, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    global Y_HORIZON                     # bisa diubah dengan --horizon
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
    ap.add_argument("--jalur-file", default="",
                    help="berkas JSON rentang jalan bahu (3 jalur). Default: <nama video>_jalur.json")
    ap.add_argument("--kalibrasi", action="store_true",
                    help="klik manual 6 titik garis di frame awal. Simpan <nama video>_kalibrasi.json (video) "
                         "atau kalibrasi_webcam.json (webcam)")
    ap.add_argument("--tanpa-kalibrasi", action="store_true",
                    help="(video) abaikan berkas kalibrasi dan pakai deteksi otomatis")
    ap.add_argument("--horizon", type=float, default=Y_HORIZON,
                    help="posisi cakrawala sebagai proporsi tinggi frame dari atas (default 0.48). "
                         "Webcam di dalam mobil biasanya sekitar 0.55-0.62")
    ap.add_argument("--cermin", action="store_true",
                    help="balik gambar kiri-kanan (kalau gambar seperti cermin, jalur kiri dan kanan tertukar)")
    ap.add_argument("--balik", action="store_true",
                    help="balik gambar atas-bawah (kalau kamera terpasang terbalik). Dengan --cermin = diputar 180 derajat")
    ap.add_argument("--pasang", action="store_true",
                    help="(webcam) bantu pasang kamera: tampilkan kotak ROI, cakrawala, dan garis acuan tanpa analisis")
    args = ap.parse_args()
    if not (0.2 <= args.horizon < Y_REF - 0.05):
        ap.error("--horizon harus antara 0.20 dan 0.75")
    Y_HORIZON = args.horizon
    if args.pasang:
        if not args.source.isdigit():
            ap.error("--pasang hanya untuk webcam (--source 0)")
        mode_pasang(args)
        return

    p = Pelacak(jalur=args.jalur)
    p.tampil_koordinat = not args.tanpa_koordinat
    p.tampil_roi = args.tampil_roi
    if args.source.lower().endswith(IMAGE_EXT) or args.source.isdigit():
        args.jalur_file = ""
    elif not args.jalur_file:
        args.jalur_file = jalur_file_default(args.source)
    if args.source.lower().endswith(IMAGE_EXT):
        proses_foto(args.source, args, p)
    else:
        proses_video(args, p)


if __name__ == "__main__":
    main()