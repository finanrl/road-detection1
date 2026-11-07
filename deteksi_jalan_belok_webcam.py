"""
Deteksi Jalur Jalan Lurus & Belok - Webcam / Video / Foto  (versi 15)
====================================================================
Alur:
  Input -> [kalibrasi ROI otomatis per video] -> Grayscale -> [CLAHE opsional]
  -> Gaussian Blur -> Canny -> ROI (satu jalan penuh)
  -> filter kecerahan/marka (marka kiri lebih ketat) -> HoughLinesP
  -> klasifikasi kiri/kanan (gating) -> pas-kan ke marka (sempit untuk kiri)
  -> fit kurva -> uji kewajaran terhadap kurva lama -> EMA
  -> validasi lebar jalur -> koordinat FINAL dengan fallback
     (deteksi / turunan / prediksi riwayat / hilang)
  -> Kalman filter pada tiap titik koordinat
  -> posisi kendaraan di jalur kiri (dari bahu sampai garis tengah)
  -> deteksi minggir ke kiri (offset + histeresis)
  -> gambar dengan warna per bagian, legenda, dan angka koordinat
  -> pemutaran video: pause, ulang, geser posisi, pilih segmen (I/O),
     dan rekam segmen ke file MP4 (--save)
  -> webcam: rekam ke file MP4 dengan tombol REKAM (klik atau tekan V).
     Setiap rekaman otomatis tersimpan sebagai <nama>_TANGGAL_JAM.mp4

Perubahan v15:
  - Garis bahu (kiri) kembali dikunci ke MARKA, bukan ke tepi aspal.
  - Ambang marka untuk sisi kiri lebih ketat (MARKA_KIRI_PUTIH_V/S), supaya batu
    dan beton terang tidak lolos sebagai marka.
  - Pas-kan garis kiri hanya mencari marka dalam jarak sempit dari prediksi
    (SNAP_WIN_KIRI_FRAC), supaya tidak tertarik ke batu yang jauh.
  - Tepi bahu dari Hough tidak lagi dipakai sebagai cadangan (LEFT_FALLBACK_HOUGH = False).
    Bila marka kiri tidak terlihat, garis kiri memakai riwayat lalu perkiraan.
  - Scan tepi aspal dari v14 tetap ada, tetapi tidak aktif secara default.
    Aktifkan dengan --bahu-scan.
  - Stage 'Mask marka kiri (ketat)' ditambahkan pada mode -p / --stages.

Perubahan v14 (tetap berlaku):
  - Scan tepi aspal per baris (BAHU_SCAN, opsi --bahu-scan).
  - Mode debug (tombol d) menampilkan titik scan bahu (merah).
  - Stage 'Mask aspal (tepi bahu)' untuk mode -p / --stages.

Perubahan v13 (tetap berlaku):
  - Garis kiri lebih halus (EMA dan Kalman dengan parameter sendiri)
  - Uji kewajaran: kurva baru harus dekat dengan kurva lama (WAJAR_TOL)
  - Garis kiri tetap terlacak saat mobil di bahu (KIRI_X_MAX_LANJUT)
  - Posisi kendaraan di jalur kiri (label_posisi), tampil di panel dan di JSON
  - Riwayat garis lebih tahan lama (MAX_MISSING = 60)

ROI otomatis: titik hilang jalan (perpotongan garis kiri dan kanan) dicari dari
beberapa frame awal, lalu ROI dibuat mulai tepat di bawah titik itu. Kalibrasi
dilakukan sekali per video. Gunakan --roi-atas untuk mengatur ROI secara manual,
atau --tanpa-roi-otomatis untuk memakai ROI default.

Mode default (--sisi dua): bahu jalan (tepi kiri, biru) dan garis tengah (kuning), dengan jalur
(hijau) di antaranya, yaitu area jalur kiri (lebar dari --lebar-jalur). Garis yang tidak terlihat
diperkirakan dari garis lainnya dan lebar jalur.

Warna pada gambar:
  biru   = bahu jalan (tepi kiri, dikunci ke marka)
  kuning = garis tengah jalan
  hijau  = jalur (oranye saat minggir ke kiri)
  putih  = batas kanan jalur (perkiraan) dan batas ROI
  merah  = titik scan bahu (hanya mode debug, tombol d, dan hanya bila --bahu-scan)

Cara pakai:
  python deteksi_jalan_belok_webcam.py --source 0
  python deteksi_jalan_belok_webcam.py --source 0 --save rekaman/jalan
  python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --koordinat koordinat.jsonl
  python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --save segmen.mp4
  python deteksi_jalan_belok_webcam.py --source jalan.jpg --stages --koordinat koordinat.jsonl
  python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --lebar-jalur 0.42

Pembanding (untuk laporan / presentasi):
  --tanpa-kalman  --tanpa-validasi-lebar  --ransac  --clahe  --tanpa-angka
  --tanpa-roi-otomatis  --bahu-scan

Tombol:
  q = keluar             | d = debug              | c = angka koordinat on/off
  s = snapshot           | p = hasil per tahap
  2 = mode 2 jalur      | 3 = mode 3 jalur (hanya garis tengah)
  (khusus webcam)
  v = rekam mulai/stop   | atau klik tombol REKAM di tengah bawah layar
  (khusus video file)
  SPASI = pause/lanjut   | r = ulang dari awal segmen | bar Frame = geser posisi
  i = tandai awal segmen | o = tandai akhir segmen    | x = reset segmen
"""

import argparse
import copy       # deepcopy state (hasil per tahap pada video)
import json       # menulis koordinat ke file JSON
import os         # membuat folder penyimpanan
import re         # membersihkan nama file tahap
import threading  # membaca kamera di thread terpisah (anti-delay)
import time       # pengatur kecepatan video & hitung FPS
from collections import deque  # riwayat deteksi untuk fallback

import cv2
import numpy as np

# ============================ PARAMETER ============================
PROCESS_WIDTH = 640

# --- Pra-pemrosesan: Grayscale -> [CLAHE] -> Gaussian Blur -> Canny
USE_CLAHE = False                  # True = perkuat kontras marka pudar (opsi --clahe)
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
BLUR_KSIZE = 5                     # harus ganjil (3, 5, 7, ...)
CANNY_LOW, CANNY_HIGH = 50, 150

# --- ROI: bagian depan jalan (proporsi tinggi gambar). ROI_Y_ATAS makin besar = ROI makin
#     pendek (lebih dekat ke kendaraan, gangguan dari jauh berkurang, tikungan terlihat lebih sedikit)
#     Nilai ini hanya default; saat berjalan, ROI dikalibrasi otomatis dari video (lihat di bawah).
Y_BOTTOM = 0.92
ROI_Y_ATAS = 0.60
_dx_roi = 0.5 * (Y_BOTTOM - ROI_Y_ATAS)        # kemiringan sisi ROI sama dengan versi sebelumnya
ROI_PROP = np.float32([[0.00, Y_BOTTOM], [_dx_roi, ROI_Y_ATAS],
                       [1.0 - _dx_roi, ROI_Y_ATAS], [1.00, Y_BOTTOM]])
Y_TOP = ROI_Y_ATAS
Y_REF = 0.88                       # tinggi acuan untuk cek lompatan & offset

# --- ROI otomatis per video (dikalibrasi dari beberapa frame)
ROI_KALIB_FRAMES = 30          # jumlah frame untuk kalibrasi (tersebar merata di video)
ROI_KALIB_MIN_FRAME = 8        # minimal frame berpasangan agar hasil kalibrasi dipakai
ROI_KALIB_Y_MIN = 0.52         # batas atas area pencarian garis (proporsi tinggi); tepi jauh dibuang
ROI_KALIB_MIN_LEN = 0.10       # segmen kalibrasi minimal 10% tinggi gambar
ROI_KALIB_OUTLIER = 0.03       # frame dengan titik hilang menyimpang > 3% tinggi dari median dibuang
ROI_HILANG_MIN = 0.40          # rentang wajar titik hilang jalan (proporsi tinggi)
ROI_HILANG_MAX = 0.50
ROI_TOP_FRAC = 0.25            # ROI atas = titik hilang + 25% jarak ke dasar
ROI_TOP_MIN = 0.60             # ROI atas tidak boleh lebih tinggi dari default lama (0.60)
ROI_TOP_MAX = 0.70
ROI_MARGIN = 0.08              # ruang tambahan di kiri & kanan garis pada sisi atas ROI (proporsi lebar)

# --- Warna (format BGR)
WARNA_GARIS_KIRI   = (255, 0, 0)       # biru: bahu jalan (tepi kiri)
WARNA_GARIS_KANAN  = (0, 255, 255)     # kuning: garis tengah jalan
WARNA_JALUR        = (0, 200, 0)       # hijau: area jalur normal
WARNA_MINGGIR_KIRI = (0, 140, 255)     # oranye: jalur saat minggir ke kiri
WARNA_ROI          = (255, 255, 255)   # putih: batas ROI

# --- Deteksi minggir ke kiri (meter; negatif = kendaraan condong ke kiri)
MINGGIR_ON_M = -0.50               # aktif jika offset < -0.50 m
MINGGIR_OFF_M = -0.30              # nonaktif jika offset > -0.30 m (histeresis)

# --- HoughLinesP
HOUGH_THRESH = 25
HOUGH_MIN_LEN = 30
HOUGH_MAX_GAP = 40
MIN_ABS_SLOPE = 0.35
BRIGHT_DELTA = 35                  # 0 = brightness filter dimatikan
BRIGHT_FRACTION = 0.4
BRIGHT_DILATE = 7

# --- Gating & fit
GATE_TOL = 0.07                    # toleransi gating saat garis baru hilang / belum stabil
GATE_TOL_STABIL = 0.04             # toleransi lebih ketat saat garis stabil (menolak garis pengganggu)
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

# --- RANSAC (fit tahan outlier). Default MATI: pada uji sintetis hasilnya campur
#     (kadang membantu, kadang tidak). Aktifkan dengan --ransac bila ingin dicoba.
USE_RANSAC = False
RANSAC_ITER = 40                   # jumlah percobaan model acak
RANSAC_THR_FRAC = 0.014            # toleransi titik dianggap "cocok" (proporsi lebar gambar)
RANSAC_MAX_CURVE = 0.08            # kelengkungan maksimum yang masuk akal (proporsi lebar)

# ===================== PERBAIKAN (v13) =====================
# 1) Garis kiri lebih halus
ALPHA_KIRI = 0.10            # EMA garis kiri (lebih kecil = lebih halus, tapi lebih telat)
ALPHA_KANAN = 0.14
KF_Q_POS_KIRI = 0.0004       # Kalman garis kiri (lebih kecil dari KF_Q_POS = 0.0008)

# 2) Pas-kan garis ke marka yang sebenarnya
SNAP_MARKA = True            # False = matikan fitur pas-kan sepenuhnya
SNAP_ROWS = 14               # jumlah baris yang dicek antara Y_TOP dan Y_BOTTOM
SNAP_WIN_FRAC = 0.035        # setengah lebar jendela pencarian marka garis tengah (proporsi lebar)

# 3) Uji kewajaran: kurva baru harus dekat dengan kurva lama
WAJAR_TOL = 0.035            # rata-rata selisih kurva (proporsi lebar)

# 4) Mobil di bahu kiri: batas x garis yang sudah terlacak dilonggarkan
KIRI_X_MAX_LANJUT = 0.85
POS_BAHU = 0.25              # batas "dekat bahu" (proporsi lebar jalur)
# ===========================================================

# ===================== PERBAIKAN (v14) =====================
BAHU_SCAN = False            # v15: nonaktif. True = garis bahu dari tepi aspal (opsi --bahu-scan)
BAHU_ROWS = 14               # jumlah baris yang di-scan antara Y_TOP dan Y_BOTTOM
BAHU_WIN_FRAC = 0.12         # setengah lebar jendela pencarian di sekitar prediksi bahu
ASPAL_S_MAX = 60             # saturasi maksimum aspal (rumput hijau biasanya > 80)
ASPAL_V_MIN = 45             # kecerahan minimum aspal (bayangan jalan tetap dihitung aspal)
ASPAL_V_MAX = 210            # kecerahan maksimum aspal (batu/beton sangat terang dikecualikan)
ASPAL_CLOSE_W = 9            # celah horizontal (px) yang dijembatani (putus-putus marka)
# ===========================================================

# ===================== PERBAIKAN (v15) =====================
SNAP_KIRI = True             # v15: garis bahu dikunci ke marka kiri (pas-kan aktif lagi)
SNAP_WIN_KIRI_FRAC = 0.02    # v15: jendela pencarian marka kiri, lebih sempit dari garis tengah
MARKA_KIRI_PUTIH_V = 200     # v15: kecerahan minimum marka putih kiri (lebih ketat dari MARKA_PUTIH_V)
MARKA_KIRI_PUTIH_S = 50      # v15: saturasi maksimum marka putih kiri
LEFT_FALLBACK_HOUGH = False  # v15: True = bila marka kiri tidak cukup, pakai tepi bahu dari Hough
# ===========================================================

# --- Smoothing EMA pada parameter garis (tahap awal)
ALPHA = 0.14                  # perubahan garis lebih halus antar-frame
MAX_JUMP = 0.08               # (tidak lagi dipakai sejak v13; diganti WAJAR_TOL)
RELOCK_AFTER = 6
MAX_MISSING = 60              # v13: sebelumnya 20

# --- Arah belok
BEND_ON, BEND_OFF = 0.018, 0.010   # hysteresis: ON > OFF
BEND_EMA = 0.15
CURVE_SIGN = 1                     # ubah ke -1 jika arah kiri/kanan terbalik
LANE_WIDTH_M = 3.5
OFFSET_OK_M = 0.8

# --- Sisi yang dideteksi: "dua" (default: bahu kiri + garis tengah = jalur kiri), "kiri" (hanya bahu), atau "kanan"
SISI = "dua"
JALUR_MODE = 2               # 2 = dua jalur (bahu kiri + garis kanan, jalur di antaranya); 3 = tiga jalur (hanya garis tengah, jalur di kanannya)
KIRI_X_MAX = 0.55        # mode kiri: garis kiri harus di kiri tengah kamera (proporsi lebar)
KIRI_PILIH_TOL = 0.03    # mode kiri saat belum ada garis: pilih kandidat dekat tengah, dalam jarak ini
KIRI_TOL = 0.04               # mode kiri: segmen dianggap garis yang sama bila posisinya dalam jarak ini (proporsi lebar)
KIRI_KELOMPOK_TOL = 0.03      # segmen dengan posisi berdekatan di garis acuan dianggap satu garis (proporsi lebar)
KIRI_MIN_PANJANG = 0.08       # kelompok segmen dengan total panjang di bawah nilai ini x tinggi gambar dibuang
KIRI_PINDAH = 0.10            # mode kiri: kelompok di sisi tengah dari garis terkunci, dalam jarak ini
KIRI_PINDAH_FRAME = 12        # v13: sebelumnya 5. ...yang muncul terus-menerus selama sekian frame -> pindah ke situ
DERIVASI_SISI_LAIN = True      # True = garis yang tidak terlihat diperkirakan dari sisi lain + lebar jalur (status 'turunan')

# --- Marka jalan: hanya garis berwarna marka (putih/kuning) yang dipakai.
#     Tepi aspal-rumput (bahu) tidak lolos karena rumput berwarna hijau, aspal kelabu.
MARKA_WAJIB = True          # False = garis apa saja boleh dipakai (perilaku lama)
MARKA_PUTIH_V = 170         # kecerahan minimum marka putih (garis tengah)
MARKA_PUTIH_S = 60          # saturasi maksimum marka putih (garis tengah)
MARKA_KUNING_V = 150        # kecerahan minimum marka kuning
MARKA_KUNING_S = 120        # saturasi minimum marka kuning (rumput kering juga bisa kuning, naikkan bila salah)
MARKA_FRAC = 0.5            # bagian titik segmen yang harus berada di atas marka
BAHU_CADANGAN = True        # True = brightness filter dimatikan (perilaku v12-v14). Lihat LEFT_FALLBACK_HOUGH

# --- Output koordinat & fallback
N_TITIK = 6                        # jumlah titik per garis (dari Y_TOP ke Y_BOTTOM)
HIST_LEN = 15                      # riwayat deteksi valid untuk prediksi yang stabil
HOLD_MAX_FRAMES = 40               # frame maksimum memakai estimasi sebelum HILANG
DECAY = 0.92                       # tren prediksi diredam bertahap (0-1)
LEBAR_PROP_DEFAULT = 0.45          # tebakan awal lebar jalur (proporsi lebar gambar di Y_REF)
LEBAR_ALPHA = 0.10                 # kecepatan kalibrasi lebar jalur (EMA)
Y_LEVELS = np.linspace(Y_TOP, Y_BOTTOM, N_TITIK)

# --- Kalman filter (smoothing tahap akhir pada tiap titik koordinat)
USE_KALMAN = True
KF_Q_POS = 0.0008                  # ketidakpastian proses posisi (proporsi lebar) per frame
KF_Q_VEL = 0.0001                  # ketidakpastian proses kecepatan (proporsi lebar)
KF_VEL_DECAY = 0.90                # kecepatan diredam tiap frame (0-1)
KF_R_DETEKSI = 0.007               # derau ukur saat deteksi langsung (proporsi lebar)
KF_R_TURUNAN = 0.012               # derau ukur saat garis diturunkan dari sisi lain
KF_R_ESTIMASI = 0.030              # derau ukur saat estimasi dari riwayat
KF_GATE = 3.0                      # ukuran loncatan (x sigma) sebelum ukuran dicurigai

# --- Validasi lebar jalur (konsistensi geometri kiri-kanan)
VALIDASI_LEBAR = False
LEBAR_DARI_KALIBRASI = True      # True = lebar jalur dipelajari dari frame yang kedua garisnya terlihat
LEBAR_TOL = 0.30                   # lebar baru boleh menyimpang +-30% dari lebar terkalibrasi
LEBAR_N_MIN = 8                    # jumlah pasangan valid sebelum lebar dianggap terkalibrasi
LEBAR_BAD_MAX = 25                 # frame tidak konsisten berturut-turut sebelum kalibrasi diulang
LEBAR_WIN_MIN, LEBAR_WIN_MAX = 0.55, 1.60   # jendela posisi garis yang baru dicari

# --- Tampilan angka koordinat pada gambar
SHOW_COORDS = True

# --- Rekam webcam
REC_FPS = 20                       # fps video hasil rekaman webcam
# ===================================================================

_RNG = np.random.default_rng(12345)   # acak tetap -> hasil RANSAC bisa diulang


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


def _ransac_inliers(t, xs, w, prior=None):
    """RANSAC: coba banyak model (garis / parabola) dari 3 titik acak yang
    tersebar di bagian bawah, tengah, dan atas data. Model dengan titik 'cocok'
    terbanyak menang; model yang dekat dengan garis frame lalu diberi bonus.
    Mengembalikan mask boolean titik inlier."""
    n = len(t)
    thr = RANSAC_THR_FRAC * w
    order = np.argsort(t)
    groups = np.array_split(order, 3)
    if any(len(g) == 0 for g in groups):
        return np.ones(n, dtype=bool)
    t_top = Y_TOP - Y_BOTTOM
    max_a = RANSAC_MAX_CURVE * w / (t_top ** 2)

    r = _RNG.random((RANSAC_ITER, 3))
    pick = [groups[k][(r[:, k] * len(groups[k])).astype(int)] for k in range(3)]
    t_mid = float(t.mean())
    best_mask, best_score = None, -1.0

    for it in range(RANSAC_ITER):
        i0, i1, i2 = pick[0][it], pick[1][it], pick[2][it]
        ta, tb, tc = t[i0], t[i1], t[i2]
        xa, xb, xc = xs[i0], xs[i1], xs[i2]
        if it % 2 == 0:                       # model garis lurus (A = 0)
            dt = tc - ta
            if abs(dt) < 1e-3:
                continue
            a = 0.0
            b = (xc - xa) / dt
            c = xa - b * ta
        else:                                 # model parabola
            m = np.array([[ta * ta, ta, 1.0], [tb * tb, tb, 1.0], [tc * tc, tc, 1.0]])
            try:
                a, b, c = np.linalg.solve(m, np.array([xa, xb, xc]))
            except np.linalg.LinAlgError:
                continue
            if abs(a) > max_a:                # kelengkungan tidak masuk akal
                continue
        mask = np.abs(xs - (a * t * t + b * t + c)) < thr
        score = float(mask.sum())
        if prior is not None:                 # bonus: konsisten dengan frame lalu
            px = prior[0] * t_mid ** 2 + prior[1] * t_mid + prior[2]
            mx = a * t_mid ** 2 + b * t_mid + c
            if abs(px - mx) < 3 * thr:
                score += 0.3 * n
        if score > best_score:
            best_score, best_mask = score, mask
    return best_mask if best_mask is not None else np.ones(n, dtype=bool)


def fit_lane(xs, ys, w, h, side=None, prior=None):
    """Fit x = A t^2 + B t + C: RANSAC -> ridge berbobot -> buang pencilan (MAD).
    Mengembalikan (theta, domain) atau None kalau data tidak cukup."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < MIN_SAMPLES:
        return None
    t = to_t(ys, h)

    keep = np.ones(len(xs), dtype=bool)
    if USE_RANSAC and len(xs) >= 2 * MIN_SAMPLES:
        k = _ransac_inliers(t, xs, w, prior)
        if k.sum() >= MIN_SAMPLES:
            keep = k

    span = t[keep].max() - t[keep].min()     # rentang vertikal titik yang dipercaya
    if span < MIN_SPAN:
        return None
    use_curve = span >= CURVE_MIN_SPAN

    # Data pendek + sudah ada garis lama: kemiringan & lengkung dikunci dari
    # garis lama, hanya posisi (C) yang dicari.
    if prior is not None and span < SHORT_SPAN:
        A0, B0 = float(prior[0]), float(prior[1])
        tk = t[keep]
        c = float(np.median(xs[keep] - (A0 * tk ** 2 + B0 * tk)))
        fit = (np.array([A0, B0, c]), (float(tk.min()), float(tk.max())))
        return _arah_ok(fit, side)

    t_top = Y_TOP - Y_BOTTOM
    sigma_a = CURVE_PRIOR_FRAC * w / (t_top ** 2)
    ridge = (NOISE_FRAC * w / sigma_a) ** 2

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


class KalmanLevels:
    """Kalman filter (model kecepatan-konstan) untuk N titik posisi x sekaligus.
    Tiap titik punya state [posisi, kecepatan]. Derau ukur (r_std) boleh beda
    untuk tiap titik dan tiap frame: ukuran yang tidak pasti -> dipercaya sedikit.
    q_pos (v13) memungkinkan parameter proses berbeda per sisi (kiri lebih halus)."""

    def __init__(self, n, q_pos=None):
        self.n = n
        self.q_pos = KF_Q_POS if q_pos is None else q_pos   # v13
        self.aktif = False
        self.x = self.v = None
        self.p00 = self.p01 = self.p10 = self.p11 = None

    def reset(self):
        self.aktif = False

    def langkah(self, z, r_std, w, decay=None):
        """z = ukuran posisi (N,), r_std = simpangan baku ukur dalam pixel (N,).
        decay = peredaman kecepatan (None -> KF_VEL_DECAY). Mengembalikan posisi halus."""
        d = KF_VEL_DECAY if decay is None else decay
        z = np.asarray(z, dtype=np.float64)
        r = np.asarray(r_std, dtype=np.float64) ** 2
        qx = (self.q_pos * w) ** 2            # v13: pakai q_pos per instance
        qv = (KF_Q_VEL * w) ** 2

        if not self.aktif:                   # inisialisasi dari ukuran pertama
            self.x = z.copy()
            self.v = np.zeros(self.n)
            self.p00 = r.copy()
            self.p01 = np.zeros(self.n)
            self.p10 = np.zeros(self.n)
            self.p11 = np.full(self.n, (5.0 * KF_Q_VEL * w) ** 2)
            self.aktif = True
            return self.x.copy()

        # --- prediksi: x' = x + v ; v' = d*v ; P' = F P F^T + Q
        x = self.x + self.v
        v = d * self.v
        p00 = self.p00 + self.p01 + self.p10 + self.p11 + qx
        p01 = d * (self.p01 + self.p11)
        p10 = d * (self.p10 + self.p11)
        p11 = d * d * self.p11 + qv

        # --- koreksi (dengan penjaga loncatan: ukuran aneh -> derau diperbesar)
        y = z - x
        s0 = p00 + r
        infl = np.maximum(1.0, y * y / (KF_GATE ** 2 * s0))
        s = p00 + r * infl
        k0 = p00 / s
        k1 = p10 / s
        self.x = x + k0 * y
        self.v = v + k1 * y
        self.p00 = (1.0 - k0) * p00
        self.p01 = (1.0 - k0) * p01
        self.p10 = p10 - k1 * p00
        self.p11 = p11 - k1 * p01
        return self.x.copy()


class LaneTrack:
    """Satu sisi garis: EMA + uji kewajaran terhadap kurva lama + persistensi.
    Saat tidak terdeteksi, kurva lama tetap disimpan sebagai acuan (v13)."""

    def __init__(self, alpha=ALPHA):
        self.alpha = alpha          # v13: alpha per sisi
        self.theta = None           # parameter [A, B, C]
        self.dom = None             # rentang t yang punya data
        self.missing = 0            # frame berturut-turut gagal
        self.rejects = 0            # penolakan berturut-turut
        self.dalam = 0              # frame berturut-turut ada segmen di sisi tengah (mode kiri)
        self.fresh = False          # True jika frame INI diterima dari deteksi

    def reset(self):
        """Buang garis ini (misalnya karena tidak konsisten dengan lebar jalur)."""
        self.theta, self.dom = None, None
        self.missing, self.rejects, self.dalam = 0, 0, 0
        self.fresh = False

    def x_at(self, y, h):
        """Posisi x garis pada ketinggian y."""
        if self.theta is None:
            return None
        return float(eval_curve(self.theta, self.dom, to_t(y, h)))

    def _wajar(self, theta, dom, w, h):
        """Kurva baru harus cocok dengan kurva lama di seluruh tinggi gambar."""
        t = to_t(np.linspace(Y_TOP * h, Y_BOTTOM * h, 8), h)
        beda = eval_curve(theta, dom, t) - eval_curve(self.theta, self.dom, t)
        return float(np.mean(np.abs(beda))) < WAJAR_TOL * w

    def update(self, fit, w, h):
        """Gabungkan hasil fitting frame baru dengan garis lama."""
        self.fresh = False
        if fit is None:
            self._miss()
            return
        theta, dom = fit
        dom = np.asarray(dom, dtype=np.float64)
        if self.theta is None:
            self.theta, self.dom = theta, dom
            self.missing = self.rejects = 0
            self.fresh = True
            return
        if not self._wajar(theta, dom, w, h):
            # Dianggap objek/tepi sesaat. Diterima hanya bila konsisten
            # selama RELOCK_AFTER frame (berarti garis memang pindah).
            self.rejects += 1
            if self.rejects >= RELOCK_AFTER:
                self.theta, self.dom = theta, dom
                self.missing = self.rejects = 0
                self.fresh = True
            else:
                self._miss()
            return
        self.theta = self.alpha * theta + (1 - self.alpha) * self.theta
        self.dom = self.alpha * dom + (1 - self.alpha) * self.dom
        self.missing = self.rejects = 0
        self.fresh = True

    def _miss(self):
        """Catat kegagalan; hapus garis jika gagal terlalu lama."""
        self.missing += 1
        if self.missing > MAX_MISSING:
            self.theta, self.dom, self.rejects = None, None, 0


class LaneState:
    """Seluruh kondisi deteksi yang dibawa dari frame ke frame."""

    def __init__(self):
        self.left = LaneTrack(ALPHA_KIRI)        # v13: alpha per sisi
        self.right = LaneTrack(ALPHA_KANAN)
        self.bend = 0.0
        self.arah = "Lurus"
        self.debug = False
        self.show_coords = SHOW_COORDS
        self.minggir = False           # True = kendaraan sedang minggir ke kiri
        self.posisi = None             # v13: (teks, warna) posisi kendaraan di jalur kiri
        self.bahu_pts = None           # v14: (xs, ys) titik scan bahu frame ini (untuk debug)
        # riwayat & status untuk fallback
        self.kiri_hist = deque(maxlen=HIST_LEN)
        self.kanan_hist = deque(maxlen=HIST_LEN)
        self.kiri_hold = 0             # frame sejak deteksi kiri terakhir
        self.kanan_hold = 0            # frame sejak deteksi kanan terakhir
        self.lebar_prop = LEBAR_PROP_DEFAULT
        self.lebar_terkalibrasi = False
        self.lebar_n = 0               # jumlah pasangan valid yang sudah dipakai kalibrasi
        self.lebar_bad = 0             # frame berturut-turut lebar tidak konsisten
        self.prev_ref = None           # (x kiri, x kanan) di Y_REF frame sebelumnya
        self.frame_idx = 0
        # Kalman filter untuk koordinat final
        self.kf_kiri = KalmanLevels(N_TITIK, KF_Q_POS_KIRI)   # v13
        self.kf_kanan = KalmanLevels(N_TITIK)


# ------------------------------------------------------------------ tahap pipeline
_CLAHE = None


def _get_clahe():
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                                 tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return _CLAHE


def preprocess(frame):
    """Grayscale -> [CLAHE] -> Gaussian Blur -> Canny.
    Mengembalikan (gray, eq, blur, edges, blur_raw):
      eq       = gray setelah CLAHE (sama dengan gray bila CLAHE mati)
      blur     = blur dari eq (dipakai Canny)
      blur_raw = blur dari gray asli (dipakai filter kecerahan)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = (BLUR_KSIZE, BLUR_KSIZE)
    blur_raw = cv2.GaussianBlur(gray, k, 0)
    if USE_CLAHE:
        eq = _get_clahe().apply(gray)
        blur = cv2.GaussianBlur(eq, k, 0)
    else:
        eq, blur = gray, blur_raw
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return gray, eq, blur, edges, blur_raw


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


def _batas_sisi(track, sisi, w):
    """Batas x (px) untuk garis sisi tertentu. Sebelum terlacak, garis kiri harus di
    kiri tengah (KIRI_X_MAX). Setelah terlacak, batas dilonggarkan agar garis bahu
    tetap dikenali saat kendaraan berada di bahu (v13)."""
    lim = KIRI_X_MAX if track.theta is None else KIRI_X_MAX_LANJUT
    return lim * w if sisi == "L" else (1.0 - lim) * w


def _fit_ok_sisi(fit, h, sisi, batas):
    """Tolak hasil fitting yang di Y_REF berada di sisi yang salah terhadap batas (v13)."""
    if fit is None:
        return None
    theta, dom = fit
    x_ref = float(eval_curve(theta, dom, to_t(Y_REF * h, h)))
    ok = x_ref < batas if sisi == "L" else x_ref > batas
    return fit if ok else None


def snap_ke_marka(track, marka, w, h, win_frac=None):
    """'Pas-kan' garis ke marka yang sebenarnya (v13). Di beberapa baris, cari kelompok
    piksel marka yang paling dekat dengan kurva garis saat ini, lalu jadikan titik
    tambahan untuk fitting. win_frac mengatur lebar jendela pencarian (None = SNAP_WIN_FRAC)."""
    xs_out, ys_out = [], []
    if not SNAP_MARKA or marka is None or track.theta is None:
        return xs_out, ys_out
    wf = SNAP_WIN_FRAC if win_frac is None else win_frac
    for yi in np.linspace(Y_TOP * h, Y_BOTTOM * h, SNAP_ROWS).astype(int):
        yi = min(max(int(yi), 0), h - 1)
        pred = track.x_at(yi, h)
        win = max(4, int(wf * w * _persp(yi, h)))
        lo = max(0, int(pred) - win)
        hi = min(w - 1, int(pred) + win)
        if hi <= lo:
            continue
        cols = np.flatnonzero(marka[yi, lo:hi + 1])
        if cols.size == 0:
            continue
        kelompok = np.split(cols, np.flatnonzero(np.diff(cols) > 3) + 1)
        terdekat = min(kelompok, key=lambda g: abs(lo + g.mean() - pred))
        xs_out.append(lo + float(terdekat.mean()))
        ys_out.append(float(yi))
    return xs_out, ys_out


def scan_bahu(aspal, track, right, lebar_prop, w, h):
    """Titik garis bahu dari TEPI ASPAL (v14, opsi --bahu-scan).
    Di tiap baris:
      1. tentukan lajur acuan (ref): 0.5 lebar lajur di kiri garis tengah bila garis
         tengah sudah terlacak, selain itu tengah gambar;
      2. cari piksel aspal di sekitar prediksi bahu (atau di kiri ref bila belum ada
         garis bahu), lalu ambil yang paling dekat ke ref sebagai 'seed';
      3. dari seed, jalan ke kiri selama masih aspal; posisi non-aspal pertama
         adalah tepi bahu.
    Baris yang tidak meyakinkan dilewati, sehingga LaneTrack memakai riwayat."""
    xs_out, ys_out = [], []
    for yi in np.linspace(Y_TOP * h, Y_BOTTOM * h, BAHU_ROWS).astype(int):
        yi = min(max(int(yi), 0), h - 1)
        row = aspal[yi]
        if right.theta is not None:
            ref = right.x_at(yi, h) - 0.5 * float(lebar_jalur(yi, h, w, lebar_prop))
        else:
            ref = 0.5 * w
        ref_i = int(np.clip(ref, 1, w - 1))
        if track.theta is not None:
            c = track.x_at(yi, h)
            win = max(4, int(BAHU_WIN_FRAC * w * _persp(yi, h)))
        else:
            c = float(ref_i)
            win = w // 2
        lo = max(0, int(c) - win)
        hi = min(ref_i, int(c) + win)
        if hi <= lo:
            continue
        cand = np.flatnonzero(row[lo:hi + 1])
        if cand.size == 0:
            continue
        seed = lo + int(cand[-1])                 # aspal terdekat ke ref
        nz = np.flatnonzero(~row[:seed])          # posisi non-aspal di kiri seed
        edge = int(nz[-1]) + 1 if nz.size else 0  # tepi kiri deretan aspal
        if edge <= 0:
            continue                              # aspal sampai tepi gambar: tepi bahu tidak terlihat
        if track.theta is not None and abs(edge - c) > 2 * win:
            continue                              # terlalu jauh dari prediksi: kemungkinan bukan bahu
        xs_out.append(float(edge))
        ys_out.append(float(yi))
    return xs_out, ys_out


def _persp(y, h):
    """Faktor perspektif: 1.0 di garis acuan, mengecil ke arah cakrawala.
    Jarak dua garis jalan (dalam px) makin kecil di bagian jauh, jadi toleransi juga."""
    return float(np.clip((y / h - Y_HORIZON) / max(Y_REF - Y_HORIZON, 0.01), 0.25, 1.0))


def label_posisi(xl_r, xr_r, w):
    """Posisi kendaraan di jalur kiri, dari bahu (p = 0) ke garis tengah (p = 1) (v13).
    Di bahu kiri (p < 0) kendaraan tetap dihitung di JALUR KIRI; yang berubah
    hanya keterangannya. Mengembalikan (teks, warna BGR) atau None."""
    lebar = xr_r - xl_r
    if lebar < 1:
        return None
    p = (w / 2.0 - xl_r) / lebar
    if p > 1.0:
        return ("Melintas garis tengah", (0, 0, 255))
    if p >= POS_BAHU:
        return ("Jalur kiri", (0, 255, 0))
    if p >= 0.0:
        return ("Jalur kiri - dekat bahu", (0, 165, 255))
    if p >= -0.6:
        return ("Jalur kiri - di bahu kiri", (0, 140, 255))
    return ("Keluar jalur ke kiri", (0, 0, 255))


def _pilih_kelompok_awal(segmen, s, w, h):
    """Belum ada garis: kelompokkan segmen menurut posisinya di garis acuan, buang
    kelompok yang pendek, lalu ambil kelompok paling dekat tengah kamera (dan paling
    panjang bila beberapa hampir sama dekatnya). s = +1 (kiri) atau -1 (kanan)."""
    segmen = sorted(segmen, key=lambda g: g["xref"])
    kelompok = [[segmen[0]]]
    for g in segmen[1:]:
        if g["xref"] - kelompok[-1][-1]["xref"] < KIRI_KELOMPOK_TOL * w:
            kelompok[-1].append(g)
        else:
            kelompok.append([g])
    ringkas = []
    for kel in kelompok:
        panjang = sum(g["len"] for g in kel)
        if panjang < KIRI_MIN_PANJANG * h:
            continue
        xc = sum(g["xref"] * g["len"] for g in kel) / panjang
        ringkas.append((xc, panjang, kel))
    if not ringkas:
        return []
    terdalam = max(s * r[0] for r in ringkas)
    dekat = [r for r in ringkas if s * r[0] > terdalam - KIRI_PILIH_TOL * w]
    return max(dekat, key=lambda r: r[1])[2]


def mask_marka(frame, putih_v=None, putih_s=None):
    """Mask piksel berwarna marka jalan: putih (kecerahan tinggi, saturasi rendah) atau
    kuning (rona kuning, saturasi tinggi). Rumput dan aspal tidak lolos.
    putih_v / putih_s: ambang khusus marka putih (None = MARKA_PUTIH_V / MARKA_PUTIH_S)."""
    pv = MARKA_PUTIH_V if putih_v is None else putih_v
    ps = MARKA_PUTIH_S if putih_s is None else putih_s
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    putih = (vv >= pv) & (ss <= ps)
    kuning = (hh >= 15) & (hh <= 35) & (ss >= MARKA_KUNING_S) & (vv >= MARKA_KUNING_V)
    mask = (putih | kuning).astype(np.uint8)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0


def mask_marka_kiri(frame):
    """Mask marka untuk garis bahu (v15): seperti mask_marka, tetapi marka putih harus
    lebih terang (MARKA_KIRI_PUTIH_V) supaya batu dan beton terang tidak ikut terbaca."""
    return mask_marka(frame, MARKA_KIRI_PUTIH_V, MARKA_KIRI_PUTIH_S)


def mask_aspal(frame):
    """Mask permukaan aspal (v14, opsi --bahu-scan): saturasi rendah dan kecerahan sedang.
    Rumput (hijau, saturasi tinggi) tidak lolos. Batu dan beton yang sangat terang
    dikecualikan lewat ASPAL_V_MAX. Celah kecil (putus-putus marka) dijembatani
    dengan operasi closing horizontal."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, ss, vv = cv2.split(hsv)
    m = ((ss <= ASPAL_S_MAX) & (vv >= ASPAL_V_MIN) & (vv <= ASPAL_V_MAX)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, ASPAL_CLOSE_W), np.uint8))
    return m > 0


def _collect_satu(lines, w, h, track, bright, sisi, marka=None):
    """Kumpulkan segmen untuk satu sisi jalur ('L' = kiri, 'R' = kanan).
      - segmen harus miring sesuai sisi dan titik tengahnya berada di sisi itu;
      - bila garis sedang dilacak: segmen yang dekat kurva garis itu (dicek di tinggi
        segmen, sehingga juga mengikuti tikungan). Segmen di sisi tengah yang muncul
        terus-menerus membuat garis pindah ke situ; segmen di sisi luar tidak;
      - bila belum ada garis: kelompok paling dekat tengah dan cukup panjang.
    Mengembalikan (xs, ys) titik segmen yang dipakai untuk fitting."""
    xs_out, ys_out = [], []
    if lines is None:
        return xs_out, ys_out
    s = 1.0 if sisi == "L" else -1.0        # s > 0: ke dalam jalur berarti x membesar
    segmen = []
    batas = _batas_sisi(track, sisi, w)     # v13
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 or dy == 0:
            continue
        slope = dy / dx
        if sisi == "L" and not (-12 <= slope <= -MIN_ABS_SLOPE):
            continue
        if sisi == "R" and not (MIN_ABS_SLOPE <= slope <= 12):
            continue
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if (sisi == "L" and xm >= batas) or (sisi == "R" and xm <= batas):   # v13
            continue
        sx, sy = sample_segment(x1, y1, x2, y2)
        if bright is not None:
            ix = np.clip(sx.astype(int), 0, w - 1)
            iy = np.clip(sy.astype(int), 0, h - 1)
            if bright[iy, ix].mean() < BRIGHT_FRACTION:
                continue
        if marka is not None:
            ix = np.clip(sx.astype(int), 0, w - 1)
            iy = np.clip(sy.astype(int), 0, h - 1)
            if marka[iy, ix].mean() < MARKA_FRAC:
                continue                      # bukan marka (mis. tepi aspal-rumput)
        segmen.append({
            "xref": float(x1 + (Y_REF * h - y1) * dx / dy),
            "len": float(np.hypot(dx, dy)),
            "xm": float(xm), "ym": float(ym), "sx": sx, "sy": sy,
        })
    if not segmen:
        return xs_out, ys_out

    if track.theta is not None:
        dekat, dalam = [], []
        for g in segmen:
            f = _persp(g["ym"], h)                        # toleransi mengikuti perspektif
            d = s * (g["xm"] - track.x_at(g["ym"], h))   # d > 0: di sisi tengah dari garis lama
            if abs(d) < KIRI_TOL * w * f:
                dekat.append(g)
            elif KIRI_TOL * w * f <= d < KIRI_PINDAH * w * f:
                dalam.append(g)
        track.dalam = (track.dalam + 1) if dalam else 0
        if dalam and track.dalam >= KIRI_PINDAH_FRAME:
            track.reset()                     # garis lama dibuang: langsung ikut yang baru
            track.dalam = 0
            pilih = dalam
        elif dekat:
            pilih = dekat
        else:
            return xs_out, ys_out
    else:
        pilih = _pilih_kelompok_awal(segmen, s, w, h)

    for g in pilih:
        xs_out.extend(g["sx"].tolist())
        ys_out.extend(g["sy"].tolist())
    return xs_out, ys_out


def collect_segments(lines, w, h, left, right, bright=None, lebar_prop=None,
                     marka=None, marka_kiri=None):
    """Segmen Hough untuk garis kiri dan kanan.
    v15: garis kiri hanya dari marka kiri (mask ketat). Tepi bahu dari Hough dipakai
    sebagai cadangan hanya bila LEFT_FALLBACK_HOUGH aktif. Bila BAHU_SCAN aktif,
    garis kiri tidak diambil dari Hough sama sekali (lihat scan_bahu).
    Garis kanan tetap dari marka."""
    if JALUR_MODE == 3 or BAHU_SCAN:
        L = ([], [])
    elif marka_kiri is not None:
        L = _collect_satu(lines, w, h, left, None, "L", marka_kiri)
        if LEFT_FALLBACK_HOUGH and len(L[0]) < MIN_SAMPLES:
            L_bahu = _collect_satu(lines, w, h, left, None, "L", None)
            if len(L_bahu[0]) >= MIN_SAMPLES:
                L = L_bahu
    else:
        L = _collect_satu(lines, w, h, left, None if BAHU_CADANGAN else bright,
                           "L", None if BAHU_CADANGAN else marka)
    R = _collect_satu(lines, w, h, right, bright, "R", marka) if SISI != "kiri" else ([], [])
    return L, R


def update_tracks(state, Lp, Rp, w, h, marka=None, aspal=None, marka_kiri=None):
    """Perbarui garis kiri & kanan (mode kiri: garis kanan dilepas).
    Garis kiri diperbarui lebih dulu agar scan bahu (bila aktif) memakai acuan garis
    tengah dari frame sebelumnya.
      - garis kiri : Hough dari marka kiri + pas-kan ke marka kiri (jendela sempit)
      - garis kanan: Hough dari marka + pas-kan ke marka (jendela standar)
      - --bahu-scan: garis kiri dari tepi aspal (scan_bahu), tanpa pas-kan."""
    state.bahu_pts = None
    for track, seg, sisi, kf in ((state.left, Lp, "L", state.kf_kiri),
                                  (state.right, Rp, "R", state.kf_kanan)):
        if (SISI == "kiri" and sisi == "R") or (JALUR_MODE == 3 and sisi == "L"):
            track.reset()
            kf.reset()
            continue
        baru = track.theta is None            # garis baru diambil (awal, setelah hilang, atau pindah)
        batas = _batas_sisi(track, sisi, w)
        if sisi == "L" and BAHU_SCAN and aspal is not None:
            xs, ys = scan_bahu(aspal, track, state.right, state.lebar_prop, w, h)
            state.bahu_pts = (xs, ys)
        else:
            if sisi == "R":
                sx, sy = snap_ke_marka(track, marka, w, h)
            elif SNAP_KIRI:
                sx, sy = snap_ke_marka(track, marka_kiri, w, h, SNAP_WIN_KIRI_FRAC)
            else:
                sx, sy = [], []
            xs = list(seg[0]) + sx
            ys = list(seg[1]) + sy
        fit = fit_lane(xs, ys, w, h, sisi, track.theta)
        track.update(_fit_ok_sisi(fit, h, sisi, batas), w, h)
        if baru and track.fresh:
            kf.reset()                        # filter Kalman mulai ulang pada garis baru


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
    """Panel informasi di pojok kiri atas: arah, kelengkungan, posisi, dan status."""
    h, w = img.shape[:2]
    s = w / 640.0

    if any_line:
        kat = abs(state.bend)
        kat_txt = (
            "-" if state.arah == "Lurus"
            else ("ringan" if kat < 0.05 else "sedang" if kat < 0.10 else "tajam")
        )

        if both and offset_m is not None:
            sisi = "kanan" if offset_m > 0 else "kiri"
            if state.minggir:
                status = ("Status: MINGGIR KIRI", WARNA_MINGGIR_KIRI, 0.45)
            elif abs(offset_m) < OFFSET_OK_M:
                status = ("Status: Baik", (0, 255, 0), 0.45)
            else:
                status = ("Status: Perhatikan posisi!", (0, 165, 255), 0.45)
            pos_txt, pos_warna = state.posisi if state.posisi else ("-", (255, 255, 255))
            lines = [
                (f"Arah: {state.arah}", (0, 255, 255), 0.6),
                (f"Kelengkungan: {kat_txt}", (255, 255, 255), 0.42),
                (f"Posisi: {pos_txt}", pos_warna, 0.42),
                (f"Offset: {abs(offset_m):.2f} m ke {sisi} tengah", (255, 255, 255), 0.42),
                status,
            ]
        else:
            if SISI == "dua":
                info = ("Baru satu garis terdeteksi", (0, 165, 255), 0.45)
            else:
                info = (f"Mode: hanya garis {SISI}", (255, 255, 255), 0.45)
            lines = [
                (f"Arah: {state.arah}", (0, 255, 255), 0.6),
                (f"Kelengkungan: {kat_txt}", (255, 255, 255), 0.42),
                info,
            ]
    else:
        lines = [("Jalur tidak terdeteksi", (0, 0, 255), 0.6)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, sc * s, 2 if sc > 0.5 else 1)[0]
             for t, _, sc in lines]
    pw = int(max(sz[0] for sz in sizes) + 24 * s)
    line_h = [int(sz[1] + 9 * s) for sz in sizes]
    ph = int(sum(line_h) + 10 * s)

    roi = img[0:ph, 0:pw]
    img[0:ph, 0:pw] = cv2.addWeighted(roi, 0.35, np.full_like(roi, 30), 0.65, 0)
    cv2.rectangle(img, (0, 0), (pw, ph), (255, 255, 255), 1)

    y = 0
    for (txt, col, sc), lh in zip(lines, line_h):
        y += lh
        cv2.putText(img, txt, (int(12 * s), int(y)), font, sc * s, col,
                    2 if sc > 0.5 else 1, cv2.LINE_AA)


def detect_lanes(frame, state):
    """Tahap deteksi jalur (mentah, dari gambar)."""
    h, w = frame.shape[:2]

    gray, eq, blur, edges, blur_raw = preprocess(frame)

    poly, roi_mask = get_roi(w, h)
    roi_edges = apply_roi(edges, roi_mask)

    bright = None
    if BRIGHT_DELTA > 0 and not BAHU_CADANGAN:   # bahu tidak harus lebih terang
        med = float(np.median(blur_raw[roi_mask > 0]))
        bright = cv2.dilate(
            (blur_raw > med + BRIGHT_DELTA).astype(np.uint8),
            _BRIGHT_KERNEL
        ) > 0
        roi_edges = roi_edges * bright.astype(np.uint8)

    lines = cv2.HoughLinesP(
        roi_edges, 1, np.pi / 180, HOUGH_THRESH,
        minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP
    )

    lebar_prop = state.lebar_prop if state.lebar_terkalibrasi else None
    marka = mask_marka(frame) if MARKA_WAJIB else None
    marka_kiri = mask_marka_kiri(frame) if MARKA_WAJIB else None
    aspal = mask_aspal(frame) if BAHU_SCAN else None
    Lp, Rp = collect_segments(lines, w, h, state.left, state.right, bright, lebar_prop,
                              marka, marka_kiri)

    update_tracks(state, Lp, Rp, w, h, marka, aspal, marka_kiri)

    update_direction(state, w)

    return lines, roi_edges, poly


# ================================================================
#   KOORDINAT OUTPUT + FALLBACK + KALMAN
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


def _lebar_konsisten(state, prop):
    """Apakah lebar pasangan garis (proporsi lebar gambar) masuk akal dan
    sesuai dengan lebar jalur yang sudah dipelajari?"""
    if not (0.12 < prop < 0.95):
        return False
    if state.lebar_n < LEBAR_N_MIN:          # masih belajar: terima yang masuk akal
        return True
    rasio = prop / max(state.lebar_prop, 1e-6)
    return (1.0 - LEBAR_TOL) < rasio < (1.0 + LEBAR_TOL)


def _kalibrasi_lebar(state, prop):
    """Rata-rata berjalan di awal, lalu EMA lambat."""
    state.lebar_n += 1
    a = max(LEBAR_ALPHA, 1.0 / state.lebar_n)
    state.lebar_prop = (1 - a) * state.lebar_prop + a * prop
    state.lebar_terkalibrasi = state.lebar_n >= LEBAR_N_MIN


def _sisi_buruk(state, xl_ref, xr_ref):
    """Bila lebar tidak konsisten, tentukan sisi mana yang dicurigai salah:
    yang paling menyimpang dari posisi frame sebelumnya (atau yang datanya
    lebih pendek bila belum ada riwayat)."""
    if state.prev_ref is not None:
        dl = abs(xl_ref - state.prev_ref[0])
        dr = abs(xr_ref - state.prev_ref[1])
        return "L" if dl > dr else "R"
    sl = float(state.left.dom[1] - state.left.dom[0])
    sr = float(state.right.dom[1] - state.right.dom[0])
    return "L" if sl < sr else "R"


def prediksi_riwayat(hist, n):
    """Prediksi posisi garis dari sampel valid terakhir dan tren robust.
    Tren dihitung dari median perubahan beberapa frame terakhir agar satu
    deteksi yang melompat tidak menarik prediksi terlalu jauh. Makin lama
    marka tertutup, kontribusi tren makin diredam."""
    arr = np.asarray(hist, dtype=np.float64)  # bentuk (K, N_TITIK)
    k = len(arr)
    if k == 0:
        return None
    terakhir = arr[-1].copy()
    if k < 2 or n <= 0:
        return terakhir
    delta = np.diff(arr[-min(k, 6):], axis=0)
    tren = np.median(delta, axis=0)
    langkah = sum(DECAY ** j for j in range(1, n + 1))
    prediksi = terakhir + tren * langkah
    # Batasi extrapolasi agar prediksi tidak meloncat keluar gambar.
    batas = max(1.0, 0.12 * PROCESS_WIDTH)
    return terakhir + np.clip(prediksi - terakhir, -batas, batas)


def r_deteksi(track, h, w):
    """Derau ukur tiap titik saat deteksi langsung. Titik yang berada di luar
    rentang data (hasil perpanjangan garis singgung) kurang dipercaya."""
    t = to_t(Y_LEVELS * h, h)
    lo, hi = float(track.dom[0]), float(track.dom[1])
    jarak = np.maximum(np.maximum(lo - t, t - hi), 0.0)
    return KF_R_DETEKSI * w * np.sqrt(1.0 + (jarak / 0.08) ** 2)


def r_estimasi(hold, w):
    """Derau ukur untuk estimasi dari riwayat: makin lama makin tidak pasti."""
    return np.full(N_TITIK, KF_R_ESTIMASI * w * np.sqrt(1.0 + hold / 10.0))


def finalize_lanes(state, w, h):
    """Tentukan koordinat FINAL kiri & kanan untuk frame ini.
      0. validasi lebar jalur (buang sisi yang tidak konsisten)
      1. kedua garis terdeteksi              -> 'deteksi'
      2. hanya satu garis terdeteksi         -> 'turunan' (garis lain = garis itu + lebar)
      3. tidak ada deteksi                   -> 'estimasi' (rata-rata riwayat + tren)
      4. riwayat habis (> HOLD_MAX_FRAMES)   -> 'hilang'
      5. Kalman filter menghaluskan semua titik
      6. status minggir ke kiri (offset + histeresis)
      7a. posisi kendaraan di jalur kiri (v13)
    """
    state.frame_idx += 1
    lv = Y_LEVELS * h
    fresh_l, fresh_r = state.left.fresh, state.right.fresh

    # 0) validasi konsistensi lebar jalur + kalibrasi
    if fresh_l and fresh_r:
        xl_ref = state.left.x_at(Y_REF * h, h)
        xr_ref = state.right.x_at(Y_REF * h, h)
        prop = (xr_ref - xl_ref) / w
        if (not VALIDASI_LEBAR) or _lebar_konsisten(state, prop):
            if LEBAR_DARI_KALIBRASI and 0.15 < prop < 0.9:
                _kalibrasi_lebar(state, prop)
            state.lebar_bad = 0
        else:
            state.lebar_bad += 1
            if _sisi_buruk(state, xl_ref, xr_ref) == "L":
                fresh_l = False
                state.left.reset()
            else:
                fresh_r = False
                state.right.reset()
            if state.lebar_bad >= LEBAR_BAD_MAX:      # kalibrasi lama mungkin salah
                state.lebar_n = 0
                state.lebar_bad = 0
                state.lebar_terkalibrasi = False

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

    lebar = lebar_jalur(lv, h, w, state.lebar_prop)

    # 2) koordinat dasar dari deteksi langsung
    xl = xr = None
    rl = rr = None
    st_l = st_r = "hilang"
    if fresh_l:
        xl, st_l, rl = xs_di_level(state.left, h), "deteksi", r_deteksi(state.left, h, w)
    if fresh_r:
        xr, st_r, rr = xs_di_level(state.right, h), "deteksi", r_deteksi(state.right, h, w)

    # 3) garis yang tidak terdeteksi -> rata-rata riwayat + tren (dihitung per sisi)
    if xl is None and state.kiri_hist and state.kiri_hold <= HOLD_MAX_FRAMES and JALUR_MODE != 3:
        xl = prediksi_riwayat(state.kiri_hist, state.kiri_hold)
        st_l, rl = "estimasi", r_estimasi(state.kiri_hold, w)
    if xr is None and state.kanan_hist and state.kanan_hold <= HOLD_MAX_FRAMES:
        xr = prediksi_riwayat(state.kanan_hist, state.kanan_hold)
        st_r, rr = "estimasi", r_estimasi(state.kanan_hold, w)

    # 4) satu sisi kosong -> turunkan dari sisi lain + lebar (hanya bila DERIVASI_SISI_LAIN)
    if not DERIVASI_SISI_LAIN:
        pass
    elif xl is None and xr is not None:
        xl = xr - lebar
        st_l = "estimasi" if st_r == "estimasi" else "turunan"
        rl = np.sqrt(rr ** 2 + (KF_R_TURUNAN * w) ** 2)
    elif xr is None and xl is not None:
        xr = xl + lebar
        st_r = "estimasi" if st_l == "estimasi" else "turunan"
        rr = np.sqrt(rl ** 2 + (KF_R_TURUNAN * w) ** 2)

    # batas longgar: garis bahu boleh bergeser ke tengah saat kendaraan di bahu (v13)
    if xl is not None:
        xl = np.minimum(xl, KIRI_X_MAX_LANJUT * w)
    if xr is not None:
        xr = np.maximum(xr, (1.0 - KIRI_X_MAX_LANJUT) * w)

    # 5) Kalman filter pada tiap titik (menghaluskan juga perpindahan antar-mode)
    if USE_KALMAN:
        if xl is not None:
            xl = state.kf_kiri.langkah(xl, rl, w, 0.0 if st_l == "estimasi" else None)
        else:
            state.kf_kiri.reset()
        if xr is not None:
            xr = state.kf_kanan.langkah(xr, rr, w, 0.0 if st_r == "estimasi" else None)
        else:
            state.kf_kanan.reset()

    # 6) status & keyakinan
    if SISI == "kiri" or JALUR_MODE == 3:
        st_u, hold_u = (st_l, state.kiri_hold) if SISI == "kiri" else (st_r, state.kanan_hold)
        status = {"deteksi": "normal", "estimasi": "perkiraan"}.get(st_u, "hilang")
        if st_u == "deteksi":
            keyakinan = 1.0
        elif st_u == "estimasi":
            keyakinan = 0.5 * max(0.0, 1.0 - hold_u / HOLD_MAX_FRAMES)
        else:
            keyakinan = 0.0
    elif st_l == "hilang" and st_r == "hilang":
        status, keyakinan = "hilang", 0.0
    elif st_l == "deteksi" and st_r == "deteksi":
        status, keyakinan = "normal", 1.0
    else:
        status = "perkiraan"
        holds = []
        if st_l == "estimasi":
            holds.append(state.kiri_hold)
        if st_r == "estimasi":
            holds.append(state.kanan_hold)
        if holds:
            keyakinan = 0.5 * max(0.0, 1.0 - max(holds) / HOLD_MAX_FRAMES)
        else:
            keyakinan = 0.7 if state.lebar_terkalibrasi else 0.5

    # 7a) posisi kendaraan di jalur kiri (v13)
    state.posisi = None
    if JALUR_MODE == 2 and SISI == "dua" and xl is not None and xr is not None:
        state.posisi = label_posisi(float(np.interp(Y_REF, Y_LEVELS, xl)),
                                    float(np.interp(Y_REF, Y_LEVELS, xr)), w)

    # 7) tengah jalur & offset (dihitung di Y_REF)
    tengah = offset_m = lebar_ref = None
    state.prev_ref = None
    lebar_ref = float(lebar_jalur(Y_REF * h, h, w, state.lebar_prop))
    if JALUR_MODE == 3:
        # tiga jalur: jalur = antara garis tengah (xr) dan batas kanan (xr + lebar jalur)
        if xr is not None:
            tengah = xr + lebar / 2.0
            xr_r = float(np.interp(Y_REF, Y_LEVELS, xr))
            offset_m = (w / 2.0 - (xr_r + lebar_ref / 2.0)) * LANE_WIDTH_M / lebar_ref
    elif xl is not None and xr is not None:
        # dua jalur: jalur = antara bahu (xl) dan garis kanan (xr)
        tengah = (xl + xr) / 2.0
        xl_r = float(np.interp(Y_REF, Y_LEVELS, xl))
        xr_r = float(np.interp(Y_REF, Y_LEVELS, xr))
        state.prev_ref = (xl_r, xr_r)
        if xr_r - xl_r > 1:
            lebar_m = xr_r - xl_r
            offset_m = (w / 2.0 - (xl_r + xr_r) / 2.0) * LANE_WIDTH_M / lebar_m
    elif xl is not None:
        tengah = xl + lebar / 2.0
        pusat = float(np.interp(Y_REF, Y_LEVELS, xl)) + lebar_ref / 2.0
        offset_m = (w / 2.0 - pusat) * LANE_WIDTH_M / lebar_ref
    elif xr is not None:
        tengah = xr - lebar / 2.0
        pusat = float(np.interp(Y_REF, Y_LEVELS, xr)) - lebar_ref / 2.0
        offset_m = (w / 2.0 - pusat) * LANE_WIDTH_M / lebar_ref

    if SISI == "kiri":
        tengah = None                 # mode kiri: garis tengah/jalur tidak ditampilkan

    # 7b) status minggir ke kiri, dengan dua ambang (histeresis) agar tidak kedip
    if offset_m is not None:
        if state.minggir:
            if offset_m > MINGGIR_OFF_M:
                state.minggir = False
        elif offset_m < MINGGIR_ON_M:
            state.minggir = True

    # skala meter tiap titik = lebar jalur pada ketinggian itu
    skala = lebar

    def bentuk(xs):
        """Ubah array x menjadi daftar titik: x, y (pixel) dan x_m (meter, + = kanan)."""
        if xs is None:
            return None
        return [
            {"x": int(round(x)), "y": int(round(y)),
             "x_m": round(float((x - w / 2.0) * LANE_WIDTH_M / lb), 3)}
            for x, y, lb in zip(xs, lv, skala)
        ]

    return {
        "frame": state.frame_idx,
        "status": status,                          # normal / perkiraan / hilang
        "keyakinan": round(float(keyakinan), 2),   # 0.0 - 1.0
        "kalibrasi_lebar": state.lebar_terkalibrasi,
        "arah": state.arah,
        "offset_m": None if offset_m is None else round(float(offset_m), 3),
        "minggir_kiri": state.minggir,
        "posisi": state.posisi[0] if state.posisi else None,      # v13
        "lebar_px": None if lebar_ref is None else round(lebar_ref, 1),
        "kiri": {"status": st_l, "titik": bentuk(xl)},
        "kanan": {"status": st_r, "titik": bentuk(xr)},
        "tengah": bentuk(tengah),
    }


def analisis_frame(frame, state):
    """Deteksi mentah + koordinat final. Mengembalikan (lines, roi_edges, poly, hasil)."""
    h, w = frame.shape[:2]
    lines, roi_edges, poly = detect_lanes(frame, state)
    hasil = finalize_lanes(state, w, h)
    return lines, roi_edges, poly, hasil


def kurva_dari_hasil(info, h, w):
    """Ubah titik-titik hasil final menjadi kurva halus (40 titik) untuk digambar.
    Memakai parabola x(y) bila cocok dengan titik-titik; kalau tidak, interpolasi."""
    titik = info["titik"]
    if titik is None:
        return None
    lv = Y_LEVELS * h
    xs_lv = np.array([p["x"] for p in titik], dtype=np.float64)
    ys = np.linspace(Y_TOP * h, Y_BOTTOM * h, 40)
    xs = np.interp(ys, lv, xs_lv)
    if len(xs_lv) >= 4:
        xq = np.polyval(np.polyfit(lv, xs_lv, 2), ys)
        if np.max(np.abs(xq - xs)) < 0.012 * w:
            xs = xq
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


def cetak_koordinat(hasil):
    """Cetak koordinat final ke terminal (dipakai pada mode foto)."""
    print(f"Status: {hasil['status']} (keyakinan {hasil['keyakinan']}) | "
          f"arah: {hasil['arah']} | offset: {hasil['offset_m']} m | "
          f"posisi: {hasil.get('posisi')} | "
          f"minggir kiri: {hasil['minggir_kiri']} | "
          f"kiri: {hasil['kiri']['status']} | kanan: {hasil['kanan']['status']}")
    kiri = hasil["kiri"]["titik"] or []
    kanan = hasil["kanan"]["titik"] or []
    tengah = hasil["tengah"] or []
    print(f"{'y':>5} {'kiri':>6} {'tengah':>7} {'kanan':>6}   x_m kiri / kanan")
    for k in range(N_TITIK):
        y = (kiri or kanan)[k]["y"] if (kiri or kanan) else "-"
        xl = kiri[k]["x"] if kiri else "-"
        xc = tengah[k]["x"] if tengah else "-"
        xr = kanan[k]["x"] if kanan else "-"
        ml = kiri[k]["x_m"] if kiri else "-"
        mr = kanan[k]["x_m"] if kanan else "-"
        print(f"{y:>5} {xl:>6} {xc:>7} {xr:>6}   {ml} / {mr}")


# ================================================================
#   TAMPILAN KOORDINAT (versi rapi)
# ================================================================
LABEL_TENGAH = True          # False = label x,y garis tengah hanya ada di tabel

FONT = cv2.FONT_HERSHEY_SIMPLEX
WARNA_SISI = {               # warna titik (BGR), sama dengan warna garis
    "kiri": (255, 0, 0),
    "tengah": (255, 0, 255),
    "kanan": (0, 255, 255),
}
WARNA_TEKS = {               # warna teks, dibuat lebih terang agar terbaca
    "kiri": (255, 190, 120),
    "tengah": (255, 150, 255),
    "kanan": (0, 255, 255),
}


def _titik(img, x, y, warna, langsung, s):
    """Penuh + tepi putih = hasil deteksi langsung.
    Ring berongga = diturunkan dari sisi lain / estimasi."""
    r = max(2, int(round(3 * s)))
    if langsung:
        cv2.circle(img, (x, y), r + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, warna, -1, cv2.LINE_AA)
    else:
        cv2.circle(img, (x, y), r + 1, warna, 2, cv2.LINE_AA)


def _label_kotak(img, x, y, teks, sisi, s):
    """Label kiri/kanan: kotak gelap dengan tepi berwarna sisi.
    Kotak kiri diletakkan di kiri titik, kotak kanan di kanan titik."""
    h, w = img.shape[:2]
    fs = 0.36 * s
    (tw, th), base = cv2.getTextSize(teks, FONT, fs, 1)
    pad = max(2, int(round(3 * s)))
    gap = max(3, int(round(6 * s)))
    bw, bh = tw + 2 * pad, th + base + 2 * pad
    if bw >= w or bh >= h:
        return
    bx = x - gap - bw if sisi == "kiri" else x + gap
    by = y - bh // 2
    bx = int(np.clip(bx, 0, w - bw - 1))
    by = int(np.clip(by, 0, h - bh - 1))
    roi = img[by:by + bh, bx:bx + bw]
    img[by:by + bh, bx:bx + bw] = cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0)
    cv2.rectangle(img, (bx, by), (bx + bw - 1, by + bh - 1), WARNA_SISI[sisi], 1, cv2.LINE_AA)
    cv2.putText(img, teks, (bx + pad, by + pad + th), FONT, fs,
                WARNA_TEKS[sisi], 1, cv2.LINE_AA)


def _label_tengah(img, x, y, teks, s):
    """Label garis tengah: teks berbingkai hitam tanpa kotak, di bawah titik,
    supaya tidak menumpuk dengan titik di baris berikutnya."""
    h, w = img.shape[:2]
    fs = 0.34 * s
    (tw, th), _ = cv2.getTextSize(teks, FONT, fs, 1)
    tx = int(np.clip(x - tw // 2, 1, max(1, w - tw - 2)))
    ty = min(h - 2, y + int(round(5 * s)) + th)
    cv2.putText(img, teks, (tx, ty), FONT, fs, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, teks, (tx, ty), FONT, fs, WARNA_TEKS["tengah"], 1, cv2.LINE_AA)


def _tabel_koordinat(img, hasil, s):
    """Tabel koordinat (x, y) di pojok kanan atas gambar. Kolom hanya untuk garis yang
    dipakai (mode kiri: hanya kolom BAHU)."""
    kiri = hasil["kiri"]["titik"]
    kanan = hasil["kanan"]["titik"]
    tengah = hasil["tengah"]
    ref = kiri or kanan
    if not ref:
        return
    h, w = img.shape[:2]
    fs_j, fs_i = 0.40 * s, 0.36 * s
    px, py = int(round(7 * s)), int(round(3 * s))

    kolom = [("BAHU", kiri, WARNA_TEKS["kiri"])]
    if SISI != "kiri":
        kolom += [("GARIS TGH", kanan, WARNA_TEKS["kanan"]),
                  ("JALUR", tengah, WARNA_TEKS["tengah"])]

    judul = "Koordinat (x, y) px"
    header = ["#"] + [nama for nama, _, _ in kolom]
    isi = [
        [str(k + 1)] + [f"{lst[k]['x']}, {ref[k]['y']}" if lst else "-"
                        for _, lst, _ in kolom]
        for k in range(len(ref))
    ]
    warna_header = [(255, 255, 255)] + [warna for _, _, warna in kolom]
    warna_isi = [(170, 170, 170)] + [(255, 255, 255)] * len(kolom)

    offset = hasil["offset_m"]
    kaki = f"Offset: {offset:+.2f} m" if offset is not None else "Offset: -"
    if offset is None:
        warna_kaki = (255, 255, 255)
    else:
        warna_kaki = (0, 255, 0) if abs(offset) < OFFSET_OK_M else (0, 165, 255)

    # ukuran kolom
    kol = [0] * len(header)
    for baris_teks in [header] + isi:
        for c, teks in enumerate(baris_teks):
            kol[c] = max(kol[c], cv2.getTextSize(teks, FONT, fs_i, 1)[0][0])
    kol_w = [v + 2 * px for v in kol]
    lebar_judul = cv2.getTextSize(judul, FONT, fs_j, 1)[0][0] + 2 * px
    if lebar_judul > sum(kol_w):
        kol_w[-1] += lebar_judul - sum(kol_w)
    pw = sum(kol_w)

    # tinggi baris
    (_, th_j), b_j = cv2.getTextSize(judul, FONT, fs_j, 1)
    (_, th_i), b_i = cv2.getTextSize("Ag", FONT, fs_i, 1)
    lh_j = th_j + b_j + 2 * py
    lh_i = th_i + b_i + 2 * py
    ph = lh_j + lh_i * (len(isi) + 2)       # judul + header + data + offset

    x0 = max(0, w - pw - int(round(6 * s)))
    y0 = int(round(28 * s))
    if y0 + ph > h:
        return

    # latar, bingkai, dan bar judul
    img[y0:y0 + ph, x0:x0 + pw] = (28, 28, 28)
    cv2.rectangle(img, (x0, y0), (x0 + pw - 1, y0 + ph - 1), (200, 200, 200), 1)
    cv2.rectangle(img, (x0, y0), (x0 + pw - 1, y0 + lh_j - 1), (120, 70, 30), -1)
    cv2.putText(img, judul, (x0 + px, y0 + py + th_j), FONT, fs_j,
                (255, 255, 255), 1, cv2.LINE_AA)

    def tulis_baris(ry, cells, warna, latar=None):
        """Tulis satu baris tabel, teks rata tengah per kolom."""
        if latar is not None:
            cv2.rectangle(img, (x0, ry), (x0 + pw - 1, ry + lh_i - 1), latar, -1)
        xk = x0
        for c, teks in enumerate(cells):
            tw = cv2.getTextSize(teks, FONT, fs_i, 1)[0][0]
            cv2.putText(img, teks, (xk + (kol_w[c] - tw) // 2, ry + py + th_i),
                        FONT, fs_i, warna[c], 1, cv2.LINE_AA)
            xk += kol_w[c]

    y = y0 + lh_j
    tulis_baris(y, header, warna_header, latar=(60, 60, 60))
    y += lh_i
    cv2.line(img, (x0, y - 1), (x0 + pw - 1, y - 1), (150, 150, 150), 1)

    for k, baris_teks in enumerate(isi):      # baris selang-seling
        tulis_baris(y, baris_teks, warna_isi,
                    latar=(40, 40, 40) if k % 2 else None)
        y += lh_i

    cv2.line(img, (x0, y - 1), (x0 + pw - 1, y - 1), (150, 150, 150), 1)
    cv2.putText(img, kaki, (x0 + px, y + py + th_i), FONT, fs_i,
                warna_kaki, 1, cv2.LINE_AA)


def gambar_koordinat(img, hasil):
    """Gambar titik (x, y) pada garis kiri, kanan, dan tengah, lalu tabel koordinat."""
    h, w = img.shape[:2]
    s = w / 640.0
    kiri_st, kanan_st = hasil["kiri"]["status"], hasil["kanan"]["status"]
    daftar = [
        ("kiri", hasil["kiri"]["titik"], kiri_st == "deteksi"),
        ("kanan", hasil["kanan"]["titik"], kanan_st == "deteksi"),
        ("tengah", hasil["tengah"], kiri_st == "deteksi" and kanan_st == "deteksi"),
    ]
    # titik dulu, baru label (label selalu tampil di atas titik)
    for nama, titik, langsung in daftar:
        for p in titik or []:
            _titik(img, p["x"], p["y"], WARNA_SISI[nama], langsung, s)
    for nama, titik, _ in daftar:
        if not titik or (nama == "tengah" and not LABEL_TENGAH):
            continue
        for p in titik:
            teks = f"{p['x']}, {p['y']}"
            if nama == "tengah":
                _label_tengah(img, p["x"], p["y"], teks, s)
            else:
                _label_kotak(img, p["x"], p["y"], teks, nama, s)
    _tabel_koordinat(img, hasil, s)
    return img


def gambar_legenda(img):
    """Keterangan warna di pojok kiri atas, di bawah status jalur."""
    s = img.shape[1] / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    items = [
        ("Bahu jalan (biru)", WARNA_GARIS_KIRI),
        ("Garis tengah (kuning)", WARNA_GARIS_KANAN),
        ("Jalur (hijau), oranye = minggir", WARNA_JALUR),
    ]
    if SISI == "kiri":
        items = items[:1]
    kotak = max(6, int(round(9 * s)))
    for k, (teks, warna) in enumerate(items):
        y = int((134 + 16 * k) * s)
        x = int(8 * s)
        cv2.rectangle(img, (x, y - kotak), (x + kotak, y), warna, -1)
        cv2.rectangle(img, (x, y - kotak), (x + kotak, y), (255, 255, 255), 1)
        tx = x + kotak + int(6 * s)
        cv2.putText(img, teks, (tx, y), font, 0.42 * s, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, teks, (tx, y), font, 0.42 * s, (255, 255, 255), 1, cv2.LINE_AA)


def gambar_hasil_akhir(img, hasil, state):
    """Status jalur, keterangan warna, dan angka koordinat."""
    s = img.shape[1] / 640.0
    teks, col = {
        "normal": ("Jalur: NORMAL", (0, 255, 0)),
        "perkiraan": ("Jalur: PERKIRAAN", (0, 165, 255)),
        "hilang": ("Jalur: HILANG - melambat", (0, 0, 255)),
    }[hasil["status"]]
    label_mode = "hanya bahu" if SISI == "kiri" else ("jalur kiri (bahu + marka tengah)" if JALUR_MODE == 2 else f"{JALUR_MODE} jalur")
    teks += f" (keyakinan {hasil['keyakinan']:.2f}) | {label_mode}"
    cv2.putText(img, teks, (int(8 * s), int(112 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, teks, (int(8 * s), int(112 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * s, col, 2, cv2.LINE_AA)
    gambar_legenda(img)
    if state.show_coords:
        gambar_koordinat(img, hasil)
    return img


# ------------------------------------------------------------------ gambar
def render_frame(frame, state, lines=None, roi_edges=None, poly=None, hasil=None):
    """Gambar area jalur (hijau normal, biru saat minggir kiri), garis kiri (merah)
    dan kanan (kuning) yang penuh, ROI (putih), dan panel info.
    Jika 'hasil' diberikan, semuanya mengikuti koordinat FINAL (termasuk estimasi)."""
    h, w = frame.shape[:2]

    if hasil is not None:
        lp = kurva_dari_hasil(hasil["kiri"], h, w)
        rp = kurva_dari_hasil(hasil["kanan"], h, w)
        minggir = bool(hasil.get("minggir_kiri", False))
    else:
        lp = curve_points(state.left, h, w) if state.left.theta is not None else None
        rp = curve_points(state.right, h, w) if state.right.theta is not None else None
        minggir = state.minggir

    th = max(3, int(round(w / 90)))
    th_kiri = th if (hasil is None or hasil["kiri"]["status"] == "deteksi") else max(2, th // 2)
    th_kanan = th if (hasil is None or hasil["kanan"]["status"] == "deteksi") else max(2, th // 2)

    # offset tersedia bila ada satu sisi atau kedua sisi
    ada_offset = hasil is not None and hasil["offset_m"] is not None
    offset_m = hasil["offset_m"] if ada_offset else 0.0
    both = ada_offset

    # 1) jalur (hijau; oranye saat minggir).
    #    Dua jalur: antara bahu (lp) dan garis kanan (rp). Tiga jalur: antara garis tengah (rp)
    #    dan batas kanan jalur perkiraan (rp + lebar jalur).
    out = frame.copy()
    warna_jalur = WARNA_MINGGIR_KIRI if minggir else WARNA_JALUR
    batas_kanan = None
    if JALUR_MODE == 3:
        if rp is not None:
            ys_c = np.linspace(Y_TOP * h, Y_BOTTOM * h, 40)
            geser = np.asarray(lebar_jalur(ys_c, h, w, state.lebar_prop)).astype(np.int32)
            batas_kanan = rp.copy()
            batas_kanan[:, 0] = rp[:, 0] + geser
            overlay = np.zeros_like(frame)
            cv2.fillPoly(overlay, [np.vstack([rp, batas_kanan[::-1]])], warna_jalur)
            out = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)
    elif lp is not None and rp is not None:
        overlay = np.zeros_like(frame)
        cv2.fillPoly(overlay, [np.vstack([lp, rp[::-1]])], warna_jalur)
        out = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

    # 2) bahu (biru), garis tengah/kanan (kuning), dan batas kanan jalur perkiraan (putih tipis)
    if lp is not None:
        cv2.polylines(out, [lp], False, WARNA_GARIS_KIRI, th_kiri, cv2.LINE_AA)
    if rp is not None:
        cv2.polylines(out, [rp], False, WARNA_GARIS_KANAN, th_kanan, cv2.LINE_AA)
    if batas_kanan is not None:
        cv2.polylines(out, [batas_kanan], False, (255, 255, 255), 1, cv2.LINE_AA)

    if poly is None:
        poly, _ = get_roi(w, h)
    cv2.polylines(out, [poly], True, WARNA_ROI, 1)

    if state.debug and roi_edges is not None:
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 1)

        ih, iw = h // 3, w // 3
        inset = cv2.cvtColor(cv2.resize(roi_edges, (iw, ih)), cv2.COLOR_GRAY2BGR)
        out[h - ih:h, w - iw:w] = inset
        cv2.rectangle(out, (w - iw, h - ih), (w - 1, h - 1), (255, 255, 255), 1)

    # titik scan bahu (hanya mode --bahu-scan dan debug)
    if state.debug and state.bahu_pts is not None:
        for x, y in zip(*state.bahu_pts):
            cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    draw_panel(out, state, offset_m, both, lp is not None or rp is not None)
    return out


def process_frame(frame, state):
    """Deteksi + koordinat final + gambar untuk satu frame.
    Mengembalikan (gambar_hasil, hasil_koordinat)."""
    lines, roi_edges, poly, hasil = analisis_frame(frame, state)
    out = render_frame(frame, state, lines, roi_edges, poly, hasil)
    out = gambar_hasil_akhir(out, hasil, state)
    return out, hasil


# =====================================================================
#            HASIL PER TAHAP (TAMPIL + SIMPAN)
# =====================================================================
def _bgr(img):
    """Pastikan gambar 3 kanal (BGR)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def build_stages(frame, state):
    """Jalankan pipeline SATU frame sambil mengumpulkan gambar tiap tahap.
    Mengembalikan (stages, hasil): stages = [(judul, gambar_BGR), ...],
    hasil = koordinat final. state ikut ter-update. Nomor tahap otomatis."""
    h, w = frame.shape[:2]
    stages = []

    def tambah(judul, img):
        stages.append((f"{len(stages) + 1:02d} {judul}", img))

    tambah("Original (resize)", frame.copy())

    gray, eq, blur, edges, blur_raw = preprocess(frame)
    tambah("Grayscale", gray)
    if USE_CLAHE:
        tambah("CLAHE (kontras adaptif)", eq)
    tambah("Gaussian Blur", blur)
    tambah("Canny Edge", edges)

    poly, roi_mask = get_roi(w, h)
    tint = np.zeros_like(frame)
    cv2.fillPoly(tint, [poly], (0, 255, 0))
    roi_vis = cv2.addWeighted(frame.copy(), 1.0, tint, 0.35, 0)
    cv2.polylines(roi_vis, [poly], True, WARNA_ROI, 2)
    tambah("ROI (area jalan)", roi_vis)

    roi_edges = apply_roi(edges, roi_mask)
    tambah("Edge di dalam ROI", roi_edges)

    bright = None
    if BRIGHT_DELTA > 0 and not BAHU_CADANGAN:   # bahu tidak harus lebih terang
        med = float(np.median(blur_raw[roi_mask > 0]))
        bright = cv2.dilate(
            (blur_raw > med + BRIGHT_DELTA).astype(np.uint8), _BRIGHT_KERNEL
        ) > 0
        tambah("Mask Brightness", bright.astype(np.uint8) * 255)
        roi_edges = roi_edges * bright.astype(np.uint8)
        tambah("Edge setelah Brightness Filter", roi_edges)

    lines = cv2.HoughLinesP(
        roi_edges, 1, np.pi / 180, HOUGH_THRESH,
        minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP
    )
    marka_vis = mask_marka(frame) if MARKA_WAJIB else np.zeros((h, w), bool)
    tambah("Mask marka (putih/kuning)", marka_vis.astype(np.uint8) * 255)
    marka_kiri = mask_marka_kiri(frame) if MARKA_WAJIB else None
    marka_kiri_vis = marka_kiri if marka_kiri is not None else np.zeros((h, w), bool)
    tambah("Mask marka kiri (ketat)", marka_kiri_vis.astype(np.uint8) * 255)
    aspal = mask_aspal(frame) if BAHU_SCAN else None
    if BAHU_SCAN:
        tambah("Mask aspal (tepi bahu)", aspal.astype(np.uint8) * 255)
    hough_vis = frame.copy()
    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(hough_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    tambah(f"HoughLinesP ({n_lines} segmen)", hough_vis)

    lebar_prop = state.lebar_prop if state.lebar_terkalibrasi else None
    marka = mask_marka(frame) if MARKA_WAJIB else None
    Lp, Rp = collect_segments(lines, w, h, state.left, state.right, bright, lebar_prop,
                              marka, marka_kiri)
    seg_vis = frame.copy()
    for x, y in zip(*Lp):
        cv2.circle(seg_vis, (int(x), int(y)), 2, WARNA_GARIS_KIRI, -1)
    for x, y in zip(*Rp):
        cv2.circle(seg_vis, (int(x), int(y)), 2, WARNA_GARIS_KANAN, -1)
    tambah("Klasifikasi Segmen (marka/Hough)", seg_vis)

    update_tracks(state, Lp, Rp, w, h, marka, aspal, marka_kiri)
    update_direction(state, w)

    fit_vis = frame.copy()
    th = max(2, int(round(w / 160)))
    if state.left.theta is not None:
        cv2.polylines(fit_vis, [curve_points(state.left, h, w)], False, WARNA_GARIS_KIRI, th)
    if state.right.theta is not None:
        cv2.polylines(fit_vis, [curve_points(state.right, h, w)], False, WARNA_GARIS_KANAN, th)
    if state.bahu_pts is not None:          # titik scan bahu (hanya --bahu-scan)
        for x, y in zip(*state.bahu_pts):
            cv2.circle(fit_vis, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    tambah("Fitting kurva (pas-kan marka)", fit_vis)

    hasil = finalize_lanes(state, w, h)

    koor_vis = frame.copy()
    for nama, col in (("kiri", WARNA_GARIS_KIRI), ("kanan", WARNA_GARIS_KANAN)):
        c = kurva_dari_hasil(hasil[nama], h, w)
        if c is not None:
            cv2.polylines(koor_vis, [c], False, col, th)
    gambar_koordinat(koor_vis, hasil)
    tambah(f"Koordinat final ({hasil['status']})", koor_vis)

    final = render_frame(frame, state, lines, roi_edges, poly, hasil)
    final = gambar_hasil_akhir(final, hasil, state)
    tambah(f"Hasil Akhir: {state.arah}", final)

    return stages, hasil


# ================================================================
#   SIMPAN HASIL PER TAHAP (gaya montage: bar judul di atas gambar)
# ================================================================
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]
WARNA_BAR = (30, 30, 30)          # bar judul (abu gelap, sama seperti montage)
WARNA_AKSEN = (200, 140, 60)      # garis aksen di bawah bar judul


def _kartu(img, judul):
    """Satu gambar tahap dengan bar judul di atasnya. Ukuran bar dan huruf
    mengikuti lebar gambar, jadi tetap rapi di file satuan maupun di montage."""
    img = _bgr(img)
    h, w = img.shape[:2]
    s = w / 640.0
    pad = max(8, int(round(10 * s)))
    skala = max(0.42, 0.6 * s)
    tebal = 2 if skala >= 0.55 else 1

    # kecilkan huruf bila judul terlalu panjang untuk lebar gambar
    (tw, _), _ = cv2.getTextSize(judul, FONT, skala, tebal)
    if tw > w - 2 * pad:
        skala *= (w - 2 * pad) / tw
    (_, th), _ = cv2.getTextSize(judul, FONT, skala, tebal)

    bar_h = th + 2 * pad
    kartu = np.zeros((bar_h + h, w, 3), np.uint8)
    kartu[bar_h:] = img
    kartu[:bar_h] = WARNA_BAR
    kartu[bar_h - max(2, int(round(3 * s))):bar_h] = WARNA_AKSEN
    cv2.putText(kartu, judul, (pad, pad + th), FONT, skala,
                (255, 255, 255), tebal, cv2.LINE_AA)
    cv2.rectangle(kartu, (0, 0), (w - 1, bar_h + h - 1), (70, 70, 70), 1)
    return kartu


def make_montage(stages, cols=3, tile_w=360, jarak=6):
    """Gabungkan semua tahap jadi satu grid (cols kolom) dengan jarak antar kartu."""
    tiles = []
    for title, img in stages:
        img = _bgr(img)
        r = tile_w / img.shape[1]
        img = cv2.resize(img, (tile_w, int(img.shape[0] * r)))
        tiles.append(_kartu(img, title))

    th = max(t.shape[0] for t in tiles)
    lebar_baris = cols * tile_w + (cols - 1) * jarak
    rows = []
    for i in range(0, len(tiles), cols):
        baris = []
        for k in range(cols):
            if k > 0:
                baris.append(np.zeros((th, jarak, 3), np.uint8))
            sel = np.zeros((th, tile_w, 3), np.uint8)
            if i + k < len(tiles):
                t = tiles[i + k]
                sel[:t.shape[0]] = t
            baris.append(sel)
        rows.append(np.hstack(baris))

    celah = np.zeros((jarak, lebar_baris, 3), np.uint8)
    susun = []
    for k, r in enumerate(rows):
        if k > 0:
            susun.append(celah)
        susun.append(r)
    return np.vstack(susun)


def _nama_file(judul):
    """'03 Gaussian Blur' -> '03_gaussian_blur'
    '10 Klasifikasi Segmen (marka/Hough)' -> '10_klasifikasi_segmen'"""
    bagian = judul.split(":")[0]
    bagian = re.sub(r"\(.*?\)", "", bagian)
    return re.sub(r"[^a-z0-9]+", "_", bagian.lower()).strip("_")


def save_stages(stages, out_dir="hasil_tahap", prefix="", hasil=None):
    """Simpan tiap tahap sebagai gambar satuan (ber-judul), montage, dan koordinat.json."""
    os.makedirs(out_dir, exist_ok=True)
    for title, img in stages:
        path = os.path.join(out_dir, f"{prefix}{_nama_file(title)}.jpg")
        cv2.imwrite(path, _kartu(img, title), JPEG_PARAMS)
    mont = os.path.join(out_dir, f"{prefix}00_semua_tahap.jpg")
    cv2.imwrite(mont, make_montage(stages), JPEG_PARAMS)
    print(f"[OK] {len(stages)} tahap tersimpan di folder: {out_dir}/")
    print(f"[OK] Rangkuman semua tahap: {mont}")
    if hasil is not None:
        jpath = os.path.join(out_dir, f"{prefix}koordinat.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(hasil, f, ensure_ascii=False, indent=2)
        print(f"[OK] Koordinat final: {jpath}")


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
    """FUNGSI UTAMA untuk Colab / notebook. Mengembalikan list tahap."""
    if sisi:
        set_sisi(sisi)
    h0, w0 = frame.shape[:2]
    if w0 > PROCESS_WIDTH:
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * PROCESS_WIDTH / w0)))
    state = LaneState()
    for _ in range(warmup):                  # foto cuma 1 frame -> ulang agar filter stabil
        analisis_frame(frame, state)
    stages, hasil = build_stages(frame, state)
    if simpan:
        save_stages(stages, out_dir, hasil=hasil)
    if tampil:
        show_stages(stages, per_stage=per_stage)
    return stages


# ------------------------------------------------------------------ input foto
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def show_still_result(frame, save_path=None,
                      window="Deteksi Jalur Jalan Lurus dan Belok",
                      koordinat_path=None):
    """Proses satu foto (dengan warmup), cetak koordinat, lalu tampilkan."""
    state = LaneState()

    for _ in range(12):
        out, hasil = process_frame(frame, state)

    cetak_koordinat(hasil)
    if koordinat_path:
        penulis = PenulisKoordinat(koordinat_path)
        penulis.tulis(hasil)
        penulis.tutup()
        print("Koordinat tersimpan ke:", koordinat_path)

    if save_path:
        cv2.imwrite(save_path, out)
        print("Tersimpan ke:", save_path)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, out)

    print("Jendela hasil terbuka. Tekan 'd' = debug, 'c' = angka koordinat, "
          "tombol lain (atau tutup jendela) untuk lanjut.")

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord("d"):
            state.debug = not state.debug
            out, _ = process_frame(frame, state)
            cv2.imshow(window, out)
            continue

        if key == ord("c"):
            state.show_coords = not state.show_coords
            out, _ = process_frame(frame, state)
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

    if args.roi_atas is None and not args.tanpa_roi_otomatis:
        terapkan_roi_otomatis([frame], atur_horizon=args.horizon is None)

    if args.stages:
        h, w = frame.shape[:2]
        state = LaneState()
        for _ in range(12):
            analisis_frame(frame, state)
        stages, hasil = build_stages(frame, state)
        save_stages(stages, args.stages_dir, hasil=hasil)
        cetak_koordinat(hasil)
        if args.koordinat:
            penulis = PenulisKoordinat(args.koordinat)
            penulis.tulis(hasil)
            penulis.tutup()
            print("Koordinat tersimpan ke:", args.koordinat)
        if args.save:
            state2 = LaneState()
            for _ in range(12):
                out, _ = process_frame(frame, state2)
            cv2.imwrite(args.save, out)
            print("Tersimpan ke:", args.save)
        show_stages(stages)
        return

    show_still_result(frame, args.save or None, koordinat_path=args.koordinat or None)
    cv2.destroyAllWindows()


def set_roi_atas(nilai):
    """Atur batas atas ROI (proporsi tinggi gambar). Nilai besar = ROI lebih pendek (bagian depan)."""
    global ROI_Y_ATAS, Y_TOP, ROI_PROP, Y_LEVELS
    ROI_Y_ATAS = nilai
    Y_TOP = nilai
    dx = 0.5 * (Y_BOTTOM - nilai)
    ROI_PROP = np.float32([[0.00, Y_BOTTOM], [dx, nilai], [1.0 - dx, nilai], [1.00, Y_BOTTOM]])
    Y_LEVELS = np.linspace(Y_TOP, Y_BOTTOM, N_TITIK)
    _ROI_CACHE.clear()


# ------------------------------------------------------------------ ROI otomatis
def set_roi_poligon(y_atas, x_kiri_atas, x_kanan_atas):
    """ROI berbentuk trapesium: sisi bawah selebar gambar, sisi atas di y_atas
    dengan batas kiri x_kiri_atas dan kanan x_kanan_atas (semua proporsi)."""
    global ROI_Y_ATAS, Y_TOP, ROI_PROP, Y_LEVELS
    ROI_Y_ATAS = Y_TOP = float(y_atas)
    ROI_PROP = np.float32([[0.0, Y_BOTTOM], [x_kiri_atas, y_atas],
                           [x_kanan_atas, y_atas], [1.0, Y_BOTTOM]])
    Y_LEVELS = np.linspace(Y_TOP, Y_BOTTOM, N_TITIK)
    _ROI_CACHE.clear()


def _kecilkan(img, width):
    h0, w0 = img.shape[:2]
    if w0 > width:
        return cv2.resize(img, (width, int(h0 * width / w0)))
    return img


def ambil_frame_kalibrasi(src, grabber, n, width, total=0):
    """Ambil n frame untuk kalibrasi.
    - Webcam: n frame terbaru dari FrameGrabber.
    - File video: n frame yang tersebar merata di seluruh video, agar intro atau layar
      gelap di awal tidak mengganggu. Dibaca lewat VideoCapture terpisah, jadi posisi
      video utama tidak berubah."""
    frames = []
    if grabber is not None:
        terakhir, batas = -1, time.perf_counter() + 5.0
        while len(frames) < n and time.perf_counter() < batas:
            ok, f, fid = grabber.read()
            if not ok:
                break
            if f is None or fid == terakhir:
                time.sleep(0.01)
                continue
            terakhir = fid
            frames.append(_kecilkan(f, width))
        return frames

    cap2 = cv2.VideoCapture(src)
    if not cap2.isOpened():
        return frames
    merata = total > n
    titik = np.linspace(0, total - 1, n).astype(int) if merata else range(n)
    for p in titik:
        if merata:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(p))
        ok, f = cap2.read()
        if not ok:
            break
        frames.append(_kecilkan(f, width))
    cap2.release()
    return frames


def _segmen_kalibrasi(lines, w, h, marka):
    """Pisahkan segmen Hough menjadi sisi kiri (bahu/tepi) dan kanan (marka tengah)."""
    kiri, kanan = [], []
    if lines is None:
        return kiri, kanan
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dx == 0 or dy == 0:
            continue
        slope = dy / dx
        xm = (x1 + x2) / 2.0
        ln = float(np.hypot(dx, dy))
        if -12 <= slope <= -MIN_ABS_SLOPE and xm < 0.5 * w:
            kiri.append((x1, y1, x2, y2, ln))
        elif MIN_ABS_SLOPE <= slope <= 12 and xm > 0.5 * w:
            if marka is not None:
                sx, sy = sample_segment(x1, y1, x2, y2)
                ix = np.clip(sx.astype(int), 0, w - 1)
                iy = np.clip(sy.astype(int), 0, h - 1)
                if marka[iy, ix].mean() < MARKA_FRAC:
                    continue
            kanan.append((x1, y1, x2, y2, ln))
    return kiri, kanan


def _fit_garis_segmen(segmen):
    """Garis x = a*y + b yang paling cocok dengan ujung-ujung segmen (berbobot panjang)."""
    if len(segmen) < 2:
        return None
    ys = np.array([v for s in segmen for v in (s[1], s[3])], dtype=np.float64)
    xs = np.array([v for s in segmen for v in (s[0], s[2])], dtype=np.float64)
    bobot = np.sqrt(np.repeat([s[4] for s in segmen], 2))
    A = np.column_stack([ys * bobot, bobot])
    a, b = np.linalg.lstsq(A, xs * bobot, rcond=None)[0]
    return float(a), float(b)


def kalibrasi_roi(frames):
    """Cari titik hilang jalan dari pasangan garis kiri-kanan pada frame yang sama.
    Frame yang titik hilangnya menyimpang dari median dibuang sebelum median akhir,
    supaya frame dengan tepi pepohonan/tiang tidak menggeser hasil.
    Mengembalikan (y_atas, x_kiri_atas, x_kanan_atas, y_hilang) dalam proporsi, atau None."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    y_hi = Y_BOTTOM * h
    min_len = ROI_KALIB_MIN_LEN * h
    area = np.zeros((h, w), np.uint8)
    area[int(ROI_KALIB_Y_MIN * h):int(y_hi), :] = 255

    pasangan = []                              # (a_kiri, b_kiri, a_kanan, b_kanan) per frame
    for f in frames:
        _, _, _, edges, _ = preprocess(f)
        lines = cv2.HoughLinesP(cv2.bitwise_and(edges, area), 1, np.pi / 180,
                                HOUGH_THRESH, minLineLength=HOUGH_MIN_LEN,
                                maxLineGap=HOUGH_MAX_GAP)
        marka = mask_marka(f) if MARKA_WAJIB else None
        sk, sn = _segmen_kalibrasi(lines, w, h, marka)
        sk = [seg for seg in sk if seg[4] >= min_len]
        sn = [seg for seg in sn if seg[4] >= min_len]
        pk, pn = _fit_garis_segmen(sk), _fit_garis_segmen(sn)
        if pk is not None and pn is not None:  # hanya frame yang kedua garisnya terlihat
            pasangan.append((pk[0], pk[1], pn[0], pn[1]))

    min_n = max(1, min(ROI_KALIB_MIN_FRAME, len(frames) // 2))
    if len(pasangan) < min_n:
        return None
    P = np.array(pasangan, dtype=np.float64)

    # titik hilang tiap frame: a_l*y + b_l = a_r*y + b_r  ->  y = (b_l - b_r) / (a_r - a_l)
    beda = P[:, 2] - P[:, 0]
    valid = beda > 1e-3
    yh = (P[:, 1] - P[:, 3]) / np.where(valid, beda, 1.0)
    valid &= (yh >= ROI_HILANG_MIN * h) & (yh <= ROI_HILANG_MAX * h)
    if valid.sum() < min_n:
        return None
    med = float(np.median(yh[valid]))
    valid &= np.abs(yh - med) <= ROI_KALIB_OUTLIER * h
    if valid.sum() < min_n:
        return None

    a_l, b_l, a_r, b_r = np.median(P[valid], axis=0)
    if a_r - a_l < 1e-3:
        return None
    y_hilang = (b_l - b_r) / (a_r - a_l)
    if not (ROI_HILANG_MIN * h <= y_hilang <= ROI_HILANG_MAX * h):
        return None

    y_atas_p = float(np.clip((y_hilang + ROI_TOP_FRAC * (y_hi - y_hilang)) / h,
                             ROI_TOP_MIN, ROI_TOP_MAX))
    yt = y_atas_p * h
    x_kiri = a_l * yt + b_l - ROI_MARGIN * w
    x_kanan = a_r * yt + b_r + ROI_MARGIN * w
    return (y_atas_p,
            float(np.clip(x_kiri / w, 0.0, 0.45)),
            float(np.clip(x_kanan / w, 0.55, 1.0)),
            float(y_hilang / h))


def terapkan_roi_otomatis(frames, atur_horizon=True):
    """Kalibrasi ROI dari frame-frame awal lalu terapkan. Jika gagal, ROI default dipakai."""
    global Y_HORIZON
    hasil = kalibrasi_roi(frames)
    if hasil is None:
        print("[ROI] Kalibrasi tidak berhasil (garis kurang jelas), memakai ROI default.")
        return
    y_atas, x_kiri, x_kanan, y_hilang = hasil
    set_roi_poligon(y_atas, x_kiri, x_kanan)
    if atur_horizon:                  # cakrawala ikut diperbarui untuk perhitungan perspektif
        Y_HORIZON = y_hilang
    print(f"[ROI] Otomatis -> atas {y_atas:.2f}, kiri {x_kiri:.2f}, "
          f"kanan {x_kanan:.2f}, cakrawala {y_hilang:.2f}")


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


# ================================================================
#   PEMUTARAN VIDEO, REKAM SEGMEN/WEBCAM, DAN INDIKATOR STATUS
# ================================================================
class PerekamVideo:
    """Merekam hasil video secara menyeluruh.
    Frame ditulis dulu ke file sementara (*_tmp.mp4). File final baru dibuat
    setelah video diproses sampai habis, sehingga rekaman yang diulang tetapi
    belum selesai tidak menimpa file yang sudah lengkap."""

    def __init__(self, path, fps):
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        self.path = path
        self.tmp = os.path.splitext(path)[0] + "_tmp.mp4"
        self.fps = fps
        self.writer = None
        self.frames = 0
        self.t0 = None                       # waktu frame pertama (rekaman webcam)
        self.t_mulai = time.perf_counter()   # waktu mulai merekam

    def tulis(self, img):
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(
                self.tmp, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError("Tidak bisa membuat file video: " + self.tmp)
        self.writer.write(img)
        self.frames += 1

    def tulis_berwaktu(self, img, detik):
        """Rekaman webcam: frame ditulis sesuai waktu nyata. Jika pemrosesan
        lebih lambat dari REC_FPS, frame diulang agar durasi video sama
        dengan lama merekam."""
        if self.t0 is None:
            self.t0 = detik
        target = int((detik - self.t0) * self.fps) + 1
        while self.frames < target:
            self.tulis(img)

    def durasi(self):
        return time.perf_counter() - self.t_mulai

    def _tutup(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _pindah(self, tujuan):
        try:
            os.replace(self.tmp, tujuan)
            return True
        except OSError as e:
            print("Gagal menyimpan ke", tujuan, "-", e)
            print("Rekaman masih ada di:", self.tmp)
            return False

    def selesai(self):
        """Segmen sudah diproses sampai akhir -> simpan sebagai file final."""
        self._tutup()
        if self.frames and self._pindah(self.path):
            print(f"[OK] Video hasil LENGKAP ({self.frames} frame) tersimpan: {self.path}")

    def hentikan_parsial(self):
        """Dihentikan sebelum segmen habis -> simpan sebagai *_parsial.mp4."""
        self._tutup()
        if self.frames:
            parsial = os.path.splitext(self.path)[0] + "_parsial.mp4"
            if self._pindah(parsial):
                print(f"[INFO] Rekaman belum lengkap ({self.frames} frame) "
                      f"disimpan sebagai: {parsial}")

    def batal(self):
        """Diulang dari awal -> buang rekaman sementara yang sedang berjalan."""
        self._tutup()
        if os.path.exists(self.tmp):
            os.remove(self.tmp)
        self.frames = 0


def gambar_status_putar(img, paused, selesai, merekam):
    """Indikator status: PAUSE / SELESAI di kiri bawah,
    titik merah REC di kanan bawah saat sedang merekam."""
    h, w = img.shape[:2]
    s = w / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    teks, warna = None, None
    if paused:
        teks, warna = "PAUSE  (SPASI = lanjut, R = ulang)", (0, 165, 255)
    elif selesai:
        teks, warna = "SELESAI  (R = ulang, Q = keluar)", (0, 255, 255)
    if teks:
        org = (int(8 * s), h - int(10 * s))
        cv2.putText(img, teks, org, font, 0.5 * s, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, teks, org, font, 0.5 * s, warna, 2, cv2.LINE_AA)
    if merekam:
        cx, cy = w - int(55 * s), h - int(14 * s)
        cv2.circle(img, (cx, cy), max(3, int(5 * s)), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "REC", (cx + int(10 * s), cy + int(5 * s)),
                    font, 0.45 * s, (0, 0, 255), 2, cv2.LINE_AA)


def fmt_waktu(detik):
    """Ubah detik jadi format MM:SS.cc (mis. 01:05.40)."""
    m = int(detik // 60)
    return f"{m:02d}:{detik - 60 * m:05.2f}"


def gambar_info_segmen(img, cur, total, fps, awal, akhir):
    """Posisi saat ini dan batas segmen (I/O), di kiri bawah."""
    h, w = img.shape[:2]
    s = w / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    baris = [f"Posisi {fmt_waktu(cur / fps)}"
             + (f" / {fmt_waktu(total / fps)}" if total > 0 else "")]
    if total > 0:
        durasi = (akhir - awal + 1) / fps
        baris.append(f"Segmen [I] {fmt_waktu(awal / fps)}  "
                     f"[O] {fmt_waktu(akhir / fps)}  durasi {fmt_waktu(durasi)}")
    for k, teks in enumerate(baris):
        y = h - int((28 + 16 * (len(baris) - 1 - k)) * s)
        org = (int(8 * s), y)
        cv2.putText(img, teks, org, font, 0.45 * s, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, teks, org, font, 0.45 * s, (255, 255, 255), 1, cv2.LINE_AA)


def nama_rekaman_baru(dasar):
    """Nama file rekaman baru: <dasar>_TANGGAL_JAM.mp4 (tidak menimpa rekaman lama)."""
    folder = os.path.dirname(dasar) or "."
    os.makedirs(folder, exist_ok=True)
    stem = os.path.basename(dasar)
    waktu = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{stem}_{waktu}.mp4")
    k = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{waktu}_{k}.mp4")
        k += 1
    return path


def gambar_tombol_rekam(img, merekam, durasi):
    """Tombol REKAM / STOP besar dan merah di tengah bawah layar (khusus webcam).
    Mengembalikan kotak tombol (x1, y1, x2, y2) dalam koordinat gambar."""
    h, w = img.shape[:2]
    s = w / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    bw, bh = int(150 * s), int(36 * s)
    x1 = (w - bw) // 2                       # tepat di tengah horizontal
    y2 = h - int(16 * s)
    y1 = y2 - bh
    x2 = x1 + bw
    cy = y1 + bh // 2

    merah = (0, 0, 220) if not merekam else (0, 0, 150)     # BGR
    cv2.rectangle(img, (x1, y1), (x2, y2), merah, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    if merekam:
        dt = int(durasi)
        teks = f"STOP  {dt // 60:02d}:{dt % 60:02d}"
    else:
        teks = "REKAM (V)"

    fs = 0.5 * s
    (tw, th), _ = cv2.getTextSize(teks, font, fs, 2)
    r = int(7 * s)                          # ukuran ikon
    jarak = int(7 * s)                      # jarak ikon ke teks
    grup = 2 * r + jarak + tw                # ikon + jarak + teks
    mulai = x1 + (bw - grup) // 2            # grup dipusatkan di dalam tombol
    cx = mulai + r

    if merekam:                              # ikon kotak = STOP
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), (255, 255, 255), -1)
    else:                                    # ikon lingkaran = REKAM
        cv2.circle(img, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(img, teks, (mulai + 2 * r + jarak, cy + th // 2),
                font, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return (x1, y1, x2, y2)


def pasang_klik_tombol(win, ui):
    """Klik kiri di dalam kotak tombol -> ui['klik'] = True. Koordinat klik
    diubah ke koordinat gambar, jadi tetap benar walau jendela diperbesar."""
    def saat_klik(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or ui["box"] is None:
            return
        try:
            _, _, ww, wh = cv2.getWindowImageRect(win)
        except (cv2.error, AttributeError):
            ww = wh = 0
        iw, ih = ui["size"]
        if ww > 0 and wh > 0:
            x, y = x * iw / ww, y * ih / wh
        x1, y1, x2, y2 = ui["box"]
        m = 12                               # area klik diperluas 12 px di tiap sisi
        if x1 - m <= x <= x2 + m and y1 - m <= y <= y2 + m:
            ui["klik"] = True
    cv2.setMouseCallback(win, saat_klik)


# ================================================================
#   MAIN
# ================================================================
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
        help="video file: simpan segmen ke .mp4; webcam: nama dasar rekaman "
             "(tiap rekaman jadi <nama>_TANGGAL_JAM.mp4); foto: simpan gambar (.jpg/.png)"
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
        help="tinggi cakrawala/titik hilang jalan (0-1 dari atas gambar, default 0.48). "
             "Bila tidak diisi, cakrawala ikut dikalibrasi otomatis dari video"
    )
    ap.add_argument(
        "--sisi", choices=["dua", "kiri", "kanan"], default="dua",
        help="garis yang dideteksi: dua (default, bahu kiri + garis tengah), kiri (hanya bahu), atau kanan"
    )
    ap.add_argument(
        "--lebar-jalur", type=float, default=None,
        help="lebar jalur dalam proporsi lebar gambar pada garis acuan (default 0.45). "
             "Dipakai menghitung posisi dan minggir saat mode kiri"
    )
    ap.add_argument(
        "--no-sync", action="store_true",
        help="(file video) proses SEMUA frame tanpa menyesuaikan kecepatan asli video"
    )
    ap.add_argument(
        "--koordinat", default="",
        help="simpan koordinat jalur ke file .jsonl (mis. koordinat.jsonl). "
             "Video/webcam: 1 baris per frame. Foto: 1 baris."
    )
    # --- opsi algoritma (untuk membandingkan sebelum/sesudah)
    ap.add_argument("--clahe", action="store_true",
                    help="aktifkan CLAHE (perkuat kontras marka pudar)")
    ap.add_argument("--tanpa-kalman", action="store_true",
                    help="matikan Kalman filter (smoothing hanya EMA)")
    ap.add_argument("--ransac", action="store_true",
                    help="aktifkan RANSAC (fit tahan outlier); default mati")
    ap.add_argument("--tanpa-validasi-lebar", action="store_true",
                    help="matikan validasi lebar jalur")
    ap.add_argument("--tanpa-angka", action="store_true",
                    help="jangan tampilkan angka koordinat di gambar (tombol c = on/off)")

    ap.add_argument("--tanpa-bahu", action="store_true",
                    help="matikan bahu jalan sebagai cadangan garis kiri (hanya marka)")
    ap.add_argument("--tanpa-marka", action="store_true",
                    help="matikan filter warna marka (garis apa saja boleh dipakai)")
    ap.add_argument("--bahu-scan", action="store_true",
                    help="(v14) garis bahu diambil dari tepi aspal per baris, bukan dari marka. "
                         "Untuk perbandingan")
    ap.add_argument("--roi-atas", type=float, default=None,
                    help="batas atas ROI manual (0-1). Makin besar makin pendek (bagian depan). "
                         "Jika diisi, kalibrasi ROI otomatis dilewati")
    ap.add_argument("--tanpa-roi-otomatis", action="store_true",
                    help="matikan kalibrasi ROI otomatis per video (pakai ROI default atau --roi-atas)")

    ap.add_argument("--jalur", type=int, choices=[2, 3], default=2,
                    help="2 = dua jalur (bahu + garis kanan, jalur di antaranya); "
                         "3 = tiga jalur (hanya garis tengah). Bisa diganti saat berjalan dengan tombol 2 / 3")

    args, _ = ap.parse_known_args()

    set_sisi(args.sisi)
    if args.lebar_jalur is not None:
        globals()["LEBAR_PROP_DEFAULT"] = args.lebar_jalur
    if args.tanpa_bahu:
        globals()["BAHU_CADANGAN"] = False
    if args.bahu_scan:
        globals()["BAHU_SCAN"] = True
    globals()["JALUR_MODE"] = args.jalur
    if args.tanpa_marka:
        globals()["MARKA_WAJIB"] = False
    if args.roi_atas is not None:
        set_roi_atas(args.roi_atas)
    if args.lebar_max is not None:
        globals()["MAX_LANE_GAP"] = args.lebar_max
    if args.horizon is not None:
        globals()["Y_HORIZON"] = args.horizon
    if args.clahe:
        globals()["USE_CLAHE"] = True
    if args.tanpa_kalman:
        globals()["USE_KALMAN"] = False
    if args.ransac:
        globals()["USE_RANSAC"] = True
    if args.tanpa_validasi_lebar:
        globals()["VALIDASI_LEBAR"] = False
    if args.tanpa_angka:
        globals()["SHOW_COORDS"] = False

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
    SEEK_WARMUP = 60             # frame yang diproses diam-diam sebelum posisi tujuan
    TANPA_BATAS = 10 ** 9

    def buka_sumber():
        """Buka webcam / file video."""
        cap_baru = (cv2.VideoCapture(src, cv2.CAP_DSHOW) if is_cam
                    else cv2.VideoCapture(src))
        if not cap_baru.isOpened():
            cap_baru = cv2.VideoCapture(src)
        if cap_baru.isOpened():
            cap_baru.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if is_cam:
                cap_baru.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap_baru.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap_baru.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
                cap_baru.set(cv2.CAP_PROP_FPS, 30)
        return cap_baru

    def buka_penulis():
        return PenulisKoordinat(args.koordinat) if args.koordinat else None

    cap = buka_sumber()
    if not cap.isOpened():
        print("Tidak bisa membuka sumber:", src)
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps_src <= 1 or fps_src > 240:
        fps_src = 30
    total = 0 if is_cam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # segmen yang dipilih: awal (I) dan akhir (O), dalam nomor frame
    awal = 0
    akhir = total - 1 if total > 0 else TANPA_BATAS

    # file video diputar sesuai kecepatan asli. Saat --save, diproses secepatnya
    sync = (not is_cam) and (not args.save) and (not args.no_sync)

    grabber = FrameGrabber(cap) if is_cam else None

    # kalibrasi ROI otomatis: dilakukan sekali per video, sebelum pemrosesan dimulai
    if args.roi_atas is None and not args.tanpa_roi_otomatis:
        frame_kalib = ambil_frame_kalibrasi(src, grabber, ROI_KALIB_FRAMES, args.width, total)
        terapkan_roi_otomatis(frame_kalib, atur_horizon=args.horizon is None)

    penulis = buka_penulis()
    perekam = None

    dasar_rekam = (os.path.splitext(args.save)[0] if args.save
                   else os.path.join("rekaman_webcam", "webcam"))

    def toggle_rekam():
        """Mulai / hentikan rekaman webcam (dari tombol V atau klik REKAM)."""
        nonlocal perekam
        if perekam is None:
            perekam = PerekamVideo(nama_rekaman_baru(dasar_rekam), REC_FPS)
            print("REKAM dimulai ->", perekam.path)
        else:
            perekam.selesai()          # file sementara -> file final
            perekam = None
            print("REKAM dihentikan.")

    WIN = "Deteksi Jalur Jalan Lurus dan Belok"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    ui = {"box": None, "size": (1, 1), "klik": False}   # status tombol REKAM (webcam)
    if is_cam:
        pasang_klik_tombol(WIN, ui)
    bar_terakhir = 0
    if total > 0:
        cv2.createTrackbar("Frame", WIN, 0, total - 1, lambda _: None)

    state = LaneState()
    pos = 0            # jumlah frame sumber yang sudah dibaca (= indeks frame berikutnya)
    src_idx = 0        # nomor frame sumber (untuk koordinat)
    i = 0              # jumlah frame yang diproses
    frame = None
    lines = roi_edges = poly = None
    hasil = None
    out = None
    last_id = -1
    paused = bool(args.save) and not is_cam      # --save: mulai PAUSE, atur segmen dulu
    ambil_satu = True                            # tampilkan frame pertama walau pause
    selesai_video = False
    t_start = time.perf_counter()
    t_prev, fps_show = t_start, 0.0

    if args.save and not is_cam:
        print("Mode --save: atur segmen (I/O) lalu tekan SPASI untuk mulai merekam.")

    def lompat(target):
        """Pindah ke frame ke-target. SEEK_WARMUP frame sebelumnya diproses diam-diam
        (tanpa ditampilkan/disimpan) agar garis, EMA, dan kalibrasi lebar sudah
        stabil ketika frame target tampil."""
        nonlocal state, pos, perekam
        if perekam is not None:
            perekam.hentikan_parsial()
            perekam = None
            print("Rekaman dihentikan karena posisi diloncat "
                  "(tersimpan sebagai *_parsial.mp4).")
        target = max(0, int(target))
        if total > 0:
            target = min(target, total - 1)
        start = max(0, target - SEEK_WARMUP)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        state = LaneState()
        for _ in range(start, target):
            ok, f = cap.read()
            if not ok:
                break
            h0, w0 = f.shape[:2]
            if w0 > args.width:
                f = cv2.resize(f, (args.width, int(h0 * args.width / w0)))
            analisis_frame(f, state)
        pos = target

    def gambar_ulang():
        """Gambar ulang frame terakhir (dipakai saat pause / selesai)."""
        img = render_frame(frame, state, lines, roi_edges, poly, hasil)
        return gambar_hasil_akhir(img, hasil, state)

    while True:

        # ---------- 1) user menggeser bar -> lompat (video otomatis pause) ----------
        if total > 0:
            bar = cv2.getTrackbarPos("Frame", WIN)
            if bar != bar_terakhir:
                bar_terakhir = bar
                lompat(bar)
                paused, selesai_video, ambil_satu = True, False, True

        # ---------- 2) ambil frame ----------
        baru = False
        if is_cam:
            ok, f_cam, fid = grabber.read()
            if not ok:
                break
            if f_cam is not None and fid != last_id:
                last_id = fid
                frame, src_idx, baru = f_cam, fid, True
        elif ambil_satu or (not paused and not selesai_video):
            ambil_satu = False
            if not paused and pos > akhir:
                selesai_video = True                     # sudah sampai O
            else:
                if sync and not paused:                  # jaga kecepatan asli video
                    due = int((time.perf_counter() - t_start) * fps_src)
                    while pos < min(due, akhir):
                        if not cap.grab():
                            break
                        pos += 1
                ok, f_file = cap.read()
                if ok:
                    pos += 1
                    frame, src_idx, baru = f_file, pos, True
                else:
                    selesai_video = True                 # file habis
        if selesai_video and perekam is not None:
            perekam.selesai()                            # rekaman segmen sudah lengkap
            perekam = None

        # ---------- 3) proses frame baru ----------
        if baru:
            h0, w0 = frame.shape[:2]
            if w0 > args.width:
                frame = cv2.resize(frame, (args.width, int(h0 * args.width / w0)))

            if total > 0:
                bar_terakhir = max(0, pos - 1)
                cv2.setTrackbarPos("Frame", WIN, bar_terakhir)

            if (not is_cam) and args.stages_frame > 0 and pos - 1 == args.stages_frame:
                stages, hs = build_stages(frame.copy(), copy.deepcopy(state))
                save_stages(stages, args.stages_dir, prefix=f"frame{src_idx:05d}_", hasil=hs)
                show_stages(stages)
                t_start = time.perf_counter() - pos / fps_src

            if args.skip <= 1 or i % args.skip == 0 or poly is None:
                lines, roi_edges, poly, hasil = analisis_frame(frame, state)
                hasil["sumber_frame"] = src_idx
                waktu = (time.perf_counter() - t_start) if is_cam else (src_idx / fps_src)
                hasil["waktu_detik"] = round(waktu, 3)
                if penulis:
                    penulis.tulis(hasil)

            out = gambar_hasil_akhir(
                render_frame(frame, state, lines, roi_edges, poly, hasil), hasil, state)
            i += 1

            # rekam video file: hanya saat diputar (bukan saat pause / lompat)
            if (not is_cam) and args.save and not paused and not selesai_video:
                if perekam is None:
                    perekam = PerekamVideo(args.save, fps_src)
                perekam.tulis(out)

            # rekam webcam: aktif setelah tombol REKAM ditekan
            if is_cam and perekam is not None:
                perekam.tulis_berwaktu(out, time.perf_counter())

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps_show = (1.0 / dt) if fps_show == 0 else (0.9 * fps_show + 0.1 / dt)

        # ---------- 4) jaga kecepatan video file ----------
        if sync and not paused and not selesai_video:
            tunggu = t_start + pos / fps_src - time.perf_counter()
            if tunggu > 0:
                time.sleep(tunggu)
        elif is_cam and not baru:
            time.sleep(0.002)

        # ---------- 5) tampilkan ----------
        if out is not None:
            tampil = out.copy()
            gambar_status_putar(tampil, paused, selesai_video, perekam is not None)
            if not is_cam:
                gambar_info_segmen(tampil, max(0, pos - 1), total, fps_src, awal, akhir)
            else:
                ui["box"] = gambar_tombol_rekam(
                    tampil, perekam is not None,
                    perekam.durasi() if perekam is not None else 0.0)
                ui["size"] = (tampil.shape[1], tampil.shape[0])
            cv2.putText(tampil, f"FPS {fps_show:.0f}", (tampil.shape[1] - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(WIN, tampil)

        # ---------- 6) tombol ----------
        key = cv2.waitKey(30 if (paused or selesai_video) else 1) & 0xFF

        if key == ord("q"):
            break

        if is_cam and (key == ord("v") or ui["klik"]):   # rekam on/off (webcam)
            ui["klik"] = False
            toggle_rekam()

        if key == ord(" ") and not is_cam:                 # pause / lanjut
            if paused or selesai_video:
                cur = max(0, pos - 1)
                if not (awal <= cur < akhir):              # di luar segmen -> dari awal
                    lompat(awal)
                paused, selesai_video = False, False
                t_start = time.perf_counter() - pos / fps_src
                print("LANJUT")
            else:
                paused = True
                print("PAUSE")

        if key == ord("r") and not is_cam:                 # ulang dari awal segmen
            if perekam is not None:
                perekam.batal()                            # rekaman sebelumnya dibuang
                perekam = None
            if penulis:
                penulis.tutup()
                penulis = buka_penulis()
            lompat(awal)
            paused, selesai_video, ambil_satu = False, False, False
            t_start = time.perf_counter() - pos / fps_src
            print("Diulang dari awal segmen.")

        if key == ord("i") and not is_cam:                 # tandai awal segmen
            awal = max(0, pos - 1)
            if awal > akhir:
                akhir = total - 1 if total > 0 else TANPA_BATAS
            print(f"Awal segmen = frame {awal}")

        if key == ord("o") and not is_cam:                 # tandai akhir segmen
            akhir = max(0, pos - 1)
            if akhir < awal:
                awal = 0
            print(f"Akhir segmen = frame {akhir}")

        if key == ord("x") and not is_cam:                 # reset segmen
            awal = 0
            akhir = total - 1 if total > 0 else TANPA_BATAS
            print("Segmen direset ke seluruh video.")

        if key == ord("d"):
            state.debug = not state.debug
            if (paused or selesai_video) and frame is not None:
                out = gambar_ulang()

        if key == ord("c"):
            state.show_coords = not state.show_coords
            if (paused or selesai_video) and frame is not None:
                out = gambar_ulang()

        if key in (ord("2"), ord("3")):                # ganti mode jumlah jalur
            globals()["JALUR_MODE"] = 2 if key == ord("2") else 3
            print("Mode jalur:", globals()["JALUR_MODE"], "jalur")
            if (paused or selesai_video) and frame is not None:
                lines, roi_edges, poly, hasil = analisis_frame(frame, state)
                out = gambar_ulang()

        if key == ord("p") and frame is not None:
            print("Mengambil hasil per tahap pada frame", src_idx)
            stages, hs = build_stages(frame.copy(), copy.deepcopy(state))
            save_stages(stages, args.stages_dir, prefix=f"frame{src_idx:05d}_", hasil=hs)
            show_stages(stages)
            t_start = time.perf_counter() - pos / fps_src

        if key == ord("s") and frame is not None:
            snap_name = f"snapshot_{src_idx:05d}.jpg"
            print("Snapshot diambil, memproses...")
            show_still_result(frame.copy(), snap_name, window="Snapshot - Deteksi Jalur")
            cv2.destroyWindow("Snapshot - Deteksi Jalur")
            t_start = time.perf_counter() - pos / fps_src

    # ---------- penutupan ----------
    if grabber:
        grabber.stop()
    cap.release()

    if perekam is not None:
        if is_cam:
            perekam.selesai()              # webcam dihentikan manual = rekaman lengkap
        else:
            perekam.hentikan_parsial()     # keluar sebelum segmen selesai

    if penulis:
        penulis.tutup()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()