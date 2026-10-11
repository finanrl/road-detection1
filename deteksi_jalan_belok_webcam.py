# =============================================================================
# CATATAN ISTILAH / GLOSARIUM KODE
# =============================================================================
# Catatan berikut hanya berupa KOMENTAR. Logika kode tidak diubah.
#
# A. ISTILAH DASAR PYTHON
# -----------------------------------------------------------------------------
# Python            : bahasa pemrograman yang digunakan untuk membuat program.
# import            : memasukkan library/modul ke dalam program.
# function/fungsi   : blok kode yang menjalankan tugas tertentu.
# def               : kata kunci untuk membuat fungsi.
# class             : cetak biru untuk membuat objek.
# object/objek      : data yang dibuat berdasarkan class.
# method            : fungsi yang berada di dalam class.
# self              : merujuk ke objek itu sendiri di dalam method class.
# __init__          : method yang otomatis jalan saat objek dibuat.
# variable          : tempat menyimpan data/nilai.
# parameter         : nilai yang diterima oleh fungsi.
# argument          : nilai yang diberikan ketika fungsi dipanggil.
# return            : mengembalikan hasil dari fungsi.
# None              : menunjukkan tidak ada nilai/objek.
# True / False      : nilai Boolean benar/salah.
# if                : menjalankan kode jika kondisi benar.
# elif              : kondisi alternatif.
# else              : dijalankan jika kondisi sebelumnya salah.
# for               : perulangan untuk setiap data.
# while             : perulangan selama kondisi benar.
# break             : menghentikan perulangan.
# continue          : melewati iterasi saat ini.
# try / except      : menangani error agar program tidak langsung berhenti.
# docstring         : teks di dalam """ """ tepat di bawah def/class untuk
#                     menjelaskan fungsi tersebut.
#
# A2. SINTAKS PYTHON YANG SERING MEMBINGUNGKAN
# -----------------------------------------------------------------------------
# a @ b             : perkalian matriks (dipakai pada phi.T @ phi).
# a ** 2            : a pangkat 2.
# a[::-1]           : membalik urutan array/list (contoh: rp[::-1]).
# arr.T             : transpose (baris jadi kolom).
# *lst              : "unpacking"; zip(*Lp) memecah (xs, ys) jadi dua argumen.
# zip(a, b)         : memasangkan elemen a dan b satu per satu.
# lambda / fungsi   : fungsi kecil. near() adalah fungsi di dalam fungsi
#   bersarang         (closure) yang boleh membaca variabel fungsi luarnya.
# all(...)          : True jika SEMUA kondisi di dalamnya True.
# f"...{x}..."      : f-string; {x} diganti nilai variabel x.
# x if cond else y  : "ternary"; pilih x jika cond benar, kalau tidak y.
# .extend(list)     : menambahkan banyak elemen sekaligus ke list.
# .tolist()         : mengubah NumPy array menjadi list Python.
# .astype(tipe)     : mengubah tipe data array (misal int32, uint8).
# .reshape(-1, 4)   : mengubah bentuk array jadi N baris x 4 kolom
#                     (-1 = hitung otomatis). Satu baris = x1, y1, x2, y2.
# & 0xFF            : mengambil 8 bit terakhir; trik standar membaca tombol
#                     dari cv2.waitKey().
# 255 (waitKey)     : artinya tidak ada tombol yang ditekan.
# if __name__ == "__main__": : kode di bawahnya hanya jalan jika file ini
#                     dijalankan langsung, bukan saat di-import.
#
# B. LIBRARY
# -----------------------------------------------------------------------------
# OpenCV (cv2)      : library untuk computer vision dan pengolahan gambar/video.
# NumPy (np)        : library untuk perhitungan numerik dan array.
# argparse          : library untuk membaca perintah dari terminal.
# os                : library untuk urusan file/folder (membuat folder simpan).
# copy              : library untuk menyalin objek; copy.deepcopy() membuat
#                     salinan penuh sehingga objek asli tidak berubah.
# matplotlib        : library untuk menampilkan gambar/grafik (dipakai untuk
#                     menampilkan hasil per tahap, cocok di Colab).
# google.colab      : modul khusus Google Colab; cv2_imshow menggantikan
#                     cv2.imshow yang tidak jalan di Colab.
#
# C. COMPUTER VISION
# -----------------------------------------------------------------------------
# Computer Vision   : teknologi agar komputer dapat menganalisis gambar/video.
# Frame             : satu gambar yang berasal dari video/webcam.
# Webcam            : kamera yang digunakan sebagai sumber video langsung.
# Pixel             : unit terkecil pada gambar digital.
# Resolution        : ukuran gambar berdasarkan lebar dan tinggi pixel.
# BGR               : urutan warna Blue-Green-Red yang digunakan OpenCV.
#                     (matplotlib memakai RGB, jadi perlu dikonversi.)
# Grayscale         : gambar dalam bentuk tingkat keabuan (1 kanal, 0-255).
# Blur              : proses mengurangi detail/noise pada gambar.
# Gaussian Blur     : metode blur menggunakan distribusi Gaussian.
# Kernel            : jendela kecil (misal 5x5) yang digeser di atas gambar.
# Edge/Tepi         : batas perubahan intensitas yang tajam pada gambar.
# Canny             : algoritma untuk mendeteksi tepi gambar.
# Threshold         : nilai batas untuk menentukan suatu kondisi.
# Mask              : array yang menentukan area mana yang diproses
#                     (255 = diproses, 0 = diabaikan).
# ROI               : Region of Interest, area gambar yang menjadi fokus.
# Polygon           : bentuk yang tersusun dari beberapa titik.
# Kanal (channel)   : lapisan warna. Gambar BGR punya 3 kanal, gray punya 1.
#
# D. DETEKSI GARIS
# -----------------------------------------------------------------------------
# Hough Transform   : metode untuk menemukan garis dari kumpulan titik.
# HoughLinesP       : versi probabilistic Hough Transform untuk mencari segmen.
# Line Segment      : bagian garis yang mempunyai titik awal dan akhir.
# Slope             : kemiringan garis (dy/dx).
# Curve             : garis yang melengkung.
# Polynomial        : persamaan yang menggunakan pangkat variabel.
# Quadratic         : polynomial dengan pangkat tertinggi 2.
# Fit/Fitting       : proses mencari model yang paling sesuai dengan data.
# Least Squares     : fitting dengan meminimalkan jumlah kuadrat selisih.
# Outlier           : data yang menyimpang jauh dari pola utama.
# Residual          : selisih antara data sebenarnya dan hasil model.
# MAD               : Median Absolute Deviation, ukuran penyebaran data.
#                     Dikali 1.4826 agar setara dengan simpangan baku.
# Ridge             : penalti pada fitting agar model tidak terlalu ekstrem.
#
# E. DETEKSI JALUR
# -----------------------------------------------------------------------------
# Lane              : batas/jalur jalan yang digunakan kendaraan.
# Left Lane         : garis batas jalur sebelah kiri.
# Right Lane        : garis batas jalur sebelah kanan.
# Lane Detection    : proses mendeteksi batas jalur dari gambar/video.
# LaneTrack         : class untuk menyimpan dan menstabilkan satu garis.
# LaneState         : menyimpan kondisi garis kiri, kanan, arah, dan debug.
# Segment           : bagian garis yang ditemukan oleh HoughLinesP.
# Gating            : menyaring segmen berdasarkan kedekatannya dengan garis
#                     yang sudah terdeteksi sebelumnya.
# Gate Tolerance    : batas jarak yang masih dianggap cukup dekat.
# Sampling          : mengambil beberapa titik dari sepanjang segmen.
# Persistensi       : mempertahankan garis lama ketika deteksi sementara gagal.
# Relock            : menerima posisi baru setelah perubahan dianggap konsisten.
# Jump              : perubahan posisi garis yang terlalu besar secara tiba-tiba.
#
# F. SMOOTHING
# -----------------------------------------------------------------------------
# Smoothing         : membuat hasil deteksi antar-frame menjadi lebih halus.
# EMA               : Exponential Moving Average, metode menggabungkan hasil
#                     frame baru dengan hasil frame sebelumnya.
#                     Rumus: baru = ALPHA*hasil_baru + (1-ALPHA)*hasil_lama
# ALPHA             : bobot hasil terbaru pada EMA.
#                     ALPHA kecil = lebih halus tetapi lebih lambat merespons.
#                     ALPHA besar = lebih responsif tetapi lebih mudah berubah.
# Missing           : frame ketika garis tidak berhasil dideteksi.
# MAX_MISSING       : jumlah frame gagal yang masih ditoleransi.
#
# G. ARAH BELOK
# -----------------------------------------------------------------------------
# Bend              : ukuran kelengkungan garis.
# BEND_ON           : ambang untuk mulai mendeteksi kondisi belok.
# BEND_OFF          : ambang untuk kembali ke kondisi lurus.
# Hysteresis        : penggunaan dua ambang berbeda agar status tidak mudah
#                     berubah bolak-balik karena noise.
# CURVE_SIGN        : menentukan tanda arah kelengkungan kiri/kanan.
#
# H. POSISI KENDARAAN
# -----------------------------------------------------------------------------
# Offset            : jarak posisi kendaraan dari tengah jalur.
# Lane Width        : perkiraan lebar jalur dalam meter.
# Center Lane       : titik tengah antara garis kiri dan kanan.
# OFFSET_OK_M       : batas offset yang masih dianggap baik.
#
# I. PARAMETER PADA PROGRAM
# -----------------------------------------------------------------------------
# PROCESS_WIDTH     : lebar frame yang digunakan saat pemrosesan.
# SHORT_SPAN        : data lebih pendek dari ini -> bentuk garis dikunci dari frame lalu.
# SLOPE_CHECK       : tolak garis yang arah miringnya salah (kanan/kiri tertukar).
# MAX_LANE_GAP      : lebar jalur maksimum; garis kanan yang terlalu jauh dari kiri ditolak.
# Y_HORIZON         : tinggi cakrawala/titik hilang jalan (dasar skala perspektif).
# SISI              : garis yang dideteksi: "dua", "kiri", atau "kanan".
# BLUR_KSIZE        : ukuran kernel Gaussian Blur.
# CANNY_LOW         : nilai threshold bawah Canny.
# CANNY_HIGH        : nilai threshold atas Canny.
# ROI_PROP          : koordinat ROI dalam bentuk proporsi frame (0-1).
# Y_TOP             : batas atas ROI.
# Y_BOTTOM          : batas bawah ROI.
# Y_REF             : posisi vertikal acuan untuk pengecekan garis.
# HOUGH_THRESH      : threshold HoughLinesP.
# HOUGH_MIN_LEN     : panjang minimum segmen garis.
# HOUGH_MAX_GAP     : jarak maksimum antar bagian garis.
# MIN_ABS_SLOPE     : kemiringan minimum segmen yang diterima.
# BRIGHT_DELTA      : perbedaan kecerahan minimum dari area sekitarnya.
# BRIGHT_FRACTION   : proporsi titik yang harus berada di area terang.
# BRIGHT_DILATE     : ukuran pelebaran area terang.
# GATE_TOL          : toleransi gating.
# MIN_LANE_GAP      : jarak minimum garis kiri dan kanan.
# MIN_SAMPLES       : jumlah minimum titik untuk fitting.
# MIN_SPAN          : rentang minimum data untuk fitting.
# CURVE_MIN_SPAN    : rentang minimum agar fitting menggunakan kurva.
# NOISE_FRAC        : perkiraan noise data.
# CURVE_PRIOR_FRAC  : batas kelengkungan yang dianggap wajar.
# ALPHA             : tingkat smoothing EMA.
# MAX_JUMP          : batas lompatan posisi antar-frame.
# RELOCK_AFTER      : jumlah penolakan sebelum relock.
# BEND_EMA          : smoothing untuk estimasi arah belok.
# LANE_WIDTH_M      : lebar jalur dalam meter.
#
# J. OPERASI NUMPY
# -----------------------------------------------------------------------------
# np.asarray()      : mengubah data menjadi NumPy array.
# np.array()        : membuat NumPy array.
# np.clip()         : membatasi nilai dalam rentang tertentu.
# np.minimum()      : memilih nilai minimum (elemen per elemen).
# np.maximum()      : memilih nilai maksimum (elemen per elemen).
# np.linspace()     : membuat angka dengan jarak yang merata.
# np.hypot()        : menghitung sqrt(x² + y²) (panjang garis).
# np.stack()        : menggabungkan beberapa array menjadi kolom/baris.
# np.hstack()       : menggabungkan array secara horizontal (kiri-kanan).
# np.vstack()       : menggabungkan array secara vertikal (atas-bawah).
# np.ones()         : membuat array berisi angka 1.
# np.zeros()        : membuat array berisi angka 0.
# np.zeros_like()   : array 0 dengan bentuk sama seperti array lain.
# np.full_like()    : array berisi nilai tertentu, bentuk sama dengan lain.
# np.median()       : mencari nilai tengah.
# np.average()      : menghitung rata-rata, dapat menggunakan bobot.
# np.linalg.solve() : menyelesaikan sistem persamaan linear.
# np.linalg.lstsq() : mencari solusi least-squares.
# np.diag()         : membuat matriks diagonal.
# np.abs()          : nilai absolut.
# np.atleast_1d()   : memastikan hasil berupa array (bukan nilai tunggal).
# .ravel()          : meratakan array banyak dimensi menjadi 1 dimensi.
#
# K. FUNGSI OPENCV
# -----------------------------------------------------------------------------
# cv2.cvtColor()      : mengubah format warna gambar.
# cv2.GaussianBlur()  : melakukan Gaussian Blur.
# cv2.Canny()         : mendeteksi tepi.
# cv2.fillPoly()      : mengisi polygon.
# cv2.bitwise_and()   : operasi AND pada gambar/mask.
# cv2.dilate()        : memperbesar area pada mask.
# cv2.HoughLinesP()   : mendeteksi segmen garis.
# cv2.polylines()     : menggambar garis/poliline.
# cv2.addWeighted()   : menggabungkan dua gambar dengan bobot (transparansi).
# cv2.putText()       : menulis teks pada gambar.
# cv2.getTextSize()   : mengukur ukuran teks sebelum digambar.
# cv2.rectangle()     : menggambar persegi panjang.
# cv2.circle()        : menggambar lingkaran (titik).
# cv2.line()          : menggambar garis.
# cv2.resize()        : mengubah ukuran gambar.
# cv2.copyMakeBorder(): menambah bingkai/padding di sekeliling gambar.
# cv2.imread()        : membaca gambar.
# cv2.imwrite()       : menyimpan gambar.
# cv2.VideoCapture()  : membuka webcam/video.
# cap.read()          : membaca frame berikutnya.
# cap.set()           : mengatur properti kamera/video.
# cap.get()           : mengambil properti kamera/video.
# cv2.VideoWriter()   : membuat video output.
# writer.write()      : menyimpan frame ke video.
# cv2.imshow()        : menampilkan gambar.
# cv2.waitKey()       : membaca input keyboard.
# cv2.namedWindow()   : membuat jendela OpenCV.
# cv2.destroyAllWindows() : menutup semua jendela.
#
# K2. ISTILAH FITUR "HASIL PER TAHAP" (BARU)
# -----------------------------------------------------------------------------
# Stage/Tahap         : satu langkah dalam pipeline beserta gambar hasilnya.
# build_stages()      : menjalankan pipeline 1 frame sambil mengumpulkan
#                       gambar tiap tahap ke dalam list [(judul, gambar), ...].
# save_stages()       : menyimpan tiap tahap menjadi file .jpg di folder.
# show_stages()       : menampilkan semua tahap (grid atau satu per satu).
# make_montage()      : menggabungkan semua tahap jadi satu gambar grid.
# proses_per_tahap()  : fungsi satu-pintu untuk Colab/notebook.
# Tile                : satu kotak gambar kecil di dalam montage/grid.
# Montage             : gambar gabungan berisi banyak tile.
# Warmup              : menjalankan pipeline beberapa kali pada foto yang
#                       sama agar EMA stabil (karena foto hanya 1 frame).
# deepcopy            : salinan state agar pengambilan tahap tidak mengubah
#                       state video yang sedang berjalan.
# prefix              : awalan nama file, misal "frame00120_".
# --stages            : argumen terminal; aktifkan mode per tahap (foto).
# --stages-frame N    : argumen terminal; ambil tahap pada frame ke-N (video).
# --stages-dir        : argumen terminal; folder penyimpanan hasil tahap.
# parse_known_args()  : membaca argumen terminal tanpa error bila ada
#                       argumen asing (aman untuk Colab/Jupyter).
# tombol 'p'          : saat video berjalan, simpan & tampilkan tahap.
#
# L. ALUR PROGRAM
# -----------------------------------------------------------------------------
# Input webcam/video/foto
#       ↓
# Resize (mengecilkan gambar agar proses lebih cepat dan ringan)
#       ↓                                            [Tahap 01 Original]
# Grayscale (gambar berwarna -> keabuan, pemrosesan lebih sederhana)
#       ↓                                            [Tahap 02 Grayscale]
# Gaussian Blur (menghaluskan gambar, mengurangi noise sebelum cari tepi)
#       ↓                                            [Tahap 03 Blur]
# Canny Edge (mendeteksi tepi / perubahan kecerahan yang tajam)
#       ↓                                            [Tahap 04 Canny]
# ROI / Region of Interest (hanya area jalan di depan kendaraan)
#       ↓                                [Tahap 05 ROI, 06 Edge dalam ROI]
# Brightness Filter (membuang tepi yang tidak terang, misal bayangan)
#       ↓                             [Tahap 07 Mask, 08 Edge setelah filter]
# HoughLinesP (mencari segmen garis dari titik-titik tepi)
#       ↓                                            [Tahap 09 Hough]
# Klasifikasi garis kiri/kanan (berdasarkan kemiringan & posisi)
#       ↓                                            [Tahap 10 Kiri/Kanan]
# Sampling titik (mengambil titik-titik dari tiap segmen)
#       ↓
# Fitting garis/kurva (mencari kurva x = A*t² + B*t + C yang paling cocok)
#       ↓
# Membuang outlier (titik menyimpang dibuang, fitting diulang)
#       ↓
# Gating (hanya menerima segmen yang dekat dengan posisi garis frame lalu)
#       ↓
# EMA Exponential Moving Average Smoothing (menghaluskan perubahan antar-frame)
#       ↓                                            [Tahap 11 Fitting + EMA]
# Menentukan kelengkungan (lurus atau melengkung, seberapa besar)
#       ↓
# Menentukan Lurus / Belok KIRI / Belok KANAN (dengan hysteresis)
#       ↓
# Menghitung offset posisi kendaraan (jarak dari tengah jalur, dalam meter)
#       ↓
# Visualisasi hasil (garis jalur, area hijau, panel arah & posisi)
#       ↓                                            [Tahap 12 Hasil Akhir]
# Simpan & tampilkan hasil per tahap (BARU)
#       ↓
#   save_stages()  -> hasil_tahap/*.jpg + 00_semua_tahap.jpg
#   show_stages()  -> grid matplotlib / cv2_imshow (Colab) / jendela OpenCV
#
# Catatan: tahap 11 mencakup sampling, fitting, buang outlier, gating, dan
# EMA karena semuanya terjadi di dalam LaneTrack.update() dan fit_lane().
#
# M. RUMUS MODEL KURVA
# -----------------------------------------------------------------------------
# Program menggunakan model:
#
#       x = A*t² + B*t + C
#
# A : koefisien kelengkungan.
# B : koefisien kemiringan.
# C : konstanta/intercept.
# t : posisi vertikal yang sudah dinormalisasi (t = y/tinggi - Y_BOTTOM,
#     sehingga t = 0 di dasar ROI dan negatif ke arah atas).
# theta : kumpulan parameter [A, B, C].
#
# Untuk garis lurus:   A = 0   sehingga   x = B*t + C
#
# Fitting ridge (menekan A agar jalan lurus tidak "mleyot"):
#       theta = (ΦᵀΦ + diag[ridge, ~0, ~0])⁻¹ Φᵀ x
# Φ = matriks [t², t, 1].
#
# N. WARNA BGR OPENCV
# -----------------------------------------------------------------------------
# (0, 0, 255)       : merah
# (0, 255, 255)     : kuning
# (0, 255, 0)       : hijau
# (255, 0, 0)       : biru
# (255, 255, 0)     : cyan
# (255, 255, 255)   : putih
# (0, 165, 255)     : oranye
#
# Pada program: garis KIRI = merah, garis KANAN = kuning,
# area jalur = hijau, batas ROI = biru, segmen Hough = cyan.
#
# O. ARGUMEN TERMINAL
# -----------------------------------------------------------------------------
# --source       : sumber webcam, video, atau foto.
# --width        : lebar frame pemrosesan.
# --save         : nama file hasil.
# --skip         : jumlah frame yang dilewati untuk mengurangi beban CPU.
# --stages       : (foto) tampilkan & simpan hasil per tahap.
# --stages-frame : (video) ambil hasil per tahap pada frame ke-N.
# --stages-dir   : folder hasil per tahap (default hasil_tahap).
#
# Contoh:
# Webcam : python deteksi_jalan_belok_webcam.py --source 0
# Video  : python deteksi_jalan_belok_webcam.py --source video_jalan.mp4
# Simpan : python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --save hasil.mp4
# Foto, per tahap  : python deteksi_jalan_belok_webcam.py --source jalan.jpg --stages
# Video, per tahap : python deteksi_jalan_belok_webcam.py --source video.mp4 --stages-frame 120
#
# Tombol: q = keluar | d = debug | s = snapshot | p = hasil per tahap
#
# Di Google Colab:
#     frame = cv2.imread("jalan.jpg")
#     proses_per_tahap(frame, out_dir="hasil_tahap")
#     proses_per_tahap(frame, per_stage=True)   # satu per satu, lebih besar
#
# =============================================================================
# P. CARA PAKAI (LENGKAP)
# -----------------------------------------------------------------------------
# 1) PERSIAPAN
#    - Python 3.8 atau lebih baru.
#    - Pasang library lewat terminal / CMD:
#          pip install opencv-python numpy matplotlib
#    - Di Google Colab, library di atas sudah tersedia (tidak perlu pip).
#    - Simpan file ini sebagai: deteksi_jalan_belok_webcam.py
#      lalu taruh satu folder dengan foto/video yang akan diproses.
#
# 2) MENJALANKAN DI KOMPUTER (TERMINAL / CMD)
#    Buka terminal di folder file ini, lalu pilih salah satu:
#
#    a. Webcam (kamera nomor 0):
#          python deteksi_jalan_belok_webcam.py --source 0
#
#    b. Video:
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4
#
#    c. Foto:
#          python deteksi_jalan_belok_webcam.py --source jalan.jpg
#
#    d. Simpan hasil akhir (video atau foto):
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --save hasil.mp4
#          python deteksi_jalan_belok_webcam.py --source jalan.jpg --save hasil.jpg
#
#    f0. Area hijau melebar sampai tepi jalan kanan (bukan marka tengah):
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --lebar-max 0.5 --horizon 0.48
#       --lebar-max : lebar jalur maksimum (proporsi lebar gambar, di bagian
#                     bawah). Makin kecil = garis kanan makin dekat ke kiri.
#                     Default 0 = dimatikan.
#       --horizon   : tinggi cakrawala / titik hilang jalan, diukur dari atas
#                     gambar (0-1). Lihat di mana jalan tampak menyempit jadi
#                     satu titik. Default 0.48.
#
#    f. Hanya deteksi garis KIRI (atau kanan), tanpa area hijau:
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --sisi kiri
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --sisi kanan
#       Di Colab: proses_per_tahap(frame, sisi="kiri")
#
#    Anti-delay & anti-loncat (otomatis, tanpa perlu opsi):
#      - Webcam dibaca di thread terpisah dan hanya frame TERBARU yang diproses,
#        jadi tidak ada delay akibat frame menumpuk. Kamera juga diatur ke
#        format MJPG 30 FPS agar lebih lancar.
#      - File video diputar dengan kecepatan ASLI: frame yang terlambat
#        dilewati, bukan slow motion. Angka FPS tampil di pojok kanan atas.
#      - Pakai --save atau --no-sync bila ingin SEMUA frame diproses
#        (video hasil menjadi lengkap, tetapi tampilan bisa lebih lambat):
#            python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --no-sync
#
#    e. Jika laptop terasa berat / lag:
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --skip 2
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --width 480
#
# 3) MENAMPILKAN & MENYIMPAN HASIL PER TAHAP (UNTUK PRESENTASI / DITANYA DOSEN)
#    a. Dari FOTO (paling mudah):
#          python deteksi_jalan_belok_webcam.py --source jalan.jpg --stages
#       Hasil: grid 12 tahap tampil di layar + tersimpan di folder hasil_tahap/
#
#    b. Dari VIDEO, otomatis pada frame tertentu (misal frame ke-120):
#          python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --stages-frame 120
#
#    c. Dari VIDEO / WEBCAM, manual: jalankan seperti biasa, lalu tekan
#       tombol  p  tepat di saat tampilan yang ingin dijelaskan.
#
#    d. Ganti folder penyimpanan:
#          python deteksi_jalan_belok_webcam.py --source jalan.jpg --stages --stages-dir tahap_foto1
#
#    Folder default = hasil_tahap/<nama_sumber>, misalnya:
#          lurus.jpg -> hasil_tahap/lurus/   |   belok.jpg -> hasil_tahap/belok/
#    sehingga hasil jalan lurus dan jalan belok TIDAK saling menimpa.
#    (Di Colab, beri out_dir berbeda: proses_per_tahap(frame, out_dir="tahap_belok"))
#
#    Isi tiap folder (foto: tanpa awalan; video: awalan frameNNNNN_):
#          00_semua_tahap.jpg  -> rangkuman 12 tahap dalam 1 gambar
#          01_original.jpg
#          02_grayscale.jpg
#          03_gaussian_blur.jpg
#          04_canny_edge.jpg
#          05_roi.jpg
#          06_edge_di_dalam_roi.jpg
#          07_mask_brightness.jpg
#          08_edge_setelah_brightness_filter.jpg
#          09_houghlinesp.jpg
#          10_klasifikasi_kiri.jpg
#          11_fitting_kurva_+_ema.jpg
#          12_hasil_akhir.jpg
#
# 4) TOMBOL KEYBOARD (SAAT JENDELA VIDEO TERBUKA)
#          q : keluar
#          d : tampilkan / sembunyikan garis Hough + inset edge (debug)
#          s : snapshot -> proses 1 frame lalu simpan snapshot_NNNNN.jpg
#          p : simpan & tampilkan hasil per tahap pada frame saat ini
#
# 5) MENJALANKAN DI GOOGLE COLAB
#    Colab tidak punya layar kamera lokal dan cv2.imshow tidak berfungsi,
#    jadi gunakan fungsi proses_per_tahap(). Tempel ke sel (cell) notebook:
#
#    --- Sel 1: unggah foto
#          from google.colab import files
#          up = files.upload()                # pilih foto jalan, mis. jalan.jpg
#
#    --- Sel 2: tempel / unggah file deteksi_jalan_belok_webcam.py ke Colab, lalu
#          from deteksi_jalan_belok_webcam import proses_per_tahap
#          import cv2
#          frame = cv2.imread("jalan.jpg")
#          proses_per_tahap(frame, out_dir="hasil_tahap")
#
#    --- Sel 3 (opsional): tampil satu per satu, ukuran besar
#          proses_per_tahap(frame, per_stage=True)
#
#    --- Sel 4 (opsional): unduh semua hasil sebagai zip
#          !zip -r hasil_tahap.zip hasil_tahap
#          files.download("hasil_tahap.zip")
#
#    Untuk VIDEO di Colab: jalankan dengan --save lalu unduh hasilnya
#    (tampilan jendela tidak tersedia di Colab):
#          !python deteksi_jalan_belok_webcam.py --source video_jalan.mp4 --save hasil.mp4
#    Atau ambil satu frame video lalu pakai proses_per_tahap():
#          cap = cv2.VideoCapture("video_jalan.mp4")
#          cap.set(cv2.CAP_PROP_POS_FRAMES, 120)   # loncat ke frame ke-120
#          ok, frame = cap.read()
#          proses_per_tahap(frame)
#
# 6) MENYESUAIKAN PARAMETER BILA HASIL KURANG PAS
#    (ubah di blok PARAMETER dekat bagian atas kode)
#    - Garis tidak terdeteksi              -> setel ROI_PROP agar mencakup jalan;
#                                             turunkan HOUGH_THRESH / HOUGH_MIN_LEN
#    - Terlalu banyak garis liar / bayangan -> naikkan BRIGHT_DELTA, atau naikkan
#                                             CANNY_LOW
#    - Marka tidak terang (jalan tanpa cat) -> set BRIGHT_DELTA = 0 untuk
#                                             mematikan brightness filter
#    - Garis bergoyang                     -> kecilkan ALPHA (mis. 0.2)
#    - Garis lambat mengikuti belokan      -> besarkan ALPHA (mis. 0.4)
#    - Belok KIRI/KANAN terbalik           -> ubah CURVE_SIGN menjadi -1
#    - Status sering berganti Lurus/Belok  -> naikkan BEND_ON, jaga BEND_OFF
#                                             tetap lebih kecil dari BEND_ON
#    - Ukuran jalur salah (posisi/offset)  -> sesuaikan LANE_WIDTH_M
#
# 7) MASALAH UMUM
#    - "Tidak bisa membuka sumber"         -> cek nama/path file video, atau coba
#                                             nomor webcam lain (--source 1)
#    - "Tidak bisa membuka foto"           -> cek nama file & ekstensinya
#    - ModuleNotFoundError: cv2            -> jalankan: pip install opencv-python
#    - Jendela tidak muncul di Colab       -> normal; pakai proses_per_tahap()
#    - Grid per tahap tidak tampil         -> pasang matplotlib; tanpanya
#                                             program memakai jendela OpenCV
#
# =============================================================================
# CATATAN: KODE ASLI DIMULAI DI BAWAH INI
# =============================================================================

"""
Deteksi Jalur Jalan Lurus & Belok - Webcam / Video / Foto
=========================================================
Alur:
  Input (webcam/video) -> Grayscale -> Gaussian Blur -> Canny Edge -> ROI
  -> HoughLinesP -> klasifikasi segmen kiri/kanan (+ gating posisi frame lalu)
  -> fit kurva berbobot (lurus / belok) -> Smoothing Filter antar-frame (EMA)
  -> visualisasi + panel arah (Lurus / Belok KIRI / Belok KANAN)

Penstabilan (smoothing filter) yang dipakai:
  1. Gating      : segmen baru hanya diterima kalau dekat dengan garis frame
                   sebelumnya, jadi tidak "meloncat" ke tepi pasir/rumput.
  2. Fit kokoh   : titik pencilan dibuang, kelengkungan diberi penalti (ridge)
                   sehingga jalan lurus tetap lurus dan tidak "mleyot".
  3. EMA         : koefisien garis digabung dengan frame sebelumnya
                   (ALPHA kecil = makin halus).
  4. Tolak lompat: perubahan mendadak ditolak; garis dipertahankan.
  5. Persistensi : garis tidak langsung hilang saat beberapa frame gagal.
"""

import argparse
import copy       # untuk deepcopy state (hasil per tahap pada video)
import os         # untuk membuat folder penyimpanan hasil tahap
import threading  # membaca kamera di thread terpisah (anti-delay)
import time       # pengatur kecepatan video & hitung FPS

import cv2
import numpy as np

# ============================ PARAMETER (silakan setel) ============================
PROCESS_WIDTH = 640

# --- Grayscale -> Gaussian Blur -> Canny
BLUR_KSIZE = 5                     # harus ganjil (3, 5, 7, ...)
CANNY_LOW, CANNY_HIGH = 50, 150    # tepi di bawah 50 dibuang, di atas 150 pasti tepi

# --- ROI (titik poligon dalam proporsi lebar/tinggi: [x, y])
# urutan titik: kiri-bawah, kiri-atas, kanan-atas, kanan-bawah (bentuk trapesium)
ROI_PROP = np.float32([[0.04, 0.92], [0.28, 0.60], [0.64, 0.60], [0.78, 0.92]])
Y_TOP, Y_BOTTOM = 0.60, 0.92
Y_REF = 0.88                       # tinggi acuan untuk cek lompatan & hitung offset

# --- HoughLinesP
HOUGH_THRESH = 25
HOUGH_MIN_LEN = 30
HOUGH_MAX_GAP = 40
MIN_ABS_SLOPE = 0.35
BRIGHT_DELTA = 35
BRIGHT_FRACTION = 0.4
BRIGHT_DILATE = 7

# --- Gating & fit
GATE_TOL = 0.07
MIN_LANE_GAP = 0.12
MAX_LANE_GAP = 0.0    # lebar jalur maksimum (proporsi lebar gambar) di ketinggian Y_REF; 0 = nonaktif
Y_HORIZON = 0.48      # tinggi cakrawala / titik hilang jalan (proporsi tinggi gambar)
MIN_SAMPLES = 8
MIN_SPAN = 0.08           # rentang data minimum agar garis dibuat (sebelumnya 0.06)
SHORT_SPAN = 0.18         # data lebih pendek dari ini -> kemiringan "dikunci" dari garis lama
SLOPE_CHECK = True        # tolak garis dengan arah miring yang salah (kanan harus
                          # turun ke kanan, kiri harus turun ke kiri)
CURVE_MIN_SPAN = 0.10
NOISE_FRAC = 0.006
CURVE_PRIOR_FRAC = 0.030

# --- Smoothing filter
ALPHA = 0.30
MAX_JUMP = 0.08
RELOCK_AFTER = 6
MAX_MISSING = 15

# --- Arah belok
BEND_ON, BEND_OFF = 0.018, 0.010   # hysteresis: ON > OFF supaya tidak bolak-balik
BEND_EMA = 0.15
CURVE_SIGN = 1                     # ubah ke -1 jika arah kiri/kanan terbalik
LANE_WIDTH_M = 3.5
OFFSET_OK_M = 0.8

# --- Sisi yang dideteksi: "dua" (kiri+kanan), "kiri" (hanya garis kiri),
#     atau "kanan" (hanya garis kanan). Bisa diganti lewat --sisi kiri
SISI = "dua"
# ==================================================================================


# ------------------------------------------------------------------ model garis
def eval_curve(theta, dom, t):
    """x = A t^2 + B t + C, dipakai hanya di dalam rentang data (dom). Di luar
    rentang, garis diperpanjang LURUS mengikuti garis singgung."""
    A, B, C = theta                          # unpacking: pecah theta jadi 3 variabel
    t_lo, t_hi = dom                         # batas bawah & atas data (domain)
    t = np.asarray(t, dtype=np.float64)
    tc = np.clip(t, t_lo, t_hi)              # t dibatasi di dalam domain
    x = A * tc ** 2 + B * tc + C             # nilai kurva di dalam domain
    # bagian di luar domain: tambahkan garis singgung (lurus) = kemiringan * jarak
    # kemiringan di ujung = turunan 2*A*t + B
    x = x + (2 * A * t_lo + B) * np.minimum(t - t_lo, 0.0)   # di bawah t_lo
    x = x + (2 * A * t_hi + B) * np.maximum(t - t_hi, 0.0)   # di atas t_hi
    return x


def to_t(y, h):
    """Ubah koordinat y (pixel) menjadi t ternormalisasi (0 di dasar ROI)."""
    return np.asarray(y, dtype=np.float64) / h - Y_BOTTOM


def _arah_ok(fit, side):
    """Tolak garis yang arah miringnya salah.
    Pada citra, garis KANAN jalur: makin ke bawah makin ke kanan (dx/dt > 0).
    Garis KIRI jalur: makin ke bawah makin ke kiri (dx/dt < 0)."""
    if fit is None or side is None or not SLOPE_CHECK:
        return fit
    th, dom = fit
    t_mid = 0.5 * (dom[0] + dom[1])
    dxdt = 2 * th[0] * t_mid + th[1]          # turunan x terhadap t
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
    if len(xs) < MIN_SAMPLES:                # titik terlalu sedikit -> gagal
        return None
    t = to_t(ys, h)
    span = t.max() - t.min()                 # rentang vertikal data
    if span < MIN_SPAN:                      # data terlalu pendek -> gagal
        return None
    use_curve = span >= CURVE_MIN_SPAN       # kurva hanya jika data cukup panjang

    # DATA PENDEK + sudah ada garis dari frame sebelumnya:
    # kemiringan & lengkung dikunci dari garis lama, hanya posisinya (C)
    # yang dicari. Kemiringan dari serpihan pendek tidak bisa dipercaya dan
    # saat diperpanjang ke bawah membuat garis bergoyang.
    if prior is not None and span < SHORT_SPAN:
        A0, B0 = float(prior[0]), float(prior[1])
        c = float(np.median(xs - (A0 * t ** 2 + B0 * t)))   # median: kebal pencilan
        fit = (np.array([A0, B0, c]), (float(t.min()), float(t.max())))
        return _arah_ok(fit, side)

    # ridge: seberapa kuat kelengkungan (A) "ditarik" ke nol
    t_top = Y_TOP - Y_BOTTOM
    sigma_a = CURVE_PRIOR_FRAC * w / (t_top ** 2)
    ridge = (NOISE_FRAC * w / sigma_a) ** 2

    keep = np.ones(len(xs), dtype=bool)      # penanda titik yang dipakai (True)
    best = None
    for _ in range(3):                       # maksimal 3 putaran buang outlier
        tt, xx = t[keep], xs[keep]           # ambil hanya titik yang dipakai
        if len(xx) < MIN_SAMPLES:
            break
        if use_curve:
            # Φ = kolom [t², t, 1]; @ = perkalian matriks; .T = transpose
            phi = np.stack([tt ** 2, tt, np.ones_like(tt)], axis=1)
            g = phi.T @ phi + np.diag([ridge, 1e-9, 1e-9])   # ridge hanya pada A
            th = np.linalg.solve(g, phi.T @ xx)              # cari [A, B, C]
        else:
            phi = np.stack([tt, np.ones_like(tt)], axis=1)
            b, c = np.linalg.lstsq(phi, xx, rcond=None)[0]   # garis lurus
            th = np.array([0.0, b, c])                       # A dipaksa 0
        best = (th, (float(tt.min()), float(tt.max())))
        res = xs - (th[0] * t ** 2 + th[1] * t + th[2])      # residual semua titik
        # MAD * 1.4826 ≈ simpangan baku yang kebal terhadap outlier
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        new_keep = np.abs(res) < max(0.012 * w, 2.5 * mad)   # titik dekat kurva
        if new_keep.sum() < MIN_SAMPLES or np.array_equal(new_keep, keep):
            break                            # tidak ada perubahan -> selesai
        keep = new_keep
    return _arah_ok(best, side)


class LaneTrack:
    """Satu sisi garis + smoothing filter (EMA) + tolak lompatan + persistensi."""

    def __init__(self):
        self.theta = None      # parameter [A, B, C] garis saat ini
        self.dom = None        # rentang t yang punya data
        self.missing = 0       # berapa frame berturut-turut gagal terdeteksi
        self.rejects = 0       # berapa kali berturut-turut hasil ditolak (lompat)

    def x_at(self, y, h):
        """Posisi x garis pada ketinggian y."""
        if self.theta is None:
            return None
        return float(eval_curve(self.theta, self.dom, to_t(y, h)))

    def update(self, fit, w, h):
        """Gabungkan hasil fitting frame baru dengan garis lama."""
        if fit is None:                      # frame ini gagal -> pertahankan garis
            self._miss()
            return
        theta, dom = fit
        dom = np.array(dom)
        if self.theta is None:               # belum punya garis -> langsung terima
            self.theta, self.dom, self.missing, self.rejects = theta, dom, 0, 0
            return
        # cek lompatan posisi pada ketinggian acuan Y_REF
        new_x = float(eval_curve(theta, dom, to_t(Y_REF * h, h)))
        jump = abs(new_x - self.x_at(Y_REF * h, h))
        if jump > MAX_JUMP * w:              # lompatan terlalu besar -> tolak
            self.rejects += 1
            if self.rejects >= RELOCK_AFTER: # konsisten berpindah -> terima (relock)
                self.theta, self.dom, self.missing, self.rejects = theta, dom, 0, 0
            else:
                self._miss()
            return
        # EMA: campur hasil baru (ALPHA) dengan hasil lama (1-ALPHA)
        self.theta = ALPHA * theta + (1 - ALPHA) * self.theta
        self.dom = ALPHA * dom + (1 - ALPHA) * np.array(self.dom)
        self.missing, self.rejects = 0, 0

    def _miss(self):
        """Catat kegagalan; hapus garis jika gagal terlalu lama."""
        self.missing += 1
        if self.missing > MAX_MISSING:
            self.theta, self.dom, self.rejects = None, None, 0


class LaneState:
    """Menyimpan seluruh kondisi deteksi yang dibawa dari frame ke frame."""

    def __init__(self):
        self.left = LaneTrack()
        self.right = LaneTrack()
        self.bend = 0.0          # kelengkungan (sudah dihaluskan EMA)
        self.arah = "Lurus"      # status arah saat ini
        self.debug = False       # tampilkan Hough + edge atau tidak


# ------------------------------------------------------------------ tahap pipeline
def preprocess(frame):
    """Grayscale -> Gaussian Blur -> Canny. Mengembalikan ketiga hasilnya."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return gray, blur, edges


_ROI_CACHE = {}   # penyimpanan sementara agar ROI tidak dihitung ulang tiap frame
# kernel dilate dibuat sekali saja (sebelumnya dibuat ulang tiap frame)
_BRIGHT_KERNEL = np.ones((BRIGHT_DILATE, BRIGHT_DILATE), np.uint8)


def get_roi(w, h):
    """Hitung polygon & mask ROI sekali saja per resolusi."""
    key = (w, h)
    if key not in _ROI_CACHE:
        # proporsi (0-1) dikali ukuran gambar -> koordinat pixel
        poly = (ROI_PROP * np.float32([w, h])).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)   # mask awal hitam semua
        cv2.fillPoly(mask, [poly], 255)           # area ROI diputihkan
        _ROI_CACHE.clear()
        _ROI_CACHE[key] = (poly, mask)
    return _ROI_CACHE[key]


def apply_roi(edges, mask):
    """Pertahankan edge hanya di dalam area ROI."""
    return cv2.bitwise_and(edges, mask)


def sample_segment(x1, y1, x2, y2, step=6.0):
    """Ambil titik-titik merata sepanjang segmen (jarak ± step pixel)."""
    n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
    return np.linspace(x1, x2, n), np.linspace(y1, y2, n)


def collect_segments(lines, w, h, left, right, bright=None):
    """Pisahkan segmen Hough ke kiri/kanan."""
    L, R = ([], []), ([], [])    # masing-masing: (daftar x, daftar y)
    if lines is None:
        return L, R
    tol = GATE_TOL * w           # toleransi gating dalam pixel

    # reshape(-1, 4): tiap baris = satu segmen [x1, y1, x2, y2]
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:              # garis tegak lurus -> slope tak terhingga
            continue

        slope = dy / dx
        # buang segmen terlalu mendatar atau terlalu tegak
        if abs(slope) < MIN_ABS_SLOPE or abs(slope) > 12:
            continue

        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0    # titik tengah segmen
        ends = ((x1, y1), (x2, y2))
        sx, sy = sample_segment(x1, y1, x2, y2)

        if bright is not None:
            # cek apakah titik-titik segmen berada di area terang
            ix = np.clip(sx.astype(int), 0, w - 1)
            iy = np.clip(sy.astype(int), 0, h - 1)
            if bright[iy, ix].mean() < BRIGHT_FRACTION:
                continue

        def near(track):
            """True jika kedua ujung segmen dekat dengan garis track."""
            return all(
                abs(x - track.x_at(y, h)) < tol
                for x, y in ends
            )

        if left.theta is not None or right.theta is not None:
            # ada garis dari frame sebelumnya -> pakai gating
            side = None

            # di citra, garis kiri miring negatif (slope<0), kanan positif
            if left.theta is not None and slope < 0 and near(left):
                side = "L"
            elif right.theta is not None and slope > 0 and near(right):
                side = "R"
            elif left.theta is None and slope < 0 and xm < 0.62 * w:
                side = "L"
            elif right.theta is None and slope > 0 and xm > 0.38 * w:
                side = "R"

            # kiri-kanan harus berjarak minimal MIN_LANE_GAP
            if side == "R" and left.theta is not None and \
                    xm < left.x_at(ym, h) + MIN_LANE_GAP * w:
                side = None

            if side == "L" and right.theta is not None and \
                    xm > right.x_at(ym, h) - MIN_LANE_GAP * w:
                side = None

            # batas lebar maksimum: garis kanan tidak boleh terlalu jauh dari
            # garis kiri (mencegah tepi jalan paling kanan ikut terdeteksi).
            # Lebar diskalakan karena perspektif: jalur tampak makin sempit
            # menuju cakrawala (Y_HORIZON), dan lebarnya 0 tepat di cakrawala.
            if MAX_LANE_GAP > 0 and side == "R" and left.theta is not None and \
                    xm > left.x_at(ym, h) + MAX_LANE_GAP * w * \
                    max(ym / h - Y_HORIZON, 0.01) / (Y_REF - Y_HORIZON):
                side = None

        else:
            # belum ada garis sama sekali -> tebak dari kemiringan & posisi
            side = (
                "L" if (slope < 0 and xm < 0.62 * w)
                else ("R" if (slope > 0 and xm > 0.38 * w) else None)
            )

        # mode satu sisi: abaikan segmen dari sisi yang tidak diinginkan
        if (side == "R" and SISI == "kiri") or (side == "L" and SISI == "kanan"):
            side = None

        if side:
            tgt = L if side == "L" else R
            tgt[0].extend(sx.tolist())   # tambah x
            tgt[1].extend(sy.tolist())   # tambah y

    return L, R


def bend_fraction(state, w):
    """Mengukur kelengkungan garis."""
    t_top = Y_TOP - Y_BOTTOM
    vals, wts = [], []

    for tr in (state.left, state.right):
        if tr.theta is not None:
            span = float(tr.dom[1] - tr.dom[0])
            vals.append(tr.theta[0] * t_top ** 2 / w)   # simpangan di ujung atas
            wts.append(span ** 2)                       # data panjang = lebih dipercaya

    if not vals:
        return None

    b = float(np.average(vals, weights=wts))

    if len(vals) == 1:       # hanya satu garis -> kurangi kepercayaan
        b *= 0.6

    return b * CURVE_SIGN


def update_direction(state, w):
    """Tentukan Lurus / Belok KIRI / Belok KANAN dengan hysteresis."""
    b = bend_fraction(state, w)

    if b is None:
        return

    # EMA pada kelengkungan agar status tidak berkedip
    state.bend = (1 - BEND_EMA) * state.bend + BEND_EMA * b

    if state.arah == "Lurus":
        # dari lurus -> belok butuh melewati ambang BESAR (BEND_ON)
        if abs(state.bend) > BEND_ON:
            state.arah = "Belok KANAN" if state.bend > 0 else "Belok KIRI"
    else:
        # dari belok -> lurus butuh turun di bawah ambang KECIL (BEND_OFF)
        if abs(state.bend) < BEND_OFF:
            state.arah = "Lurus"
        else:
            state.arah = "Belok KANAN" if state.bend > 0 else "Belok KIRI"


# ------------------------------------------------------------------ gambar
def curve_points(track, h, w):
    """Titik-titik kurva (40 titik dari Y_TOP ke Y_BOTTOM) siap digambar."""
    y = np.linspace(Y_TOP * h, Y_BOTTOM * h, 40)
    x = eval_curve(track.theta, track.dom, to_t(y, h))
    return np.stack([x, y], axis=1).astype(np.int32)


def draw_panel(img, state, offset_m, both, any_line):
    """Gambar panel informasi (arah, kelengkungan, posisi, status) di pojok kiri atas."""
    h, w = img.shape[:2]
    s = w / 640.0            # faktor skala agar teks menyesuaikan ukuran gambar

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

        # tiap baris: (teks, warna BGR, ukuran font)
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
                "Status: Baik"
                if baik
                else "Status: Perhatikan posisi!",
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
    # ukur dulu semua teks agar panel pas ukurannya
    sizes = [
        cv2.getTextSize(
            t,
            font,
            sc * s,
            2 if sc > 0.5 else 1
        )[0]
        for t, _, sc in lines
    ]

    pw = int(max(sz[0] for sz in sizes) + 24 * s)       # lebar panel
    line_h = [int(sz[1] + 9 * s) for sz in sizes]       # tinggi tiap baris
    ph = int(sum(line_h) + 10 * s)                      # tinggi panel

    # latar panel gelap semi-transparan
    roi = img[0:ph, 0:pw]
    img[0:ph, 0:pw] = cv2.addWeighted(
        roi,
        0.35,
        np.full_like(roi, 30),
        0.65,
        0
    )

    cv2.rectangle(
        img,
        (0, 0),
        (pw, ph),
        (255, 255, 255),
        1
    )

    y = 0

    for (txt, col, sc), lh in zip(lines, line_h):
        y += lh
        cv2.putText(
            img,
            txt,
            (int(12 * s), int(y)),
            font,
            sc * s,
            col,
            2 if sc > 0.5 else 1,
            cv2.LINE_AA          # LINE_AA = tepi huruf halus (anti-aliasing)
        )


def detect_lanes(frame, state):
    """Tahap deteksi jalur."""
    h, w = frame.shape[:2]

    gray, blur, edges = preprocess(frame)

    poly, roi_mask = get_roi(w, h)
    roi_edges = apply_roi(edges, roi_mask)

    bright = None

    if BRIGHT_DELTA > 0:
        # median kecerahan di dalam ROI = patokan kecerahan "jalan biasa"
        med = float(np.median(blur[roi_mask > 0]))

        # piksel yang lebih terang dari patokan (+BRIGHT_DELTA), lalu dilebarkan
        bright = cv2.dilate(
            (blur > med + BRIGHT_DELTA).astype(np.uint8),
            _BRIGHT_KERNEL
        ) > 0

        # edge yang tidak berada di area terang dibuang (dikali 0)
        roi_edges = roi_edges * bright.astype(np.uint8)

    lines = cv2.HoughLinesP(
        roi_edges,
        1,                  # resolusi jarak (pixel)
        np.pi / 180,        # resolusi sudut (1 derajat)
        HOUGH_THRESH,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP
    )

    Lp, Rp = collect_segments(
        lines,
        w,
        h,
        state.left,
        state.right,
        bright
    )

    state.left.update(
        fit_lane(Lp[0], Lp[1], w, h, "L", state.left.theta),
        w,
        h
    )

    state.right.update(
        fit_lane(Rp[0], Rp[1], w, h, "R", state.right.theta),
        w,
        h
    )

    update_direction(state, w)

    return lines, roi_edges, poly


def render_frame(frame, state, lines=None, roi_edges=None, poly=None):
    """Gambar garis jalur, area hijau, ROI, panel info di atas frame."""
    h, w = frame.shape[:2]

    overlay = np.zeros_like(frame)   # lapisan transparan tempat menggambar

    lp = (
        curve_points(state.left, h, w)
        if state.left.theta is not None
        else None
    )

    rp = (
        curve_points(state.right, h, w)
        if state.right.theta is not None
        else None
    )

    th = max(3, int(round(w / 90)))  # ketebalan garis mengikuti lebar gambar

    if lp is not None:
        cv2.polylines(
            overlay,
            [lp],
            False,
            (0, 0, 255),     # kiri = merah
            th
        )

    if rp is not None:
        cv2.polylines(
            overlay,
            [rp],
            False,
            (0, 255, 255),   # kanan = kuning
            th
        )

    offset_m, both = 0.0, lp is not None and rp is not None

    if both:
        # isi area di antara garis kiri dan kanan (rp dibalik agar poligon menutup)
        cv2.fillPoly(
            overlay,
            [np.vstack([lp, rp[::-1]])],
            (0, 255, 0)
        )

        xl = state.left.x_at(Y_REF * h, h)
        xr = state.right.x_at(Y_REF * h, h)

        if xr - xl > 1:
            # (tengah gambar - tengah jalur) pixel -> meter
            # skala: lebar jalur dalam pixel (xr-xl) = LANE_WIDTH_M meter
            offset_m = (
                (w / 2.0 - (xl + xr) / 2.0)
                * LANE_WIDTH_M
                / (xr - xl)
            )

    # tempel overlay ke gambar dengan transparansi 40%
    out = cv2.addWeighted(
        frame,
        1.0,
        overlay,
        0.4,
        0
    )

    if poly is None:
        poly, _ = get_roi(w, h)

    cv2.polylines(
        out,
        [poly],
        True,                # True = poligon tertutup
        (255, 0, 0),         # biru
        1
    )

    if state.debug and roi_edges is not None:

        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(
                    out,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    1
                )

        # inset: edge ROI diperkecil 1/3 dan ditempel di pojok kanan bawah
        ih, iw = h // 3, w // 3

        inset = cv2.cvtColor(
            cv2.resize(roi_edges, (iw, ih)),
            cv2.COLOR_GRAY2BGR
        )

        out[h - ih:h, w - iw:w] = inset

        cv2.rectangle(
            out,
            (w - iw, h - ih),
            (w - 1, h - 1),
            (255, 255, 255),
            1
        )

    draw_panel(
        out,
        state,
        offset_m,
        both,
        lp is not None or rp is not None
    )

    return out


def process_frame(frame, state):
    """Deteksi + gambar untuk satu frame."""
    lines, roi_edges, poly = detect_lanes(frame, state)
    return render_frame(
        frame,
        state,
        lines,
        roi_edges,
        poly
    )


# =====================================================================
#            FITUR BARU: HASIL PER TAHAP (TAMPIL + SIMPAN)
# =====================================================================
def _bgr(img):
    """Pastikan gambar 3 kanal (BGR) agar bisa digabung/ditampilkan.
    Gambar grayscale (2 dimensi) diubah jadi 3 kanal."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _label(img, text):
    """Tulis judul tahap di pojok kiri atas gambar (pada bar gelap)."""
    out = _bgr(img)
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (30, 30, 30), -1)  # -1 = isi penuh
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_stages(frame, state):
    """Jalankan pipeline SATU frame sambil mengumpulkan gambar tiap tahap.
    Logikanya sama dengan detect_lanes() + render_frame().
    Mengembalikan list [(judul, gambar_BGR), ...]. state ikut ter-update."""
    h, w = frame.shape[:2]
    stages = [("01 Original (resize)", frame.copy())]

    # --- 02-04: grayscale, blur, canny
    gray, blur, edges = preprocess(frame)
    stages.append(("02 Grayscale", gray))
    stages.append(("03 Gaussian Blur", blur))
    stages.append(("04 Canny Edge", edges))

    # --- 05-06: ROI
    poly, roi_mask = get_roi(w, h)
    roi_vis = frame.copy()
    tint = np.zeros_like(frame)
    cv2.fillPoly(tint, [poly], (0, 255, 0))              # area ROI diwarnai hijau
    roi_vis = cv2.addWeighted(roi_vis, 1.0, tint, 0.35, 0)
    cv2.polylines(roi_vis, [poly], True, (255, 0, 0), 2)
    stages.append(("05 ROI (area jalan)", roi_vis))

    roi_edges = apply_roi(edges, roi_mask)
    stages.append(("06 Edge di dalam ROI", roi_edges))

    # --- 07-08: brightness filter
    bright = None
    if BRIGHT_DELTA > 0:
        med = float(np.median(blur[roi_mask > 0]))
        bright = cv2.dilate(
            (blur > med + BRIGHT_DELTA).astype(np.uint8),
            np.ones((BRIGHT_DILATE, BRIGHT_DILATE), np.uint8)
        ) > 0
        # bool * 255 -> gambar hitam-putih agar bisa dilihat
        stages.append(("07 Mask Brightness", bright.astype(np.uint8) * 255))
        roi_edges = roi_edges * bright.astype(np.uint8)
        stages.append(("08 Edge setelah Brightness Filter", roi_edges))

    # --- 09: HoughLinesP
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

    # --- 10: klasifikasi kiri / kanan (+ sampling titik)
    Lp, Rp = collect_segments(lines, w, h, state.left, state.right, bright)
    seg_vis = frame.copy()
    for x, y in zip(*Lp):    # zip(*Lp) = pasangkan (x, y) dari (daftar_x, daftar_y)
        cv2.circle(seg_vis, (int(x), int(y)), 2, (0, 0, 255), -1)      # kiri = merah
    for x, y in zip(*Rp):
        cv2.circle(seg_vis, (int(x), int(y)), 2, (0, 255, 255), -1)    # kanan = kuning
    stages.append(("10 Klasifikasi Kiri(merah)/Kanan(kuning)", seg_vis))

    # --- update state (fitting + buang outlier + gating + EMA + arah)
    state.left.update(fit_lane(Lp[0], Lp[1], w, h, "L", state.left.theta), w, h)
    state.right.update(fit_lane(Rp[0], Rp[1], w, h, "R", state.right.theta), w, h)
    update_direction(state, w)

    # --- 11: fitting garis/kurva
    fit_vis = frame.copy()
    th = max(2, int(round(w / 160)))
    if state.left.theta is not None:
        cv2.polylines(fit_vis, [curve_points(state.left, h, w)], False, (0, 0, 255), th)
    if state.right.theta is not None:
        cv2.polylines(fit_vis, [curve_points(state.right, h, w)], False, (0, 255, 255), th)
    stages.append(("11 Fitting kurva + EMA", fit_vis))

    # --- 12: hasil akhir
    final = render_frame(frame, state, lines, roi_edges, poly)
    stages.append((f"12 Hasil Akhir: {state.arah}", final))

    return stages


def make_montage(stages, cols=3, tile_w=360):
    """Gabungkan semua tahap jadi satu gambar grid (cols kolom)."""
    tiles = []
    for title, img in stages:
        img = _bgr(img)
        r = tile_w / img.shape[1]                          # rasio pengecilan
        img = cv2.resize(img, (tile_w, int(img.shape[0] * r)))
        tiles.append(_label(img, title))
    th = max(t.shape[0] for t in tiles)                    # tinggi tile terbesar
    rows = []
    for i in range(0, len(tiles), cols):                   # loncat per `cols` tile
        row = tiles[i:i + cols]
        # samakan tinggi semua tile dengan bingkai hitam di bawah
        row = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
               for t in row]
        while len(row) < cols:                             # isi sisa kolom kosong
            row.append(np.zeros((th, tile_w, 3), np.uint8))
        rows.append(np.hstack(row))                        # gabung kiri-kanan
    return np.vstack(rows)                                 # gabung atas-bawah


def save_stages(stages, out_dir="hasil_tahap", prefix=""):
    """Simpan tiap tahap sebagai file .jpg + satu gambar rangkuman."""
    os.makedirs(out_dir, exist_ok=True)      # buat folder; tidak error jika sudah ada
    for title, img in stages:
        # judul -> nama file, misal "04 Canny Edge" -> "04_canny_edge"
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
    """Tampilkan hasil per tahap.
    - Colab / Jupyter : memakai cv2_imshow / matplotlib (inline)
    - Komputer biasa  : matplotlib grid; kalau tidak ada -> jendela OpenCV
    per_stage=True -> tampilkan satu per satu (lebih besar)."""
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
                # matplotlib memakai RGB, OpenCV memakai BGR -> perlu dikonversi
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

    # tampilan grid (semua tahap sekaligus)
    if plt is not None:
        cols = 3
        rows = (len(stages) + cols - 1) // cols    # pembulatan ke atas
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")                         # sembunyikan sumbu semua kotak
        for ax, (title, img) in zip(axes, stages):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")        # gambar 1 kanal -> skala abu-abu
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
    """FUNGSI UTAMA untuk Colab / notebook.
    frame   : gambar BGR (hasil cv2.imread / frame video)
    warmup  : berapa kali state 'dipanaskan' agar EMA stabil (untuk foto)
    Mengembalikan list tahap [(judul, gambar), ...]."""
    if sisi:                      # contoh: sisi="kiri"
        set_sisi(sisi)
    h0, w0 = frame.shape[:2]
    if w0 > PROCESS_WIDTH:
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * PROCESS_WIDTH / w0)))
    state = LaneState()
    for _ in range(warmup):          # foto cuma 1 frame -> ulang agar EMA konvergen
        detect_lanes(frame, state)
    stages = build_stages(frame, state)
    if simpan:
        save_stages(stages, out_dir)
    if tampil:
        show_stages(stages, per_stage=per_stage)
    return stages


# ------------------------------------------------------------------ input foto / video
IMAGE_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)


def show_still_result(
    frame,
    save_path=None,
    window="Deteksi Jalur Jalan Lurus dan Belok"
):
    """Proses satu foto (dengan warmup) lalu tampilkan di jendela OpenCV."""
    state = LaneState()

    for _ in range(12):              # warmup agar hasil EMA stabil
        out = process_frame(frame, state)

    if save_path:
        cv2.imwrite(save_path, out)
        print("Tersimpan ke:", save_path)

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL            # jendela boleh diubah ukurannya
    )

    cv2.imshow(window, out)

    print(
        "Jendela hasil terbuka. Tekan 'd' untuk debug, "
        "tombol apa saja lain (atau tutup jendelanya) untuk lanjut."
    )

    while True:
        key = cv2.waitKey(50) & 0xFF   # & 0xFF: ambil kode tombol saja

        if key == ord("d"):
            state.debug = not state.debug

            out = process_frame(
                frame,
                state
            )

            cv2.imshow(window, out)
            continue

        if key != 255:                 # 255 = tidak ada tombol; selain itu keluar
            break

        try:
            # jendela ditutup pengguna (klik X) -> keluar
            if cv2.getWindowProperty(
                window,
                cv2.WND_PROP_VISIBLE
            ) < 1:
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
        frame = cv2.resize(
            frame,
            (
                args.width,
                int(h0 * args.width / w0)
            )
        )

    # --- MODE PER TAHAP (baru)
    if args.stages:
        proses_per_tahap(frame, out_dir=args.stages_dir)
        if args.save:
            state = LaneState()
            for _ in range(12):
                out = process_frame(frame, state)
            cv2.imwrite(args.save, out)
            print("Tersimpan ke:", args.save)
        return

    show_still_result(
        frame,
        args.save or None
    )

    cv2.destroyAllWindows()


def set_sisi(nilai):
    """Ubah mode sisi garis: 'dua', 'kiri', atau 'kanan'."""
    global SISI
    SISI = nilai


# ------------------------------------------------------------------ pembaca kamera
class FrameGrabber:
    """Membaca webcam di thread terpisah dan hanya menyimpan frame TERBARU.
    Tanpa ini, frame menumpuk di buffer saat proses lambat sehingga gambar
    yang tampil tertinggal (delay) dan terlihat loncat-loncat."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None        # frame terbaru
        self.fid = 0             # nomor urut frame terbaru
        self.ok = True           # False jika kamera berhenti
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self.lock:      # lock: aman dibaca thread utama
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
        "--source",
        default="0",
        help=(
            "0 = webcam, path video "
            "(mis. video_jalan.mp4), atau path FOTO "
            "(mis. jalan.jpg)"
        )
    )

    ap.add_argument(
        "--width",
        type=int,
        default=PROCESS_WIDTH,
        help="lebar frame proses (default 640)"
    )

    ap.add_argument(
        "--save",
        default="",
        help=(
            "simpan hasil: video (.mp4) untuk sumber "
            "video/webcam, atau gambar (.jpg/.png) "
            "untuk sumber foto"
        )
    )

    ap.add_argument(
        "--skip",
        type=int,
        default=1,
        help=(
            "proses 1 dari tiap N frame "
            "(mis. 2 = lewati separuh frame); "
            "pakai kalau laptop terasa berat/lag. "
            "Default 1 (semua frame diproses)"
        )
    )

    # --- argumen baru untuk hasil per tahap
    ap.add_argument(
        "--stages",
        action="store_true",     # tanpa nilai: cukup ditulis --stages
        help="(foto) tampilkan & simpan hasil tiap tahap proses"
    )

    ap.add_argument(
        "--stages-frame",
        type=int,
        default=0,
        help="(video) otomatis ambil hasil per tahap pada frame ke-N "
             "(0 = tidak otomatis; bisa juga tekan 'p')"
    )

    ap.add_argument(
        "--stages-dir",
        default="",
        help="folder penyimpanan hasil per tahap (default: "
             "hasil_tahap/<nama_sumber>, jadi tiap foto/video punya folder sendiri)"
    )

    ap.add_argument(
        "--lebar-max",
        type=float,
        default=None,
        help="lebar jalur maksimum (0-1, proporsi lebar gambar; default 0.50). "
             "Kecilkan bila garis kanan masih menangkap tepi jalan yang jauh; "
             "0 = nonaktif"
    )

    ap.add_argument(
        "--horizon",
        type=float,
        default=None,
        help="tinggi cakrawala/titik hilang jalan (0-1 dari atas gambar, "
             "default 0.48); dipakai bersama --lebar-max"
    )

    ap.add_argument(
        "--sisi",
        choices=["dua", "kiri", "kanan"],
        default="dua",
        help="garis yang dideteksi: dua (default), kiri, atau kanan"
    )

    ap.add_argument(
        "--no-sync",
        action="store_true",
        help="(file video) proses SEMUA frame tanpa menyesuaikan kecepatan "
             "asli video (otomatis aktif bila memakai --save)"
    )

    # parse_known_args: argumen asing diabaikan (aman untuk Colab/Jupyter)
    args, _ = ap.parse_known_args()

    set_sisi(args.sisi)
    if args.lebar_max is not None:
        globals()["MAX_LANE_GAP"] = args.lebar_max
    if args.horizon is not None:
        globals()["Y_HORIZON"] = args.horizon

    # folder hasil per tahap: kalau tidak diisi, otomatis hasil_tahap/<nama_sumber>
    # contoh: lurus.jpg -> hasil_tahap/lurus ; belok.jpg -> hasil_tahap/belok
    # (supaya hasil proses berikutnya tidak menimpa hasil sebelumnya)
    if not args.stages_dir:
        nama = os.path.splitext(os.path.basename(args.source))[0]
        if args.source.isdigit():
            nama = f"webcam_{args.source}"
        args.stages_dir = os.path.join("hasil_tahap", nama)

    # sumber berakhiran .jpg/.png/dst. = foto
    if args.source.lower().endswith(IMAGE_EXT):
        run_on_image(
            args.source,
            args
        )
        return

    is_cam = args.source.isdigit()   # "0", "1" = nomor webcam

    src = (
        int(args.source)
        if is_cam
        else args.source
    )

    # CAP_DSHOW = backend kamera Windows (lebih cepat terbuka)
    cap = (
        cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if is_cam
        else cv2.VideoCapture(src)
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(src)   # coba lagi tanpa DSHOW

    if not cap.isOpened():
        print(
            "Tidak bisa membuka sumber:",
            src
        )
        return

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,   # buffer 1 frame = tidak ada delay menumpuk
        1
    )

    if is_cam:
        # MJPG: kamera mengirim frame terkompresi -> FPS lebih tinggi & stabil
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            args.width
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            int(args.width * 9 / 16)   # rasio 16:9
        )

        cap.set(cv2.CAP_PROP_FPS, 30)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps_src <= 1 or fps_src > 240:      # nilai FPS aneh -> pakai 30
        fps_src = 30

    # Sinkron waktu: hanya untuk FILE video yang tidak sedang disimpan.
    # Frame yang terlambat dilewati (tanpa decode) supaya kecepatan video
    # tetap normal, bukan slow motion. Saat --save, semua frame diproses.
    sync = (not is_cam) and (not args.save) and (not args.no_sync)

    grabber = FrameGrabber(cap) if is_cam else None

    WIN = "Deteksi Jalur Jalan Lurus dan Belok"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    state, i = LaneState(), 0    # i = jumlah frame yang sudah diproses

    lines = roi_edges = poly = None
    writer = None

    last_id = -1                 # nomor frame webcam terakhir yang diproses
    pos = 0                      # jumlah frame sumber yang sudah dibaca
    t_start = time.perf_counter()
    t_prev, fps_show = t_start, 0.0

    while True:

        if is_cam:
            ok, frame, fid = grabber.read()
            if not ok:                       # kamera berhenti
                break
            if frame is None or fid == last_id:
                # belum ada frame baru -> tunggu sebentar (jangan proses 2x)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.002)
                continue
            last_id = fid
        else:
            if sync:
                # frame ke berapa yang seharusnya tampil sekarang?
                due = int((time.perf_counter() - t_start) * fps_src)
                while pos < due:             # lewati frame yang terlambat
                    if not cap.grab():       # grab = ambil tanpa decode (cepat)
                        break
                    pos += 1
            ok, frame = cap.read()
            if not ok:                       # video habis
                break
            pos += 1

        h0, w0 = frame.shape[:2]

        if w0 > args.width:
            frame = cv2.resize(
                frame,
                (
                    args.width,
                    int(h0 * args.width / w0)
                )
            )

        # --- per tahap otomatis pada frame ke-N (memakai salinan state
        #     supaya video yang sedang berjalan tidak terganggu)
        ambil_tahap = (args.stages_frame > 0 and i == args.stages_frame)

        if ambil_tahap:
            stages = build_stages(frame.copy(), copy.deepcopy(state))
            save_stages(stages, args.stages_dir, prefix=f"frame{i:05d}_")
            show_stages(stages)
            t_start = time.perf_counter() - pos / fps_src   # abaikan waktu jeda

        # --skip: deteksi hanya tiap N frame; frame lain memakai hasil terakhir
        if (
            args.skip <= 1
            or i % args.skip == 0
            or poly is None
        ):
            lines, roi_edges, poly = detect_lanes(
                frame,
                state
            )

        out = render_frame(
            frame,
            state,
            lines,
            roi_edges,
            poly
        )

        i += 1

        if args.save:

            if writer is None:     # writer dibuat sekali, saat frame pertama

                fps = (
                    cap.get(cv2.CAP_PROP_FPS)
                    or 30          # jika FPS terbaca 0, pakai 30
                )

                writer = cv2.VideoWriter(
                    args.save,
                    cv2.VideoWriter_fourcc(*"mp4v"),   # codec MP4
                    fps,
                    (
                        out.shape[1],   # lebar
                        out.shape[0]    # tinggi
                    )
                )

            writer.write(out)      # disimpan SEBELUM teks FPS ditambahkan

        # hitung FPS tampilan (dihaluskan agar angkanya tidak berkedip)
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps_show = (1.0 / dt) if fps_show == 0 else (0.9 * fps_show + 0.1 / dt)
        cv2.putText(
            out,
            f"FPS {fps_show:.0f}",
            (out.shape[1] - 90, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        if sync:
            # tahan sampai waktu tampil frame ini -> kecepatan video normal
            tunggu = t_start + pos / fps_src - time.perf_counter()
            if tunggu > 0:
                time.sleep(tunggu)

        cv2.imshow(
            WIN,
            out
        )

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
            t_start = time.perf_counter() - pos / fps_src   # abaikan waktu jeda

        if key == ord("s"):

            snap_name = (
                f"snapshot_{i:05d}.jpg"   # :05d = angka 5 digit, diisi 0 di depan
            )

            print(
                "Snapshot diambil, memproses..."
            )

            show_still_result(
                frame.copy(),
                snap_name,
                window="Snapshot - Deteksi Jalur"
            )

            cv2.destroyWindow(
                "Snapshot - Deteksi Jalur"
            )
            t_start = time.perf_counter() - pos / fps_src   # abaikan waktu jeda

    if grabber:
        grabber.stop()

    cap.release()

    if writer:
        writer.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()