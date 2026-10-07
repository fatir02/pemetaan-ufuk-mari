"""
web_app.py
Aplikasi Web Mobile & Desktop Pemetaan Profil Ufuk Mar'i Berbasis Computer Vision.
Dapat diakses langsung dari browser Smartphone (Chrome/Safari) melalui Wi-Fi lokal.
"""

import os
import sys
import io
import time
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, render_template_string

# Pastikan path modul terdaftar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cv_engine import HorizonDetector
from core.falak_calc import calculate_dip, classify_obstacle, format_dms
from core.report_generator import export_pdf_report, export_csv_data
from core.sample_generator import generate_synthetic_horizon

app = Flask(__name__)
detector = HorizonDetector()

# Cache penyimpanan hasil analisis terakhir untuk ekspor PDF/CSV
LAST_ANALYSIS = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Pemetaan Ufuk Mar'i - Web Mobile</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #0b0f17; color: #e2e8f0; padding-bottom: 50px; }
    .header { background: linear-gradient(135deg, #0f172a, #1e293b); padding: 18px 16px; border-bottom: 1px solid #334155; text-align: center; }
    .header h1 { font-size: 1.15rem; color: #38bdf8; font-weight: 800; letter-spacing: 0.5px; }
    .header p { font-size: 0.78rem; color: #94a3b8; margin-top: 3px; }
    .container { max-width: 680px; margin: 0 auto; padding: 16px; }
    .card { background-color: #131b2a; border: 1px solid #233147; border-radius: 14px; padding: 16px; margin-bottom: 16px; }
    .card-title { font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }
    .form-group { margin-bottom: 12px; }
    label { display: block; font-size: 0.78rem; color: #94a3b8; margin-bottom: 5px; font-weight: 600; }
    input[type="number"], input[type="text"], select {
      width: 100%; background-color: #1a2436; border: 1px solid #334155; border-radius: 8px; color: #ffffff;
      padding: 10px 12px; font-size: 0.9rem; outline: none;
    }
    input:focus { border-color: #38bdf8; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    .btn {
      display: block; width: 100%; border: none; border-radius: 10px; padding: 12px 18px; font-size: 0.92rem;
      font-weight: 700; cursor: pointer; text-align: center; text-decoration: none; transition: 0.2s;
    }
    .btn-primary { background: linear-gradient(135deg, #0284c7, #0369a1); color: #ffffff; }
    .btn-success { background: linear-gradient(135deg, #10b981, #059669); color: #ffffff; }
    .btn-secondary { background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; }
    .btn-sm { padding: 6px 12px; font-size: 0.78rem; border-radius: 6px; width: auto; }
    .photo-area {
      border: 2px dashed #334155; border-radius: 12px; padding: 20px; text-align: center;
      background-color: #0f172a; margin-bottom: 12px; cursor: pointer;
    }
    .photo-area.active { border-color: #38bdf8; background-color: #132238; }
    .preview-img { width: 100%; border-radius: 8px; display: none; margin-top: 10px; border: 1px solid #334155; }
    .badge {
      display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;
    }
    .badge-success { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-warning { background-color: #451a03; color: #fbbf24; border: 1px solid #b45309; }
    .badge-danger { background-color: #450a0a; color: #f87171; border: 1px solid #dc2626; }
    .stat-card { background-color: #1a2436; border: 1px solid #28374d; border-radius: 10px; padding: 10px; text-align: center; }
    .stat-val { font-size: 1.15rem; font-weight: 800; color: #38bdf8; margin-top: 4px; }
    .stat-lbl { font-size: 0.72rem; color: #94a3b8; }
    #loading { display: none; text-align: center; padding: 24px 0; color: #38bdf8; font-weight: 700; }
    .spinner {
      border: 4px solid rgba(56, 189, 248, 0.2); border-top: 4px solid #38bdf8; border-radius: 50%;
      width: 36px; height: 36px; animation: spin 1s linear infinite; margin: 0 auto 10px;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .slider-container { margin: 15px 0; }
    input[type=range] { width: 100%; accent-color: #38bdf8; }
  </style>
</head>
<body>

  <div class="header">
    <h1>🔭 PEMETAAN UFUK MAR'I</h1>
    <p>Aplikasi Falak Digital & Deteksi Halangan Rukyat (Web Mobile)</p>
  </div>

  <div class="container">

    <!-- KARTU 1: PENGAMBILAN CITRA & KAMERA HP -->
    <div class="card">
      <div class="card-title">
        <span>📸 1. Jepret / Muat Foto Ufuk</span>
        <button type="button" class="btn btn-secondary btn-sm" onclick="loadSamplePhoto()">🖼️ Pakai Contoh</button>
      </div>

      <div class="photo-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
        <p style="font-size: 2rem; margin-bottom: 6px;">📷</p>
        <p style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">Ketuk untuk Potret Kamera HP atau Pilih Foto</p>
        <p style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Mendukung kamera langsung & galeri gambar</p>
      </div>

      <input type="file" id="fileInput" accept="image/*" capture="environment" style="display: none;" onchange="onFileSelected(this)">
      <img id="rawPreview" class="preview-img" alt="Preview Mentah">
    </div>

    <!-- KARTU 2: PARAMETER BIDIKAN & SENSOR HP -->
    <div class="card">
      <div class="card-title">
        <span>🧭 2. Parameter Sensor & Bidikan</span>
        <button type="button" class="btn btn-secondary btn-sm" onclick="getGPS()">📍 Ambil GPS HP</button>
      </div>

      <div class="grid-2">
        <div class="form-group">
          <label>Azimuth Bidikan (0 - 360°):</label>
          <div style="display: flex; gap: 6px;">
            <input type="number" id="inpAzimuth" value="270.0" step="0.1">
            <button type="button" class="btn btn-secondary btn-sm" onclick="document.getElementById('inpAzimuth').value='270.0'">270° Barat</button>
          </div>
        </div>
        <div class="form-group">
          <label>Kemiringan / Tilt (°):</label>
          <input type="number" id="inpTilt" value="0.0" step="0.1">
        </div>
      </div>

      <div class="grid-3">
        <div class="form-group">
          <label>Lintang (Lat):</label>
          <input type="number" id="inpLat" value="-7.9800" step="0.0001">
        </div>
        <div class="form-group">
          <label>Bujur (Lon):</label>
          <input type="number" id="inpLon" value="110.3061" step="0.0001">
        </div>
        <div class="form-group">
          <label>Tinggi (mdpl):</label>
          <input type="number" id="inpAlt" value="45.0" step="1">
        </div>
      </div>

      <button type="button" id="btnProcess" class="btn btn-primary" onclick="processImage()" style="margin-top: 6px;">
        🚀 PROSES EKSTRAKSI KONTUR UFUK
      </button>
    </div>

    <div id="loading">
      <div class="spinner"></div>
      <p>Sedang mengekstrak profil ufuk mar'i dengan Computer Vision...</p>
    </div>

    <!-- KARTU 3: HASIL OLAH & KURVA PROFIL -->
    <div id="resultSection" style="display: none;">
      
      <div class="card">
        <div class="card-title">
          <span>⛰️ 3. Citra Hasil Kontur Ufuk Mar'i</span>
          <span id="resBadge" class="badge badge-success">Analisis Selesai</span>
        </div>
        <img id="resOverlayImg" style="width: 100%; border-radius: 8px; border: 1px solid #334155;" alt="Hasil Kontur">
      </div>

      <div class="card">
        <div class="card-title">
          <span>📈 4. Kurva Profil Elevasi vs Azimuth</span>
        </div>
        <img id="resPlotImg" style="width: 100%; border-radius: 8px; border: 1px solid #334155;" alt="Grafik Profil">
        
        <div class="slider-container">
          <label>Inspeksi Titik Azimut Sasaran Hilal: <b id="lblSliderAz" style="color: #38bdf8;">270.00°</b></label>
          <input type="range" id="sliderAz" min="262.5" max="277.5" step="0.05" value="270.0" oninput="onSliderTargetChanged(this.value)">
        </div>

        <div class="grid-3">
          <div class="stat-card">
            <div class="stat-lbl">Tinggi Halangan</div>
            <div class="stat-val" id="resTargetAlt">+0.14°</div>
          </div>
          <div class="stat-card">
            <div class="stat-lbl">Ufuk Laut (Dip)</div>
            <div class="stat-val" id="resDip">-0.20°</div>
          </div>
          <div class="stat-card">
            <div class="stat-lbl">Status Rukyat</div>
            <div class="stat-val" id="resStatusTxt" style="font-size: 0.95rem; color: #34d399;">Cukup Layak</div>
          </div>
        </div>

        <div style="background-color: #0f172a; padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 0.8rem; line-height: 1.4; color: #cbd5e1;">
          <p id="resDescText">Memuat deskripsi kelayakan...</p>
        </div>
      </div>

      <!-- KARTU 4: EKSPOR RESMI PDF & CSV -->
      <div class="card">
        <div class="card-title">
          <span>📄 5. Berkas Laporan Resmi</span>
        </div>
        <div class="grid-2">
          <a id="btnDownloadPdf" href="/api/download_pdf" target="_blank" class="btn btn-success">
            🖨️ Unduh PDF Laporan
          </a>
          <a id="btnDownloadCsv" href="/api/download_csv" target="_blank" class="btn btn-secondary">
            📊 Unduh Data CSV
          </a>
        </div>
      </div>

    </div>

  </div>

  <script>
    let selectedImageBase64 = null;
    let currentProfileData = null;

    function onFileSelected(input) {
      if (input.files && input.files[0]) {
        const file = input.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
          selectedImageBase64 = e.target.result;
          const preview = document.getElementById('rawPreview');
          preview.src = selectedImageBase64;
          preview.style.display = 'block';
          document.getElementById('dropArea').classList.add('active');
        };
        reader.readAsDataURL(file);
      }
    }

    function loadSamplePhoto() {
      fetch('/api/sample_image')
        .then(res => res.json())
        .then(data => {
          selectedImageBase64 = data.image_base64;
          const preview = document.getElementById('rawPreview');
          preview.src = selectedImageBase64;
          preview.style.display = 'block';
          document.getElementById('dropArea').classList.add('active');
        });
    }

    function getGPS() {
      if (!navigator.geolocation) {
        alert("Geolocation tidak didukung pada browser ini.");
        return;
      }
      navigator.geolocation.getCurrentPosition(
        pos => {
          document.getElementById('inpLat').value = pos.coords.latitude.toFixed(5);
          document.getElementById('inpLon').value = pos.coords.longitude.toFixed(5);
          if (pos.coords.altitude) {
            document.getElementById('inpAlt').value = Math.round(pos.coords.altitude);
          }
          alert("Lokasi GPS HP berhasil disinkronkan!");
        },
        err => {
          alert("Gagal membaca GPS: " + err.message + ". Silakan isi secara manual.");
        },
        { enableHighAccuracy: true }
      );
    }

    function processImage() {
      if (!selectedImageBase64) {
        alert("Silakan potret foto atau pilih gambar terlebih dahulu.");
        return;
      }

      document.getElementById('loading').style.display = 'block';
      document.getElementById('btnProcess').disabled = true;

      const payload = {
        image_base64: selectedImageBase64,
        azimuth: parseFloat(document.getElementById('inpAzimuth').value) || 270.0,
        tilt: parseFloat(document.getElementById('inpTilt').value) || 0.0,
        latitude: parseFloat(document.getElementById('inpLat').value) || -7.98,
        longitude: parseFloat(document.getElementById('inpLon').value) || 110.306,
        altitude: parseFloat(document.getElementById('inpAlt').value) || 45.0
      };

      fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(res => res.json())
      .then(data => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('btnProcess').disabled = false;

        if (data.error) {
          alert("Gagal memproses gambar: " + data.error);
          return;
        }

        currentProfileData = data;
        document.getElementById('resOverlayImg').src = data.overlay_base64;
        document.getElementById('resPlotImg').src = data.plot_base64;
        
        // Atur batas slider
        const slider = document.getElementById('sliderAz');
        slider.min = data.az_min.toFixed(2);
        slider.max = data.az_max.toFixed(2);
        slider.value = payload.azimuth.toFixed(2);
        document.getElementById('lblSliderAz').innerText = payload.azimuth.toFixed(2) + "°";

        updateTargetDisplay(payload.azimuth, data.target_analysis);

        document.getElementById('resultSection').style.display = 'block';
        document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
      })
      .catch(err => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('btnProcess').disabled = false;
        alert("Terjadi kesalahan jaringan atau server.");
      });
    }

    function onSliderTargetChanged(val) {
      document.getElementById('lblSliderAz').innerText = parseFloat(val).toFixed(2) + "°";
      fetch('/api/recalculate_target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_az: parseFloat(val) })
      })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          document.getElementById('resPlotImg').src = data.plot_base64;
          document.getElementById('resOverlayImg').src = data.overlay_base64;
          updateTargetDisplay(parseFloat(val), data.target_analysis);
        }
      });
    }

    function updateTargetDisplay(targetAz, analysis) {
      document.getElementById('resTargetAlt').innerText = (analysis.target_alt >= 0 ? "+" : "") + analysis.target_alt.toFixed(2) + "°";
      document.getElementById('resDip').innerText = "-" + (analysis.dip_deg || 0.20).toFixed(2) + "°";
      document.getElementById('resStatusTxt').innerText = analysis.severity || analysis.status || "Layak";
      const cat = analysis.category || analysis.status || "Ufuk Terbuka";
      const rec = analysis.recommendation || (analysis.target_alt <= 0.0 ? "Lokasi REKOMENDED (LAYAK) untuk rukyatul hilal." : "PERLU DIPERHATIKAN, terdapat halangan daratan/bukit.");
      document.getElementById('resDescText').innerText = 
        `Pada azimut ${targetAz.toFixed(2)}°, rintangan ufuk setinggi ${(analysis.target_alt >= 0 ? "+" : "") + analysis.target_alt.toFixed(2)}°. Kategori: ${cat}. ${rec}`;
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/sample_image", methods=["GET"])
def api_sample_image():
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sample_horizon.jpg")
    if not os.path.exists(sample_path):
        img_bgr = generate_synthetic_horizon(1280, 720)
    else:
        img_bgr = cv2.imread(sample_path)
    
    _, buffer = cv2.imencode(".jpg", img_bgr)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
    return jsonify({"image_base64": b64_str})


def render_plot_image(azimuths, elevations, dip_deg, target_az=None, target_alt=None):
    fig = plt.figure(figsize=(7, 3.2), dpi=120)
    fig.patch.set_facecolor("#131b2a")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0b0f17")

    min_y = min(-dip_deg - 0.5, float(np.min(elevations)) - 0.5, -1.0)
    max_y = max(float(np.max(elevations)) + 0.8, 2.0)
    ax.set_xlim(azimuths[0], azimuths[-1])
    ax.set_ylim(min_y, max_y)

    ax.plot([azimuths[0], azimuths[-1]], [0.0, 0.0], color="#38bdf8", linestyle="--", linewidth=1.2, label="Ufuk Hakiki (0.00°)")
    if dip_deg > 0:
        ax.plot([azimuths[0], azimuths[-1]], [-dip_deg, -dip_deg], color="#fbbf24", linestyle=":", linewidth=1.3, label=f"Ufuk Laut Dip (-{dip_deg:.2f}°)")

    ax.plot(azimuths, elevations, color="#ff5252", linewidth=2.0, label="Kontur Ufuk Mar'i")
    ax.fill_between(azimuths, elevations, min_y, color="#ff1744", alpha=0.15)

    if target_az is not None:
        ax.plot([target_az, target_az], [min_y, max_y], color="#e040fb", linestyle="-.", linewidth=1.5, label=f"Sasaran ({target_az:.2f}°)")
        if target_alt is not None:
            ax.scatter([target_az], [target_alt], color="#e040fb", s=50, zorder=5)

    ax.set_xlabel("Rentang Azimut Bidikan (°)", color="#94a3b8", fontsize=8, fontweight="bold")
    ax.set_ylabel("Sudut Elevasi (°)", color="#94a3b8", fontsize=8, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#233147")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1a2436", edgecolor="#334155", labelcolor="#e2e8f0")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    global LAST_ANALYSIS
    data = request.json or {}
    raw_b64 = data.get("image_base64", "")
    azimuth = float(data.get("azimuth", 270.0))
    tilt = float(data.get("tilt", 0.0))
    lat = float(data.get("latitude", -7.98))
    lon = float(data.get("longitude", 110.3061))
    alt = float(data.get("altitude", 45.0))

    if not raw_b64:
        return jsonify({"error": "Citra kosong"}), 400

    # Decode base64 ke BGR OpenCV
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",")[1]
    img_bytes = base64.b64decode(raw_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return jsonify({"error": "Format gambar tidak didukung"}), 400

    h, w = image_bgr.shape[:2]

    # Jalankan pendeteksi kontur ufuk
    horizon_y, mask = detector.detect_horizon(
        image_bgr,
        method=detector.METHOD_GRADIENT,
        threshold_offset=0,
        blur_kernel=5,
        smooth_window=15
    )

    profile_data = detector.compute_horizon_profile(
        horizon_y=horizon_y,
        img_w=w,
        img_h=h,
        az_center=azimuth,
        hfov=15.0,
        vfov=8.5,
        tilt_center=tilt,
        elevation_m=alt
    )

    target_az = azimuth
    az_arr = profile_data["azimuths"]
    el_arr = profile_data["elevations"]
    t_idx = int(np.argmin(np.abs(az_arr - target_az)))
    target_alt = float(el_arr[t_idx])
    target_analysis = classify_obstacle(target_alt, profile_data["dip_deg"])
    target_analysis["target_az"] = target_az
    target_analysis["target_alt"] = target_alt
    target_analysis["dip_deg"] = profile_data["dip_deg"]

    overlay_bgr = detector.render_overlay(
        image_bgr=image_bgr,
        horizon_y=horizon_y,
        azimuths=az_arr,
        elevations=el_arr,
        target_az=target_az,
        dip_deg=profile_data["dip_deg"],
        az_center=azimuth,
        hfov=15.0,
        vfov=8.5,
        tilt_center=tilt
    )

    _, ov_buf = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(ov_buf).decode("utf-8")

    plot_b64 = render_plot_image(az_arr, el_arr, profile_data["dip_deg"], target_az, target_alt)

    # Simpan di cache untuk ekspor berkas
    LAST_ANALYSIS = {
        "image_bgr": image_bgr,
        "overlay_bgr": overlay_bgr,
        "horizon_y": horizon_y,
        "profile_data": profile_data,
        "target_analysis": target_analysis,
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "azimuth": azimuth,
        "tilt": tilt,
        "w": w,
        "h": h,
    }

    return jsonify({
        "success": True,
        "overlay_base64": overlay_b64,
        "plot_base64": plot_b64,
        "az_min": float(az_arr[0]),
        "az_max": float(az_arr[-1]),
        "target_analysis": target_analysis
    })


@app.route("/api/recalculate_target", methods=["POST"])
def api_recalculate_target():
    global LAST_ANALYSIS
    if not LAST_ANALYSIS:
        return jsonify({"error": "Belum ada analisis aktif"}), 400

    data = request.json or {}
    target_az = float(data.get("target_az", LAST_ANALYSIS["azimuth"]))

    prof = LAST_ANALYSIS["profile_data"]
    az_arr = prof["azimuths"]
    el_arr = prof["elevations"]
    t_idx = int(np.argmin(np.abs(az_arr - target_az)))
    target_alt = float(el_arr[t_idx])
    target_analysis = classify_obstacle(target_alt, prof["dip_deg"])
    target_analysis["target_az"] = target_az
    target_analysis["target_alt"] = target_alt
    target_analysis["dip_deg"] = prof["dip_deg"]

    overlay_bgr = detector.render_overlay(
        image_bgr=LAST_ANALYSIS["image_bgr"],
        horizon_y=LAST_ANALYSIS["horizon_y"],
        azimuths=az_arr,
        elevations=el_arr,
        target_az=target_az,
        dip_deg=prof["dip_deg"],
        az_center=LAST_ANALYSIS["azimuth"],
        hfov=15.0,
        vfov=8.5,
        tilt_center=LAST_ANALYSIS["tilt"]
    )
    _, ov_buf = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(ov_buf).decode("utf-8")

    plot_b64 = render_plot_image(az_arr, el_arr, prof["dip_deg"], target_az, target_alt)

    LAST_ANALYSIS["overlay_bgr"] = overlay_bgr
    LAST_ANALYSIS["target_analysis"] = target_analysis

    return jsonify({
        "success": True,
        "overlay_base64": overlay_b64,
        "plot_base64": plot_b64,
        "target_analysis": target_analysis
    })


@app.route("/api/download_pdf", methods=["GET"])
def api_download_pdf():
    global LAST_ANALYSIS
    if not LAST_ANALYSIS:
        return "Belum ada data analisis aktif", 400

    exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(exports_dir, exist_ok=True)
    pdf_path = os.path.join(exports_dir, f"Laporan_Ufuk_Web_{int(time.time())}.pdf")

    ta = LAST_ANALYSIS["target_analysis"]
    desc = f"Analisis berbasis Computer Vision pada azimut {ta['target_az']:.2f}°. Kondisi: {ta.get('category', 'Ufuk Mar-i')}."
    rec = ta.get("recommendation", "Lokasi rukyatul hilal tervalidasi.")

    export_pdf_report(
        output_pdf_path=pdf_path,
        location_name="Pos Observasi Falak (Web Mobile)",
        latitude=LAST_ANALYSIS["latitude"],
        longitude=LAST_ANALYSIS["longitude"],
        elevation_m=LAST_ANALYSIS["altitude"],
        az_center=LAST_ANALYSIS["azimuth"],
        hfov=15.0,
        vfov=8.5,
        tilt_center=LAST_ANALYSIS["tilt"],
        dip_deg=LAST_ANALYSIS["profile_data"]["dip_deg"],
        profile_data=LAST_ANALYSIS["profile_data"],
        overlay_img_bgr=LAST_ANALYSIS["overlay_bgr"],
        target_analysis=ta,
        observer_notes=desc,
        recommendation_text=rec,
    )

    return send_file(pdf_path, as_attachment=True, download_name="Laporan_Ufuk_Mar'i.pdf")


@app.route("/api/download_csv", methods=["GET"])
def api_download_csv():
    global LAST_ANALYSIS
    if not LAST_ANALYSIS:
        return "Belum ada data analisis aktif", 400

    exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(exports_dir, exist_ok=True)
    csv_path = os.path.join(exports_dir, f"Data_Ufuk_Web_{int(time.time())}.csv")

    export_csv_data(
        output_csv_path=csv_path,
        profile_data=LAST_ANALYSIS["profile_data"],
        location_name="Pos Observasi Falak (Web Mobile)",
        az_center=LAST_ANALYSIS["azimuth"],
        dip_deg=LAST_ANALYSIS["profile_data"]["dip_deg"],
    )

    return send_file(csv_path, as_attachment=True, download_name="Data_Profil_Ufuk.csv")


def main():
    print("=" * 60)
    print("   APLIKASI WEB PEMETAAN UFUK MAR'I (MOBILE & DESKTOP)")
    print("   Buka dari Browser Laptop : http://localhost:5000")
    print("   Buka dari Browser HP     : http://<IP_LAPTOP>:5000")
    print("   Contoh IP Laptop Anda    : http://192.168.0.105:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
