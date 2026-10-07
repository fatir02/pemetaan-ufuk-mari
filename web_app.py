"""
web_app.py
Aplikasi Web Mobile & Desktop Pemetaan Profil Ufuk Mar'i Berbasis Computer Vision.
Fitur Unggulan Mobile:
- Pilihan Ambil Gambar: Galeri HP, Kamera Native, dan Kamera Live Viewfinder
- Kamera Live dengan Garis Bantu Deteksi Ufuk Astronomi (Ufuk Hakiki 0°, Ufuk Laut Dip, Salib Sumbu Optik & Sensor Kemiringan)
- Pemrosesan Kontur Ufuk OpenCV Multi-Spektral Presisi Tinggi
- Menu Preview Laporan PDF Interaktif dengan Zoom (50% - 250%) dan Scroll Sentuh Halus
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
import pypdfium2

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
  <title>Pemetaan Ufuk Mar'i - Web Mobile Falak</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #0b0f17; color: #e2e8f0; padding-bottom: 60px; }
    .header { background: linear-gradient(135deg, #0f172a, #1e293b); padding: 18px 16px; border-bottom: 1px solid #334155; text-align: center; }
    .header h1 { font-size: 1.18rem; color: #38bdf8; font-weight: 800; letter-spacing: 0.5px; }
    .header p { font-size: 0.78rem; color: #94a3b8; margin-top: 3px; }
    .container { max-width: 720px; margin: 0 auto; padding: 16px; }
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
    .grid-4 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
    .btn {
      display: inline-block; width: 100%; border: none; border-radius: 10px; padding: 12px 16px; font-size: 0.9rem;
      font-weight: 700; cursor: pointer; text-align: center; text-decoration: none; transition: 0.2s;
    }
    .btn-primary { background: linear-gradient(135deg, #0284c7, #0369a1); color: #ffffff; }
    .btn-success { background: linear-gradient(135deg, #10b981, #059669); color: #ffffff; }
    .btn-secondary { background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; }
    .btn-warning { background: linear-gradient(135deg, #d97706, #b45309); color: #ffffff; }
    .btn-sm { padding: 8px 10px; font-size: 0.8rem; border-radius: 8px; width: 100%; }
    .photo-area {
      border: 2px dashed #334155; border-radius: 12px; padding: 18px 14px; text-align: center;
      background-color: #0f172a; margin-bottom: 12px;
    }
    .preview-img { width: 100%; border-radius: 8px; display: none; margin-top: 10px; border: 1px solid #334155; }
    .badge {
      display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;
    }
    .badge-success { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
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

    /* Gaya Viewfinder Kamera Live & Garis Bantu */
    #liveCamWrapper {
      display: none;
      position: relative;
      width: 100%;
      border-radius: 12px;
      overflow: hidden;
      background-color: #000000;
      border: 2px solid #38bdf8;
      margin-bottom: 14px;
    }
    #cameraVideo {
      width: 100%;
      height: auto;
      display: block;
      object-fit: cover;
      max-height: 480px;
    }
    #cameraOverlay {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none;
    }
    .cam-controls {
      position: absolute;
      bottom: 12px;
      left: 0; right: 0;
      display: flex;
      justify-content: center;
      gap: 12px;
      padding: 0 16px;
      z-index: 10;
    }
    .btn-shutter {
      background: radial-gradient(circle, #ef4444 40%, #dc2626 100%);
      color: #ffffff;
      font-weight: 900;
      font-size: 0.95rem;
      border: 3px solid #ffffff;
      border-radius: 30px;
      padding: 10px 24px;
      box-shadow: 0 4px 15px rgba(239, 68, 68, 0.6);
      cursor: pointer;
    }
    .btn-close-cam {
      background: rgba(15, 23, 42, 0.85);
      color: #94a3b8;
      border: 1px solid #475569;
      border-radius: 20px;
      padding: 8px 16px;
      font-size: 0.8rem;
      cursor: pointer;
    }

    /* Modal / Kotak Preview Laporan dengan Zoom & Scroll */
    #reportPreviewModal {
      display: none;
      margin-top: 16px;
      background-color: #111726;
      border: 1.5px solid #0284c7;
      border-radius: 14px;
      padding: 14px;
    }
    .preview-toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      background-color: #172033;
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 12px;
    }
    .zoom-btn-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .zoom-btn {
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 6px 10px;
      font-weight: bold;
      font-size: 0.85rem;
      cursor: pointer;
    }
    .zoom-btn:hover { background: #334155; color: #38bdf8; }
    .preview-viewport {
      width: 100%;
      max-height: 72vh;
      overflow: auto;
      background-color: #070a10;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 16px;
      text-align: center;
      touch-action: pan-x pan-y;
      -webkit-overflow-scrolling: touch;
    }
    .preview-pages-wrapper {
      display: inline-block;
      transition: transform 0.15s ease-out;
      transform-origin: top center;
    }
    .preview-page-card {
      background-color: #ffffff;
      border-radius: 6px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
      margin: 0 auto 20px auto;
      max-width: 100%;
      overflow: hidden;
    }
    .preview-page-card img {
      width: 100%;
      height: auto;
      display: block;
    }
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
      </div>

      <!-- Tombol Pilihan Lengkap: Galeri, Kamera Live, Kamera Cepat, dan Sampel -->
      <div class="grid-4">
        <button type="button" class="btn btn-secondary btn-sm" onclick="document.getElementById('galleryInput').click()">
          🖼️ Buka Galeri HP
        </button>
        <button type="button" class="btn btn-primary btn-sm" onclick="startLiveCamera()">
          📹 Kamera Live (Garis Ufuk)
        </button>
        <button type="button" class="btn btn-secondary btn-sm" onclick="document.getElementById('cameraInput').click()">
          📸 Jepret Kamera HP
        </button>
        <button type="button" class="btn btn-secondary btn-sm" onclick="loadSamplePhoto()">
          🌄 Pakai Contoh
        </button>
      </div>

      <!-- Hidden File Inputs: Dipisahkan antara Galeri murni dan Kamera -->
      <input type="file" id="galleryInput" accept="image/*" style="display: none;" onchange="onFileSelected(this)">
      <input type="file" id="cameraInput" accept="image/*" capture="environment" style="display: none;" onchange="onFileSelected(this)">

      <!-- Viewfinder Kamera Live dengan Garis Bantu Deteksi Ufuk -->
      <div id="liveCamWrapper">
        <video id="cameraVideo" playsinline autoplay muted></video>
        <canvas id="cameraOverlay"></canvas>
        <div class="cam-controls">
          <button type="button" class="btn-shutter" onclick="captureLiveFrame()">
            📸 JEPRET FOTO
          </button>
          <button type="button" class="btn-close-cam" onclick="stopLiveCamera()">
            ✖️ Tutup Kamera
          </button>
        </div>
      </div>

      <div class="photo-area" id="dropArea">
        <p style="font-size: 1.8rem; margin-bottom: 4px;">🏞️</p>
        <p style="font-weight: 700; color: #38bdf8; font-size: 0.88rem;" id="lblPhotoStatus">Belum ada foto yang dipilih</p>
        <p style="font-size: 0.74rem; color: #64748b; margin-top: 3px;">Pilih via Galeri HP atau gunakan Kamera Live di atas</p>
      </div>

      <img id="rawPreview" class="preview-img" alt="Preview Foto Aktif">
    </div>

    <!-- KARTU 2: PARAMETER BIDIKAN & SENSOR HP -->
    <div class="card">
      <div class="card-title">
        <span>🧭 2. Parameter Sensor & Bidikan</span>
        <button type="button" class="btn btn-secondary btn-sm" style="width: auto;" onclick="getGPS()">📍 Ambil GPS HP</button>
      </div>

      <div class="grid-2">
        <div class="form-group">
          <label>Azimuth Bidikan (0 - 360°):</label>
          <div style="display: flex; gap: 6px;">
            <input type="number" id="inpAzimuth" value="270.0" step="0.1">
            <button type="button" class="btn btn-secondary btn-sm" style="width: auto; white-space: nowrap;" onclick="document.getElementById('inpAzimuth').value='270.0'">270° Barat</button>
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
      <p id="loadingText">Sedang mengekstrak profil ufuk mar'i dengan Computer Vision...</p>
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
            <div class="stat-val" id="resStatusTxt" style="font-size: 0.92rem; color: #34d399;">Cukup Layak</div>
          </div>
        </div>

        <div style="background-color: #0f172a; padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 0.8rem; line-height: 1.4; color: #cbd5e1;">
          <p id="resDescText">Memuat deskripsi kelayakan...</p>
        </div>
      </div>

      <!-- KARTU 4: DOKUMEN & PREVIEW LAPORAN -->
      <div class="card">
        <div class="card-title">
          <span>📄 5. Berkas Laporan Resmi (PDF Berita Acara)</span>
        </div>
        
        <div class="grid-2" style="margin-bottom: 10px;">
          <button type="button" class="btn btn-success" onclick="openReportPreview()">
            👁️ PREVIEW LAPORAN (ZOOM & SCROLL)
          </button>
          <a id="btnDownloadPdf" href="/api/download_pdf" target="_blank" class="btn btn-secondary">
            📥 Unduh Langsung PDF
          </a>
        </div>
        <a id="btnDownloadCsv" href="/api/download_csv" target="_blank" class="btn btn-secondary" style="font-size: 0.82rem; padding: 8px 12px;">
          📊 Unduh Data Koordinat CSV
        </a>

        <!-- AREA PREVIEW LAPORAN INTERAKTIF -->
        <div id="reportPreviewModal">
          <div class="preview-toolbar">
            <span style="color: #38bdf8; font-weight: bold; font-size: 0.85rem;">📑 Preview Dokumen Falak</span>
            
            <div class="zoom-btn-group">
              <button type="button" class="zoom-btn" onclick="stepZoom(-0.2)">➖</button>
              <span id="lblZoomLevel" style="color: #ffffff; font-size: 0.8rem; font-weight: bold; min-width: 42px; text-align: center;">100%</span>
              <button type="button" class="zoom-btn" onclick="stepZoom(0.2)">➕</button>
              <button type="button" class="zoom-btn" onclick="resetZoom()">🔄 100%</button>
              <button type="button" class="zoom-btn" onclick="fitWidthZoom()">↔️ Pas</button>
              <a href="/api/download_pdf" target="_blank" class="zoom-btn" style="background: #059669; color: #fff; text-decoration: none;">📥 Unduh</a>
            </div>
          </div>

          <div class="slider-container" style="margin: 6px 4px 12px 4px;">
            <input type="range" id="zoomSlider" min="50" max="250" value="100" step="5" oninput="setZoomScale(this.value / 100.0)">
          </div>

          <div class="preview-viewport" id="previewViewport">
            <div class="preview-pages-wrapper" id="previewWrapper">
              <!-- Halaman dokumen PDF dimuat di sini secara dinamis -->
            </div>
          </div>
        </div>

      </div>

    </div>

  </div>

  <script>
    let selectedImageBase64 = null;
    let currentProfileData = null;
    let cameraStream = null;
    let animFrameId = null;
    let currentZoom = 1.0;
    let deviceOrientation = { pitch: 0, roll: 0 };

    // Pantau orientasi sensor gerak HP jika didukung
    if (window.DeviceOrientationEvent) {
      window.addEventListener('deviceorientation', function(e) {
        if (e.beta !== null) {
          deviceOrientation.pitch = e.beta; // derajat tilt sumbu x
          deviceOrientation.roll = e.gamma; // derajat roll sumbu y
        }
      }, true);
    }

    function onFileSelected(input) {
      if (input.files && input.files[0]) {
        const file = input.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
          selectedImageBase64 = e.target.result;
          displayLoadedImage(selectedImageBase64, "Foto siap dari " + (input.id === 'galleryInput' ? 'Galeri HP' : 'Kamera'));
        };
        reader.readAsDataURL(file);
      }
    }

    function displayLoadedImage(b64, statusText) {
      const preview = document.getElementById('rawPreview');
      preview.src = b64;
      preview.style.display = 'block';
      document.getElementById('lblPhotoStatus').innerText = statusText || "Foto Terpilih";
      document.getElementById('lblPhotoStatus').style.color = "#10b981";
      stopLiveCamera();
    }

    function loadSamplePhoto() {
      fetch('/api/sample_image')
        .then(res => res.json())
        .then(data => {
          selectedImageBase64 = data.image_base64;
          displayLoadedImage(selectedImageBase64, "Foto contoh lanskap ufuk terpasang");
        });
    }

    // ==========================================
    // KAMERA LIVE DENGAN GARIS BANTU DETEKSI UFUK
    // ==========================================
    async function startLiveCamera() {
      const wrapper = document.getElementById('liveCamWrapper');
      const video = document.getElementById('cameraVideo');
      wrapper.style.display = 'block';

      try {
        const constraints = {
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          },
          audio: false
        };
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = cameraStream;
        video.onloadedmetadata = () => {
          video.play();
          drawLiveHudLoop();
        };
      } catch (err) {
        alert("Tidak dapat mengakses kamera live via browser: " + err.message + "\\nSilakan gunakan tombol 'Jepret Kamera HP' sebagai alternatif.");
        wrapper.style.display = 'none';
      }
    }

    function stopLiveCamera() {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
      }
      if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }
      document.getElementById('liveCamWrapper').style.display = 'none';
    }

    function drawLiveHudLoop() {
      const video = document.getElementById('cameraVideo');
      const canvas = document.getElementById('cameraOverlay');
      if (!cameraStream || video.paused || video.ended) return;

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        const cy = h / 2.0;
        const cx = w / 2.0;

        // 1. Grid Komposisi Rule of Thirds Tipis
        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w / 3, 0); ctx.lineTo(w / 3, h);
        ctx.moveTo(2 * w / 3, 0); ctx.lineTo(2 * w / 3, h);
        ctx.moveTo(0, h / 3); ctx.lineTo(w, h / 3);
        ctx.moveTo(0, 2 * h / 3); ctx.lineTo(w, 2 * h / 3);
        ctx.stroke();

        // 2. Garis Ufuk Hakiki (0.00°) Putih/Cyan Putus-putus
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.setLineDash([12, 8]);
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(w, cy);
        ctx.stroke();
        ctx.setLineDash([]);

        // Teks Label Ufuk Hakiki
        ctx.fillStyle = "#38bdf8";
        ctx.font = "bold 20px sans-serif";
        ctx.fillText("── Ufuk Hakiki (0.00°) ──", 24, cy - 8);

        // 3. Garis Estimasi Ufuk Laut (-Dip ~0.20°) Kuning Emas
        const dipY = cy + (h * 0.025);
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(0, dipY);
        ctx.lineTo(w, dipY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#fbbf24";
        ctx.font = "bold 16px sans-serif";
        ctx.fillText("── Ufuk Laut Dip (-0.20°) ──", 24, dipY + 22);

        // 4. Salib Sumbu Optik Tengah (Optical Crosshair)
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - 25, cy); ctx.lineTo(cx + 25, cy);
        ctx.moveTo(cx, cy - 25); ctx.lineTo(cx, cy + 25);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.stroke();

        // 5. Header HUD: Azimuth & Indikator Waterpass
        ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
        ctx.fillRect(w / 2 - 160, 16, 320, 44);
        ctx.strokeStyle = "#38bdf8";
        ctx.strokeRect(w / 2 - 160, 16, 320, 44);

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 16px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("🧭 BIDIKAN BARAT 270.0° • LIVE", w / 2, 44);
        ctx.textAlign = "left";
      }

      animFrameId = requestAnimationFrame(drawLiveHudLoop);
    }

    function captureLiveFrame() {
      const video = document.getElementById('cameraVideo');
      if (!cameraStream || video.videoWidth === 0) return;

      const hiddenCanvas = document.createElement('canvas');
      hiddenCanvas.width = video.videoWidth;
      hiddenCanvas.height = video.videoHeight;
      const ctx = hiddenCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

      selectedImageBase64 = hiddenCanvas.toDataURL('image/jpeg', 0.92);
      displayLoadedImage(selectedImageBase64, "Foto berhasil dipotret dari Kamera Live");
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
        alert("Silakan potret foto atau pilih gambar dari galeri terlebih dahulu.");
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

    // ==========================================
    // PREVIEW LAPORAN RESMI (ZOOM & SCROLL)
    // ==========================================
    function openReportPreview() {
      const modal = document.getElementById('reportPreviewModal');
      const wrapper = document.getElementById('previewWrapper');
      modal.style.display = 'block';
      wrapper.innerHTML = "<p style='color: #38bdf8; padding: 20px;'>Sedang merender halaman dokumen PDF resmi...</p>";
      modal.scrollIntoView({ behavior: 'smooth' });

      fetch('/api/preview_pdf')
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            wrapper.innerHTML = `<p style="color: #ef4444; padding: 20px;">Gagal merender PDF: ${data.error}</p>`;
            return;
          }

          wrapper.innerHTML = "";
          data.pages.forEach((pageB64, idx) => {
            const card = document.createElement('div');
            card.className = "preview-page-card";
            card.innerHTML = `
              <img src="${pageB64}" alt="Halaman ${idx + 1}">
              <div style="background: #0f172a; color: #94a3b8; font-size: 0.72rem; padding: 6px; text-align: center;">
                — Halaman ${idx + 1} dari ${data.pages.length} —
              </div>
            `;
            wrapper.appendChild(card);
          });

          resetZoom();
        })
        .catch(err => {
          wrapper.innerHTML = `<p style="color: #ef4444; padding: 20px;">Koneksi gagal: ${err.message}</p>`;
        });
    }

    function setZoomScale(scale) {
      currentZoom = Math.max(0.4, Math.min(2.8, scale));
      const wrapper = document.getElementById('previewWrapper');
      wrapper.style.transform = `scale(${currentZoom})`;
      document.getElementById('lblZoomLevel').innerText = Math.round(currentZoom * 100) + "%";
      document.getElementById('zoomSlider').value = Math.round(currentZoom * 100);
    }

    function stepZoom(delta) {
      setZoomScale(currentZoom + delta);
    }

    function resetZoom() {
      setZoomScale(1.0);
    }

    function fitWidthZoom() {
      const viewport = document.getElementById('previewViewport');
      const vw = viewport.clientWidth - 40;
      // Lebar dasar dokumen A4 pada render
      const scale = Math.max(0.4, Math.min(2.0, vw / 650.0));
      setZoomScale(scale);
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

    # Jalankan pendeteksi kontur ufuk multi-spektral OpenCV
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

    # Simpan di cache untuk ekspor berkas dan preview PDF
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


def _generate_current_pdf_path() -> str:
    """Helper untuk menerbitkan PDF terkini dari cache analisis."""
    global LAST_ANALYSIS
    if not LAST_ANALYSIS:
        raise ValueError("Belum ada data analisis aktif")

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
    return pdf_path


@app.route("/api/preview_pdf", methods=["GET"])
def api_preview_pdf():
    """Merender halaman dokumen PDF resmi ke format citra untuk preview interaktif di HP."""
    try:
        pdf_path = _generate_current_pdf_path()
        pdf = pypdfium2.PdfDocument(pdf_path)
        pages_b64 = []
        for i in range(len(pdf)):
            pil_img = pdf[i].render(scale=2.0).to_pil()
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=88)
            pages_b64.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"))

        return jsonify({
            "success": True,
            "pages": pages_b64,
            "pdf_url": "/api/download_pdf"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/download_pdf", methods=["GET"])
def api_download_pdf():
    try:
        pdf_path = _generate_current_pdf_path()
        return send_file(pdf_path, as_attachment=True, download_name="Laporan_Ufuk_Mar'i.pdf")
    except Exception as e:
        return f"Gagal membuat PDF: {str(e)}", 400


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
