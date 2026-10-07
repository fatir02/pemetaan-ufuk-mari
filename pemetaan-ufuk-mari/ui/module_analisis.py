"""
module_analisis.py
MODUL 2: ANALISIS UFUK
Menangani:
- Inspeksi sudut halangan pada azimut sasaran (misal Azimut Hilal / Matahari terbenam)
- Sinkronisasi kursor garis vertikal interaktif pada gambar dan grafik profil
- Kalkulasi real-time tinggi halangan dan klasifikasi status pandangan
- Input catatan lapangan & rekomendasi kelayakan lokasi rukyat
- Penerbitan laporan resmi PDF lengkap via ReportLab
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QSlider, QTextEdit,
    QDoubleSpinBox, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal

from core.falak_calc import format_dms, classify_obstacle
from core.cv_engine import HorizonDetector
from core.report_generator import export_pdf_report, export_csv_data
from ui.widgets.image_viewer import InteractiveImageViewer
from ui.widgets.plot_canvas import HorizonPlotCanvas


class ModuleAnalisisUfuk(QWidget):
    """
    Modul 2: Inspeksi Sudut Halangan & Evaluasi Kelayakan Lokasi Rukyat.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.cv_detector = HorizonDetector()
        self.profile_data: Optional[Dict[str, Any]] = None
        self.source_image: Optional[np.ndarray] = None
        self.current_overlay_bgr: Optional[np.ndarray] = None
        self.current_target_az: float = 270.0
        self.current_target_alt: float = 0.0
        self.current_analysis: Optional[Dict[str, Any]] = None

        self._updating_widgets = False

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # Splitter horizontal: Panel Kiri (Visualisasi Citra & Grafik) dan Panel Kanan (Inspeksi & Laporan)
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.setChildrenCollapsible(False)

        # ==========================================
        # PANEL KIRI: VISUALISASI OVERLAY & GRAFIK
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Header visual
        v_title = QLabel("INSPEKSI VISUAL KONTUR & KURSOR SASARAN")
        v_title.setStyleSheet("font-weight: bold; color: #ff5252; font-size: 13px;")
        left_layout.addWidget(v_title)

        # Image Viewer dengan overlay kontur dan kursor
        self.image_viewer = InteractiveImageViewer(
            placeholder_text="Belum ada data ekstraksi ufuk.\nSilakan tangkap gambar pada Modul 1 terlebih dahulu."
        )
        self.image_viewer.setMinimumHeight(260)
        self.image_viewer.pixel_clicked.connect(self._on_image_pixel_clicked)
        left_layout.addWidget(self.image_viewer, stretch=3)

        # Matplotlib Plot Canvas
        self.plot_canvas = HorizonPlotCanvas(self, width=8, height=2.4)
        self.plot_canvas.azimuth_clicked.connect(self._on_plot_azimuth_clicked)
        left_layout.addWidget(self.plot_canvas, stretch=2)

        # Petunjuk interaksi
        hint_label = QLabel("💡 Tips: Anda dapat mengklik langsung pada gambar atau grafik untuk memilih Azimut Sasaran.")
        hint_label.setStyleSheet("color: #78909c; font-style: italic; font-size: 11px;")
        left_layout.addWidget(hint_label)

        h_splitter.addWidget(left_widget)

        # ==========================================
        # PANEL KANAN: KONTROL AZIMUT & EVALUASI
        # ==========================================
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # Group 1: Pemilihan Azimut Sasaran
        target_group = QGroupBox("1. Pemilihan Azimut Sasaran (Hilal / Objek Falak)")
        target_layout = QVBoxLayout(target_group)
        target_layout.setSpacing(8)

        # Spinbox & Slider Azimut Sasaran
        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel("Azimut Sasaran:"))
        self.spin_target_az = QDoubleSpinBox()
        self.spin_target_az.setRange(0.0, 360.0)
        self.spin_target_az.setDecimals(2)
        self.spin_target_az.setValue(270.0)
        self.spin_target_az.setSingleStep(0.1)
        self.spin_target_az.setSuffix("°")
        self.spin_target_az.valueChanged.connect(self._on_spin_az_changed)
        spin_layout.addWidget(self.spin_target_az, 1)
        target_layout.addLayout(spin_layout)

        # Slider Azimut
        self.slider_az = QSlider(Qt.Horizontal)
        self.slider_az.setRange(26000, 28000)  # Presisi 0.01 derajat
        self.slider_az.setValue(27000)
        self.slider_az.valueChanged.connect(self._on_slider_az_changed)
        target_layout.addWidget(self.slider_az)

        # Rentang label slider
        self.label_az_range = QLabel("Rentang Tercover: 262.50° s.d. 277.50°")
        self.label_az_range.setStyleSheet("color: #90a4ae; font-size: 11px;")
        target_layout.addWidget(self.label_az_range)

        # Tombol Preset Azimut
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(5)

        self.btn_preset_center = QPushButton("Pusat (270°)")
        self.btn_preset_center.clicked.connect(lambda: self.set_target_azimuth(self.profile_data.get("az_center", 270.0) if self.profile_data else 270.0))
        preset_layout.addWidget(self.btn_preset_center)

        self.btn_preset_min = QPushButton("Titik Terendah")
        self.btn_preset_min.clicked.connect(self._select_min_elevation_azimuth)
        preset_layout.addWidget(self.btn_preset_min)

        self.btn_preset_max = QPushButton("Titik Tertinggi")
        self.btn_preset_max.clicked.connect(self._select_max_elevation_azimuth)
        preset_layout.addWidget(self.btn_preset_max)

        target_layout.addLayout(preset_layout)
        right_layout.addWidget(target_group)

        # Group 2: Kartu Hasil Kalkulasi Real-Time
        calc_group = QGroupBox("2. Data Kalkulasi Sudut Halangan Real-Time")
        calc_layout = QVBoxLayout(calc_group)
        calc_layout.setSpacing(6)

        # Kartu status status banner
        self.status_card = QFrame()
        self.status_card.setStyleSheet("""
            QFrame {
                background-color: #1e2430;
                border: 2px solid #00e676;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        card_layout = QVBoxLayout(self.status_card)
        card_layout.setSpacing(4)

        self.lbl_status_badge = QLabel("STATUS: BEBAS HALANGAN")
        self.lbl_status_badge.setStyleSheet("color: #00e676; font-size: 14px; font-weight: bold;")
        card_layout.addWidget(self.lbl_status_badge)

        self.lbl_status_detail = QLabel("Ufuk mar'i terbuka dan bebas dari rintangan.")
        self.lbl_status_detail.setWordWrap(True)
        self.lbl_status_detail.setStyleSheet("color: #b0bec5; font-size: 12px;")
        card_layout.addWidget(self.lbl_status_detail)

        calc_layout.addWidget(self.status_card)

        # Rincian Parameter Angka
        grid_data = QVBoxLayout()
        grid_data.setSpacing(4)

        self.val_target_az = self._create_info_row(grid_data, "Azimut Sasaran Terpilih:", "270.00°")
        self.val_target_alt = self._create_info_row(grid_data, "Tinggi Halangan Mar'i (Alt):", "+0.00° (00° 00' 00\")")
        self.val_true_horizon_diff = self._create_info_row(grid_data, "Selisih thd Ufuk Hakiki (0°):", "+0.00°")
        self.val_sea_horizon_diff = self._create_info_row(grid_data, "Selisih thd Ufuk Laut (-Dip):", "+0.00°")

        calc_layout.addLayout(grid_data)
        right_layout.addWidget(calc_group)

        # Group 3: Catatan Manual & Rekomendasi Kelayakan
        notes_group = QGroupBox("3. Evaluasi & Catatan Lapangan Astronomi")
        notes_layout = QVBoxLayout(notes_group)
        notes_layout.setSpacing(6)

        notes_layout.addWidget(QLabel("Deskripsi Manual / Kondisi Atmosfer:"))
        self.txt_manual_desc = QTextEdit()
        self.txt_manual_desc.setPlaceholderText("Misal: Pandangan ufuk barat cerah, terdapat bukit karang di sisi selatan...")
        self.txt_manual_desc.setMaximumHeight(65)
        notes_layout.addWidget(self.txt_manual_desc)

        notes_layout.addWidget(QLabel("Rekomendasi Kelayakan Lokasi Rukyat:"))
        self.txt_recommendation = QTextEdit()
        self.txt_recommendation.setPlaceholderText("Rekomendasi otomatis akan dihasilkan di sini dan dapat Anda edit...")
        self.txt_recommendation.setMaximumHeight(80)
        notes_layout.addWidget(self.txt_recommendation)

        right_layout.addWidget(notes_group)

        # Tombol Aksi Utama Cetak Laporan PDF Resmi
        self.btn_export_pdf = QPushButton("🖨️ CETAK DATA LAPORAN PEMETAAN UFUK (PDF RESMI)")
        self.btn_export_pdf.setStyleSheet("""
            QPushButton {
                background-color: #b71c1c;
                border: 1px solid #d32f2f;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 11px 18px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
                border-color: #ff5252;
            }
            QPushButton:pressed {
                background-color: #8b0000;
            }
        """)
        self.btn_export_pdf.clicked.connect(self._export_official_pdf_report)
        right_layout.addWidget(self.btn_export_pdf)

        # Tombol Simpan Gambar Overlay
        self.btn_save_overlay = QPushButton("💾 Simpan Gambar Ber-Overlay (JPG)")
        self.btn_save_overlay.clicked.connect(self._save_overlay_image)
        right_layout.addWidget(self.btn_save_overlay)

        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        h_splitter.addWidget(right_scroll)

        h_splitter.setStretchFactor(0, 6)
        h_splitter.setStretchFactor(1, 4)
        main_layout.addWidget(h_splitter)

    def _create_info_row(self, parent_layout: QVBoxLayout, label_text: str, default_val: str) -> QLabel:
        """Membuat baris label & value yang seragam."""
        h = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #cfd8dc; font-weight: 500;")
        val = QLabel(default_val)
        val.setStyleSheet("color: #80d8ff; font-weight: bold;")
        h.addWidget(lbl)
        h.addStretch()
        h.addWidget(val)
        parent_layout.addLayout(h)
        return val

    # ==========================================
    # PENERIMA DATA DARI MODUL 1
    # ==========================================
    def load_profile_data(self, profile_data: Dict[str, Any]):
        """Memuat data ekstraksi dari Modul 1 dan menginisialisasi parameter analisis."""
        self.profile_data = profile_data
        self.source_image = profile_data.get("source_image")

        azimuths = profile_data["azimuths"]
        az_min = float(azimuths[0])
        az_max = float(azimuths[-1])
        az_center = float(profile_data.get("az_center", 270.0))

        # Atur rentang slider dan spinbox sesuai FOV teramati
        self._updating_widgets = True
        self.spin_target_az.setRange(az_min, az_max)
        self.spin_target_az.setValue(az_center)

        self.slider_az.setRange(int(az_min * 100), int(az_max * 100))
        self.slider_az.setValue(int(az_center * 100))
        self._updating_widgets = False

        self.label_az_range.setText(f"Rentang Tercover: {az_min:.2f}° s.d. {az_max:.2f}° (FOV {profile_data.get('hfov', 15):.1f}°)")

        # Hitung analisis pada posisi awal
        self.set_target_azimuth(az_center)

    # ==========================================
    # LOGIKA INSPEKSI & KALKULASI REAL-TIME
    # ==========================================
    def set_target_azimuth(self, target_az: float):
        """Memperbarui azimut sasaran dan menjalankan inspeksi real-time."""
        if self.profile_data is None:
            return

        azimuths = self.profile_data["azimuths"]
        elevations = self.profile_data["elevations"]
        horizon_y = self.profile_data["horizon_y"]
        w = self.profile_data["img_w"]
        h = self.profile_data["img_h"]
        dip_deg = self.profile_data.get("dip_deg", 0.0)

        # Pastikan berada di dalam batas
        target_az = max(float(azimuths[0]), min(float(azimuths[-1]), target_az))
        self.current_target_az = target_az

        # Cari indeks piksel x terdekat untuk azimut tersebut
        idx = int(np.argmin(np.abs(azimuths - target_az)))
        target_x = idx
        target_y = int(horizon_y[idx])
        target_alt = float(elevations[idx])
        self.current_target_alt = target_alt

        # Klasifikasikan halangan
        analysis = classify_obstacle(target_alt, dip_deg)
        analysis["target_az"] = target_az
        analysis["alt_obstacle"] = target_alt
        analysis["target_x"] = target_x
        analysis["target_y"] = target_y
        self.current_analysis = analysis

        # Update nilai UI
        if not self._updating_widgets:
            self._updating_widgets = True
            self.spin_target_az.setValue(target_az)
            self.slider_az.setValue(int(target_az * 100))
            self._updating_widgets = False

        # Perbarui Kartu Status
        self._update_status_card(analysis, target_az, target_alt, dip_deg)

        # Render overlay citra dengan garis target
        if self.source_image is not None:
            overlay = self.cv_detector.render_overlay(
                image_bgr=self.source_image,
                horizon_y=horizon_y,
                azimuths=azimuths,
                elevations=elevations,
                target_az=target_az,
                target_x=target_x,
                target_y=target_y,
                show_true_horizon=True,
                dip_deg=dip_deg,
                az_center=self.profile_data.get("az_center", 270.0),
                hfov=self.profile_data.get("hfov", 15.0),
                vfov=self.profile_data.get("vfov", 8.5),
                tilt_center=self.profile_data.get("tilt_center", 0.0),
            )
            self.current_overlay_bgr = overlay
            self.image_viewer.set_cv_image(overlay)

        # Perbarui kurva Matplotlib
        self.plot_canvas.update_plot(
            azimuths=azimuths,
            elevations=elevations,
            dip_deg=dip_deg,
            target_az=target_az,
            target_alt=target_alt,
        )

    def _update_status_card(self, analysis: dict, target_az: float, target_alt: float, dip_deg: float):
        """Memperbarui teks dan warna pada kartu status kalkulasi."""
        color = analysis.get("color", "#00e676")
        status_text = analysis.get("status", "Bebas Halangan").upper()
        severity = analysis.get("severity", "Baik")

        self.lbl_status_badge.setText(f"STATUS: {status_text} ({severity})")
        self.lbl_status_badge.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")

        self.lbl_status_detail.setText(analysis.get("category", "-"))

        self.status_card.setStyleSheet(f"""
            QFrame {{
                background-color: #1e2430;
                border: 2px solid {color};
                border-radius: 6px;
                padding: 10px;
            }}
        """)

        # Update teks rincian
        self.val_target_az.setText(f"{target_az:.2f}°")
        self.val_target_alt.setText(f"{target_alt:+.2f}° ({format_dms(target_alt)})")
        self.val_true_horizon_diff.setText(f"{target_alt:+.2f}°")
        self.val_sea_horizon_diff.setText(f"{target_alt - (-dip_deg):+.2f}°")

        # Perbarui rekomendasi otomatis jika pengguna belum mengubah secara manual
        auto_rec = analysis.get("recommendation", "")
        if not self.txt_recommendation.toPlainText().strip() or "MABIMS" in self.txt_recommendation.toPlainText() or "Ufuk" in self.txt_recommendation.toPlainText():
            self.txt_recommendation.setPlainText(auto_rec)

    def _on_spin_az_changed(self, val: float):
        if not self._updating_widgets:
            self.set_target_azimuth(val)

    def _on_slider_az_changed(self, int_val: int):
        if not self._updating_widgets:
            self.set_target_azimuth(int_val / 100.0)

    def _on_plot_azimuth_clicked(self, az: float):
        self.set_target_azimuth(az)

    def _on_image_pixel_clicked(self, x: int, y: int):
        """Menerima klik langsung pada citra untuk mengatur azimut target."""
        if self.profile_data is None:
            return
        w = self.profile_data["img_w"]
        x_clamped = max(0, min(w - 1, x))
        azimuths = self.profile_data["azimuths"]
        target_az = float(azimuths[x_clamped])
        self.set_target_azimuth(target_az)

    def _select_min_elevation_azimuth(self):
        """Preset ke titik dengan halangan paling rendah (paling terbuka)."""
        if self.profile_data:
            st = self.profile_data.get("stats", {})
            min_az = st.get("min_azimuth", 270.0)
            self.set_target_azimuth(min_az)

    def _select_max_elevation_azimuth(self):
        """Preset ke titik dengan halangan paling tinggi."""
        if self.profile_data:
            st = self.profile_data.get("stats", {})
            max_az = st.get("max_azimuth", 270.0)
            self.set_target_azimuth(max_az)

    def _save_overlay_image(self):
        """Menyimpan gambar hasil tangkap dengan overlay garis ufuk."""
        if self.current_overlay_bgr is None:
            QMessageBox.information(self, "Perhatian", "Belum ada gambar overlay yang tersedia.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Simpan Citra Overlay Kontur Ufuk", "citra_overlay_ufuk.jpg", "JPEG Image (*.jpg);;PNG Image (*.png)"
        )
        if filepath:
            cv2.imwrite(filepath, self.current_overlay_bgr)
            QMessageBox.information(self, "Sukses", f"Citra overlay berhasil disimpan ke:\n{filepath}")

    def _export_official_pdf_report(self):
        """Mencetak laporan resmi pemetaan ufuk mar'i dalam format PDF lengkap."""
        if self.profile_data is None or self.current_overlay_bgr is None:
            QMessageBox.information(self, "Perhatian", "Belum ada data ekstraksi ufuk untuk dicetak.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Cetak Laporan PDF Pemetaan Ufuk", "laporan_resmi_pemetaan_ufuk.pdf", "PDF Document (*.pdf)"
        )
        if filepath:
            try:
                p = self.profile_data
                export_pdf_report(
                    output_pdf_path=filepath,
                    location_name=p.get("location_name", "Titik Pengamatan Falak"),
                    latitude=p.get("latitude", 0.0),
                    longitude=p.get("longitude", 0.0),
                    elevation_m=p.get("elevation_m", 0.0),
                    az_center=p.get("az_center", 270.0),
                    hfov=p.get("hfov", 15.0),
                    vfov=p.get("vfov", 8.5),
                    tilt_center=p.get("tilt_center", 0.0),
                    dip_deg=p.get("dip_deg", 0.0),
                    profile_data=p,
                    overlay_img_bgr=self.current_overlay_bgr,
                    target_analysis=self.current_analysis,
                    observer_notes=self.txt_manual_desc.toPlainText(),
                    recommendation_text=self.txt_recommendation.toPlainText(),
                )
                QMessageBox.information(
                    self, "Sukses Menerbitkan Laporan",
                    f"Dokumen Berita Acara & Laporan Pemetaan Ufuk Mar'i resmi berhasil diterbitkan ke:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Gagal Mencetak", f"Terjadi kesalahan saat menyusun PDF: {str(e)}")
