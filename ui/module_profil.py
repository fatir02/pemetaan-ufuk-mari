"""
module_profil.py
MODUL 1: PROFIL UFUK
Menangani:
- Live streaming kamera USB / eyepiece teleskop via QThread
- Form input parameter geografis falak & optik kamera
- Tangkap frame & ekstraksi kontur ufuk mar'i dengan algoritma Computer Vision
- Plotting kurva kontur ufuk mar'i pada Matplotlib canvas
- Ekspor cepat PDF / PNG grafik
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QSlider, QComboBox, QCheckBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal

from core.falak_calc import parse_dms, format_dms, calculate_dip
from core.cv_engine import HorizonDetector
from core.camera_worker import CameraWorker
from core.sample_generator import generate_synthetic_horizon
from core.report_generator import export_pdf_report, export_csv_data
from ui.widgets.image_viewer import InteractiveImageViewer
from ui.widgets.plot_canvas import HorizonPlotCanvas


class ModuleProfilUfuk(QWidget):
    """
    Modul 1: Pengambilan Citra & Plotting Profil Ufuk Mar'i.
    """
    # Signal untuk mengirim hasil ekstraksi ke Modul 2 (Analisis Ufuk)
    profile_extracted = pyqtSignal(dict)
    switch_to_analysis = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cv_detector = HorizonDetector()
        self.camera_worker = CameraWorker(camera_index=0, is_simulation=True)
        self.camera_worker.frame_ready.connect(self._on_camera_frame)
        self.camera_worker.fps_updated.connect(self._on_fps_updated)
        self.camera_worker.error_occurred.connect(self._on_camera_error)
        self.camera_worker.status_changed.connect(self._on_camera_status)

        # State data
        self.active_frame_bgr: Optional[np.ndarray] = None
        self.captured_frame_bgr: Optional[np.ndarray] = None
        self.extracted_profile_data: Optional[Dict[str, Any]] = None
        self.last_overlay_img: Optional[np.ndarray] = None

        self._init_ui()
        self._load_sample_landscape()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # Splitter horizontal: Panel Kiri (Stream & Kontur) & Panel Kanan (Form Parameter & CV Settings)
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.setChildrenCollapsible(False)

        # ==========================================
        # PANEL KIRI: TAMPILAN KAMERA & KONTUR
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Sub-header Panel Kiri
        cam_head_layout = QHBoxLayout()
        cam_title = QLabel("LIVE STREAM KAMERA USB / VIEWPORT CITRA")
        cam_title.setStyleSheet("font-weight: bold; color: #ff5252; font-size: 13px;")
        cam_head_layout.addWidget(cam_title)
        cam_head_layout.addStretch()

        self.fps_label = QLabel("FPS: -- | Mode: Standby")
        self.fps_label.setStyleSheet("color: #80d8ff; font-weight: 600; font-size: 11px;")
        cam_head_layout.addWidget(self.fps_label)
        left_layout.addLayout(cam_head_layout)

        # Viewer citra interaktif
        self.image_viewer = InteractiveImageViewer(placeholder_text="Kamera Siap. Klik 'Mulai Kamera' atau 'Muat Contoh'.")
        self.image_viewer.setMinimumHeight(280)
        left_layout.addWidget(self.image_viewer, stretch=3)

        # Kontrol Kamera Toolbar
        cam_controls = QHBoxLayout()
        cam_controls.setSpacing(6)

        self.combo_cam_source = QComboBox()
        self.combo_cam_source.addItems([
            "Mode Simulasi Falak (Bawaan)",
            "Kamera USB 0 (/dev/video0)",
            "Kamera USB 1 (/dev/video1)",
            "Kamera USB 2 (/dev/video2)",
        ])
        cam_controls.addWidget(QLabel("Sumber:"), 0)
        cam_controls.addWidget(self.combo_cam_source, 1)

        self.btn_toggle_cam = QPushButton("Mulai Kamera")
        self.btn_toggle_cam.setProperty("class", "btn-indigo")
        self.btn_toggle_cam.clicked.connect(self._toggle_camera)
        cam_controls.addWidget(self.btn_toggle_cam)

        self.btn_load_file = QPushButton("Buka Foto/File...")
        self.btn_load_file.clicked.connect(self._load_image_dialog)
        cam_controls.addWidget(self.btn_load_file)

        self.btn_load_sample = QPushButton("Muat Contoh")
        self.btn_load_sample.clicked.connect(self._load_sample_landscape)
        cam_controls.addWidget(self.btn_load_sample)

        left_layout.addLayout(cam_controls)

        # Tombol Aksi Utama Penangkapan Frame
        self.btn_capture_extract = QPushButton("📸 TANGKAP GAMBAR & EKSTRAKSI KONTUR UFUK MAR'I")
        self.btn_capture_extract.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                border: 1px solid #ff1744;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 16px;
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
        self.btn_capture_extract.clicked.connect(self._capture_and_extract)
        left_layout.addWidget(self.btn_capture_extract)

        h_splitter.addWidget(left_widget)

        # ==========================================
        # PANEL KANAN: FORM PARAMETER & CV TUNING
        # ==========================================
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # Group 1: Parameter Geografis & Titik Pengamatan
        geo_group = QGroupBox("1. Parameter Titik Pengamatan (Geografis)")
        geo_layout = QVBoxLayout(geo_group)
        geo_layout.setSpacing(6)

        # Nama Lokasi
        l_loc = QLabel("Nama Lokasi / Titik Pengamatan:")
        self.input_location = QLineEdit("Pos Observasi Hilal Parangtritis / Depok")
        geo_layout.addWidget(l_loc)
        geo_layout.addWidget(self.input_location)

        # Lintang (Phi) & Bujur (Lambda)
        coords_layout = QHBoxLayout()
        coords_layout.setSpacing(8)

        v_lat = QVBoxLayout()
        v_lat.addWidget(QLabel("Lintang Tempat (phi):"))
        self.input_lat = QLineEdit("-07° 58' 48.00\" LS")
        self.input_lat.setPlaceholderText("-07° 58' 48\" atau -7.98")
        self.input_lat.editingFinished.connect(self._update_dip_info)
        v_lat.addWidget(self.input_lat)
        coords_layout.addLayout(v_lat)

        v_lon = QVBoxLayout()
        v_lon.addWidget(QLabel("Bujur Tempat (lambda):"))
        self.input_lon = QLineEdit("110° 18' 22.00\" BT")
        self.input_lon.setPlaceholderText("110° 18' 22\" atau 110.306")
        v_lon.addWidget(self.input_lon)
        coords_layout.addLayout(v_lon)
        geo_layout.addLayout(coords_layout)

        # Ketinggian tempat (h) & Indikator Dip
        elev_layout = QHBoxLayout()
        v_h = QVBoxLayout()
        v_h.addWidget(QLabel("Ketinggian Tempat (h dalam meter dpl):"))
        self.spin_elev_m = QDoubleSpinBox()
        self.spin_elev_m.setRange(0.0, 5000.0)
        self.spin_elev_m.setValue(45.0)
        self.spin_elev_m.setSuffix(" m")
        self.spin_elev_m.valueChanged.connect(self._update_dip_info)
        v_h.addWidget(self.spin_elev_m)
        elev_layout.addLayout(v_h)

        v_dip = QVBoxLayout()
        v_dip.addWidget(QLabel("Kerendahan Ufuk (Dip):"))
        self.label_dip_info = QLabel("0.197° (11.81')")
        self.label_dip_info.setStyleSheet("color: #ffb74d; font-weight: bold; padding-top: 6px;")
        v_dip.addWidget(self.label_dip_info)
        elev_layout.addLayout(v_dip)
        geo_layout.addLayout(elev_layout)

        right_layout.addWidget(geo_group)

        # Group 2: Spesifikasi Optik Kamera Eyepiece
        optic_group = QGroupBox("2. Orientasi & Spesifikasi Optik Kamera")
        optic_layout = QVBoxLayout(optic_group)
        optic_layout.setSpacing(6)

        # Azimut Tengah Kamera
        az_layout = QHBoxLayout()
        v_az = QVBoxLayout()
        v_az.addWidget(QLabel("Arah Azimut Tengah Kamera (°):"))
        self.spin_az_center = QDoubleSpinBox()
        self.spin_az_center.setRange(0.0, 360.0)
        self.spin_az_center.setDecimals(2)
        self.spin_az_center.setValue(270.0)
        self.spin_az_center.setSuffix("° (Barat)")
        v_az.addWidget(self.spin_az_center)
        az_layout.addLayout(v_az)

        v_tilt = QVBoxLayout()
        v_tilt.addWidget(QLabel("Kemiringan Optik (Tilt):"))
        self.spin_tilt = QDoubleSpinBox()
        self.spin_tilt.setRange(-20.0, 20.0)
        self.spin_tilt.setDecimals(2)
        self.spin_tilt.setValue(0.0)
        self.spin_tilt.setSuffix("° (Level)")
        v_tilt.addWidget(self.spin_tilt)
        az_layout.addLayout(v_tilt)
        optic_layout.addLayout(az_layout)

        # FOV (Horizontal & Vertikal)
        fov_layout = QHBoxLayout()
        v_hfov = QVBoxLayout()
        v_hfov.addWidget(QLabel("Horizontal FOV (HFOV):"))
        self.spin_hfov = QDoubleSpinBox()
        self.spin_hfov.setRange(1.0, 90.0)
        self.spin_hfov.setValue(15.0)
        self.spin_hfov.setSuffix("°")
        v_hfov.addWidget(self.spin_hfov)
        fov_layout.addLayout(v_hfov)

        v_vfov = QVBoxLayout()
        v_vfov.addWidget(QLabel("Vertical FOV (VFOV):"))
        vfov_h = QHBoxLayout()
        self.spin_vfov = QDoubleSpinBox()
        self.spin_vfov.setRange(0.5, 60.0)
        self.spin_vfov.setValue(8.5)
        self.spin_vfov.setSuffix("°")
        vfov_h.addWidget(self.spin_vfov, 1)

        self.btn_auto_vfov = QPushButton("Auto Rasio")
        self.btn_auto_vfov.setToolTip("Hitung VFOV otomatis dari rasio aspek frame/sensor")
        self.btn_auto_vfov.clicked.connect(self._auto_calc_vfov)
        vfov_h.addWidget(self.btn_auto_vfov)
        v_vfov.addLayout(vfov_h)

        fov_layout.addLayout(v_vfov)
        optic_layout.addLayout(fov_layout)

        right_layout.addWidget(optic_group)

        # Group 3: Penyetelan Algoritma Computer Vision
        cv_group = QGroupBox("3. Penyetelan Deteksi Computer Vision")
        cv_layout = QVBoxLayout(cv_group)
        cv_layout.setSpacing(6)

        # Metode deteksi
        m_layout = QHBoxLayout()
        m_layout.addWidget(QLabel("Metode CV:"))
        self.combo_cv_method = QComboBox()
        self.combo_cv_method.addItems(HorizonDetector.ALL_METHODS)
        m_layout.addWidget(self.combo_cv_method, 1)
        cv_layout.addLayout(m_layout)

        # Slider Sensitivitas
        s_layout = QHBoxLayout()
        s_layout.addWidget(QLabel("Sensitivitas / Offset:"))
        self.slider_sens = QSlider(Qt.Horizontal)
        self.slider_sens.setRange(-50, 50)
        self.slider_sens.setValue(0)
        self.label_sens_val = QLabel("0")
        self.label_sens_val.setFixedWidth(28)
        self.slider_sens.valueChanged.connect(lambda v: self.label_sens_val.setText(str(v)))
        s_layout.addWidget(self.slider_sens)
        s_layout.addWidget(self.label_sens_val)
        cv_layout.addLayout(s_layout)

        # Slider Smoothing Window
        sm_layout = QHBoxLayout()
        sm_layout.addWidget(QLabel("Filter Smoothing:"))
        self.slider_smooth = QSlider(Qt.Horizontal)
        self.slider_smooth.setRange(3, 45)
        self.slider_smooth.setValue(15)
        self.label_smooth_val = QLabel("15")
        self.label_smooth_val.setFixedWidth(28)
        self.slider_smooth.valueChanged.connect(lambda v: self.label_smooth_val.setText(str(v)))
        sm_layout.addWidget(self.slider_smooth)
        sm_layout.addWidget(self.label_smooth_val)
        cv_layout.addLayout(sm_layout)

        # Checkbox referensi visual
        chk_layout = QHBoxLayout()
        self.chk_invert = QCheckBox("Invert Polaritas")
        self.chk_show_true = QCheckBox("Garis 0° Hakiki")
        self.chk_show_true.setChecked(True)
        self.chk_show_dip = QCheckBox("Garis Dip Laut")
        self.chk_show_dip.setChecked(True)
        chk_layout.addWidget(self.chk_invert)
        chk_layout.addWidget(self.chk_show_true)
        chk_layout.addWidget(self.chk_show_dip)
        cv_layout.addLayout(chk_layout)

        right_layout.addWidget(cv_group)
        right_layout.addStretch()

        right_scroll.setWidget(right_widget)
        h_splitter.addWidget(right_scroll)

        # Set ukuran awal splitter (60% kiri, 40% kanan)
        h_splitter.setStretchFactor(0, 6)
        h_splitter.setStretchFactor(1, 4)
        main_layout.addWidget(h_splitter, stretch=3)

        # ==========================================
        # PANEL BAWAH: MATPLOTLIB PLOT & STATISTIK
        # ==========================================
        bottom_box = QGroupBox("Hasil Plotting Kontur Ufuk Mar'i (Sumbu X = Azimut °, Sumbu Y = Elevasi °)")
        bottom_layout = QVBoxLayout(bottom_box)
        bottom_layout.setContentsMargins(10, 8, 10, 8)
        bottom_layout.setSpacing(6)

        # Canvas Matplotlib
        self.plot_canvas = HorizonPlotCanvas(self, width=8, height=2.6)
        self.plot_canvas.azimuth_clicked.connect(self._on_plot_azimuth_clicked)
        bottom_layout.addWidget(self.plot_canvas)

        # Baris Ringkasan Statistik Ufuk & Tombol Ekspor
        stat_bar = QHBoxLayout()
        self.label_stats = QLabel("Statistik Ufuk: Belum diekstraksi.")
        self.label_stats.setStyleSheet("color: #4fc3f7; font-weight: 600; font-size: 11px;")
        stat_bar.addWidget(self.label_stats, stretch=1)

        self.btn_export_plot_png = QPushButton("💾 Ekspor Grafik (PNG)")
        self.btn_export_plot_png.clicked.connect(self._export_plot_png)
        stat_bar.addWidget(self.btn_export_plot_png)

        self.btn_export_csv = QPushButton("📄 Ekspor CSV")
        self.btn_export_csv.clicked.connect(self._export_csv)
        stat_bar.addWidget(self.btn_export_csv)

        self.btn_export_pdf = QPushButton("📑 Cetak PDF Profil")
        self.btn_export_pdf.setProperty("class", "btn-indigo")
        self.btn_export_pdf.clicked.connect(self._export_quick_pdf)
        stat_bar.addWidget(self.btn_export_pdf)

        self.btn_go_analysis = QPushButton("➡️ ANALISIS UFUK SASARAN")
        self.btn_go_analysis.setStyleSheet("""
            QPushButton {
                background-color: #00796b;
                border: 1px solid #009688;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #009688;
            }
        """)
        self.btn_go_analysis.clicked.connect(self._go_to_analysis_module)
        stat_bar.addWidget(self.btn_go_analysis)

        bottom_layout.addLayout(stat_bar)
        main_layout.addWidget(bottom_box, stretch=2)

        self._update_dip_info()

    # ==========================================
    # LOGIKA & EVENT HANDLER
    # ==========================================
    def _auto_calc_vfov(self):
        """Menghitung VFOV otomatis dari HFOV berdasarkan rasio aspek piksel sensor."""
        hfov = self.spin_hfov.value()
        w, h = 1280, 720
        if self.active_frame_bgr is not None:
            h, w = self.active_frame_bgr.shape[:2]
        import math
        hfov_rad = math.radians(hfov)
        vfov_rad = 2.0 * math.atan((h / w) * math.tan(hfov_rad / 2.0))
        self.spin_vfov.setValue(round(math.degrees(vfov_rad), 2))

    def _update_dip_info(self):
        """Menghitung dan memperbarui informasi kerendahan ufuk laut."""
        h = self.spin_elev_m.value()
        dip_deg, dip_arcmin = calculate_dip(h)
        self.label_dip_info.setText(f"{dip_deg:.3f}° ({dip_arcmin:.2f}')")

    def _toggle_camera(self):
        """Memulai atau menghentikan streaming kamera."""
        if self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.btn_toggle_cam.setText("Mulai Kamera")
            self.btn_toggle_cam.setStyleSheet("")
            self.fps_label.setText("FPS: -- | Kamera Berhenti")
        else:
            idx = self.combo_cam_source.currentIndex()
            is_sim = (idx == 0)
            actual_cam_idx = max(0, idx - 1)
            self.camera_worker.set_camera_index(actual_cam_idx, is_simulation=is_sim)
            self.camera_worker.start()
            self.btn_toggle_cam.setText("Hentikan Kamera")
            self.btn_toggle_cam.setStyleSheet("background-color: #c62828; border-color: #ff1744;")

    def _on_camera_frame(self, frame_bgr: np.ndarray):
        """Menerima frame baru dari QThread kamera."""
        self.active_frame_bgr = frame_bgr
        self.image_viewer.set_cv_image(frame_bgr)

    def _on_fps_updated(self, fps: float):
        """Memperbarui counter FPS di header."""
        mode_str = "Simulasi" if self.camera_worker.is_simulation else f"USB #{self.camera_worker.camera_index}"
        self.fps_label.setText(f"FPS: {fps:.1f} | Mode: {mode_str}")

    def _on_camera_error(self, err_msg: str):
        self.fps_label.setText(f"Status: {err_msg}")

    def _on_camera_status(self, is_running: bool, status_msg: str):
        if not is_running:
            self.btn_toggle_cam.setText("Mulai Kamera")
            self.btn_toggle_cam.setStyleSheet("")
        self.fps_label.setText(status_msg)

    def _load_sample_landscape(self):
        """Memuat citra lanskap simulasi ufuk mar'i default."""
        sample_path = "/home/alfattir/Projects/pemetaan-ufuk-mari/assets/sample_horizon.jpg"
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
        else:
            img = generate_synthetic_horizon(1280, 720)
            cv2.imwrite(sample_path, img)

        self.active_frame_bgr = img
        self.image_viewer.set_cv_image(img)
        self.fps_label.setText("Mode: Sampel Citra Senja Falak Termuat")

    def _load_image_dialog(self):
        """Membuka dialog file untuk memuat citra dari penyimpanan lokal."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Buka File Foto Ufuk Mar'i", "", "Citra (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if filepath:
            img = cv2.imread(filepath)
            if img is not None:
                self.active_frame_bgr = img
                self.image_viewer.set_cv_image(img)
                self.fps_label.setText(f"File: {os.path.basename(filepath)} ({img.shape[1]}x{img.shape[0]})")
            else:
                QMessageBox.warning(self, "Gagal Memuat", "Citra tidak dapat dibaca oleh OpenCV.")

    def _capture_and_extract(self):
        """Mengambil frame aktif dan mengekstraksi kontur ufuk mar'i."""
        if self.active_frame_bgr is None:
            QMessageBox.information(self, "Perhatian", "Tidak ada frame citra aktif. Silakan mulai kamera atau muat gambar.")
            return

        self.captured_frame_bgr = self.active_frame_bgr.copy()
        img = self.captured_frame_bgr
        h, w = img.shape[:2]

        # Ambil parameter input
        az_center = self.spin_az_center.value()
        tilt_center = self.spin_tilt.value()
        hfov = self.spin_hfov.value()
        vfov = self.spin_vfov.value()
        elev_m = self.spin_elev_m.value()

        method = self.combo_cv_method.currentText()
        thresh_offset = self.slider_sens.value()
        smooth_w = self.slider_smooth.value()
        invert = self.chk_invert.isChecked()

        try:
            # 1. Deteksi kontur ufuk mar'i
            horizon_y, mask = self.cv_detector.detect_horizon(
                image_bgr=img,
                method=method,
                threshold_offset=thresh_offset,
                smooth_window=smooth_w,
                invert_mask=invert,
            )

            # 2. Hitung profil sudut astronomi
            profile_data = self.cv_detector.compute_horizon_profile(
                horizon_y=horizon_y,
                img_w=w,
                img_h=h,
                az_center=az_center,
                hfov=hfov,
                vfov=vfov,
                tilt_center=tilt_center,
                elevation_m=elev_m,
            )

            # Tambahkan metadata geografis ke profile_data
            profile_data["location_name"] = self.input_location.text().strip()
            profile_data["latitude"] = parse_dms(self.input_lat.text())
            profile_data["longitude"] = parse_dms(self.input_lon.text())
            profile_data["elevation_m"] = elev_m
            profile_data["source_image"] = img

            self.extracted_profile_data = profile_data

            # 3. Render overlay kontur pada viewport citra
            overlay = self.cv_detector.render_overlay(
                image_bgr=img,
                horizon_y=horizon_y,
                azimuths=profile_data["azimuths"],
                elevations=profile_data["elevations"],
                show_true_horizon=self.chk_show_true.isChecked(),
                dip_deg=profile_data["dip_deg"] if self.chk_show_dip.isChecked() else 0.0,
                az_center=az_center,
                hfov=hfov,
                vfov=vfov,
                tilt_center=tilt_center,
            )
            self.last_overlay_img = overlay
            self.image_viewer.set_cv_image(overlay)

            # 4. Plot ke canvas Matplotlib
            self.plot_canvas.update_plot(
                azimuths=profile_data["azimuths"],
                elevations=profile_data["elevations"],
                dip_deg=profile_data["dip_deg"],
            )

            # 5. Tampilkan ringkasan statistik
            st = profile_data["stats"]
            self.label_stats.setText(
                f"Ufuk Mar'i: Azimuth {st['az_range_min']:.2f}° s.d. {st['az_range_max']:.2f}° | "
                f"Elevasi Min: {st['min_elevation']:+.2f}° (Az {st['min_azimuth']:.2f}°) | "
                f"Elevasi Max: {st['max_elevation']:+.2f}° (Az {st['max_azimuth']:.2f}°) | "
                f"Rata-rata: {st['mean_elevation']:+.2f}°"
            )

            # Pancarkan signal ke modul lain
            self.profile_extracted.emit(profile_data)

        except Exception as e:
            QMessageBox.critical(self, "Kesalahan Ekstraksi", f"Gagal mengekstraksi kontur ufuk: {str(e)}")

    def _on_plot_azimuth_clicked(self, az: float):
        """Menanggapi klik pengguna pada kurva profil untuk inspeksi cepat."""
        if self.extracted_profile_data is None:
            return
        # Cari elevasi terdekat pada azimut tersebut
        az_arr = self.extracted_profile_data["azimuths"]
        alt_arr = self.extracted_profile_data["elevations"]
        idx = int(np.argmin(np.abs(az_arr - az)))
        target_alt = float(alt_arr[idx])
        self.plot_canvas.set_target_azimuth(az, target_alt)

    def _export_plot_png(self):
        """Menyimpan gambar kurva grafik ke format PNG."""
        if self.extracted_profile_data is None:
            QMessageBox.information(self, "Perhatian", "Belum ada data ekstraksi untuk diekspor.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Simpan Grafik Profil Ufuk (PNG)", "grafik_profil_ufuk.png", "PNG Image (*.png)"
        )
        if filepath:
            success = self.plot_canvas.save_figure(filepath)
            if success:
                QMessageBox.information(self, "Sukses", f"Grafik profil berhasil disimpan ke:\n{filepath}")
            else:
                QMessageBox.warning(self, "Gagal", "Gagal menyimpan file grafik.")

    def _export_csv(self):
        """Mengekspor data numerik profil ufuk ke format CSV."""
        if self.extracted_profile_data is None:
            QMessageBox.information(self, "Perhatian", "Belum ada data ekstraksi untuk diekspor.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Ekspor Data Koordinat Ufuk (CSV)", "data_profil_ufuk.csv", "CSV File (*.csv)"
        )
        if filepath:
            try:
                export_csv_data(
                    filepath,
                    self.input_location.text(),
                    self.spin_az_center.value(),
                    self.extracted_profile_data
                )
                QMessageBox.information(self, "Sukses", f"Data profil berhasil diekspor ke:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Gagal", f"Gagal mengekspor data: {str(e)}")

    def _export_quick_pdf(self):
        """Mengekspor laporan PDF cepat dari modul profil."""
        if self.extracted_profile_data is None or self.last_overlay_img is None:
            QMessageBox.information(self, "Perhatian", "Silakan lakukan ekstraksi kontur terlebih dahulu sebelum mencetak PDF.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Cetak Laporan PDF Profil Ufuk", "laporan_profil_ufuk.pdf", "PDF Document (*.pdf)"
        )
        if filepath:
            try:
                p = self.extracted_profile_data
                export_pdf_report(
                    output_pdf_path=filepath,
                    location_name=p["location_name"],
                    latitude=p["latitude"],
                    longitude=p["longitude"],
                    elevation_m=p["elevation_m"],
                    az_center=p["az_center"],
                    hfov=p["hfov"],
                    vfov=p["vfov"],
                    tilt_center=p["tilt_center"],
                    dip_deg=p["dip_deg"],
                    profile_data=p,
                    overlay_img_bgr=self.last_overlay_img,
                )
                QMessageBox.information(self, "Sukses", f"Laporan PDF resmi berhasil diterbitkan ke:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Gagal", f"Gagal membuat laporan PDF: {str(e)}")

    def _go_to_analysis_module(self):
        """Beralih ke Modul 2 (Analisis Ufuk)."""
        if self.extracted_profile_data is None:
            self._capture_and_extract()
        self.switch_to_analysis.emit()

    def closeEvent(self, event):
        """Pastikan thread kamera dihentikan saat widget ditutup."""
        if self.camera_worker.isRunning():
            self.camera_worker.stop()
        super().closeEvent(event)
