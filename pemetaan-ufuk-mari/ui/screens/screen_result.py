"""
screen_result.py
LAYAR 3: HASIL OLAH & KURVA PROFIL UFUK
Tampilan terpadu dalam satu layar:
- Foto jepretan dengan overlay garis kontur ufuk mar'i (OpenCV).
- Data numerik sensor & koordinat (Azimuth, Elevasi, GPS, Alt, Dip).
- Visualisasi grafik kurva profil ufuk (Matplotlib).
- Slider azimut sasaran hilal & status kelayakan falak.
- Tombol aksi: [Foto Ulang] & [Lanjut ke Cetak PDF].
"""

from typing import Optional, Dict, Any
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSlider, QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal

from core.falak_calc import classify_obstacle, format_dms
from ui.widgets.image_viewer import InteractiveImageViewer
from ui.widgets.plot_canvas import HorizonPlotCanvas


class ScreenResult(QWidget):
    """Layar Hasil Olah & Kurva Profil Ufuk Mar'i."""
    retake_requested = pyqtSignal()
    proceed_to_report = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_package: Optional[Dict[str, Any]] = None
        self.target_azimuth: float = 270.0

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # ==========================================
        # 1. HEADER RINGKAS LAYAR HASIL
        # ==========================================
        header_bar = QHBoxLayout()

        self.btn_retake = QPushButton("↺  Foto Ulang")
        self.btn_retake.setStyleSheet("""
            QPushButton {
                background-color: #1e2634; color: #e2e8f0;
                border: 1px solid #334155; border-radius: 8px;
                padding: 8px 18px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #2a3445; color: #ffffff; }
        """)
        self.btn_retake.clicked.connect(self.retake_requested.emit)
        header_bar.addWidget(self.btn_retake)

        header_bar.addStretch()

        lbl_head = QLabel("HASIL EKSTRAKSI KONTUR & PROFIL UFUK MAR'I")
        lbl_head.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: 800; letter-spacing: 0.5px;")
        header_bar.addWidget(lbl_head)

        header_bar.addStretch()

        self.btn_to_report = QPushButton("📑  Lanjut ke Cetak Laporan ➡️")
        self.btn_to_report.setCursor(Qt.PointingHandCursor)
        self.btn_to_report.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
                color: #ffffff; font-size: 13px; font-weight: bold;
                border: 1px solid #38bdf8; border-radius: 8px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
            }
        """)
        self.btn_to_report.clicked.connect(self._on_proceed_report)
        header_bar.addWidget(self.btn_to_report)

        main_layout.addLayout(header_bar)

        # ==========================================
        # 2. RESPONSIF CONTAINER: KIRI (FOTO & DATA) & KANAN (GRAFIK KURVA)
        # ==========================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_content = QWidget()
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_content_layout.setSpacing(10)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # PANEL KIRI: Foto Overlay & Tabel Numerik
        self.left_box = QWidget()
        left_layout = QVBoxLayout(self.left_box)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        lbl_foto_title = QLabel("Citra Kontur Ufuk Mar'i (Garis Kuning = Batas Daratan):")
        lbl_foto_title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        left_layout.addWidget(lbl_foto_title)

        self.image_viewer = InteractiveImageViewer(placeholder_text="Menunggu Data Hasil...")
        self.image_viewer.setMinimumHeight(240)
        left_layout.addWidget(self.image_viewer, stretch=3)

        # Kartu Data Numerik Terukur
        data_card = QFrame()
        data_card.setStyleSheet("""
            background-color: #121721;
            border: 1px solid #1e2634;
            border-radius: 10px;
            padding: 8px 12px;
        """)
        card_l = QVBoxLayout(data_card)
        card_l.setSpacing(4)

        lbl_data_title = QLabel("DATA SENSOR & KOORDINAT PENGAMATAN")
        lbl_data_title.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        card_l.addWidget(lbl_data_title)

        self.row_az = self._create_data_row("Sudut Azimuth Bidikan", "270.00°")
        self.row_alt = self._create_data_row("Kemiringan Optik (Tilt)", "0.00°")
        self.row_gps = self._create_data_row("Koordinat GPS (φ, λ)", "-07° 58' 48\", 110° 18' 22\"")
        self.row_height = self._create_data_row("Ketinggian Tempat (h)", "45.0 mdpl")
        self.row_dip = self._create_data_row("Kerendahan Ufuk Laut (Dip)", "-0.20°")

        for r in [self.row_az, self.row_alt, self.row_gps, self.row_height, self.row_dip]:
            card_l.addWidget(r)

        left_layout.addWidget(data_card, stretch=2)
        self.splitter.addWidget(self.left_box)

        # PANEL KANAN: Grafik Kurva Matplotlib & Analisis Sasaran
        self.right_box = QWidget()
        right_layout = QVBoxLayout(self.right_box)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        lbl_chart_title = QLabel("Kurva Profil Ufuk Mar'i (Elevasi vs Azimuth):")
        lbl_chart_title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        right_layout.addWidget(lbl_chart_title)

        self.plot_canvas = HorizonPlotCanvas(self, width=6, height=3)
        self.plot_canvas.setMinimumHeight(240)
        self.plot_canvas.azimuth_clicked.connect(self._on_plot_azimuth_clicked)
        right_layout.addWidget(self.plot_canvas, stretch=3)

        # Inspeksi Titik Azimut Sasaran Hilal
        inspect_card = QFrame()
        inspect_card.setStyleSheet("""
            background-color: #121721;
            border: 1px solid #1e2634;
            border-radius: 10px;
            padding: 10px 14px;
        """)
        inspect_l = QVBoxLayout(inspect_card)
        inspect_l.setSpacing(6)

        slider_row = QHBoxLayout()
        lbl_slider_txt = QLabel("Inspeksi Azimut Hilal/Objek:")
        lbl_slider_txt.setStyleSheet("color: #e2e8f0; font-weight: bold; font-size: 11px;")
        slider_row.addWidget(lbl_slider_txt)

        self.lbl_target_az_val = QLabel("270.00°")
        self.lbl_target_az_val.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        slider_row.addWidget(self.lbl_target_az_val)

        slider_row.addStretch()
        inspect_l.addLayout(slider_row)

        self.target_slider = QSlider(Qt.Horizontal)
        self.target_slider.setRange(26000, 28000)
        self.target_slider.setValue(27000)
        self.target_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #1e2634; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #38bdf8; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #ff5252; width: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px;
            }
        """)
        self.target_slider.valueChanged.connect(self._on_target_slider_changed)
        inspect_l.addWidget(self.target_slider)

        # Status Hasil Analisis Pada Titik Tersebut
        status_row = QHBoxLayout()
        self.lbl_obstruction_elev = QLabel("Tinggi Halangan: --")
        self.lbl_obstruction_elev.setStyleSheet("color: #fbbf24; font-weight: bold; font-size: 12px;")
        status_row.addWidget(self.lbl_obstruction_elev)

        status_row.addStretch()

        self.lbl_recommendation_badge = QLabel("Status: Menganalisis...")
        self.lbl_recommendation_badge.setStyleSheet("""
            background-color: #10b981; color: #ffffff;
            font-size: 11px; font-weight: bold;
            padding: 4px 12px; border-radius: 10px;
        """)
        status_row.addWidget(self.lbl_recommendation_badge)

        inspect_l.addLayout(status_row)
        right_layout.addWidget(inspect_card, stretch=2)

        self.splitter.addWidget(self.right_box)
        scroll_content_layout.addWidget(self.splitter)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Responsif: jika lebar jendela sempit (< 850px, misalnya layar HP/tablet vertikal),
        # otomatis ubah susunan menjadi vertikal: foto di atas, grafik di bawah
        if self.width() < 850:
            if self.splitter.orientation() != Qt.Vertical:
                self.splitter.setOrientation(Qt.Vertical)
                if hasattr(self, "left_box") and self.left_box.layout():
                    self.left_box.layout().setContentsMargins(0, 0, 0, 8)
                if hasattr(self, "right_box") and self.right_box.layout():
                    self.right_box.layout().setContentsMargins(0, 8, 0, 0)
        else:
            if self.splitter.orientation() != Qt.Horizontal:
                self.splitter.setOrientation(Qt.Horizontal)
                if hasattr(self, "left_box") and self.left_box.layout():
                    self.left_box.layout().setContentsMargins(0, 0, 8, 0)
                if hasattr(self, "right_box") and self.right_box.layout():
                    self.right_box.layout().setContentsMargins(8, 0, 0, 0)

    def _create_data_row(self, label: str, value: str) -> QFrame:
        """Baris item data numerik."""
        row = QFrame()
        l = QHBoxLayout(row)
        l.setContentsMargins(0, 2, 0, 2)
        lbl_k = QLabel(label)
        lbl_k.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_v = QLabel(value)
        lbl_v.setObjectName("Val")
        lbl_v.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 11px;")
        l.addWidget(lbl_k)
        l.addStretch()
        l.addWidget(lbl_v)
        return row

    def load_data(self, data_package: Dict[str, Any]):
        """Memuat paket data ekstraksi dari Layar 2."""
        self.data_package = data_package
        self.img_color = data_package.get("overlay_image")

        profile_data = data_package["profile_data"]
        azimuth = data_package["azimuth"]
        elevation = data_package["elevation"]
        lat = data_package["latitude"]
        lon = data_package["longitude"]
        alt = data_package["altitude"]
        dip_deg = profile_data.get("dip_deg", 0.197)

        # 1. Update foto overlay citra langsung dengan warna asli & kontur kuning OpenCV
        if self.img_color is not None:
            self.image_viewer.set_cv_image(self.img_color)

        # 2. Update baris data numerik
        self._set_row_val(self.row_az, f"{azimuth:.2f}°")
        self._set_row_val(self.row_alt, f"{elevation:+.2f}°")
        self._set_row_val(self.row_gps, f"{format_dms(lat, is_lat=True)}, {format_dms(lon, is_lon=True)}")
        self._set_row_val(self.row_height, f"{alt:.1f} mdpl")
        self._set_row_val(self.row_dip, f"-{dip_deg:.3f}°")

        # 3. Update range slider azimut sasaran sesuai data profil
        az_array = profile_data["azimuths"]
        az_min = float(az_array[0])
        az_max = float(az_array[-1])
        self.target_slider.setRange(int(az_min * 100), int(az_max * 100))
        mid_val = int(azimuth * 100)
        self.target_slider.setValue(max(int(az_min * 100), min(int(az_max * 100), mid_val)))

        # 4. Update grafik Matplotlib
        self._update_analysis_at_azimuth(azimuth)

    def _set_row_val(self, row_frame: QFrame, text: str):
        lbl = row_frame.findChild(QLabel, "Val")
        if lbl:
            lbl.setText(text)

    def _on_target_slider_changed(self, val_int: int):
        target_az = val_int / 100.0
        self.lbl_target_az_val.setText(f"{target_az:.2f}°")
        self._update_analysis_at_azimuth(target_az)

    def _on_plot_azimuth_clicked(self, az: float):
        self.target_slider.setValue(int(az * 100))

    def _update_analysis_at_azimuth(self, target_az: float):
        """Memperbarui nilai halangan dan grafik berdasarkan azimut sasaran terpilih."""
        if self.data_package is None:
            return

        profile_data = self.data_package["profile_data"]
        az_array = profile_data["azimuths"]
        el_array = profile_data["elevations"]
        dip_deg = profile_data.get("dip_deg", 0.0)

        # Cari indeks azimut terdekat
        idx = int(np.argmin(np.abs(az_array - target_az)))
        obstacle_alt = float(el_array[idx])

        # Klasifikasi halangan
        obs_result = classify_obstacle(obstacle_alt, dip_deg)
        status_text = obs_result["status"]
        category_text = obs_result["category"]
        is_clear = (status_text == "Bebas Halangan")

        self.lbl_obstruction_elev.setText(f"Tinggi Halangan: {obstacle_alt:+.2f}° ({format_dms(obstacle_alt)})")

        if is_clear:
            self.lbl_recommendation_badge.setText(f"✅ {status_text} (Layak)")
            self.lbl_recommendation_badge.setStyleSheet("""
                background-color: #10b981; color: #ffffff;
                font-size: 11px; font-weight: bold; padding: 4px 12px; border-radius: 10px;
            """)
        elif "Rendah" in status_text:
            self.lbl_recommendation_badge.setText(f"⚠️ {status_text}")
            self.lbl_recommendation_badge.setStyleSheet("""
                background-color: #f59e0b; color: #ffffff;
                font-size: 11px; font-weight: bold; padding: 4px 12px; border-radius: 10px;
            """)
        else:
            self.lbl_recommendation_badge.setText(f"❌ {status_text}")
            self.lbl_recommendation_badge.setStyleSheet("""
                background-color: #ef4444; color: #ffffff;
                font-size: 11px; font-weight: bold; padding: 4px 12px; border-radius: 10px;
            """)

        # Update grafik plot canvas
        self.plot_canvas.update_plot(
            azimuths=az_array,
            elevations=el_array,
            dip_deg=dip_deg,
            target_az=target_az,
            target_alt=obstacle_alt
        )

        # Simpan state analisis ke paket data
        self.data_package["target_analysis"] = {
            "target_az": target_az,
            "target_alt": obstacle_alt,
            "alt_obstacle": obstacle_alt,
            "is_clear": is_clear,
            "category": category_text,
            "status": status_text,
            "description": obs_result.get("recommendation", category_text),
            "severity": obs_result.get("severity", ""),
            "color": obs_result.get("color", "#00e676"),
            "recommendation": obs_result.get("recommendation", ""),
        }

    def _on_proceed_report(self):
        """Lanjut ke Layar 4 (Laporan)."""
        if self.data_package is not None:
            self.proceed_to_report.emit(self.data_package)
