"""
main_window.py
Jendela Utama Aplikasi Pemetaan Profil Ufuk Mar'i Berbasis Computer Vision.
Mengelola alur kerja 4 Layar (Stepper Wizard) yang bersih, modern, dan mudah dipahami:
  1. Beranda (Dashboard)
  2. Pengamatan (Kamera & Sensor Azimuth 270°)
  3. Hasil & Profil Ufuk (Foto Kontur & Kurva Elevasi)
  4. Laporan Resmi (Cetak PDF & CSV)
"""

import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont

from ui.styles import DARK_THEME_QSS
from ui.screens.screen_dashboard import ScreenDashboard
from ui.screens.screen_observation import ScreenObservation
from ui.screens.screen_result import ScreenResult
from ui.screens.screen_report import ScreenReport


class MainWindow(QMainWindow):
    """
    Jendela Utama Aplikasi Pemetaan Profil Ufuk Mar'i.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("APLIKASI PEMETAAN PROFIL UFUK MAR'I BERBASIS COMPUTER VISION")
        self.resize(1200, 780)
        self.setMinimumSize(360, 480)

        # Terapkan stylesheet tema dark modern
        self.setStyleSheet(DARK_THEME_QSS)

        self._init_ui()
        self._init_clock_timer()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # STACKED WIDGET (4 LAYAR ALUR UTAMA WIZARD)
        # ==========================================
        self.stacked_widget = QStackedWidget()

        # Inisialisasi 4 Layar
        self.screen_dashboard = ScreenDashboard(self)
        self.screen_observation = ScreenObservation(self)
        self.screen_result = ScreenResult(self)
        self.screen_report = ScreenReport(self)

        # Tambahkan ke QStackedWidget
        self.stacked_widget.addWidget(self.screen_dashboard)    # Index 0
        self.stacked_widget.addWidget(self.screen_observation)  # Index 1
        self.stacked_widget.addWidget(self.screen_result)       # Index 2
        self.stacked_widget.addWidget(self.screen_report)       # Index 3

        # Sambungkan sinyal antar layar
        # Layar 1 -> Layar 2
        self.screen_dashboard.start_requested.connect(lambda: self.switch_screen(1))

        # Layar 2 -> Layar 1 atau Layar 3
        self.screen_observation.back_to_dashboard.connect(lambda: self.switch_screen(0))
        self.screen_observation.proceed_to_result.connect(self._on_observation_finished)

        # Layar 3 -> Layar 2 atau Layar 4
        self.screen_result.retake_requested.connect(lambda: self.switch_screen(1))
        self.screen_result.proceed_to_report.connect(self._on_result_to_report)

        # Layar 4 -> Layar 3 atau Layar 2
        self.screen_report.back_to_result.connect(lambda: self.switch_screen(2))
        self.screen_report.start_new_observation.connect(lambda: self.switch_screen(1))

        main_layout.addWidget(self.stacked_widget)

    def _stepper_btn_style(self, active: bool = False) -> str:
        if active:
            return """
                QPushButton {
                    background-color: #172030;
                    color: #38bdf8;
                    border: none;
                    border-bottom: 3px solid #0284c7;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 10px 18px;
                    border-radius: 0px;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: transparent;
                    color: #94a3b8;
                    border: none;
                    border-bottom: 3px solid transparent;
                    font-size: 12px;
                    font-weight: 500;
                    padding: 10px 18px;
                    border-radius: 0px;
                }
                QPushButton:hover {
                    color: #ffffff;
                    background-color: #161c27;
                }
            """

    def switch_screen(self, index: int):
        """Berpindah layar wizard alur kerja aplikasi."""
        # Jika user langsung melompat ke Layar 3 atau Layar 4 sebelum proses manual,
        # otomatis siapkan data pengamatan dari foto/sampel default agar layar tidak kosong
        if index in (2, 3) and (self.screen_result.data_package is None or self.screen_report.data_package is None):
            package = self.screen_observation.build_observation_package()
            if package:
                if self.screen_result.data_package is None:
                    self.screen_result.load_data(package)
                if self.screen_report.data_package is None:
                    self.screen_report.load_data(self.screen_result.data_package)

        self.stacked_widget.setCurrentIndex(index)

    def _on_observation_finished(self, package: dict):
        """Saat proses ekstraksi citra selesai di Layar 2 -> muat data ke Layar 3."""
        self.screen_result.load_data(package)
        self.switch_screen(2)

    def _on_result_to_report(self, package: dict):
        """Saat user melanjutkan dari Layar 3 -> muat data ke Layar 4."""
        self.screen_report.load_data(package)
        self.switch_screen(3)

    def _init_clock_timer(self):
        """Memperbarui jam astronomi setiap detik."""
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

    def _update_clock(self):
        t_loc = time.localtime()
        if hasattr(self, "lbl_clock"):
            self.lbl_clock.setText(time.strftime("%H:%M:%S WIB", t_loc))

    def closeEvent(self, event):
        """Menutup worker kamera dengan aman."""
        if hasattr(self, "screen_observation"):
            self.screen_observation.closeEvent(event)
        event.accept()
