"""
screen_dashboard.py
LAYAR 1: DASHBOARD UTAMA (BERANDA)
Tampilan awal yang bersih dan minimalis sesuai permintaan alur simpel:
- Brand aplikasi astronomi & instrumen falak.
- 1 Tombol Aksi Utama: [MULAI PENGAMATAN].
- Status ringkas kesiapan instrumen.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap


class ScreenDashboard(QWidget):
    """Layar Beranda / Dashboard Minimalis."""
    # Sinyal untuk berpindah ke Layar 2 (Pengamatan)
    start_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setAlignment(Qt.AlignCenter)

        # Kartu Kontainer Tengah
        card = QFrame()
        card.setObjectName("DashboardCard")
        card.setStyleSheet("""
            #DashboardCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #151922, stop:1 #0e1117);
                border: 1px solid #232a38;
                border-radius: 20px;
                padding: 40px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(18)
        card_layout.setAlignment(Qt.AlignCenter)

        # Badge Label Kategori
        lbl_badge = QLabel("INSTRUMEN FALAK DIGITAL • COMPUTER VISION")
        lbl_badge.setAlignment(Qt.AlignCenter)
        lbl_badge.setStyleSheet("""
            background-color: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 12px;
            padding: 6px 16px;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1.5px;
        """)
        card_layout.addWidget(lbl_badge, alignment=Qt.AlignCenter)

        # Ikon Eksplorasi Ufuk
        lbl_icon = QLabel("🔭")
        lbl_icon.setAlignment(Qt.AlignCenter)
        lbl_icon.setStyleSheet("font-size: 72px; margin: 10px 0;")
        card_layout.addWidget(lbl_icon, alignment=Qt.AlignCenter)

        # Judul Utama Aplikasi
        lbl_title = QLabel("APLIKASI PEMETAAN UFUK MAR'I")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("""
            color: #ffffff;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 0.5px;
        """)
        card_layout.addWidget(lbl_title)

        # Deskripsi Singkat
        lbl_desc = QLabel(
            "Sistem Deteksi Kontur Batas Langit & Daratan Secara Presisi\n"
            "Untuk Analisis Sudut Halangan Lapangan & Kelayakan Rukyatul Hilal"
        )
        lbl_desc.setAlignment(Qt.AlignCenter)
        lbl_desc.setStyleSheet("""
            color: #94a3b8;
            font-size: 13px;
            line-height: 1.5;
            margin-bottom: 15px;
        """)
        card_layout.addWidget(lbl_desc)

        # Satu Tombol Utama Sesuai Spesifikasi: [MULAI PENGAMATAN]
        self.btn_mulai = QPushButton("🚀  MULAI PENGAMATAN")
        self.btn_mulai.setCursor(Qt.PointingHandCursor)
        self.btn_mulai.setMinimumHeight(56)
        self.btn_mulai.setMinimumWidth(320)
        self.btn_mulai.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                letter-spacing: 1px;
                border: 1px solid #38bdf8;
                border-radius: 14px;
                padding: 12px 28px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
                border: 1px solid #7dd3fc;
            }
            QPushButton:pressed {
                background-color: #075985;
            }
        """)
        self.btn_mulai.clicked.connect(self.start_requested.emit)
        card_layout.addWidget(self.btn_mulai, alignment=Qt.AlignCenter)

        # Baris Info Fitur Cepat di Bawah Tombol
        info_layout = QHBoxLayout()
        info_layout.setSpacing(20)
        info_layout.setAlignment(Qt.AlignCenter)

        features = [
            ("📡 Sensor Azimuth 270°", "Lock otomatis tepat Barat"),
            ("⛰️ Computer Vision", "Deteksi profil kontur bukit"),
            ("📄 Laporan Resmi PDF", "Format berita acara falak"),
        ]

        for title, sub in features:
            box = QFrame()
            box.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.03);
                border: 1px solid #1e2634;
                border-radius: 10px;
                padding: 10px 14px;
            """)
            box_l = QVBoxLayout(box)
            box_l.setSpacing(2)
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet("color: #e2e8f0; font-weight: bold; font-size: 12px;")
            s_lbl = QLabel(sub)
            s_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
            box_l.addWidget(t_lbl)
            box_l.addWidget(s_lbl)
            info_layout.addWidget(box)

        card_layout.addSpacing(10)
        card_layout.addLayout(info_layout)

        main_layout.addWidget(card)
