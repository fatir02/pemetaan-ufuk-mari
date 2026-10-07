"""
screen_report_preview.py
Layar 5: Menu Preview Laporan Resmi (Berita Acara Falak) Interaktif.
Menyediakan penampil dokumen PDF resolusi tinggi dengan kemampuan:
- Zoom In / Zoom Out bebas (30% - 300%)
- Fit Width (Pas Lebar) & Fit Page (Pas Halaman)
- Scroll vertikal dan horizontal halus
- Drag to pan (geser dengan kursor tangan)
- Cetak / Buka Dokumen di PDF Viewer sistem
- Simpan salinan berkas PDF ke direktori tujuan
"""

import os
import shutil
from typing import Optional, List, Dict, Any
import pypdfium2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QFrame, QSlider,
    QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QCursor, QWheelEvent, QMouseEvent


class DraggableScrollArea(QScrollArea):
    """
    QScrollArea yang mendukung drag-to-scroll (pan dengan mouse drag)
    dan zoom menggunakan Ctrl + Mouse Wheel.
    """
    zoom_delta_requested = pyqtSignal(int)  # +1 untuk zoom in, -1 untuk zoom out

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("""
            QScrollArea {
                background-color: #0b0f17;
                border: none;
            }
            QScrollBar:vertical {
                background: #111622;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                min-height: 24px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
            QScrollBar:horizontal {
                background: #111622;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #334155;
                min-width: 24px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #475569;
            }
        """)

        self._drag_active = False
        self._last_pos = QPoint()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_delta_requested.emit(1)
            elif delta < 0:
                self.zoom_delta_requested.emit(-1)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_active:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ScreenReportPreview(QWidget):
    """
    Layar 5: Menu Preview Dokumen Laporan Resmi
    dengan kontrol Zoom, Scroll halus, dan Aksi Berkas.
    """
    back_to_report = pyqtSignal()
    start_new_observation = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pdf_path: Optional[str] = None
        self.data_package: Optional[Dict[str, Any]] = None
        self.base_pixmaps: List[QPixmap] = []
        self.page_labels: List[QLabel] = []
        self.zoom_factor: float = 1.0  # 1.0 = 100%

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # 1. TOOLBAR ATAS (NAVIGASI, STATUS & ZOOM)
        # ==========================================
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #111622, stop:1 #161d2b);
                border-bottom: 1px solid #232c3d;
                padding: 6px 12px;
            }
        """)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 8, 12, 8)
        tb_layout.setSpacing(10)

        # Tombol Kembali ke Layar 4 Form Laporan
        self.btn_back = QPushButton("⬅️ Kembali ke Form")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setStyleSheet(self._btn_style("#1e2634", "#334155", "#94a3b8"))
        self.btn_back.clicked.connect(self.back_to_report.emit)
        tb_layout.addWidget(self.btn_back)

        # Judul & File Info
        v_title = QVBoxLayout()
        v_title.setSpacing(1)
        lbl_title = QLabel("PREVIEW DOKUMEN LAPORAN RESMI (PDF)")
        lbl_title.setStyleSheet("color: #38bdf8; font-weight: 800; font-size: 13px; letter-spacing: 0.5px;")
        self.lbl_file_info = QLabel("Memuat berkas dokumen...")
        self.lbl_file_info.setStyleSheet("color: #64748b; font-size: 11px;")
        v_title.addWidget(lbl_title)
        v_title.addWidget(self.lbl_file_info)
        tb_layout.addLayout(v_title)

        tb_layout.addStretch()

        # Group Kontrol Zoom
        zoom_bar = QHBoxLayout()
        zoom_bar.setSpacing(6)

        lbl_zoom_icon = QLabel("🔍")
        lbl_zoom_icon.setStyleSheet("font-size: 14px;")
        zoom_bar.addWidget(lbl_zoom_icon)

        self.btn_zoom_out = QPushButton("➖")
        self.btn_zoom_out.setToolTip("Zoom Out (Perkecil) [Ctrl + -]")
        self.btn_zoom_out.setCursor(Qt.PointingHandCursor)
        self.btn_zoom_out.setFixedSize(32, 32)
        self.btn_zoom_out.setStyleSheet(self._btn_icon_style())
        self.btn_zoom_out.clicked.connect(lambda: self._step_zoom(-0.15))
        zoom_bar.addWidget(self.btn_zoom_out)

        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(30, 250)
        self.slider_zoom.setValue(100)
        self.slider_zoom.setFixedWidth(110)
        self.slider_zoom.setToolTip("Geser untuk mengatur tingkat perbesaran dokumen")
        self.slider_zoom.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #1e2634;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #0284c7;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        self.slider_zoom.valueChanged.connect(self._on_slider_zoom_changed)
        zoom_bar.addWidget(self.slider_zoom)

        self.btn_zoom_in = QPushButton("➕")
        self.btn_zoom_in.setToolTip("Zoom In (Perbesar) [Ctrl + +]")
        self.btn_zoom_in.setCursor(Qt.PointingHandCursor)
        self.btn_zoom_in.setFixedSize(32, 32)
        self.btn_zoom_in.setStyleSheet(self._btn_icon_style())
        self.btn_zoom_in.clicked.connect(lambda: self._step_zoom(0.15))
        zoom_bar.addWidget(self.btn_zoom_in)

        self.lbl_zoom_pct = QLabel("100%")
        self.lbl_zoom_pct.setFixedWidth(46)
        self.lbl_zoom_pct.setAlignment(Qt.AlignCenter)
        self.lbl_zoom_pct.setStyleSheet("color: #e2e8f0; font-weight: bold; font-size: 11px;")
        zoom_bar.addWidget(self.lbl_zoom_pct)

        self.btn_fit_width = QPushButton("↔️ Pas Lebar")
        self.btn_fit_width.setToolTip("Sesuaikan lebar dokumen ke layar")
        self.btn_fit_width.setCursor(Qt.PointingHandCursor)
        self.btn_fit_width.setStyleSheet(self._btn_style("#172030", "#253347", "#38bdf8"))
        self.btn_fit_width.clicked.connect(self._fit_to_width)
        zoom_bar.addWidget(self.btn_fit_width)

        self.btn_fit_page = QPushButton("↕️ 100%")
        self.btn_fit_page.setToolTip("Reset ukuran asli (100%)")
        self.btn_fit_page.setCursor(Qt.PointingHandCursor)
        self.btn_fit_page.setStyleSheet(self._btn_style("#172030", "#253347", "#38bdf8"))
        self.btn_fit_page.clicked.connect(lambda: self.set_zoom(1.0))
        zoom_bar.addWidget(self.btn_fit_page)

        tb_layout.addLayout(zoom_bar)

        tb_layout.addSpacing(10)

        # Tombol Aksi Dokumen
        self.btn_open_external = QPushButton("🖨️ Buka / Cetak PDF")
        self.btn_open_external.setCursor(Qt.PointingHandCursor)
        self.btn_open_external.setStyleSheet(self._btn_style("#059669", "#10b981", "#ffffff", bold=True))
        self.btn_open_external.clicked.connect(self._open_in_external_viewer)
        tb_layout.addWidget(self.btn_open_external)

        self.btn_save_as = QPushButton("💾 Simpan Sebagai...")
        self.btn_save_as.setCursor(Qt.PointingHandCursor)
        self.btn_save_as.setStyleSheet(self._btn_style("#0284c7", "#38bdf8", "#ffffff"))
        self.btn_save_as.clicked.connect(self._save_copy_as)
        tb_layout.addWidget(self.btn_save_as)

        main_layout.addWidget(toolbar)

        # ==========================================
        # 2. VIEWPORT / SCROLL AREA DOKUMEN
        # ==========================================
        self.scroll_area = DraggableScrollArea(self)
        self.scroll_area.setCursor(Qt.OpenHandCursor)
        self.scroll_area.zoom_delta_requested.connect(self._on_wheel_zoom)

        # Kontainer Halaman di Dalam Scroll Area
        self.pages_container = QWidget()
        self.pages_container.setStyleSheet("background-color: transparent;")
        self.pages_layout = QVBoxLayout(self.pages_container)
        self.pages_layout.setContentsMargins(20, 24, 20, 24)
        self.pages_layout.setSpacing(24)
        self.pages_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self.scroll_area.setWidget(self.pages_container)
        main_layout.addWidget(self.scroll_area)

    def _btn_style(self, bg: str, border: str, text: str, bold: bool = False) -> str:
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 7px;
                padding: 6px 12px;
                font-size: 11px;
                {'font-weight: bold;' if bold else ''}
            }}
            QPushButton:hover {{
                border-color: #38bdf8;
                background-color: #1e293b;
            }}
        """

    def _btn_icon_style(self) -> str:
        return """
            QPushButton {
                background-color: #1e2634;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2b384c;
                border-color: #38bdf8;
            }
        """

    def load_pdf(self, pdf_path: str, data_package: Optional[Dict[str, Any]] = None):
        """
        Memuat dan merender halaman-halaman dokumen PDF resolusi tinggi.
        """
        self.pdf_path = pdf_path
        self.data_package = data_package

        if not os.path.exists(pdf_path):
            QMessageBox.critical(self, "Error", f"Berkas PDF tidak ditemukan:\n{pdf_path}")
            return

        file_size = os.path.getsize(pdf_path) / 1024.0
        file_name = os.path.basename(pdf_path)

        # Kosongkan halaman sebelumnya
        self.base_pixmaps.clear()
        for lbl in self.page_labels:
            self.pages_layout.removeWidget(lbl)
            lbl.deleteLater()
        self.page_labels.clear()

        try:
            # Render PDF dengan pypdfium2 pada skala tinggi (2.2x) untuk ketajaman teks saat di-zoom
            pdf = pypdfium2.PdfDocument(pdf_path)
            num_pages = len(pdf)
            self.lbl_file_info.setText(f"{file_name} • {num_pages} Halaman • {file_size:.1f} KB")

            for i in range(num_pages):
                page = pdf[i]
                # Render ke PIL image pada skala 2.2
                pil_img = page.render(scale=2.2).to_pil()
                data = pil_img.convert("RGBA").tobytes("raw", "RGBA")
                qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimg)
                self.base_pixmaps.append(pixmap)

                # Buat widget kartu halaman
                page_card = QFrame()
                page_card.setStyleSheet("""
                    QFrame {
                        background-color: #ffffff;
                        border: 1px solid #334155;
                        border-radius: 4px;
                    }
                """)
                card_layout = QVBoxLayout(page_card)
                card_layout.setContentsMargins(0, 0, 0, 0)
                card_layout.setSpacing(0)

                lbl_page = QLabel()
                lbl_page.setAlignment(Qt.AlignCenter)
                lbl_page.setStyleSheet("background-color: #ffffff; border-radius: 4px;")
                card_layout.addWidget(lbl_page)

                # Label penanda halaman di bawah kartu
                lbl_footer = QLabel(f"— Halaman {i + 1} dari {num_pages} —")
                lbl_footer.setAlignment(Qt.AlignCenter)
                lbl_footer.setStyleSheet("color: #64748b; font-size: 11px; margin-top: 4px; margin-bottom: 12px;")

                self.pages_layout.addWidget(page_card, alignment=Qt.AlignHCenter)
                self.pages_layout.addWidget(lbl_footer, alignment=Qt.AlignHCenter)
                self.page_labels.append(lbl_page)

            # Terapkan ukuran awal
            self.zoom_factor = 1.0
            self.slider_zoom.setValue(100)
            self._render_zoom()

        except Exception as e:
            QMessageBox.critical(self, "Gagal Merender PDF", f"Terjadi kesalahan saat memuat halaman PDF:\n{str(e)}")

    def set_zoom(self, factor: float):
        """Mengatur faktor perbesaran langsung."""
        factor = max(0.3, min(3.0, factor))
        self.zoom_factor = factor
        val_int = int(round(factor * 100))
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(val_int)
        self.slider_zoom.blockSignals(False)
        self.lbl_zoom_pct.setText(f"{val_int}%")
        self._render_zoom()

    def _step_zoom(self, delta: float):
        """Menambah / mengurangi perbesaran."""
        self.set_zoom(self.zoom_factor + delta)

    def _on_wheel_zoom(self, direction: int):
        """Dipicu saat Ctrl + Mouse wheel diputar."""
        step = 0.15 if direction > 0 else -0.15
        self._step_zoom(step)

    def _on_slider_zoom_changed(self, val_int: int):
        """Dipicu saat slider zoom digeser."""
        self.zoom_factor = val_int / 100.0
        self.lbl_zoom_pct.setText(f"{val_int}%")
        self._render_zoom()

    def _fit_to_width(self):
        """Menyesuaikan lebar halaman dengan lebar viewport scroll area."""
        if not self.base_pixmaps:
            return
        viewport_w = self.scroll_area.viewport().width() - 60  # margin cadangan
        if viewport_w > 100:
            orig_w = self.base_pixmaps[0].width()
            # orig_w adalah piksel dari render 2.2x (misal ~1300px)
            # Standar A4 72 DPI adalah 595px
            target_scale = float(viewport_w) / float(orig_w)
            self.set_zoom(target_scale * 1.5)

    def _render_zoom(self):
        """Memperbarui resolusi tampilan semua halaman sesuai faktor zoom saat ini."""
        if not self.base_pixmaps:
            return

        for i, pixmap in enumerate(self.base_pixmaps):
            if i < len(self.page_labels):
                lbl = self.page_labels[i]
                target_w = int(round(pixmap.width() * (self.zoom_factor / 1.5)))
                target_h = int(round(pixmap.height() * (self.zoom_factor / 1.5)))

                scaled_pix = pixmap.scaled(
                    target_w, target_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                lbl.setPixmap(scaled_pix)
                lbl.setFixedSize(target_w, target_h)

    def _open_in_external_viewer(self):
        """Membuka dokumen PDF di pembaca default sistem untuk dicetak langsung."""
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            QMessageBox.warning(self, "Peringatan", "Berkas PDF tidak ditemukan.")
            return

        try:
            os.startfile(self.pdf_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal membuka berkas:\n{str(e)}")

    def _save_copy_as(self):
        """Menyimpan salinan berkas PDF ke lokasi yang dipilih pengguna."""
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            QMessageBox.warning(self, "Peringatan", "Berkas PDF belum tersedia.")
            return

        default_name = os.path.basename(self.pdf_path)
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Simpan Salinan Dokumen PDF Laporan",
            default_name,
            "Dokumen PDF (*.pdf);;Semua Berkas (*.*)"
        )
        if dest_path:
            try:
                shutil.copyfile(self.pdf_path, dest_path)
                QMessageBox.information(
                    self,
                    "Sukses Simpan",
                    f"Salinan dokumen PDF berhasil disimpan ke:\n\n{dest_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Gagal Simpan", f"Terjadi kesalahan saat menyalin berkas:\n{str(e)}")
