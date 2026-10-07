"""
test_app.py
Skrip pengujian otomatis alur kerja 4 Layar Aplikasi Pemetaan Ufuk Mar'i:
1. Layar 1: Dashboard Minimalis
2. Layar 2: Pengamatan (Kamera, HUD Azimuth 270°, Sensor) & Ekstraksi Kontur
3. Layar 3: Hasil Olah, Foto Kontur, Data Numerik & Kurva Profil Ufuk Matplotlib
4. Layar 4: Form Evaluasi & Penerbitan Dokumen PDF / CSV Resmi
"""

import sys
import os
import cv2
import numpy as np

# Pastikan path modul terdaftar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def test_full_pipeline():
    print("=== MEMULAI TEST LENGKAP APLIKASI PEMETAAN UFUK MAR'I (ALUR 4 LAYAR) ===")

    # Jalankan dengan platform offscreen agar aman di lingkungan headless / non-interaktif
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    print("1. Inisialisasi MainWindow (4-Screen Wizard): BERHASIL")

    # ==========================================
    # TEST LAYAR 1: DASHBOARD
    # ==========================================
    assert window.stacked_widget.currentIndex() == 0, "Layar awal harus Layar 1 (Dashboard)!"
    print("2. Layar 1 (Dashboard Beranda) Aktif: BERHASIL")

    # Simulasikan klik tombol [Mulai Pengamatan]
    window.screen_dashboard.btn_mulai.click()
    assert window.stacked_widget.currentIndex() == 1, "Pindah ke Layar 2 gagal!"
    print("3. Navigasi ke Layar 2 (Pengamatan): BERHASIL")

    # ==========================================
    # TEST LAYAR 2: PENGAMATAN & EKSTRAKSI
    # ==========================================
    screen2 = window.screen_observation
    assert len(screen2.captured_images) > 0, "Slot foto awal harus memiliki citra terisi!"
    assert screen2.azimuth_val == 270.0, "Azimuth default harus 270.0° (Barat)!"
    print(f"4. Layar 2 Siap dengan {len(screen2.captured_images)} foto & Bidikan Azimuth {screen2.azimuth_val}° (Barat)")

    # Picu proses ekstraksi kontur ufuk mar'i
    screen2._process_and_proceed()
    assert window.stacked_widget.currentIndex() == 2, "Pindah ke Layar 3 gagal!"
    print("5. Ekstraksi Kontur Computer Vision & Navigasi ke Layar 3: BERHASIL")

    # ==========================================
    # TEST LAYAR 3: HASIL OLAH & KURVA PROFIL
    # ==========================================
    screen3 = window.screen_result
    assert screen3.data_package is not None, "Data hasil tidak termuat di Layar 3!"
    prof = screen3.data_package["profile_data"]
    print("6. Data Terpadu Layar 3 Tervalidasi:")
    print(f"   - Rentang Azimut: {prof['stats']['az_range_min']:.2f}° s.d. {prof['stats']['az_range_max']:.2f}°")
    print(f"   - Elevasi Min: {prof['stats']['min_elevation']:.2f}°, Max: {prof['stats']['max_elevation']:.2f}°")
    print(f"   - Kerendahan Ufuk Laut (Dip): {prof['dip_deg']:.3f}°")

    # Uji pemilihan azimut sasaran hilal
    screen3._update_analysis_at_azimuth(270.5)
    target_info = screen3.data_package["target_analysis"]
    print(f"7. Inspeksi Azimut Sasaran 270.5°:")
    print(f"   - Tinggi Halangan: {target_info['target_alt']:+.2f}°")
    print(f"   - Kategori: {target_info['category']} ({target_info['description']})")

    # Simulasikan klik tombol [Lanjut ke Cetak Laporan]
    screen3.btn_to_report.click()
    assert window.stacked_widget.currentIndex() == 3, "Pindah ke Layar 4 gagal!"
    print("8. Navigasi ke Layar 4 (Laporan PDF): BERHASIL")

    # ==========================================
    # TEST LAYAR 4: LAPORAN & CETAK DOKUMEN
    # ==========================================
    screen4 = window.screen_report
    assert screen4.data_package is not None, "Data tidak termuat di Layar 4!"
    assert len(screen4.text_desc.toPlainText()) > 10, "Draft deskripsi evaluasi tidak terisi otomatis!"
    assert len(screen4.text_rec.toPlainText()) > 10, "Draft rekomendasi tidak terisi otomatis!"
    print("9. Form Evaluasi & Rekomendasi Terisi Otomatis: BERHASIL")

    # Uji Cetak PDF
    exports_dir = os.path.abspath("exports")
    os.makedirs(exports_dir, exist_ok=True)
    pdf_out = os.path.join(exports_dir, "verifikasi_laporan_ufuk.pdf")
    from core.report_generator import export_pdf_report
    export_pdf_report(
        output_pdf_path=pdf_out,
        location_name="Pos Observasi Falak Parangtritis",
        latitude=screen4.data_package["latitude"],
        longitude=screen4.data_package["longitude"],
        elevation_m=screen4.data_package["altitude"],
        az_center=screen4.data_package["azimuth"],
        hfov=15.0,
        vfov=8.5,
        tilt_center=screen4.data_package["elevation"],
        dip_deg=prof["dip_deg"],
        profile_data=prof,
        overlay_img_bgr=screen4.data_package["overlay_image"],
        target_analysis=screen4.data_package["target_analysis"],
        observer_notes=screen4.text_desc.toPlainText(),
        recommendation_text=screen4.text_rec.toPlainText(),
    )
    assert os.path.exists(pdf_out), "File PDF tidak terbentuk!"
    print(f"10. Penerbitan Laporan PDF Resmi: BERHASIL ({os.path.getsize(pdf_out)} bytes)")

    # Uji Ekspor CSV
    csv_out = os.path.join(exports_dir, "verifikasi_data_ufuk.csv")
    from core.report_generator import export_csv_data
    export_csv_data(
        output_csv_path=csv_out,
        profile_data=prof,
        location_name="Pos Observasi Falak Parangtritis",
        az_center=screen4.data_package["azimuth"],
        dip_deg=prof["dip_deg"]
    )
    assert os.path.exists(csv_out), "File CSV tidak terbentuk!"
    print(f"11. Penerbitan Data Koordinat CSV: BERHASIL ({os.path.getsize(csv_out)} bytes)")

    # Uji Simpan Plot PNG
    png_out = os.path.join(exports_dir, "verifikasi_plot_ufuk.png")
    screen3.plot_canvas.save_figure(png_out)
    assert os.path.exists(png_out), "File PNG plot tidak terbentuk!"
    print(f"12. Penerbitan Gambar Grafik PNG: BERHASIL ({os.path.getsize(png_out)} bytes)")

    window.close()
    print("=== SEMUA PENGUJIAN 4-LAYAR BERHASIL 100%! ===")


if __name__ == "__main__":
    test_full_pipeline()
