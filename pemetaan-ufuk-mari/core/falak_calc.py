"""
falak_calc.py
Modul perhitungan matematika astronomi & ilmu falak:
- Perhitungan kerendahan ufuk (Dip of Horizon).
- Konversi koordinat sudut (DMS <-> Desimal).
- Proyeksi optik kamera (Piksel <-> Azimut & Elevasi).
- Klasifikasi halangan ufuk mar'i untuk kriteria rukyat hilal.
"""

import math
import re
from typing import Tuple, Optional, Union
import numpy as np


def parse_dms(text: Union[str, float, int]) -> float:
    """
    Mengonversi string koordinat DMS (Derajat Menit Detik) atau desimal ke float.
    Mendukung format:
      - '-07° 15\' 30.5"'
      - '7 15 30.5 S' atau '110 24 15 E'
      - '-7:15:30.5'
      - '-7.25833'
    """
    if isinstance(text, (int, float)):
        return float(text)

    s = str(text).strip()
    if not s:
        return 0.0

    # Cek jika format desimal murni
    try:
        return float(s)
    except ValueError:
        pass

    upper = s.upper()
    is_negative = False
    # Cek indikator negatif (LS, BB, S, W, atau tanda minus)
    if "LS" in upper or "BB" in upper or "S" in upper or "W" in upper or s.strip().startswith("-"):
        is_negative = True

    # Bersihkan simbol derajat, menit, detik dan arah mata angin (ID / EN)
    cleaned = re.sub(r"[°\'\"NSEWnsewLuUbBtT]", " ", s)
    cleaned = cleaned.replace(":", " ").replace(",", ".")
    parts = [float(p) for p in cleaned.split() if p.strip()]

    if not parts:
        return 0.0

    deg = abs(parts[0])
    minute = parts[1] if len(parts) > 1 else 0.0
    second = parts[2] if len(parts) > 2 else 0.0

    val = deg + (minute / 60.0) + (second / 3600.0)
    return -val if is_negative else val


def format_dms(val: float, is_lat: Optional[bool] = None, is_lon: Optional[bool] = None) -> str:
    """
    Mengonversi float derajat desimal ke string DMS yang rapi.
    Contoh:
      - format_dms(-7.25833, is_lat=True) -> '07° 15\' 30.00" LS'
      - format_dms(110.4042, is_lon=True) -> '110° 24\' 15.12" BT'
      - format_dms(2.35) -> '+02° 21\' 00.00"'
    """
    sign = "-" if val < 0 else "+"
    abs_val = abs(val)

    deg = int(abs_val)
    remainder = (abs_val - deg) * 60.0
    minute = int(remainder)
    second = (remainder - minute) * 60.0

    if is_lat is not None:
        direction = "LS" if val < 0 else "LU"
        return f"{deg:02d}° {minute:02d}' {second:05.2f}\" {direction}"
    elif is_lon is not None:
        direction = "BB" if val < 0 else "BT"
        return f"{deg:03d}° {minute:02d}' {second:05.2f}\" {direction}"
    else:
        return f"{sign}{deg:02d}° {minute:02d}' {second:05.2f}\""


def calculate_dip(elevation_m: float) -> Tuple[float, float]:
    """
    Menghitung kerendahan ufuk (Dip of horizon) berdasarkan ketinggian tempat (h).
    Rumus standar Falak:
      Dip (menit busur) = 1.76 * sqrt(h)
      Dip (derajat)     = (1.76 / 60.0) * sqrt(h) ≈ 0.029333 * sqrt(h)
    
    Returns:
      (dip_deg, dip_arcmin)
    """
    if elevation_m <= 0:
        return 0.0, 0.0

    dip_arcmin = 1.76 * math.sqrt(elevation_m)
    dip_deg = dip_arcmin / 60.0
    return dip_deg, dip_arcmin


def pixel_to_azimuth_elevation(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    img_w: int,
    img_h: int,
    az_center: float,
    hfov: float,
    vfov: Optional[float] = None,
    tilt_center: float = 0.0,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Mengonversi koordinat piksel (x, y) menjadi (Azimut, Elevasi) dalam derajat.
    Menggunakan proyeksi pinhole optik standar:
      tan(ΔAz)  = ((x - xc) / (W/2)) * tan(HFOV / 2)
      tan(ΔAlt) = ((yc - y) / (H/2)) * tan(VFOV / 2)
    
    Kamera:
      - x=0 (sisi kiri) -> Azimut lebih kecil (Selatan jika menghadap Barat)
      - x=W-1 (sisi kanan) -> Azimut lebih besar (Utara jika menghadap Barat)
      - y=0 (puncak gambar) -> Elevasi positif / lebih tinggi
      - y=H-1 (dasar gambar) -> Elevasi negatif / lebih rendah
    """
    if img_w <= 0 or img_h <= 0:
        return az_center, tilt_center

    hfov_rad = math.radians(hfov)
    if vfov is None or vfov <= 0:
        # Hitung VFOV otomatis dari rasio aspek
        vfov_rad = 2.0 * math.atan((img_h / img_w) * math.tan(hfov_rad / 2.0))
    else:
        vfov_rad = math.radians(vfov)

    xc = (img_w - 1) / 2.0
    yc = (img_h - 1) / 2.0

    tan_half_hfov = math.tan(hfov_rad / 2.0)
    tan_half_vfov = math.tan(vfov_rad / 2.0)

    # Perhitungan delta Azimut
    norm_x = (x - xc) / (img_w / 2.0)
    if isinstance(x, np.ndarray):
        delta_az_rad = np.arctan(norm_x * tan_half_hfov)
        azimuth = (az_center + np.degrees(delta_az_rad)) % 360.0
    else:
        delta_az_rad = math.atan(norm_x * tan_half_hfov)
        azimuth = (az_center + math.degrees(delta_az_rad)) % 360.0

    # Perhitungan delta Elevasi
    norm_y = (yc - y) / (img_h / 2.0)
    if isinstance(y, np.ndarray):
        delta_alt_rad = np.arctan(norm_y * tan_half_vfov)
        elevation = tilt_center + np.degrees(delta_alt_rad)
    else:
        delta_alt_rad = math.atan(norm_y * tan_half_vfov)
        elevation = tilt_center + math.degrees(delta_alt_rad)

    return azimuth, elevation


def azimuth_elevation_to_pixel(
    az: float,
    alt: float,
    img_w: int,
    img_h: int,
    az_center: float,
    hfov: float,
    vfov: Optional[float] = None,
    tilt_center: float = 0.0,
) -> Tuple[int, int]:
    """
    Mengonversi sudut (Azimut, Elevasi) menjadi koordinat piksel (x, y).
    Mengembalikan koordinat bulat (integer).
    """
    if img_w <= 0 or img_h <= 0:
        return int(img_w / 2), int(img_h / 2)

    hfov_rad = math.radians(hfov)
    if vfov is None or vfov <= 0:
        vfov_rad = 2.0 * math.atan((img_h / img_w) * math.tan(hfov_rad / 2.0))
    else:
        vfov_rad = math.radians(vfov)

    xc = (img_w - 1) / 2.0
    yc = (img_h - 1) / 2.0

    tan_half_hfov = math.tan(hfov_rad / 2.0)
    tan_half_vfov = math.tan(vfov_rad / 2.0)

    # Selisih azimut
    delta_az = az - az_center
    # Tangani wrapping 360 derajat jika melintasi batas 0/360
    if delta_az > 180.0:
        delta_az -= 360.0
    elif delta_az < -180.0:
        delta_az += 360.0

    delta_az_rad = math.radians(delta_az)
    norm_x = math.tan(delta_az_rad) / tan_half_hfov
    x = xc + (norm_x * (img_w / 2.0))

    # Selisih elevasi
    delta_alt = alt - tilt_center
    delta_alt_rad = math.radians(delta_alt)
    norm_y = math.tan(delta_alt_rad) / tan_half_vfov
    y = yc - (norm_y * (img_h / 2.0))

    return int(round(x)), int(round(y))


def classify_obstacle(alt_obstacle: float, dip_deg: float = 0.0) -> dict:
    """
    Mengklasifikasikan status halangan ufuk mar'i pada azimut tertentu
    dan memberikan rekomendasi falakiah terkait kelayakan rukyat hilal.
    
    Kriteria Acuan:
    - Ufuk Hakiki = 0.0°
    - Ufuk Mar'i Laut = -dip_deg
    - Kriteria Baru MABIMS (Tinggi Hilal minimal 3°, Elongasi 6.4°)
    """
    sea_horizon = -dip_deg

    if alt_obstacle <= sea_horizon + 0.05:
        category = "Ufuk Terbuka (Laut/Lembah Bebas Halangan)"
        status = "Bebas Halangan"
        color = "#00e676"  # Hijau terang
        severity = "Sangat Baik"
        recommendation = (
            "Ufuk mar'i berada tepat di kerendahan ufuk laut. Lokasi sangat ideal "
            "untuk rukyatul hilal, terbenamnya matahari terpantau sempurna tanpa rintangan."
        )
    elif alt_obstacle <= 0.0:
        category = "Ufuk Rendah Terbuka (Di Bawah Ufuk Hakiki)"
        status = "Bebas Halangan"
        color = "#29b6f6"  # Biru langit terang
        severity = "Sangat Baik"
        recommendation = (
            "Halangan berada di bawah ufuk hakiki (0°). Pandangan ke arah ghurub "
            "sangat leluasa untuk observasi benda langit."
        )
    elif alt_obstacle <= 1.2:
        category = "Halangan Rendah (Bukit Jauh / Garis Pantai)"
        status = "Halangan Rendah"
        color = "#ffca28"  # Kuning amber
        severity = "Cukup Layak"
        recommendation = (
            f"Terdapat siluet halangan rendah setinggi {alt_obstacle:.2f}°. "
            "Hilal dengan ketinggian di atas 1.5° masih aman terpantau."
        )
    elif alt_obstacle <= 3.0:
        category = "Halangan Sedang (Perbukitan / Pohon)"
        status = "Terhalang Bukit/Pohon"
        color = "#ff7043"  # Oranye kemerahan
        severity = "Rawan / Perlu Waspada"
        recommendation = (
            f"Ketinggian halangan mencapai {alt_obstacle:.2f}°. "
            "Kritis untuk hilal awal bulan (ambang MABIMS 3° berada tepat di tepi halangan). "
            "Diperlukan evaluasi sudut lintang bulan saat rukyat."
        )
    else:
        category = "Halangan Tinggi (Gunung / Vegetasi Rimbun / Bangunan)"
        status = "Terhalang Bukit/Bangunan Tinggi"
        color = "#ff1744"  # Merah terang
        severity = "Tidak Direkomendasikan"
        recommendation = (
            f"Tinggi halangan signifikan ({alt_obstacle:.2f}° di atas ufuk hakiki). "
            "Sangat berisiko menutupi posisi hilal saat matahari terbenam. "
            "Disarankan mencari titik pengamatan yang lebih tinggi atau bebas rintangan."
        )

    return {
        "status": status,
        "category": category,
        "color": color,
        "severity": severity,
        "recommendation": recommendation,
        "altitude": alt_obstacle,
        "sea_horizon": sea_horizon,
        "above_sea_horizon": alt_obstacle - sea_horizon,
        "above_true_horizon": alt_obstacle,
    }
