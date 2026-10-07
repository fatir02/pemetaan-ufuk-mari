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
    Menggunakan proyeksi pinhole optik 3D terkopel rotasi pitch kamera:
      fx = (W/2) / tan(HFOV/2)
      fy = (H/2) / tan(VFOV/2)
      Xc = (x - xc) / fx, Yc = (yc - y) / fy, Zc = 1.0
      Rotasi pitch terhadap sumbu X:
      Xw = Xc, Yw = Yc*cos(tilt) + sin(tilt), Zw = -Yc*sin(tilt) + cos(tilt)
      Az = az_center + arctan2(Xw, Zw)
      Alt = arctan2(Yw, sqrt(Xw^2 + Zw^2))
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

    fx = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
    fy = (img_h / 2.0) / math.tan(vfov_rad / 2.0)

    theta = math.radians(tilt_center)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        xc_norm = (x - xc) / fx
        yc_norm = (yc - y) / fy
        xw = xc_norm
        yw = yc_norm * cos_t + sin_t
        zw = -yc_norm * sin_t + cos_t

        delta_az_deg = np.degrees(np.arctan2(xw, zw))
        azimuth = az_center + delta_az_deg
        elevation = np.degrees(np.arctan2(yw, np.sqrt(xw**2 + zw**2)))
    else:
        xc_norm = (float(x) - xc) / fx
        yc_norm = (yc - float(y)) / fy
        xw = xc_norm
        yw = yc_norm * cos_t + sin_t
        zw = -yc_norm * sin_t + cos_t

        delta_az_deg = math.degrees(math.atan2(xw, zw))
        azimuth = az_center + delta_az_deg
        elevation = math.degrees(math.atan2(yw, math.sqrt(xw**2 + zw**2)))

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
    Menggunakan proyeksi balik pinhole optik 3D terkopel pitch kamera.
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

    fx = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
    fy = (img_h / 2.0) / math.tan(vfov_rad / 2.0)

    # Selisih azimut dinormalisasi ke rentang [-180, 180] derajat
    delta_az = (az - az_center + 180.0) % 360.0 - 180.0
    delta_az_rad = math.radians(delta_az)
    alt_rad = math.radians(alt)
    theta = math.radians(tilt_center)

    # Vektor arah 3D di koordinat dunia
    xw = math.cos(alt_rad) * math.sin(delta_az_rad)
    yw = math.sin(alt_rad)
    zw = math.cos(alt_rad) * math.cos(delta_az_rad)

    # Rotasi balik pitch kamera (-theta di sumbu X) ke koordinat kamera
    xc_ray = xw
    yc_ray = yw * math.cos(theta) - zw * math.sin(theta)
    zc_ray = yw * math.sin(theta) + zw * math.cos(theta)

    if zc_ray <= 1e-6:
        return -9999, -9999

    x = xc + fx * (xc_ray / zc_ray)
    y = yc - fy * (yc_ray / zc_ray)

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
