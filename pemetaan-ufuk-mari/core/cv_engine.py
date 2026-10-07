"""
cv_engine.py
Engine Computer Vision untuk segmentasi dan ekstraksi profil kontur ufuk mar'i.
Mengidentifikasi batas pertemuan langit (sky) dan daratan/halangan (ground/terrain)
per kolom horizontal, mengonversi ke profil elevasi sudut astronomi.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from scipy.signal import medfilt

from core.falak_calc import (
    pixel_to_azimuth_elevation,
    calculate_dip,
    classify_obstacle,
)


class HorizonDetector:
    """
    Kelas pendeteksi ufuk mar'i dengan beberapa algoritma computer vision:
    1. Gradien Vertikal (Optimal untuk senja, siluet bukit, dan ufuk rukyat)
    2. Adaptive / Otsu Thresholding (Segmentasi biner intensitas)
    3. Canny Edge + Morfologi Kontur
    """

    METHOD_GRADIENT = "Gradien Vertikal (Optimal Senja/Siluet)"
    METHOD_OTSU = "Otsu Adaptive Threshold"
    METHOD_CANNY = "Canny Edge + Morfologi"

    ALL_METHODS = [METHOD_GRADIENT, METHOD_OTSU, METHOD_CANNY]

    def __init__(self):
        self.last_horizon_y = None
        self.last_azimuths = None
        self.last_elevations = None
        self.last_bw_data = None

    def preprocess_bw(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Pengolahan citra ilmiah menggunakan OpenCV untuk akurasi maksimal:
        1. Konversi ke Grayscale (hitam-putih).
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk menonjolkan
           kontras tipis batas langit mendung vs bukit/daratan.
        3. Bilateral Filter untuk menghilangkan noise sensor tanpa mengaburkan tepi batas ufuk.
        4. Otsu Adaptive Binarization untuk segmentasi biner hitam-putih.

        Returns:
            Dict dengan:
            - 'enhanced_bw': Citra grayscale kontras tinggi 3-channel BGR.
            - 'binary_bw': Citra biner hitam-putih murni 3-channel BGR.
            - 'gray': Citra 1-channel grayscale untuk kalkulasi gradien presisi.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE untuk mendongkrak kontras lokal
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Bilateral Filter: pereduksi noise yang menjaga ketajaman garis tepi ufuk
        denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

        # 3. Segmentasi Biner Otsu
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        binary_bgr = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)

        return {
            "enhanced_bw": enhanced_bgr,
            "binary_bw": binary_bgr,
            "gray": denoised,
        }

    def detect_horizon(
        self,
        image_bgr: np.ndarray,
        method: str = METHOD_GRADIENT,
        threshold_offset: int = 0,
        blur_kernel: int = 5,
        smooth_window: int = 15,
        invert_mask: bool = False,
        roi_top_pct: float = 0.08,
        roi_bottom_pct: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mendeteksi batas ufuk mar'i untuk setiap kolom horizontal citra.
        Menggunakan pra-pengolahan citra hitam-putih OpenCV (CLAHE + Bilateral)
        untuk menjamin akurasi tinggi pada kondisi langit apapun.
        """
        h, w = image_bgr.shape[:2]
        if h < 20 or w < 20:
            raise ValueError("Dimensi gambar terlalu kecil untuk ekstraksi ufuk.")

        # Jalankan pra-pengolahan citra hitam-putih OpenCV
        bw_data = self.preprocess_bw(image_bgr)
        self.last_bw_data = bw_data
        gray = bw_data["gray"]

        y_min = int(h * np.clip(roi_top_pct, 0.0, 0.4))
        y_max = int(h * np.clip(roi_bottom_pct, 0.6, 1.0))
        if y_max <= y_min + 10:
            y_min, y_max = int(h * 0.1), int(h * 0.9)

        mask = np.zeros((h, w), dtype=np.uint8)
        horizon_y = np.full(w, int(h * 0.5), dtype=np.int32)

        if method == self.METHOD_GRADIENT:
            # Algoritma Dynamic Programming (DP) Skyline Extraction
            # Normalisasi kolom + penalti diskontinuitas kuadratik
            # Bebas dari lonjakan palsu ke langit (overcast) maupun kebocoran ke sawah/jalan
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sign = -1.0 if not invert_mask else 1.0
            grad = sign * sobel_y[y_min:y_max, :]
            grad_pos = np.clip(grad, 0, None)

            # Normalisasi per kolom untuk menjaga sensitivitas kontras lokal
            col_max = np.max(grad_pos, axis=0, keepdims=True)
            col_max[col_max == 0] = 1.0
            grad_norm = grad_pos / col_max

            cost = 1.0 - grad_norm
            roi_h = y_max - y_min

            dp = np.zeros((roi_h, w), dtype=np.float32)
            parent = np.zeros((roi_h, w), dtype=np.int32)
            dp[:, 0] = cost[:, 0]

            max_jump = max(4, int(roi_h * 0.04))
            jump_penalty = 0.5

            for x in range(1, w):
                for y in range(roi_h):
                    y_start = max(0, y - max_jump)
                    y_end = min(roi_h, y + max_jump + 1)
                    prev_costs = dp[y_start:y_end, x - 1] + jump_penalty * (np.abs(np.arange(y_start, y_end) - y) ** 1.2)
                    min_idx = int(np.argmin(prev_costs))
                    dp[y, x] = cost[y, x] + prev_costs[min_idx]
                    parent[y, x] = y_start + min_idx

            horizon_roi = np.zeros(w, dtype=np.int32)
            horizon_roi[-1] = int(np.argmin(dp[:, -1]))
            for x in range(w - 2, -1, -1):
                horizon_roi[x] = parent[horizon_roi[x + 1], x + 1]

            horizon_y = y_min + horizon_roi
            for x in range(w):
                mask[horizon_y[x]:, x] = 255

        elif method == self.METHOD_CANNY:
            # Metode Canny Edge + Pencarian Kontur Teratas
            v = float(np.median(gray))
            lower = int(max(0, (0.66 + threshold_offset / 100.0) * v))
            upper = int(min(255, (1.33 + threshold_offset / 100.0) * v))
            edges = cv2.Canny(gray, lower, upper)

            # Morfologi closing horizontal untuk menghubungkan tepi bukit
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

            # Temukan tepi pertama dari atas di dalam rentang ROI
            for x in range(w):
                col_edges = np.where(closed[y_min:y_max, x] > 0)[0]
                if len(col_edges) > 0:
                    horizon_y[x] = y_min + col_edges[0]
                else:
                    # Fallback ke baris rata-rata
                    horizon_y[x] = int(h * 0.55)
                mask[horizon_y[x]:, x] = 255

        else:
            # METHOD_OTSU: Adaptive Thresholding
            # Pisahkan langit di bagian atas dan daratan di bagian bawah
            # Hitung threshold lokal atau Otsu pada wilayah pencarian
            roi_gray = gray[y_min:y_max, :]
            otsu_val, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            target_thresh = int(np.clip(otsu_val + threshold_offset, 10, 245))

            if not invert_mask:
                _, raw_mask = cv2.threshold(gray, target_thresh, 255, cv2.THRESH_BINARY_INV)
            else:
                _, raw_mask = cv2.threshold(gray, target_thresh, 255, cv2.THRESH_BINARY)

            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
            cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

            # Ambil piksel daratan teratas di dalam ROI
            for x in range(w):
                col = cleaned_mask[y_min:y_max, x]
                ground_idxs = np.where(col > 0)[0]
                if len(ground_idxs) > 0:
                    horizon_y[x] = y_min + ground_idxs[0]
                else:
                    horizon_y[x] = y_max
                mask[horizon_y[x]:, x] = 255

        # Smoothing kontur untuk membuang artefak tajam/noise kabel tanpa mengubah lekuk bukit
        sw = max(3, smooth_window | 1)
        if sw >= 3 and len(horizon_y) > sw:
            # Menggunakan boundary mode 'nearest' agar tepi kiri dan kanan tidak loncat/anjlok
            from scipy.ndimage import median_filter, uniform_filter1d
            horizon_y = median_filter(horizon_y.astype(np.float64), size=sw, mode="nearest")
            smoothed = uniform_filter1d(horizon_y, size=5, mode="nearest")
            horizon_y = np.clip(smoothed, y_min, y_max).astype(np.int32)

        self.last_horizon_y = horizon_y
        return horizon_y, mask

    def compute_horizon_profile(
        self,
        horizon_y: np.ndarray,
        img_w: int,
        img_h: int,
        az_center: float,
        hfov: float,
        vfov: Optional[float] = None,
        tilt_center: float = 0.0,
        elevation_m: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Menghitung profil sudut (Azimut dan Elevasi) lengkap dari array horizon_y.
        """
        x_indices = np.arange(img_w)
        azimuths, elevations = pixel_to_azimuth_elevation(
            x=x_indices,
            y=horizon_y,
            img_w=img_w,
            img_h=img_h,
            az_center=az_center,
            hfov=hfov,
            vfov=vfov,
            tilt_center=tilt_center,
        )

        dip_deg, dip_arcmin = calculate_dip(elevation_m)

        min_idx = int(np.argmin(elevations))
        max_idx = int(np.argmax(elevations))

        stats = {
            "min_elevation": float(np.min(elevations)),
            "min_azimuth": float(azimuths[min_idx]),
            "max_elevation": float(np.max(elevations)),
            "max_azimuth": float(azimuths[max_idx]),
            "mean_elevation": float(np.mean(elevations)),
            "az_range_min": float(azimuths[0]),
            "az_range_max": float(azimuths[-1]),
            "hfov": hfov,
            "vfov": vfov if vfov is not None else float(np.ptp(elevations)),
        }

        self.last_azimuths = azimuths
        self.last_elevations = elevations

        return {
            "azimuths": azimuths,
            "elevations": elevations,
            "horizon_y": horizon_y,
            "dip_deg": dip_deg,
            "dip_arcmin": dip_arcmin,
            "stats": stats,
            "img_w": img_w,
            "img_h": img_h,
            "az_center": az_center,
            "hfov": hfov,
            "vfov": vfov,
            "tilt_center": tilt_center,
            "elevation_m": elevation_m,
        }

    def render_overlay(
        self,
        image_bgr: np.ndarray,
        horizon_y: np.ndarray,
        azimuths: Optional[np.ndarray] = None,
        elevations: Optional[np.ndarray] = None,
        target_az: Optional[float] = None,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None,
        show_true_horizon: bool = True,
        dip_deg: float = 0.0,
        az_center: float = 270.0,
        hfov: float = 15.0,
        vfov: Optional[float] = None,
        tilt_center: float = 0.0,
    ) -> np.ndarray:
        """
        Merender kontur ufuk mar'i dan garis referensi astronomi pada gambar.
        - Garis Cyan/Neon: Kontur Ufuk Mar'i
        - Garis Putih Putus-putus: Ufuk Hakiki (0°)
        - Garis Kuning Tipis: Kerendahan Ufuk Laut (-Dip)
        - Garis Merah Vertikal & Titik Bidik: Posisi Azimut Sasaran Terpilih
        - Salib tengah optik (Center Optical Crosshair)
        """
        h, w = image_bgr.shape[:2]
        canvas = image_bgr.copy()

        # 1. Gambar Ufuk Hakiki (0°) jika diminta
        if show_true_horizon:
            from core.falak_calc import azimuth_elevation_to_pixel
            _, y_true = azimuth_elevation_to_pixel(
                az_center, 0.0, w, h, az_center, hfov, vfov, tilt_center
            )
            if 0 <= y_true < h:
                dash_len = 16
                for dx in range(0, w, dash_len * 2):
                    cv2.line(canvas, (dx, y_true), (min(w - 1, dx + dash_len), y_true), (220, 220, 220), 1)
                cv2.putText(
                    canvas, "Ufuk Hakiki (0.00 deg)", (15, max(20, y_true - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
                )

            # Garis Dip (Ufuk Mar'i Laut) jika ketinggian > 0
            if dip_deg > 0:
                _, y_dip = azimuth_elevation_to_pixel(
                    az_center, -dip_deg, w, h, az_center, hfov, vfov, tilt_center
                )
                if 0 <= y_dip < h:
                    dash_len = 12
                    for dx in range(0, w, dash_len * 2):
                        cv2.line(canvas, (dx, y_dip), (min(w - 1, dx + dash_len), y_dip), (0, 220, 255), 1)
                    cv2.putText(
                        canvas, f"Ufuk Laut Dip (-{dip_deg:.2f} deg)", (15, min(h - 10, y_dip + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA
                    )

        # 2. Gambar Salib Sumbu Tengah Optik (Optical Center Crosshair)
        cx, cy = int((w - 1) / 2), int((h - 1) / 2)
        cv2.line(canvas, (cx - 15, cy), (cx + 15, cy), (120, 120, 120), 1)
        cv2.line(canvas, (cx, cy - 15), (cx, cy + 15), (120, 120, 120), 1)

        # 3. Gambar Kontur Ufuk Mar'i (Polylines Kuning Terang Ber-outline Hitam)
        # Hitung ketebalan proporsional agar tetap terlihat tajam pada foto resolusi tinggi
        line_thick = max(3, int(round(w / 350.0)))
        outline_thick = line_thick + max(2, int(line_thick * 0.6))

        pts = np.column_stack((np.arange(w), horizon_y)).reshape((-1, 1, 2))
        # Garis outline hitam agar kontras di latar langit terang/awan
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 0), thickness=outline_thick, lineType=cv2.LINE_AA)
        # Garis kontur kuning neon terang
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 255), thickness=line_thick, lineType=cv2.LINE_AA)

        # 4. Kursor Azimut Sasaran (Target Azimuth Line & Marker)
        if target_x is not None and 0 <= target_x < w:
            ty = horizon_y[target_x] if target_y is None else target_y
            cv2.line(canvas, (target_x, 0), (target_x, h - 1), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (target_x, ty), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (target_x, ty), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (target_x, ty), 15, (0, 0, 255), 1, cv2.LINE_AA)

            if target_az is not None and elevations is not None:
                alt_val = elevations[target_x]
                txt = f"Az: {target_az:.2f} deg | Alt: {alt_val:+.2f} deg"
                tx_box = min(w - 245, max(10, target_x - 120))
                ty_box = max(35, ty - 25)
                cv2.rectangle(canvas, (tx_box, ty_box - 20), (tx_box + 240, ty_box + 8), (20, 20, 25), -1)
                cv2.rectangle(canvas, (tx_box, ty_box - 20), (tx_box + 240, ty_box + 8), (0, 50, 255), 1)
                cv2.putText(
                    canvas, txt, (tx_box + 6, ty_box - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

        return canvas
