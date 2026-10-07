"""
sample_generator.py
Generator citra simulasi lanskap ufuk mar'i untuk pengujian instrumen falak.
Menghasilkan citra pemandangan ufuk barat saat senja (sunset/twilight)
dengan perbukitan realistis, pohon, dan langit bergradien.
"""

import numpy as np
import cv2


def generate_synthetic_horizon(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Menghasilkan citra pemandangan ufuk realistis:
    - Langit senja bergradien halus (oranye keemasan di ufuk hingga biru gelap di atas).
    - Siluet jajaran perbukitan dengan ketinggian bervariasi.
    - Pohon dan kontur lanskap alami.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Gradien Langit Senja (Sky Gradient)
    # Atas: Biru gelap laut (Deep Navy Blue)
    # Tengah: Jingga/Ungu senja (Dusk Violet/Orange)
    # Dekat ufuk: Emas kekuningan/oranye terang (Sunset Golden Glow)
    c_top = np.array([45, 20, 15], dtype=np.float32)       # BGR: Dark Blue
    c_mid = np.array([60, 60, 190], dtype=np.float32)      # BGR: Warm Twilight Magenta
    c_low = np.array([80, 160, 255], dtype=np.float32)     # BGR: Golden Sunset Amber

    mid_y = int(height * 0.45)
    for y in range(height):
        if y < mid_y:
            t = y / max(1, mid_y)
            color = (1.0 - t) * c_top + t * c_mid
        else:
            t = (y - mid_y) / max(1, height - mid_y)
            color = (1.0 - t) * c_mid + t * c_low
        img[y, :] = color.astype(np.uint8)

    # 2. Buat Kontur Bukit Belakang (Hills in the distance - layer 1)
    x = np.arange(width)
    # Kombinasi beberapa fungsi sinus untuk kontur bukit alami
    base_hill1 = (
        height * 0.58
        + 40 * np.sin(2 * np.pi * x / (width * 0.8) + 0.5)
        + 25 * np.cos(2 * np.pi * x / (width * 0.35) + 1.2)
        + 15 * np.sin(2 * np.pi * x / (width * 0.15) + 2.0)
    ).astype(np.int32)

    hill1_color = (40, 50, 90)  # BGR: Bayangan bukit jauh kebiruan
    for col in range(width):
        y_start = max(0, min(height - 1, base_hill1[col]))
        img[y_start:, col] = hill1_color

    # 3. Buat Kontur Bukit Depan & Pepohonan (Foregound hills with terrain details - layer 2)
    base_hill2 = (
        height * 0.65
        + 55 * np.sin(2 * np.pi * x / (width * 0.9) + 2.8)
        + 30 * np.sin(2 * np.pi * x / (width * 0.4) + 0.4)
        + 10 * np.cos(2 * np.pi * x / (width * 0.1) + 1.5)
    ).astype(np.int32)

    # Tambahkan sedikit variasi acak halus untuk tekstur dedaunan
    np.random.seed(42)
    noise = np.random.normal(0, 1.2, width)
    base_hill2 = np.clip(base_hill2 + noise, 0, height - 1).astype(np.int32)

    hill2_color = (20, 25, 45)  # BGR: Siluet bukit depan lebih gelap
    for col in range(width):
        y_start = max(0, min(height - 1, base_hill2[col]))
        img[y_start:, col] = hill2_color

    # 4. Tambahkan beberapa siluet pohon khas pengamatan ufuk
    tree_positions = [int(width * 0.22), int(width * 0.72), int(width * 0.76)]
    for tx in tree_positions:
        ty = base_hill2[min(tx, width - 1)]
        tree_h = 35
        # Kanopi pohon
        cv2.circle(img, (tx, max(0, ty - tree_h + 10)), 16, (15, 20, 35), -1)
        cv2.circle(img, (tx - 8, max(0, ty - tree_h + 15)), 12, (15, 20, 35), -1)
        cv2.circle(img, (tx + 8, max(0, ty - tree_h + 15)), 12, (15, 20, 35), -1)
        # Batang
        cv2.line(img, (tx, ty), (tx, max(0, ty - tree_h + 10)), (10, 15, 25), 3)

    # 5. Berikan sedikit Gaussian blur halus agar transisi menyerupai lensa kamera asli
    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    return img


if __name__ == "__main__":
    test_img = generate_synthetic_horizon()
    cv2.imwrite("/home/alfattir/Projects/pemetaan-ufuk-mari/assets/sample_horizon.jpg", test_img)
    print("Sample horizon image successfully generated at assets/sample_horizon.jpg")
