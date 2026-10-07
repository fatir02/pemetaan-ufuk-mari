import numpy as np
import cv2

# Buat ikon 128x128 falak
icon = np.zeros((128, 128, 4), dtype=np.uint8)

# Latar belakang lingkaran gelap merah maroon / crimson
cv2.circle(icon, (64, 64), 60, (25, 20, 30, 255), -1, cv2.LINE_AA)
cv2.circle(icon, (64, 64), 58, (45, 25, 180, 255), 2, cv2.LINE_AA)

# Siluet bukit ufuk
pts = np.array([
    [10, 85], [35, 70], [65, 80], [95, 65], [118, 78],
    [118, 118], [10, 118]
], dtype=np.int32)
cv2.fillPoly(icon, [pts], (30, 30, 40, 255))

# Garis kontur cyan menyala
contour_pts = np.array([
    [10, 85], [35, 70], [65, 80], [95, 65], [118, 78]
], dtype=np.int32)
cv2.polylines(icon, [contour_pts], False, (255, 220, 0, 255), 3, cv2.LINE_AA)

# Hilal (Bulan Sabit) di langit senja
cv2.circle(icon, (60, 42), 16, (0, 215, 255, 255), -1, cv2.LINE_AA)
cv2.circle(icon, (66, 38), 14, (25, 20, 30, 255), -1, cv2.LINE_AA)

# Garis bidik optik merah tipis
cv2.line(icon, (95, 20), (95, 108), (0, 50, 255, 200), 2, cv2.LINE_AA)
cv2.circle(icon, (95, 65), 5, (0, 0, 255, 255), -1, cv2.LINE_AA)

cv2.imwrite('/home/alfattir/Projects/pemetaan-ufuk-mari/assets/icon.png', icon)
print('Icon generated successfully!')
