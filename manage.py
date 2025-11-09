import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load image
# -------------------------------
img = cv2.imread("sample_underwater.jpg")

if img is None:
    print("Error: Image not found.")
    exit()

img = cv2.resize(img, (800, 600))

# -------------------------------
# Step 2: Auto Gray-World Color Correction
# -------------------------------
def gray_world_correction(image):
    b, g, r = cv2.split(image)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3
    scale_b, scale_g, scale_r = mean_gray / mean_b, mean_gray / mean_g, mean_gray / mean_r
    b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))

img_corrected = gray_world_correction(img)

# -------------------------------
# Step 3: CLAHE (Contrast Enhancement)
# -------------------------------
lab = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
lab_clahe = cv2.merge((l_clahe, a, b))
img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# -------------------------------
# Step 4: White Balance Tuning
# -------------------------------
b, g, r = cv2.split(img_clahe)
r = cv2.addWeighted(r, 1.3, r, 0, 0)
g = cv2.addWeighted(g, 1.1, g, 0, 0)
b = cv2.addWeighted(b, 0.9, b, 0, 0)
color_corrected = cv2.merge((b, g, r))

# -------------------------------
# Step 5: Adaptive Gamma Correction
# -------------------------------
gray_mean = np.mean(cv2.cvtColor(color_corrected, cv2.COLOR_BGR2GRAY)) / 255
gamma = np.clip(1.2 + (0.5 - gray_mean), 0.8, 1.8)
gamma_correction = np.array(255 * (color_corrected / 255) ** (1 / gamma), dtype='uint8')

# -------------------------------
# Step 6: Wavelet-Based Detail Enhancement
# -------------------------------
gray = cv2.cvtColor(gamma_correction, cv2.COLOR_BGR2GRAY)
coeffs = pywt.dwt2(gray, 'db1')
cA, (cH, cV, cD) = coeffs
detail = cH + cV + cD
detail_enhanced = np.clip(detail * 2.0, 0, 255).astype(np.uint8)
detail_resized = cv2.resize(detail_enhanced, (gray.shape[1], gray.shape[0]))
wavelet_merge = cv2.addWeighted(gray, 0.8, detail_resized, 0.2, 0)
wavelet_merge = cv2.cvtColor(wavelet_merge, cv2.COLOR_GRAY2BGR)

# -------------------------------
# Step 7: Adaptive Weighted Fusion
# -------------------------------
gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
gray_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY) / 255.0
gray_wavelet = cv2.cvtColor(wavelet_merge, cv2.COLOR_BGR2GRAY) / 255.0

contrast_base = cv2.Laplacian(gray_base, cv2.CV_64F).var()
contrast_clahe = cv2.Laplacian(gray_clahe, cv2.CV_64F).var()
contrast_wavelet = cv2.Laplacian(gray_wavelet, cv2.CV_64F).var()

brightness_base = np.mean(gray_base)
brightness_clahe = np.mean(gray_clahe)
brightness_wavelet = np.mean(gray_wavelet)

w1 = 0.3 + 0.4 * contrast_clahe + 0.2 * brightness_clahe
w2 = 0.3 + 0.4 * contrast_wavelet + 0.2 * brightness_wavelet
w3 = 1.0 - (w1 + w2) / 2
sum_w = w1 + w2 + w3
w1, w2, w3 = w1/sum_w, w2/sum_w, w3/sum_w

adaptive_fused = cv2.addWeighted(img_clahe, w1, wavelet_merge, w2, 0)
adaptive_fused = cv2.addWeighted(adaptive_fused, 1.0, gamma_correction, w3, 0)

# -------------------------------
# Step 8: Final Bilateral Smoothing
# -------------------------------
final_output = cv2.bilateralFilter(adaptive_fused, 9, 75, 75)

# -------------------------------
# Step 9: Show and Save Results
# -------------------------------
cv2.imshow("Original Image", img)
cv2.imshow("CLAHE + Color Corrected", img_clahe)
cv2.imshow("Wavelet Enhanced", wavelet_merge)
cv2.imshow("Final Enhanced Output", final_output)

cv2.imwrite("underwater_enhanced_day4.jpg", final_output)

# Optional: Comparison Grid
comparison = np.hstack((img, img_clahe, final_output))
cv2.imwrite("comparison_grid_day4.jpg", comparison)

# Optional: Histogram Plot (for analysis)
plt.figure(figsize=(10,4))
plt.title("Histogram Before vs After Enhancement")
plt.hist(img.ravel(), bins=256, color='gray', alpha=0.5, label='Original')
plt.hist(final_output.ravel(), bins=256, color='blue', alpha=0.5, label='Enhanced')
plt.legend()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
