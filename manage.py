import cv2
import numpy as np
import pywt  # new import for wavelet transform

# Step 1–6: existing preprocessing pipeline
img = cv2.imread("sample_underwater.jpg")
if img is None:
    print("Error: Image not found.")
    exit()

img = cv2.resize(img, (800, 600))
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
img_clahe = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

b, g, r = cv2.split(img_clahe)
r = cv2.addWeighted(r, 1.4, r, 0, 0)
g = cv2.addWeighted(g, 1.1, g, 0, 0)
b = cv2.addWeighted(b, 0.9, b, 0, 0)
color_corrected = cv2.merge((b, g, r))

gamma = 1.2
gamma_correction = np.array(255 * (color_corrected / 255) ** (1 / gamma), dtype='uint8')
sharp = cv2.bilateralFilter(gamma_correction, 9, 75, 75)

# -------------------------------------------
# 🔹 Step 7: Wavelet-based detail enhancement
# -------------------------------------------
def wavelet_detail_enhancement(image):
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Perform single-level DWT on luminance (Y) channel
    coeffs2 = pywt.dwt2(y.astype(np.float32), 'db1')
    LL, (LH, HL, HH) = coeffs2

    # Enhance high-frequency sub-bands slightly
    boost_factor = 1.5
    LH *= boost_factor
    HL *= boost_factor
    HH *= boost_factor

    # Reconstruct enhanced Y channel
    y_enhanced = pywt.idwt2((LL, (LH, HL, HH)), 'db1')
    y_enhanced = np.clip(y_enhanced, 0, 255).astype(np.uint8)

    # Merge back and convert to BGR
    merged = cv2.merge((y_enhanced, cr, cb))
    result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return result

wavelet_enhanced = wavelet_detail_enhancement(sharp)

# Step 8: Display results
cv2.imshow("Original Underwater Image", img)
cv2.imshow("Enhanced + Wavelet Detail", wavelet_enhanced)
cv2.imwrite("underwater_enhanced_wavelet.jpg", wavelet_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
