import cv2

# Step 1: Load the underwater image
img = cv2.imread("sample_underwater.jpg")

# Step 2: Show original image
cv2.imshow("Original Underwater Image", img)

# Step 3: Convert to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
cv2.imshow("Grayscale Image", gray_img)
cv2.imwrite("day1_gray.jpg", gray_img)  # save result

# Step 4: Adjust Brightness & Contrast
# alpha = contrast factor, beta = brightness factor
alpha = 1.5   # contrast (1.0 = normal, >1 = higher contrast)
beta = 30     # brightness (0 = no change, +ve = brighter, -ve = darker)

bright_contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imshow("Brightness + Contrast", bright_contrast_img)
cv2.imwrite("day1_bright_contrast.jpg", bright_contrast_img)

# Step 5: End program when key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
