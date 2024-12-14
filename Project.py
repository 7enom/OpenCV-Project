import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to apply motion blur
def motion_blur(image, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

# Function for Wiener filter deblurring
def wiener_filter(image, kernel, K=0.01):
    kernel /= kernel.sum()  # Normalize the kernel
    restored_channels = []  # Store results for each channel

    for c in range(image.shape[2]):  # Process each color channel
        image_fft = np.fft.fft2(image[:, :, c])  # Fourier Transform of the channel
        kernel_fft = np.fft.fft2(kernel, s=image.shape[:2])  # Kernel FFT
        kernel_fft_conj = np.conj(kernel_fft)  # Conjugate of the kernel FFT

        denominator = kernel_fft * kernel_fft_conj + K  # Wiener filter formula
        result_fft = image_fft * kernel_fft_conj / denominator  # Apply Wiener filter
        restored_channel = np.abs(np.fft.ifft2(result_fft))  # Inverse FFT
        restored_channels.append(restored_channel)

    # Combine the restored channels into a single image
    restored_image = np.stack(restored_channels, axis=2)
    return np.uint8(np.clip(restored_image, 0, 255))  # Ensure valid pixel range

# Apply motion blur
kernel_size = 15
motion_blurred = motion_blur(image, kernel_size)

# Define the motion blur kernel for deblurring
motion_kernel = np.zeros((kernel_size, kernel_size))
motion_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size

# Apply Wiener filter to deblur the image
motion_deblurred = wiener_filter(motion_blurred, motion_kernel)

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(motion_blurred)
plt.title("Motion Blurred Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(motion_deblurred)
plt.title("Deblurred Image")
plt.axis("off")

plt.tight_layout()
plt.show()
