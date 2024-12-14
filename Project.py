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

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image = cv2.imread("./Images/car.jpg", cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Function to apply a box blur
# def box_blur(image, kernel_size):
#     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
#     return cv2.filter2D(image, -1, kernel)

# # Function to apply a motion blur
# def motion_blur(image, kernel_size):
#     kernel = np.zeros((kernel_size, kernel_size))
#     kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
#     kernel /= kernel_size
#     return cv2.filter2D(image, -1, kernel)

# # Function for Wiener filter deblurring
# def wiener_filter(image, kernel, K=0.01):
#     kernel /= kernel.sum()  # Normalize the kernel
#     restored_channels = []  # Store results for each channel

#     for c in range(image.shape[2]):  # Process each color channel
#         image_fft = np.fft.fft2(image[:, :, c])  # Fourier Transform of the channel
#         kernel_fft = np.fft.fft2(kernel, s=image.shape[:2])  # Kernel FFT
#         kernel_fft_conj = np.conj(kernel_fft)  # Conjugate of the kernel FFT

#         denominator = kernel_fft * kernel_fft_conj + K  # Wiener filter formula
#         result_fft = image_fft * kernel_fft_conj / denominator  # Apply Wiener filter
#         restored_channel = np.abs(np.fft.ifft2(result_fft))  # Inverse FFT
#         restored_channels.append(restored_channel)

#     # Combine the restored channels into a single image
#     restored_image = np.stack(restored_channels, axis=2)
#     return np.uint8(np.clip(restored_image, 0, 255))  # Ensure valid pixel range


# # Apply blurring techniques
# box_blurred = box_blur(image, kernel_size=15)
# motion_blurred = motion_blur(image, kernel_size=15)

# # Estimate deblurring (assuming kernel is known)
# motion_kernel = np.zeros((15, 15))
# motion_kernel[7, :] = np.ones(15) / 15  # Custom motion blur kernel
# motion_deblurred = wiener_filter(motion_blurred, motion_kernel)

# # Display results
# plt.figure(figsize=(15, 10))

# plt.subplot(2, 3, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(2, 3, 2)
# plt.imshow(box_blurred)
# plt.title("Box Blurred")
# plt.axis("off")

# plt.subplot(2, 3, 3)
# plt.imshow(motion_blurred)
# plt.title("Motion Blurred")
# plt.axis("off")

# plt.subplot(2, 3, 4)
# plt.imshow(motion_deblurred)
# plt.title("Deblurred (Wiener Filter)")
# plt.axis("off")

# plt.tight_layout()
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load an image
# image = cv2.imread('./Images/car.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Display the original image
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 3, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.axis("off")

# # Gaussian Blur
# gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)
# plt.subplot(2, 3, 2)
# plt.imshow(gaussian_blur)
# plt.title("Gaussian Blur")
# plt.axis("off")

# # Median Blur
# median_blur = cv2.medianBlur(image, 15)
# plt.subplot(2, 3, 3)
# plt.imshow(median_blur)
# plt.title("Median Blur")
# plt.axis("off")

# # Motion Blur
# size = 15
# kernel_motion_blur = np.zeros((size, size))
# kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
# kernel_motion_blur /= size
# motion_blur = cv2.filter2D(image, -1, kernel_motion_blur)
# plt.subplot(2, 3, 4)
# plt.imshow(motion_blur)
# plt.title("Motion Blur")
# plt.axis("off")

# # Deblurring using Inverse Filtering
# def inverse_filter(image, kernel):
#     # Normalize the kernel
#     kernel /= np.sum(kernel)
    
#     # Expand kernel dimensions to match the image shape (for RGB channels)
#     kernel = np.repeat(kernel[:, :, np.newaxis], 3, axis=2)
    
#     # Perform Fourier Transform on the kernel and image
#     kernel_fft = np.fft.fft2(kernel, s=image.shape[:2], axes=(0, 1))
#     image_fft = np.fft.fft2(image, axes=(0, 1))
    
#     # Avoid division by zero using a small epsilon (1e-3)
#     restored_fft = image_fft / (kernel_fft + 1e-3)
    
#     # Perform Inverse Fourier Transform to obtain the deblurred image
#     restored = np.fft.ifft2(restored_fft, axes=(0, 1))
    
#     # Take absolute value and convert to uint8 for valid image representation
#     restored = np.abs(restored).astype(np.uint8)
    
#     return restored


# restored_image = inverse_filter(motion_blur, kernel_motion_blur)
# plt.subplot(2, 3, 5)
# plt.imshow(restored_image)
# plt.title("Restored Image")
# plt.axis("off")

# # Show all results
# plt.tight_layout()
# plt.show() 
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image = cv2.imread("./Images/car.jpg", cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# # Sobel Edge Detection
# sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X-direction
# sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y-direction
# sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# # Canny Edge Detection
# edges_canny = cv2.Canny(gray, 100, 200)

# # Laplacian Edge Detection
# laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

# # Edge Enhancement
# edges_enhanced = cv2.addWeighted(gray, 0.5, cv2.convertScaleAbs(sobel_combined), 0.5, 0)

# # Plot Results
# plt.figure(figsize=(15, 10))

# plt.subplot(2, 3, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(2, 3, 2)
# plt.imshow(gray, cmap='gray')
# plt.title("Grayscale Image")
# plt.axis("off")

# plt.subplot(2, 3, 3)
# plt.imshow(sobel_combined, cmap='gray')
# plt.title("Sobel Edge Detection")
# plt.axis("off")

# plt.subplot(2, 3, 4)
# plt.imshow(edges_canny, cmap='gray')
# plt.title("Canny Edge Detection")
# plt.axis("off")

# plt.subplot(2, 3, 5)
# plt.imshow(laplacian, cmap='gray')
# plt.title("Laplacian Edge Detection")
# plt.axis("off")

# plt.subplot(2, 3, 6)
# plt.imshow(edges_enhanced, cmap='gray')
# plt.title("Enhanced Edges")
# plt.axis("off")

# plt.tight_layout()
# plt.show()