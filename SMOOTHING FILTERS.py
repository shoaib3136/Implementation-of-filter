
###Developed By :Shaik Shoaib Nawaz 
###Register Number:212222240094
### Smoothing Filters
# In[1]:Using Averaging Filter


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("path/to/your/image.jpg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Create the averaging kernel
kernel = np.ones((3, 3)) / 9  # 3x3 kernel with all elements equal to 1/9

# Apply the averaging filter
image3 = cv2.filter2D(image2, -1, kernel)

# Create the figure and subplots
plt.figure(figsize=(10, 5))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the filtered image
plt.subplot(1, 2, 2)
plt.imshow(image3, cmap='gray')
plt.title("Averaged Image")
plt.axis("off")

# Show the plot
plt.show()


# In[2]:Using Weighted Averaging Filter


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("Desktop/dipt/Implementation-of-filter/nature.jpg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Create the weighted averaging kernel
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16  # Normalized weights for better visualization

# Apply the weighted averaging filter
image3 = cv2.filter2D(image2, -1, kernel1)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the filtered image
plt.subplot(1, 2, 2)
plt.imshow(image3, cmap='gray')
plt.title("Weighted Average Filter Image")
plt.axis("off")

# Show the plot
plt.show()


# In[3]:Using Gaussian Filter


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("Desktop/dipt/Implementation-of-filter/nature.jpg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with a kernel size of 5x5 and sigmaX (standard deviation) of 0 (calculated automatically)
gaussian_blur = cv2.GaussianBlur(image2, (5, 5), 0)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the Gaussian blurred image
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title("Gaussian Blur")
plt.axis("off")

# Show the plot
plt.show()


# In[4]:Using Median Filter


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("Desktop/dipt/Implementation-of-filter/nature.jpg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply median filter with a kernel size of 3x3
median = cv2.medianBlur(image2, 3)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the median filtered image
plt.subplot(1, 2, 2)
plt.imshow(median, cmap='gray')
plt.title("Median Filter")
plt.axis("off")

# Show the plot
plt.show()






