import numpy as np
from PIL import Image
import cv2

# Load image
img = Image.open("input.png").convert("RGB")
pixels = np.array(img, dtype=np.float64)

height, width, channels = pixels.shape
print(f"Loaded: {width}x{height}, {channels} channels")

# Display with OpenCV (BGR format)
display = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imshow("Input", display)
print("Press any key to close and save...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
output = Image.fromarray(pixels.astype(np.uint8))
output.save("output.png")
print("Saved to output.png")