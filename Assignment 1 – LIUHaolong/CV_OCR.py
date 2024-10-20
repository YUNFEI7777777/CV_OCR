import cv2
import pytesseract
import numpy as np
import re

# Image path
image_path = 'image.jpg'  # Ensure this file is in your working directory

# Read the image
img = cv2.imread(image_path)

# Add custom configuration options for Tesseract OCR
custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine and set page segmentation mode to 6 (assumes a single uniform block of text)

# Perform OCR on the original image
text = pytesseract.image_to_string(img, config=custom_config)
print("OCR Result:\n", text)  # Print the OCR result

# Define preprocessing functions for enhancing the image before OCR
def get_grayscale(image):
    """Convert the image to grayscale.
    Grayscale Image
    Effect: Converts the original image from color to grayscale, where each pixel is represented by a shade of gray, from black to white.
    Purpose: Color images can contain irrelevant details like colors and shadows.
    Grayscale conversion simplifies the image by retaining only the brightness information,
    removing color details. This makes subsequent processing steps like thresholding or edge detection more effective.
    OCR typically works better on grayscale images since color is not required for text recognition."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    """Apply Otsu's thresholding to binarize the image.
    Effect: Uses Otsu’s thresholding method to convert the grayscale image into a binary (black and white) image, making text areas white and the background black.
    Purpose: Thresholding enhances the contrast between text and the background,
    making the text stand out more clearly. This is crucial for OCR because it can better differentiate between characters and background noise.
    Otsu’s method is adaptive, meaning it automatically chooses the optimal threshold value based on the image’s brightness distribution,
    which is useful for a variety of text recognition scenarios."""
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def opening(image):
    """Perform morphological opening (erosion followed by dilation) to remove small objects.
    Effect: A morphological operation that first erodes and then dilates the image, removing small noise areas while preserving larger text areas.
    Purpose: Morphological opening is useful for removing small noise points in the image while retaining the main text regions.
    The erosion step reduces the size of the foreground (typically the text), while dilation restores the main structure of the text.
    This step helps improve the clarity of the text areas by eliminating tiny noise spots that could interfere with OCR recognition."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    """Apply Canny edge detection to highlight edges in the image.
    Effect: Uses the Canny edge detection algorithm to identify and highlight the edges in the image, with edges shown as white lines.
    Purpose: Edge detection enhances the boundaries and outlines in the image. For text recognition,
    edge detection can help in cases where the contrast between the text and the background is low.
    While not a typical preprocessing step for OCR, Canny edge detection can improve recognition in scenarios where the text has clear outlines,
    making it more visible to the OCR engine."""
    return cv2.Canny(image, 100, 200)

# Preprocess the image and perform OCR
""" Summary:
Grayscale: Simplifies the image by removing unnecessary color details, focusing only on brightness.
Thresholding: Increases contrast between text and background, making the text clearer.
Morphological Opening: Removes noise and improves text region integrity.
Canny Edge Detection: Highlights edges and contours, useful for high-contrast text detection.
"""
gray = get_grayscale(img)  # Convert to grayscale
thresh = thresholding(gray)  # Apply thresholding
opening_img = opening(gray)  # Perform opening
canny_img = canny(gray)  # Apply Canny edge detection


# Display the results of preprocessing
cv2.imshow('Grayscale', gray)  # Show the grayscale image
cv2.imshow('Thresholded', thresh)  # Show the thresholded image
cv2.imshow('Opened', opening_img)  # Show the opened image
cv2.imshow('Canny', canny_img)  # Show the Canny edge-detected image
cv2.waitKey(0)  # Wait for a key press to close the windows
cv2.destroyAllWindows()  # Close all OpenCV windows

# Perform OCR on the preprocessed images and store the results
ocr_results = [
    pytesseract.image_to_string(gray, config=custom_config),  # OCR on grayscale image
    pytesseract.image_to_string(thresh, config=custom_config),  # OCR on thresholded image
    pytesseract.image_to_string(opening_img, config=custom_config),  # OCR on opened image
    pytesseract.image_to_string(canny_img, config=custom_config)  # OCR on Canny image
]

# Print each OCR result
for idx, result in enumerate(ocr_results, 1):
    print(f"OCR Result {idx}:\n", result)

# Get text boxes and draw rectangles around recognized text
d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)  # Get detailed OCR output
n_boxes = len(d['text'])  # Number of text boxes detected
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:  # Filter based on confidence level
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])  # Get box coordinates
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around detected text

# Display the image with rectangles around text
cv2.imshow('Image with Boxes', img)  # Show the image with text boxes
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows

# Function to read text from the image and save it to a file
def read_text_from_image(image):
    """Reads text from an image file and outputs found text to a text file."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)  # Otsu's thresholding
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))  # Define a rectangular kernel for dilation
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)  # Dilation to enhance contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours

    image_copy = image.copy()  # Make a copy of the image for processing
    with open("results.txt", "a") as file:  # Open the results file in append mode
        for contour in contours:  # Loop through each contour found
            x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box for the contour
            cropped = image_copy[y:y + h, x:x + w]  # Crop the image to the bounding box
            text = pytesseract.image_to_string(cropped)  # Perform OCR on the cropped image
            file.write(text + "\n")  # Write the recognized text to the file

# Read text from the image
read_text_from_image(img)

# Display OCR results from the text file
with open("results.txt", "r") as f:
    lines = f.readlines()  # Read all lines from the results file
    lines.reverse()  # Reverse the order of lines for display
    for line in lines:
        print(line)  # Print each line

