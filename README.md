# Dependencies:
  * OpenCV
  * Matplotlib
  * Numpy
  * Imutils

# Usage:
Using an image from test_imgs folder:
```
python segmentation.py -i <image>
```

# How it Works:
  * Select ROI (Region of Interest)
  * Adjust Gamma Levels
  * Increase Contrast
  * Extract Foreground
  * Get Pixels of Extracted Foreground
  * Color the Pixels  

# Select ROI (Region of Interest):
  * Once the image loads up, your rectangular ROI can be selected by clicking the top left corner of your object of interest and dragging down to the bottom right corner of your object. Currently, a rectangle is drawn as you click and drag to help visualize the region. **NOTE: The rectangle freezes for the first second, but will draw smoothly afterwards.**
  ![Alt text](/results/roi.png?raw=true "Selected ROI")

# Adjust Gamma Levels:
  * Adjusting gamma levels of an image should help with bright/dark images. The gamma value needed is found dynamically by analyzing the image and determining how bright/dark it is. **NOTE: More experimentation is needed to determine effectiveness.**
  ![Alt text](/results/gamma.png?raw=true "Adjust gamma")

# Increase Contrast:
  * Adjusting the contrast helps distinguish foreground objects from background objects. **NOTE: More experimentation is needed to determine effectiveness.**
  ![Alt text](/results/contrast.png?raw=true "Adjust contrast")
  
# Extract Foreground
  * Once the image has been pre-processed, the foreground object of choice can be extracted using OpenCV's built in grabcut function. The function yields a mask of the extracted object that can be used in the later steps.
    ![Alt text](/results/mask.png?raw=true "Mask")
  
# Get Pixels of Extracted Foreground
  * After dilating and eroding the mask to smooth it out, the mask's contour is found as well as the pixels within the mask.
    ![Alt text](/results/smooth_mask.png?raw=true "Smooth mask")
  
# Color the Pixels:
  * A transparent overlay is then added to the pixels within the contour boundary to complete the segmentation.
  ![Alt text](/results/result_4.png?raw=true "Result")
  
