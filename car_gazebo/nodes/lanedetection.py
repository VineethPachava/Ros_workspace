

#importing some useful packages

import numpy as np
import cv2









def reject_outliers(data, m=2, min_data_length=4):
    """
    The mean of a distribution will be biased by outliers.
    
    `data` Data set that represent lines as a points(hough transform) from where we want to exclude all line outliers
    
    `m` : how much we scale from standard deviation.
    
    `min_data_length` : how much entries the dataset should have before start excluding outliers
    
    The result is filtered dataset representing lines without otuliers

    """
    if len(data)>min_data_length:
        data = np.array(data)

        #filter intercept outliers
        intercepts = data[:,1]
        filter_data=data[abs(intercepts - np.mean(intercepts)) < m * np.std(intercepts)]

        #filter slope outliers
        slopes = filter_data[:,0]
        filter_data=filter_data[abs(slopes- np.mean(slopes)) < m * np.std(slopes)]
        return filter_data.tolist()
    else: 
        return data

def is_outlier(point, data, m=2, min_data_length=4):
    """
    The mean of a distribution will be biased by outliers.
     
    `point` outlier candidate point representing the line in hough transform, 
    
    `data` is the data set representing lines as a points(hough transform)
    
    `m` : how much we scale from standard deviation.
    
    `min_data_length` : how much entries the data should have before start excluding outliers
    
    The result is True if the candidate point is outlier, false otherwise

    """
    if len(data)<=min_data_length:
        return False

    data = np.array(data)
    
    slopes = data[:,0]
    intercepts = data[:,1]

    slope_diff = np.abs(point[0] - np.mean(slopes))
    intercept_diff = np.abs(point[1] - np.mean(intercepts))

    slope_std= m * np.std(slopes)
    intercept_std= m * np.std(intercepts)

    is_outlier = slope_diff >= slope_std or intercept_diff>=intercept_std

    return is_outlier


# # Mean Line
#     1. We keep the lines formed from the latest X frames into a buffer
#     2. Then we can draw the average line among all lines in the buffer

# In[4]:


def get_mean_line_from_buffer(buffer, frames, y_min, y_max):
    """
    We should keep the lines formed from the latest frames.
    Then we can draw the average line among all lines in the buffer
    
    `buffer` list containing the lines in form : (slope, intercept) from all previous frames
    
    `frames` how much frames we should consider for calculating the mean line
    
    The result is mean line in form : (slope, intercept)
    """
  
    #get the mean line from the frame buffer
    mean_line = np.mean(np.array(buffer[-frames:]), axis=0).tolist()
    mean_slope = mean_line[0]   
    mean_intercept = mean_line[1]
    
    #calculate the X coordinates of the line
    x1 = int((y_min-mean_intercept)/mean_slope)
    x2 = int((y_max-mean_intercept)/mean_slope)
    return x1, x2


# In[5]:


# buffer cache through frames
line_low_avg_cache = []
line_high_avg_cache = []

#calculate max y coordinate for drawing and constructing the line
factor_height = 0.62


# ## Helper Functions

# In[6]:


import math

def grayscale(img):

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global line_low_avg_cache
    global line_high_avg_cache
    
    #get min y and max y for drawing the line
    y_min = img_height = img.shape[0]
    y_max = int(img_height*factor_height)

    lines_left = []
    lines_right = []
    
    #iterate all the lines
    for line in lines:

        for x1,y1,x2,y2 in line:
            
            #calculate slope
            slope = (y2 - y1)/(x2 - x1)
            
            #exclude non-valid slopes
            if (abs(slope)<0.5 or abs(slope)>0.8):
                continue
                
            if(math.isnan(slope) or math.isinf(slope)):
                continue
             
            #calculate intercept
            intercept = y1 - x1*slope
            
            if(slope<0):
                #it's right line
                lines_left.append([slope,intercept])
                
            else :
                #it's left line
                lines_right.append([slope,intercept])

    
    #clean left lines from noise
    lines_low = reject_outliers(lines_right,m=1.7)
    
    #clean right lines from noise
    lines_high = reject_outliers(lines_left, m=1.7)
    
    
    #add left lines to the frame buffer only if they are not outliers inside
    if lines_high:
        for element in lines_high:
            if not is_outlier(element,line_high_avg_cache,m=2.6):
                line_high_avg_cache.append(element)
    
    #add right lines to the frame buffer only if they are not outliers inside
    if lines_low:
         for element in lines_low:
            if not is_outlier(element,line_low_avg_cache,m=2.6):
                line_low_avg_cache.append(element)

    if line_high_avg_cache:
        x1, x2 = get_mean_line_from_buffer(buffer=line_high_avg_cache, 
                                           frames=20,
                                           y_min = y_min,
                                           y_max = y_max)
        #line extrapolation
        cv2.line(img,(x1, y_min),(x2, y_max),color,thickness)
    
    if line_low_avg_cache:
        x1, x2 = get_mean_line_from_buffer(buffer=line_low_avg_cache, 
                                           frames=20,
                                           y_min = y_min,
                                           y_max = y_max)
        #line extrapolation
        cv2.line(img,(x1, y_min),(x2, y_max),color,thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.0):

    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)





import os
os.listdir("./")


#canny edge detection params
low_threshold = 50
high_threshold = 150
kernel_size = 5

#hough transform params
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 17   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 80  #minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

#set mask scale factor
mask_scale_factor = 0.60
mask_width_factor = 0.5


# In[9]:


def detect_segments(image):
    
    #get image shape
    imshape = image.shape
    img_width = image.shape[1]
    img_height = image.shape[0]
    
    #apply greyscale transform
    grayscale_transform_image = image
    
    #apply gausian transform
    gausian_transform_image = gaussian_blur(grayscale_transform_image, kernel_size)
    
    #perform canny transform
    canny_transform_image = canny(gausian_transform_image, low_threshold, high_threshold
                                 )
    #get mask x, y coordinates
    mask_y = int(img_height*mask_scale_factor)
    mask_x = int(img_width*mask_width_factor)
    
    #compose mask vertices
    vertices = np.array([[(0,img_height),(mask_x, mask_y), (mask_x, mask_y), (img_width,img_height)]])
    
    #perform region of intersect
    masked_edges = region_of_interest(canny_transform_image, vertices)

    # Hough transform on edge detected image
    lines_transform_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    cv2.imshow("lol",lines_transform_image)
    #merge tranformed image with the current image
    # result_image = weighted_img(image, lines_transform_image)
    
    return lines_transform_image





def clear_cache():
    line_low_avg_cache.clear()
    line_high_avg_cache.clear()




def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = detect_segments(image)
    return result




#reading in an image
image = cv2.imread('solidWhiteCurve.jpg',0)

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
# cv2.imshow('image',image) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

lane_image = process_image(image)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',lane_image)
cv2.waitKey(0)
cv2.destroyAllWindows()










