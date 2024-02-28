'''DetectHelpers.py

   These are helper functions for detection to keep the detection file clean.
   
   The Helper functions are:

    init_processing 
        Creates initial binary for a color, erodes, dilates, and finds contours

    get_rects
        Given contours, returns a list of min-area rectangles found from those contours.
        It can also filter out rectangles that are too square.

    get_circs
        Given contours, returns a list of min-enclosing circles found from those contours.
        Filters out contours that have an aspect ratio that is too high

    get_largest_green_rect
        Used with arm camera only. Given a list of green rectangles, it obtains 
        the largest contour and its center and corner locations. 

   

'''
import cv2
import numpy as np

'''
Creates initial binary for a color, erodes, dilates, and finds contours

Arguments:
hsv         - the OpenCV image from the camera in HSV format
hsv_limits  - the HSV limits of the desired color
iter        - the number of iterations of erode/dilate to do

Returns:
contours    - the detected contours
binary      - the binary produced by this detection
'''
def init_processing(hsv, hsv_limits, iter):
    # Threshold in Hmin/max, Smin/max, Vmin/max
    binary = cv2.inRange(hsv, hsv_limits[:,0], hsv_limits[:,1])

    binary = cv2.erode( binary, None, iterations=iter)
    binary = cv2.dilate(binary, None, iterations=3*iter)
    binary = cv2.erode( binary, None, iterations=iter)

    # Find contours in the mask
    (contours, _) = cv2.findContours(binary, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours on the original image for debugging.
    #cv2.drawContours(frame, contours, -1, self.blue, 2)

    return (contours, binary) 

'''
Given contours, returns a list of min-area rectangles found from those contours
It can also filter out rectangles that are too square.
Arguments:
contours    - the detected contours

Returns:
rectangles  - min-area rectangular contours that aren't too square
'''

def get_rects(contours):
   rectangles = []
   if len(contours) > 0:
      for cnt in contours:
            # comparing min rectangle and contour areas
            rotated_rect = cv2.minAreaRect(cnt)

            # aspect ratio of contour, if large then its a rectangle
            # works better than area comparison for rects
            (_, (width, height), _) = rotated_rect

            if width > height:
               aspect_ratio = width / height
            else:
               aspect_ratio = height / width

            if aspect_ratio > 1.1:
               rectangles.append(cnt)

   return rectangles


'''
Given contours, returns a list of min-enclosing circles found from those contours.
Ignores contours that aren't circular enough
Arguments:
contours    - the detected contours

Returns:
circles  - min-enclosing circle
'''

def get_circs(contours):
   circles = []
   if len(contours) > 0:
      for cnt in contours:
            rotated_rect = cv2.minAreaRect(cnt)

            # aspect ratio of contour, if large then its a rectangle
            # works better than area comparison for rects
            (_, (width, height), _) = rotated_rect

            if width > height:
               aspect_ratio = width / height
            else:
               aspect_ratio = height / width

            if aspect_ratio < 1.4: # to avoid finding a purple rectangle
               circles.append(cnt)
   return circles



'''
Used with arm camera only. Finds the largest green rectangle in view.
Arguments:
    green_rectangles - list of green rectangular contours
    frame            - OpenCV image
    color            - color to make highlighted contours
    
Return:
    green_corners   - list of points representing the corners of the rectangle
'''
def get_largest_green_rect(green_rectangles, frame, color):
   if len(green_rectangles) > 0:
      largest_rect = max(green_rectangles, key = cv2.contourArea)
      rotatedrectangle = cv2.minAreaRect(largest_rect)
      ((u_green, v_green), (wm, hm), angle_green) = cv2.minAreaRect(largest_rect)
                  
      box = np.int0(cv2.boxPoints(rotatedrectangle))
      cv2.drawContours(frame, [box], 0, color, 2)
      green_corners = box.tolist()
      return green_corners
   else:
      return None


'''
Used with arm camera only. Finds the largest purple circle in view.
Arguments:
    purple_circles  - list of purple circular contours
    frame           - OpenCV image
    color           - color to make highlighted contours
    
Return:
    purple_circCenter  - point representing center of purple circular contour
'''
def get_largest_purple_circ(purple_circles, frame, color):
   if len(purple_circles) > 0:
      largest_circle = max(purple_circles, key = cv2.contourArea)
      ((u_purple, v_purple), radius) = cv2.minEnclosingCircle(largest_circle)
      purple_circCenter = [u_purple, v_purple]
      
      cv2.circle(frame, (int(u_purple), int(v_purple)), int(radius), color,  1)
      return purple_circCenter
   else:
      return None
            

'''
Used with ceiling camera only. Finds center and world angle of a track.
Arguments:
    rectangle - a rotated rectangular contour

Returns:
    rectCenter - world coordinates of center of rectangle
    world_angle - true pose of rectangle
'''
def get_track(rectangle):
    pass
    # rotatedrectangle = cv2.minAreaRect(rectangle)
    # max_rect_area = rotatedrectangle[1][0] * rotatedrectangle[1][1]
            
    # ((um, vm), (wm, hm), angle) = cv2.minAreaRect(largest_rotated_rectangle)
    
    # # Draw the largest rotated rectangle on the original image
    # box = np.int0(cv2.boxPoints(rotatedrectangle))
    # #self.get_logger().info(str(box))
    # cv2.circle(frame, (int(um), int(vm)), 1, self.blue, 2)
    # cv2.drawContours(frame, [box], 0, self.red, 2)
    # if wm < hm:
    #     angle += 90
    # orange_rectCenter = self.pixelToWorld(frame, um, vm, x0, y0, markerCorners, markerIds, self.ceil_camK, self.ceil_camD, angle = angle)
    # world_coords = []
    # for coord in box:
    #     transformed_pt = self.pixelToWorld(frame, coord[0], coord[1], x0, y0, markerCorners,
    #                                     markerIds, self.ceil_camK, self.ceil_camD, angle = angle, annotateImage=False)
    #     world_coords.append(transformed_pt)

    # norm1 = 0
    # norm2 = 0
    # if world_coords[0] is not None:
    #     norm1 = np.linalg.norm(world_coords[0] - world_coords[1])
    #     norm2 = np.linalg.norm(world_coords[1] - world_coords[2])
