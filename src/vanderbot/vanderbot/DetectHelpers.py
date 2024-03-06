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
        the largest contour and its center. It returns the corner locations. 
        If no green rect in view, returns None.

    get_largest_purple_circ
        Used with arm camera only. Finds the largest purple circle in view.
        It returns the center. If no purple circle in view, returns None.

    get_track
        Takes rectangular contour of track. Returns world coordinates of its
        center, as well as its true world angle.

    get_rect_pose_msg
        Takes rect center and world angle.
        Prepares and returns a pose message to publish a rectangle.
'''
import cv2
import numpy as np

from geometry_msgs.msg  import Point, Pose2D, Pose, Quaternion, Polygon, Point32

LOWER_Y_THRESHOLD = 0

'''
Creates initial binary for a color, erodes, dilates

Arguments:
hsv         - the OpenCV image from the camera in HSV format
hsv_limits  - the HSV limits of the desired color
iter        - the number of iterations of erode/dilate to do

Returns:
binary      - the binary produced by this detection
'''
def init_processing(hsv, hsv_limits, iter):
    # Threshold in Hmin/max, Smin/max, Vmin/max
    binary = cv2.inRange(hsv, hsv_limits[:,0], hsv_limits[:,1])

    if (iter > 0):
        binary = cv2.erode( binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=3*iter)
        binary = cv2.erode( binary, None, iterations=iter)

    return binary

'''
Given contours, returns a list of min-area rectangles found from those contours
It can also filter out rectangles that are too square.
Arguments:
binary - the binary used to do detection

Returns:
rectangles  - min-area rectangular contours that aren't too square
'''

def get_rects(binary):
    # Find contours in the mask
    (contours, _) = cv2.findContours(binary, 
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours on the original image for debugging.
    #cv2.drawContours(frame, contours, -1, self.blue, 2)
   
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
binary - the binary used to do detection

Returns:
circles  - min-enclosing circle
'''

def get_circs(binary):
    # Find contours in the mask
    (contours, _) = cv2.findContours(binary, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours on the original image for debugging.
    #cv2.drawContours(frame, contours, -1, self.blue, 2)
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
def get_largest_purple_circ(purple_circles, frame, color, thresh):
   if len(purple_circles) > 0:
      largest_circle = max(purple_circles, key = cv2.contourArea)
      area = cv2.contourArea(largest_circle)
      if area < thresh:
         return None, None
      ((u_purple, v_purple), radius) = cv2.minEnclosingCircle(largest_circle)
      purple_circCenter = [u_purple, v_purple]
      
      cv2.circle(frame, (int(u_purple), int(v_purple)), int(radius), color,  1)
      return purple_circCenter, area
   else:
      return None, None
            

'''
Used with ceiling camera only. Finds center and world angle of a track.
Arguments:
    rectangle - a rotated rectangular contour
    frame     - openCV image to draw on
    color1    - the color to draw the center of the rectangle
    color2    - the color to draw the outline of the rectangle
    pixelToWorld - the function mapping pixel to world space
    center    - the center of four ArUco. (x0, y0)
    camK      - K matrix of camera
    camD      - distortion matrix of camera
    min_rect_area - minimum area of acceptable rectangle
    markerCorners - ArUco marker corners
    markerIds     - IDs of the ArUco markers


Returns:
    rectCenter - world coordinates of center of rectangle
    world_angle - true pose of rectangle
    frame       - openCV image
'''
def get_track(rectangle, frame, color1, color2, pixelToWorld, center, camK, camD, min_rect_area, markerCorners, markerIds):   
    (x0, y0) = center

    ((um, vm), (wm, hm), angle) = cv2.minAreaRect(rectangle)
    rect_area = wm * hm

    if rect_area < min_rect_area:
       return (None, None, frame) # don't want this rectangle

    box = np.int0(cv2.boxPoints(((um, vm), (wm, hm), angle)))
    cv2.circle(frame, (int(um), int(vm)), 1, color1, 2)

    if wm < hm:
        angle += 90
    rectCenter = pixelToWorld(frame, um, vm, x0, y0, markerCorners, markerIds, camK, camD, angle = angle)
    
    # catch if rectangle is on the wrong table
    if rectCenter is not None:
        y = rectCenter[1]
        if y <= LOWER_Y_THRESHOLD:
            world_angle = None
            rectCenter = None
            return (rectCenter, world_angle, frame) # don't want this rectangle
        
    # Draw the rotated rectangle on the original image ONLY if we want it
    cv2.drawContours(frame, [box], 0, color2, 2)

    world_coords = []

    for coord in box:
        transformed_pt = pixelToWorld(frame, coord[0], coord[1], x0, y0, markerCorners,
                                        markerIds, camK, camD, angle = angle, annotateImage=False)
        world_coords.append(transformed_pt)

    norm1 = 0
    norm2 = 0
    if world_coords[0] is not None:
        norm1 = np.linalg.norm(world_coords[0] - world_coords[1])
        norm2 = np.linalg.norm(world_coords[1] - world_coords[2])

        if norm1 <= norm2:
            delta_y = world_coords[1][1] - world_coords[2][1]
            delta_x = world_coords[1][0] - world_coords[2][0]
            world_angle = np.pi - np.arctan(delta_y / delta_x)
        else:
            delta_y = world_coords[0][1] - world_coords[1][1]
            delta_x = world_coords[0][0] - world_coords[1][0]
            world_angle = np.arctan(delta_y / delta_x)
    else:
        world_angle = None
        rectCenter = None # error in detection

    return (rectCenter, world_angle, frame)


'''
Prepare a pose message to publish a rectangle.
Arguments:
    rectCenter (the center of the rectangle in world space)
    world_angle (the world angle of the rectangle)

Returns:
    pose_msg (the pose message to publish)
'''
def get_rect_pose_msg(rectCenter, world_angle):
    (xc, yc) = rectCenter

    pose_msg = Pose()
    rect_pt = Point()
    rect_angle = Quaternion()
    rect_pt.x = float(xc)
    rect_pt.y = float(yc)
    rect_pt.z = 0.0
    rect_angle.x = 0.0
    rect_angle.y = 0.0
    rect_angle.z = float(np.sin(world_angle / 2))
    rect_angle.w = float(np.cos(world_angle / 2))
    pose_msg.position = rect_pt
    pose_msg.orientation = rect_angle

    return pose_msg