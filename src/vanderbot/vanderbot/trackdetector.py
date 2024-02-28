#!/usr/bin/env python3
#
#   trackdetector.py
#
#   Detect Brio train track pieces with OpenCV
#
#   Node:           /trackdetector
#   Subscribers:    /ceilcam/image_raw          Source image from overhead cam
#                   /armcam/image_raw           Source image from arm camera
#   Publishers:     /trackdetector/ceil_binary_orange  Intermediate ceiling binary image (orange)
#                   /trackdetector/ceil_binary_pink     Intermediate ceiling binary image (pink)
#                   /trackdetector/ceil_image_raw  Debug (marked up) ceiling image
#                   /trackdetector/arm_binary   Intermediate arm binary image
#                   /trackdetector/arm_image_raw  Debug (marked up) arm image
#
import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose2D, Pose, Quaternion, Polygon, Point32
from sensor_msgs.msg import CameraInfo # get rectification calibration (from checkerboard)
import vanderbot.DetectHelpers as dh
#
#  Detector Node Class
#
class DetectorNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)
    MIN_RECT_AREA = 800

    # Thresholds in Hmin/max, Smin/max, Vmin/max 
    HSV_ORANGE = np.array([[4, 18], [80, 255], [30, 255]])
    HSV_PINK = np.array([[150, 180], [80, 255], [120, 255]])
    HSV_PURPLE = np.array([[160, 180], [140, 255], [120, 255]])
    HSV_GREEN = np.array([[18, 150], [80, 255], [72, 255]])

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create publishers, saving 3 images
        self.create_publishers(name, 3)

        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()
        
        # Get camera info
        self.get_camera_info()    

        # # Report the ceiling camera calibration parameters.
        # self.report_cal_params("CEILING CAMERA", self.ceil_camD, self.ceil_camK, self.ceil_camw, self.ceil_camh)
        # self.report_cal_params("ARM CAMERA", self.arm_camD, self.arm_camK, self.arm_camw, self.arm_camh)

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.ceil_sub = self.create_subscription(
            Image, '/ceil_image_raw', self.ceil_process, 1)
        
        self.arm_sub = self.create_subscription(
            Image, '/arm_image_raw', self.arm_process, 1)

        # Report.
        self.get_logger().info("Track detector running...")

    # Create publishers for the processed images and binaries for detectors
    # Store up to n_images, just in case.
    def create_publishers(self, name, n_images):
        self.ceil_pubrgb = self.create_publisher(Image, name+'/ceil_image_raw', n_images)
        self.ceil_pubbin_orange = self.create_publisher(Image, name+'/ceil_binary_orange', n_images)
        self.ceil_pubbin_pink = self.create_publisher(Image, name+'/ceil_binary_pink', n_images)

        self.arm_pubrgb = self.create_publisher(Image, name+'/arm_image_raw', n_images)
        self.arm_pubbin_purple = self.create_publisher(Image, name+'/arm_binary_purple', n_images)
        self.arm_pubbin_green = self.create_publisher(Image, name+'/arm_binary_green', n_images)

        # Publish data on tracks detected
        self.rect_pose_orange = self.create_publisher(Pose, "/StraightTrackOrange", n_images)
        self.rect_pose_pink = self.create_publisher(Pose, "/StraightTrackPink", n_images)
        self.purple_circ = self.create_publisher(Point, "/PurpleCirc", n_images)
        self.green_rect = self.create_publisher(Polygon, "/GreenRect", n_images)
            # Difference in centers of purple and green rectangles in pixel space
        self.pg_diff = self.create_publisher(Point, "/PurpleGreenDiff", n_images) 

    # Get camera info by subscribing to camera info topic
    def get_camera_info(self):
        # Temporarily subscribe to get just one message for each camera.
        self.get_logger().info("Waiting for camera info...")
        sub = self.create_subscription(CameraInfo, '/ceilcam/camera_info', self.cb_ceil, 1)
        self.ceil_caminfoready = False
        while not self.ceil_caminfoready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        self.get_logger().info("Waiting for camera info...")
        sub = self.create_subscription(CameraInfo, '/armcam/camera_info', self.cb_arm, 1)
        self.arm_caminfoready = False
        while not self.arm_caminfoready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

    # Create a temporary handler to grab the rectification info.
    def cb_ceil(self, msg):
        self.ceil_camD = np.array(msg.d).reshape(5)
        self.ceil_camK = np.array(msg.k).reshape((3,3))
        self.ceil_camw = msg.width
        self.ceil_camh = msg.height
        self.ceil_caminfoready = True

    def cb_arm(self, msg):
        self.arm_camD = np.array(msg.d).reshape(5)
        self.arm_camK = np.array(msg.k).reshape((3,3))
        self.arm_camw = msg.width
        self.arm_camh = msg.height
        self.arm_caminfoready = True

    # Report the camera calibration parameters.
    def report_cal_params(self, cam_name, camD, camK, camw, camh):
        self.get_logger().info(cam_name + " CALIBRATION PARAMETERS")
        self.get_logger().info("Received Distortion: \n %s" % (camD))
        self.get_logger().info("Received Camera Matrix: \n %s" % (camK))
        self.get_logger().info("Image size is (%7.2f, %7.2f)" %
        (float(camw), float(camh)))
        self.get_logger().info("Image center at (%7.2f, %7.2f)" %
        (camK[0][2], camK[1][2]))
        self.get_logger().info("FOV %6.2fdeg horz, %6.2fdeg vert" %
        (np.rad2deg(2*np.arctan(camw/2/camK[0][0])),
        np.rad2deg(2*np.arctan(camh/2/camK[1][1]))))


    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()

    def pixelToWorld(self, image, u, v, x0, y0, markerCorners, markerIds, K, D, annotateImage=True, angle = None):
            '''
            Used with the CEILING camera only.
            Convert the (u,v) pixel position into (x,y) world coordinates
            Inputs:
            image: The image as seen by the camera
            u:     The horizontal (column) pixel coordinate
            v:     The vertical (row) pixel coordinate
            x0:    The x world coordinate in the center of the marker paper
            y0:    The y world coordinate in the center of the marker paper
            annotateImage: Annotate the image with the marker information

            Outputs:
            point: The (x,y) world coordinates matching (u,v), or None

            Return None for the point if not all the Aruco markers are detected
            '''

            # Detect the Aruco markers (using the 4X4 dictionary).
            if annotateImage:
                cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

            # Abort if not all markers are detected.
            if (markerIds is None or len(markerIds) != 4 or
                set(markerIds.flatten()) != set([1,2,3,4])):
                # self.get_logger().info("Cannot see Aruco")
                return None


            # Determine the center of the marker pixel coordinates.
            uvMarkers = np.zeros((4,2), dtype='float32')
            
            for i in range(4):
                uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

            uvMarkersUndistorted = cv2.undistortPoints(uvMarkers, K, D) 

            # Calculate the matching World coordinates of the 4 Aruco markers.
            DX = 0.708
            DY = 0.303
            xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                                    [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])

            # Create the perspective transform.
            M = cv2.getPerspectiveTransform(uvMarkersUndistorted, xyMarkers)

            # Map the object in question.
            uvObj = np.float32([u, v])

            # Undistort coords 
            uvObj = cv2.undistortPoints(uvObj, K, D) 
            xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)

            # Mark the detected coordinates.
            if annotateImage:
                #.circle(image, (u, v), 5, (0, 0, 0), -1)
                if angle is not None:
                    s = "(%7.4f, %7.4f, %7.4f)" % (xyObj[0], xyObj[1], angle)
                    cv2.putText(image, s, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    #self.get_logger().info(f"Pt: ({world_pt1Obj[0]}, {world_pt1Obj[1]}), Center:({xyObj[0]}, {xyObj[1]})")
                else:
                    s = "(%7.4f, %7.4f)" % (xyObj[0], xyObj[1])
                    cv2.putText(image, s, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)

            return xyObj

    '''
    Helper function to handle first part of image processing
    Arguments:
        msg  - message that triggered the processing step in the first place
    Returns: 
        frame   - the OpenCV image, 
        hsv     - HSV version of the image,
    '''
    def start_process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))
        
        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Mark the center of the image.
        cv2.circle(frame, (uc, vc), 5, self.red, -1)

        # Help to determine the HSV range...
        frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
        frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)

        # Report the center HSV values.  Note the row comes first.
        # self.get_logger().info(
        #     "HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        return (frame, hsv)

    # Process the ceiling camera image (detect the track).
    def ceil_process(self, msg):
        # Capture, convert image, and mark its center
        (frame, hsv) = self.start_process(msg)
        
        (orange_rectCenter, pink_rectCenter) = (None, None)

        # Assume the center of marker sheet is at the world origin.
        x0 = 0.0
        y0 = 0.382

        (contours_orange, binary_orange) = dh.init_processing(hsv, self.HSV_ORANGE, iter=1)
        (contours_pink,   binary_pink)   = dh.init_processing(hsv, self.HSV_PINK,   iter=1)
        
        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...

        orange_rectangles = dh.get_rects(contours_orange)
        pink_rectangles = dh.get_rects(contours_pink)
                    
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
                frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

        if len(orange_rectangles) > 0:
            largest_rotated_rectangle = max(orange_rectangles, key=cv2.contourArea)
            rotatedrectangle = cv2.minAreaRect(largest_rotated_rectangle)
            max_rect_area = rotatedrectangle[1][0] * rotatedrectangle[1][1]
                    
            ((um, vm), (wm, hm), angle) = cv2.minAreaRect(largest_rotated_rectangle)
            
            # Draw the largest rotated rectangle on the original image
            box = np.int0(cv2.boxPoints(rotatedrectangle))
            #self.get_logger().info(str(box))
            cv2.circle(frame, (int(um), int(vm)), 1, self.blue, 2)
            cv2.drawContours(frame, [box], 0, self.red, 2)
            if wm < hm:
                angle += 90
            orange_rectCenter = self.pixelToWorld(frame, um, vm, x0, y0, markerCorners, markerIds, self.ceil_camK, self.ceil_camD, angle = angle)
            world_coords = []
            for coord in box:
                transformed_pt = self.pixelToWorld(frame, coord[0], coord[1], x0, y0, markerCorners,
                                                markerIds, self.ceil_camK, self.ceil_camD, angle = angle, annotateImage=False)
                world_coords.append(transformed_pt)

            norm1 = 0
            norm2 = 0
            if world_coords[0] is not None:
                norm1 = np.linalg.norm(world_coords[0] - world_coords[1])
                norm2 = np.linalg.norm(world_coords[1] - world_coords[2])


        # Report the mapping.
        if orange_rectCenter is None or max_rect_area < self.MIN_RECT_AREA:
            #self.get_logger().info("Unable to execute rectangle mapping-Orange")
            pass
        else:
            (xc, yc) = orange_rectCenter
            world_angle = 0
            if norm1 <= norm2:
                delta_y = world_coords[1][1] - world_coords[2][1]
                delta_x = world_coords[1][0] - world_coords[2][0]
                world_angle = np.pi - np.arctan(delta_y / delta_x)
            else:
                delta_y = world_coords[0][1] - world_coords[1][1]
                delta_x = world_coords[0][0] - world_coords[1][0]
                world_angle = np.arctan(delta_y / delta_x)
            #self.get_logger().info(str(world_angle * 180 / np.pi))
            # pt1 = rectCenter[1]
            # delta_y = pt1[1] - yc
            # delta_x = pt1[0] - xc
            #world_angle = np.arctan2(delta_y, delta_x)
            #self.get_logger().info("Camera pointed at rectangle of (%f,%f)" % (xc, yc))
            #self.get_logger().info(f"{world_angle * 180 / np.pi}")
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
            self.rect_pose_orange.publish(pose_msg)

        if len(pink_rectangles) > 0:
            largest_rotated_rectangle = max(pink_rectangles, key=cv2.contourArea)
            rotatedrectangle = cv2.minAreaRect(largest_rotated_rectangle)
            max_rect_area = rotatedrectangle[1][0] * rotatedrectangle[1][1]
                
            #self.get_logger().info(str(max_rect_area))
            ((um, vm), (wm, hm), angle) = cv2.minAreaRect(largest_rotated_rectangle)
            
            # Draw the largest rotated rectangle on the original image
            box = np.int0(cv2.boxPoints(rotatedrectangle))
            #self.get_logger().info(str(box))
            cv2.circle(frame, (int(um), int(vm)), 1, self.blue, 2)
            cv2.drawContours(frame, [box], 0, self.yellow, 2)

            point = cv2.undistortPoints(np.float32((um, vm)), self.arm_camK, self.arm_camD)
            undist_x = float(point[0][0][0])
            undist_y = float(point[0][0][1])
            
            # self.get_logger().info(f"Pink Track Center {undist_x}, {undist_y}")




            if wm < hm:
                angle += 90
            pink_rectCenter = self.pixelToWorld(frame, um, vm, x0, y0, markerCorners, markerIds, self.ceil_camK, self.ceil_camD, angle = angle)
            world_coords = []
            for coord in box:
                transformed_pt = self.pixelToWorld(frame, coord[0], coord[1], x0, y0, markerCorners,
                                                markerIds, self.ceil_camK, self.ceil_camD, angle = angle, annotateImage=False)
                world_coords.append(transformed_pt)

            norm1 = 0
            norm2 = 0
            if world_coords[0] is not None:
                norm1 = np.linalg.norm(world_coords[0] - world_coords[1])
                norm2 = np.linalg.norm(world_coords[1] - world_coords[2])
            #self.get_logger().info(f"{norm1}, {norm2}")
            # self.get_logger().info(str(um))
            # self.get_logger().info(str(vm))
            # if len(markerIds) == 4:
            #     rectCenter = self.pixelToWorld(frame, um, vm, x0, y0, markerCorners, markerIds)
            #     s = "(%7.4f, %7.4f, %7.4f)" % (rectCenter[0], rectCenter[1], angle)
            #     cv2.putText(frame, s, (int(rectCenter[0] - 80), int(rectCenter[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, (255, 0, 0), 2, cv2.LINE_AA)
            # if rectCenter is not None:
            #     self.get_logger().info(
            #                 "Found Rectangle enclosed by width %d and height %d about (%d,%d)" %
            #                 (wm, hm, rectCenter[0], rectCenter[1]))
    
        # Report the mapping.
        if pink_rectCenter is None or max_rect_area < self.MIN_RECT_AREA:
            # self.get_logger().info("Unable to execute rectangle mapping-Pink")
            pass
        else:
            # self.get_logger().info("Saw Pink track")
            (xc, yc) = pink_rectCenter
            world_angle = 0
            if norm1 <= norm2:
                delta_y = world_coords[1][1] - world_coords[2][1]
                delta_x = world_coords[1][0] - world_coords[2][0]
                world_angle = np.pi - np.arctan(delta_y / delta_x)
            else:
                delta_y = world_coords[0][1] - world_coords[1][1]
                delta_x = world_coords[0][0] - world_coords[1][0]
                world_angle = np.arctan(delta_y / delta_x)
            #self.get_logger().info(str(world_angle * 180 / np.pi))
            # pt1 = rectCenter[1]
            # delta_y = pt1[1] - yc
            # delta_x = pt1[0] - xc
            #world_angle = np.arctan2(delta_y, delta_x)
            #self.get_logger().info("Camera pointed at rectangle of (%f,%f)" % (xc, yc))
            #self.get_logger().info(f"{world_angle * 180 / np.pi}")
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
            self.rect_pose_pink.publish(pose_msg)

        # Convert the frame back into a ROS image and republish.
        self.ceil_pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the binary (black/white) image.
        self.ceil_pubbin_orange.publish(self.bridge.cv2_to_imgmsg(binary_orange))
        self.ceil_pubbin_pink.publish(self.bridge.cv2_to_imgmsg(binary_pink))

    def arm_process(self, msg):
        (frame, hsv) = self.start_process(msg)
        
        (contours_purple, binary_purple) = dh.init_processing(hsv, self.HSV_PURPLE, iter=1)
        (contours_green,  binary_green)  = dh.init_processing(hsv, self.HSV_GREEN,  iter=1)
        
        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...
        purple_circles = dh.get_circs(contours_purple)
        green_rectangles = dh.get_rects(contours_green)

        green_rectCorners = dh.get_largest_green_rect(green_rectangles, frame, self.red)
        purple_circCenter = dh.get_largest_purple_circ(purple_circles, frame, self.yellow)

        # Report the mapping.
        if green_rectCorners is None:
            pass
        else:
            green_polygon_msg = Polygon()
            for point in green_rectCorners:
                p = Point32()
                point = np.float32(point)
                point = cv2.undistortPoints(point, self.arm_camK, self.arm_camD)
                p.x = float(point[0][0][0])
                p.y = float(point[0][0][1])
                green_polygon_msg.points.append(p)

            self.green_rect.publish(green_polygon_msg)

        if purple_circCenter is None:
            pass
        else:
            point_msg = Point()
            # Map the object in question.
            uv_purple = np.float32(purple_circCenter)
            # self.get_logger().info(f"PURPLE UV 1{uv_purple}")

            # Undistort coords and report xbar, ybar, 1
            uv_purple = cv2.undistortPoints(uv_purple, self.arm_camK, self.arm_camD)
            # self.get_logger().info(f"PURPLE UV 2{uv_purple}")
            # self.get_logger().info(f"PURPLE corners type{uv_purple[0][0][0]}")
            point_msg.x = float(uv_purple[0][0][0])
            point_msg.y = float(uv_purple[0][0][1])
            point_msg.z = 1.0
            # self.get_logger().info(f"PurpleCenter {u_purple}, {v_purple}")
            self.purple_circ.publish(point_msg)

        # Convert the frame back into a ROS image and republish.
        self.arm_pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the binary (black/white) image.
        self.arm_pubbin_purple.publish(self.bridge.cv2_to_imgmsg(binary_purple))
        self.arm_pubbin_green.publish(self.bridge.cv2_to_imgmsg(binary_green))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('trackdetector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
