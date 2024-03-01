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
from geometry_msgs.msg  import Point, Pose2D, Pose, Quaternion, Polygon, Point32, PoseArray
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

    # Relative value of camera resolution in pixels relative to 640x480
    # Allows us to adjust pixel thresholds for contour size when we change resolution
    # A value of 0.5 indicates a resolution of 320x240, for example
    RESOLUTION = 0.5

    # Minimum contour pixel areas, adjusted for resolution
    MIN_TRACK_AREA = 700
    MIN_NUB_AREA = 800 * RESOLUTION * RESOLUTION

    # Thresholds in Hmin/max, Smin/max, Vmin/max 
    HSV_ORANGE = np.array([[4, 18], [80, 255], [30, 255]])
    HSV_PINK = np.array([[150, 180], [80, 255], [120, 255]])
    HSV_PURPLE = np.array([[160, 180], [60, 255], [120, 255]])
    HSV_GREEN = np.array([[20, 100], [0, 255], [0, 240]])
    HSV_BLUE = np.array([[87, 107], [170, 220], [175, 255]])

    # Take center of ArUco relative to world origin. X0, Y0
    CENTER = (0.0, 0.382)

    

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
        self.ceil_pubbin_blue = self.create_publisher(Image, name+'/ceil_binary_blue', n_images)

        self.arm_pubrgb = self.create_publisher(Image, name+'/arm_image_raw', n_images)
        self.arm_pubbin_purple = self.create_publisher(Image, name+'/arm_binary_purple', n_images)
        self.arm_pubbin_green = self.create_publisher(Image, name+'/arm_binary_green', n_images)

        # Publish data on tracks detected
        self.rect_pose_orange = self.create_publisher(PoseArray, "/LeftTracksOrange", n_images)
        self.rect_pose_pink = self.create_publisher(PoseArray, "/RightTracksPink", n_images)
        self.rect_pose_blue = self.create_publisher(PoseArray, "/StraightTracksBlue", n_images)
        self.purple_circ = self.create_publisher(Point, "/PurpleCirc", n_images)
        self.green_rect = self.create_publisher(Polygon, "/GreenRect", n_images) 

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

        (contours_orange, binary_orange) = dh.init_processing(hsv, self.HSV_ORANGE, iter=1)
        (contours_pink,   binary_pink)   = dh.init_processing(hsv, self.HSV_PINK,   iter=1)
        (contours_blue,   binary_blue)   = dh.init_processing(hsv, self.HSV_BLUE,   iter=1)
        
        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...

        orange_rectangles = dh.get_rects(contours_orange)
        pink_rectangles = dh.get_rects(contours_pink)
        blue_rectangles = dh.get_rects(contours_blue)
                    
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
                frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

        # TODO start looping through all rectangles instead of taking the largest
        # for orange_rec in orange_rectangles:

        orange_poses = PoseArray()
        if len(orange_rectangles) is not 0:
            for orange_rec in orange_rectangles:
                #orange_rec = max(orange_rectangles, key=cv2.contourArea)
                (rectCenter, world_angle, frame) = dh.get_track(orange_rec, 
                                                                frame, 
                                                                self.red, 
                                                                self.blue, 
                                                                self.pixelToWorld, 
                                                                self.CENTER, 
                                                                self.ceil_camK, 
                                                                self.ceil_camD,
                                                                self.MIN_TRACK_AREA,
                                                                markerCorners,
                                                                markerIds)
                if rectCenter is not None:
                    pose_msg = dh.get_rect_pose_msg(rectCenter, world_angle)
                    orange_poses.poses.append(pose_msg)
                    #self.rect_pose_orange.publish(pose_msg)

        if len(orange_poses.poses) is not 0:
            self.rect_pose_orange.publish(orange_poses)
        # for pink_rec in pink_rectangles:

        pink_poses = PoseArray()
        if len(pink_rectangles) is not 0:
            for pink_rec in pink_rectangles:
                #pink_rec = max(pink_rectangles, key=cv2.contourArea)
                (rectCenter, world_angle, frame) = dh.get_track(pink_rec, 
                                                                frame, 
                                                                self.red, 
                                                                self.blue, 
                                                                self.pixelToWorld, 
                                                                self.CENTER, 
                                                                self.ceil_camK, 
                                                                self.ceil_camD,
                                                                self.MIN_TRACK_AREA,
                                                                markerCorners,
                                                                markerIds)

                if rectCenter is not None:
                    pose_msg = dh.get_rect_pose_msg(rectCenter, world_angle)
                    pink_poses.poses.append(pose_msg)
                    #self.rect_pose_pink.publish(pose_msg)

        if len(pink_poses.poses) is not 0:
            self.rect_pose_pink.publish(pink_poses)

        blue_poses = PoseArray()
        if len(blue_rectangles) is not 0:
            for blue_rec in blue_rectangles:
                #blue_rec = max(blue_rectangles, key=cv2.contourArea)
                (rectCenter, world_angle, frame) = dh.get_track(blue_rec, 
                                                                frame, 
                                                                self.red, 
                                                                self.blue, 
                                                                self.pixelToWorld, 
                                                                self.CENTER, 
                                                                self.ceil_camK, 
                                                                self.ceil_camD,
                                                                self.MIN_TRACK_AREA,
                                                                markerCorners,
                                                                markerIds)

                if rectCenter is not None:
                    pose_msg = dh.get_rect_pose_msg(rectCenter, world_angle)
                    blue_poses.poses.append(pose_msg)

        if len(blue_poses.poses) is not 0:
            # self.get_logger().info("Blue track angle %f" % np.arcsin(blue_poses.poses[0].orientation.z))
            self.rect_pose_blue.publish(blue_poses)

        # Convert the frame back into a ROS image and republish.
        self.ceil_pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the binary (black/white) image.
        self.ceil_pubbin_orange.publish(self.bridge.cv2_to_imgmsg(binary_orange))
        self.ceil_pubbin_pink.publish(self.bridge.cv2_to_imgmsg(binary_pink))
        self.ceil_pubbin_blue.publish(self.bridge.cv2_to_imgmsg(binary_blue))

    def arm_process(self, msg):
        (frame, hsv) = self.start_process(msg)
        
        (contours_purple, binary_purple) = dh.init_processing(hsv, self.HSV_PURPLE, iter=1)
        (contours_green,  binary_green)  = dh.init_processing(hsv, self.HSV_GREEN,  iter=1)
        
        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...
        purple_circles = dh.get_circs(contours_purple)
        green_rectangles = dh.get_rects(contours_green)

        green_rectCorners = dh.get_largest_green_rect(green_rectangles, frame, self.red)
        purple_circCenter, purple_area = dh.get_largest_purple_circ(purple_circles, frame, self.yellow, self.MIN_NUB_AREA)
        
        # Report the mapping.
        if green_rectCorners is not None:
            green_polygon_msg = Polygon()
            for point in green_rectCorners:
                p = Point32()
                point = 2*np.float32(point) # we multiply by two because we halved the camera resolution to improve framerate
                point = cv2.undistortPoints(point, self.arm_camK, self.arm_camD)
                p.x = float(point[0][0][0])
                p.y = float(point[0][0][1])
                green_polygon_msg.points.append(p)

            self.green_rect.publish(green_polygon_msg)

        if purple_circCenter is not None:
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
