#!/usr/bin/env python3
#
#   balldetector.py
#
#   Detect the tennis balls with OpenCV.
#
#   Node:           /balldetector
#   Subscribers:    /usb_cam/image_raw          Source image
#   Publishers:     /balldetector/binary        Intermediate binary image
#                   /balldetector/image_raw     Debug (marked up) image
#
import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose2D, Pose, Quaternion


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

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Thresholds in Hmin/max, Smin/max, Vmin/max
        self.hsvlimits = np.array([[9, 15], [110, 255], [120, 255]])

        # Create a publisher for the processed (debugging) images.
        # Store up to three images, just in case.
        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)
        self.pubbin = self.create_publisher(Image, name+'/binary',    3)
        self.circ_point = self.create_publisher(Point, "/Circle", 3)
        self.rect_pose = self.create_publisher(Pose, "/Rectangle", 3)

        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

        # Report.
        self.get_logger().info("Ball detector running...")

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()

    def pixelToWorld(self, image, u, v, x0, y0, markerCorners, markerIds, annotateImage=True, angle = None, w = None, h = None):
            '''
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
                return None


            # Determine the center of the marker pixel coordinates.
            uvMarkers = np.zeros((4,2), dtype='float32')
            for i in range(4):
                uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

            # Calculate the matching World coordinates of the 4 Aruco markers.
            DX = 0.1016
            DY = 0.06985
            xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                                    [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])


            # Create the perspective transform.
            M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)

            # Map the object in question.
            uvObj = np.float32([u, v])
            xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)

            pt1 = [0, 0]
            if angle is not None:
                pt1[0] = u + min(h, w) / 2 * np.cos(angle)
                pt1[1] = v + max(h, w) / 2 * np.sin(angle)


            pt1Obj = np.float32(pt1)

            world_pt1Obj = cv2.perspectiveTransform(pt1Obj.reshape(1,1,2), M).reshape(2)

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
    
    def calculate_circularity(contour):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if perimeter == 0:
            return 0

        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        return circularity

    # Process the image (detect the ball).
    def process(self, msg):
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

        # Assume the center of marker sheet is at the world origin.
        x0 = 0.0
        y0 = 0.314

        rectCenter = None
        discCenter = None

        # Mark the center of the image.
        cv2.circle(frame, (uc, vc), 5, self.red, -1)


        # Help to determine the HSV range...
        if True:
            # Draw the center lines.  Note the row is the first dimension.
            frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
            frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)

            # Report the center HSV values.  Note the row comes first.
            # self.get_logger().info(
            #     "HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        
        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary = cv2.inRange(hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])

        # Erode and Dilate. Definitely adjust the iterations!
        iter = 4
        binary = cv2.erode( binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=3*iter)
        binary = cv2.erode( binary, None, iterations=iter)


        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw all contours on the original image for debugging.
        cv2.drawContours(frame, contours, -1, self.blue, 2)
        
        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...
        rectangles =[]
        circles = []
        if len(contours) > 0:
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)

                # comparing min rectangle and contour areas
                # not used rn
                rotated_rect = cv2.minAreaRect(cnt)
                #rotated_rect = cv2.boxPoints(cnt)
                rect_area = rotated_rect[1][0] * rotated_rect[1][1]
                rect_ratio = cnt_area /rect_area
                
                # comparing min circles and contour areas
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circ_area = np.pi * (radius ** 2)
                circ_ratio = cnt_area / circ_area

                # aspect ratio of contour, if large then its a rectangle
                # works better than area comparison for rects
                ((cx, cy), (width, height), angle) = rotated_rect

                if width > height:
                    aspect_ratio = width / height
                else:
                    aspect_ratio = height / width

                if aspect_ratio > 1.1:
                    rectangles.append(cnt)
                    # if rect_ratio > 0.95 or rect_ratio < 1.05:
                    #     rectangles.append(cnt)
                elif circ_ratio > 0.95 or circ_ratio < 1.05:
                    circles.append(cnt)
 
        # if rectCenter == None:
        #     self.get_logger().info(
        #     "rect is none")

        # if discCenter is None:
        #     self.get_logger().info(
        #     "disc is none")
                    
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
                frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

        if len(circles) > 0:
            largest_circle = max(circles, key=cv2.contourArea)
            ((ur, vr), radius) = cv2.minEnclosingCircle(largest_circle)
            ur     = int(ur)
            vr     = int(vr)
            radius = int(radius)
            #self.get_logger().info(str(ur))
            #self.get_logger().info(str(vr))

            # Draw the circle (yellow) and centroid (red) on the
            # original image.
            cv2.circle(frame, (ur, vr), int(radius), self.yellow,  2)
            cv2.circle(frame, (ur, vr), 5,           self.red,    -1)

            # Convert the center of the image into world coordinates.
            discCenter = self.pixelToWorld(frame, ur, vr, x0, y0, markerCorners, markerIds)

        
            # Report.
            # if discCenter is not None:
            #     self.get_logger().info(
            #         "Found Disk enclosed by radius %d about (%d,%d)" %
            #         (radius, discCenter[0], discCenter[1]))
            # else:
            #     self.get_logger().info("Problem")

        if len(rectangles) > 0:
            largest_rotated_rectangle = max(rectangles, key=cv2.contourArea)
            rotatedrectangle = cv2.minAreaRect(largest_rotated_rectangle)
            ((um, vm), (wm, hm), angle) = cv2.minAreaRect(largest_rotated_rectangle)
            
            # Draw the largest rotated rectangle on the original image
            box = np.int0(cv2.boxPoints(rotatedrectangle))
            #self.get_logger().info(str(box))
            cv2.drawContours(frame, [box], 0, self.red, 2)
            rectCenter = self.pixelToWorld(frame, um, vm, x0, y0, markerCorners, markerIds, angle = angle, w = wm, h = hm)
            world_coords = []
            for coord in box:
                transformed_pt = self.pixelToWorld(frame, coord[0], coord[1], x0, y0, markerCorners, markerIds, angle = angle, w = wm, h = hm)
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
        if rectCenter is None:
            # self.get_logger().info("Unable to execute rectangle mapping")
            pass
        else:
            (xc, yc) = rectCenter
            world_angle = 0
            if norm1 <= norm2:
                delta_y = world_coords[1][1] - world_coords[2][1]
                delta_x = world_coords[1][0] - world_coords[2][0]
                world_angle = np.arctan(delta_y / delta_x)
            else:
                delta_y = world_coords[0][1] - world_coords[1][1]
                delta_x = world_coords[0][0] - world_coords[1][0]
                world_angle = np.arctan(delta_y / delta_x)
            # pt1 = rectCenter[1]
            # delta_y = pt1[1] - yc
            # delta_x = pt1[0] - xc
            #world_angle = np.arctan2(delta_y, delta_x)
            self.get_logger().info("Camera pointed at rectangle of (%f,%f)" % (xc, yc))
            self.get_logger().info(f"{world_angle * 180 / np.pi}")
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
            self.rect_pose.publish(pose_msg)


        if discCenter is None:
            # self.get_logger().info("Unable to execute disc mapping")
            pass
        else:
            (xc, yc) = discCenter
            self.get_logger().info("Camera pointed at disc of (%f,%f)" % (xc, yc))
            pt_msg = Point()
            pt_msg.x = float(xc)
            pt_msg.y = float(yc)
            pt_msg.z = 0.0
            self.circ_point.publish(pt_msg)

        # Convert the frame back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the binary (black/white) image.
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('demodetector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
