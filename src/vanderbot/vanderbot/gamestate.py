#!/usr/bin/env python3

import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion, Polygon, PoseArray, Point32
import vanderbot.DetectHelpers as dh
from vanderbot.hexagonalplanner import GridNode, HexagonalGrid, Planner, Track

RATE = 100
THRESH = 0.1 #If the x or y coord of a track is at least THRESH away from a track in self.placed_track, ignore track
T_POS = 0.25 #Filtering coefficient for filtering position
T_ANGLE = 0.1 #Filtering coefficient for filtering angle

class GameState(Node):
    STATES = {"Straight" : 0.0, "Right" : 1.0, "Left" : -1.0}
    START_LOC = (0.6, 0.2)
    DIST_THRESH = 1
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        self.orange_track_sub = self.create_subscription(
            PoseArray, '/LeftTracksOrange', self.recvtracks_orange, 10)
        
        self.pink_track_sub = self.create_subscription(
            PoseArray, '/RightTracksPink', self.recvtracks_pink, 10)
        
        self.blue_track_sub = self.create_subscription(
            PoseArray, '/StraightTracksBlue', self.recvtracks_blue, 10)
        
        self.placed_track_sub = self.create_subscription(Pose, '/PlacedTrack', self.placed_track, 10)

        self.green_rect = self.create_subscription(
            Polygon, '/GreenRect', self.recvgreenrect, 10)
        
        # self.purple_circ = self.create_subscription(
        #     Point, '/PurpleCirc', self.recvpurplecirc, 10)
        
        self.orange_tracks = [] #List of tracks of type Track
        self.pink_tracks = [] #List of tracks of type Track
        self.blue_tracks = [] #List of tracks of type Track
        self.green_rect = None #Last Polygon message of green rectangle sent from trackdetector
        self.purple_circ = None #Last point message of purple circle sent from trackdetector

        # self.orange_track_pub = self.create_publisher(PoseArray, '/OrangeTrack', 10)
        # self.pink_track_pub = self.create_publisher(PoseArray, '/PinkTrack', 10)
        # self.blue_track_pub = self.create_publisher(PoseArray, '/BlueTrack', 10)

        self.sent_track_pub = self.create_publisher(PoseArray, '/SentTrack', 10)
        self.green_rect_filtered = self.create_publisher(Polygon, "/GreenRectFiltered", 3) 
        self.purple_circ_filtered = self.create_publisher(Point, "/PurpleCircFiltered", 3)

        self.important_tracks = [] #List of tracks that is in the path
        self.placed_tracks = [] #List of tracks of type Track

        self.timer = self.create_timer(1 / RATE, self.cb_timer)
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        self.blue_pos_time = None  #Intialize times for all of the filters
        self.blue_angle_time = None
        self.pink_pos_time = None
        self.pink_angle_time = None
        self.orange_pos_time = None
        self.orange_angle_time = None
        self.green_rect_time = None
        self.purple_circ_time = None
        self.important_pos_time = None
        self.important_angle_time = None
        
    def remove_placed_tracks(self, track_list):
        """
        Removes all the tracks that have the same (x, y) coords as any pose message
        in self.placed_tracks
        Inputs: track_list (list of tracks of type Track)
        Returns: a new track list (list of tracks of type Track)
        """
        track_list_c = track_list.copy()
        for track in track_list:
            for p_track in self.placed_tracks:
                if abs(track.pose.position.x - p_track.pose.position.x) < THRESH:
                    if abs(track.pose.position.y - p_track.pose.position.y) < THRESH:
                        track_list_c.remove(track)
                        break
        
        return track_list_c   

    def filter_position(self, x, y, x_new, y_new, t):
        #Get current time
        t_now = self.get_clock().now().nanoseconds * 1e-9

        #Calculate weighting
        c = (t_now - t) / T_POS

        #Reset self.curpos_time for next iteration of filter
        t = t_now
        
        #Calculate new x and y positions
        x_filtered = x + min(1,c)*(x_new - x)
        y_filtered = y + min(1,c)*(y_new - y)

        return x_filtered, y_filtered 
    
    def wrap180(self, angle):
        return angle - np.pi * round(angle/(np.pi))
    
    def filter_angle(self, theta, theta_new, t):
        #Get current time
        t_now = self.get_clock().now().nanoseconds * 1e-9

        #Calculate weighting
        c = (t_now - t) / T_ANGLE

        #Reset self.curangle_time for next iteration of filter
        t = t_now
        
        #Calculate new theta
        return self.wrap180(theta + min(1,c) * self.wrap180((theta_new - theta)))

    def find_closest_track(self, track, track_list):
        """
        Finds the track in the track_list that is closest to
        to the given track
        Inputs: track: Track object
                track_list: list of track objects 
        Returns index of the closest track in track_list, distance
        between track and closest track, angle between track and closest track
        """
        if len(track_list) == 1:
            dx = track.pose.position.x - track_list[0].pose.position.x
            dy = track.pose.position.y - track_list[0].pose.position.y
            angle = 2 * np.arcsin(track_list[0].pose.orientation.z)
            angle -= 2 * np.arcsin(track.pose.orientation.z)
            #self.get_logger().info("here")
            return 0, np.sqrt(dx ** 2 + dy ** 2), angle
        
        distances = []
        angles = []

        for i in range(len(track_list)):
            dx = track.pose.position.x - track_list[i].pose.position.x
            dy = track.pose.position.y - track_list[i].pose.position.y
            distances.append(np.sqrt(dx ** 2 + dy ** 2))
            angle = 2 * np.arcsin(track.pose.orientation.z)
            angle -= 2 * np.arcsin(track_list[i].pose.orientation.z)
            angles.append(angle)

        closest_dist_idx = distances.index(min(distances))
        closest_angle_idx = angles.index(min(angles))


        return closest_dist_idx, distances[closest_dist_idx], angles[closest_dist_idx]
        #return closest_angle_idx, distances[closest_angle_idx], angles[closest_angle_idx]
    

    def track_filtering(self, tracks, new_tracks, pos_time, angle_time):
        #Removes the appropriate track from tracks when a track is removed
        if len(new_tracks) < len(tracks):
            distances = []
            for i in range(len(tracks)):
                _, dist, _ = self.find_closest_track(tracks[i], new_tracks)
                distances.append(dist)
            
            max_dist_index = distances.index(max(distances))
            tracks.pop(max_dist_index)

        #Adds the appropriate track from new_tracks to tracks when a track is added
        elif len(new_tracks) > len(tracks):
            distances = []
            for i in range(len(tracks)):
                _, dist, _ = self.find_closest_track(new_tracks[i], tracks)
                distances.append(dist)
            
            max_dist_index = distances.index(max(distances))
            tracks.append(new_tracks[max_dist_index])
        
        else:
            #Filters the positions and angles of the tracks in tracks
            for i in range(len(tracks)):
                idx, dist, _ = self.find_closest_track(tracks[i], new_tracks)
                #self.get_logger().info(f"{dist}")
                new_track = new_tracks[idx]
                x_new, y_new = self.filter_position(tracks[i].pose.position.x,
                                           tracks[i].pose.position.y,
                                           new_track.pose.position.x,
                                           new_track.pose.position.y, pos_time)
                theta_new = self.filter_angle(2 * np.arcsin(tracks[i].pose.orientation.z),
                                           2 * np.arcsin(new_track.pose.orientation.z), angle_time)
                tracks[i].pose.position.x = x_new
                tracks[i].pose.position.y = y_new
                tracks[i].pose.orientation.z = float(np.sin(theta_new / 2))
                tracks[i].pose.orientation.w = float(np.cos(theta_new / 2))
                
        #self.get_logger().info(f"{tracks[0]}") 
        #self.get_logger().info(f"{len(tracks)}")     

    def recvtracks_orange(self, posemsg):
        #Initalizes self.orange_tracks when program first starts
        if self.orange_angle_time is None:
            self.orange_angle_time = self.get_clock().now().nanoseconds * 1e-9
        if self.orange_pos_time is None:
            self.orange_pos_time = self.get_clock().now().nanoseconds * 1e-9
        if len(self.orange_tracks) == 0:
            self.orange_tracks = self.remove_placed_tracks([Track(pose, "Left") for pose in posemsg.poses]) 
            pass

        new_orange_tracks = self.remove_placed_tracks([Track(pose, "Left") for pose in posemsg.poses])
        if len(self.orange_tracks) > 0 and len(new_orange_tracks) > 0:
            self.track_filtering(self.orange_tracks, new_orange_tracks, self.orange_pos_time, self.orange_angle_time)
        test_pos = (round(self.orange_tracks[0].pose.position.x, 3), round(self.orange_tracks[0].pose.position.y, 3))
        test_angle = round(2 * np.arcsin(self.orange_tracks[0].pose.orientation.z) * 180 / np.pi, 3)
        #self.get_logger().info(f"{test_pos} @ {test_angle} deg.") 
        
    def recvtracks_pink(self, posemsg):
        #Initalizes self.pink_tracks when program first starts
        if self.pink_angle_time is None:
            self.pink_angle_time = self.get_clock().now().nanoseconds * 1e-9
        if self.pink_pos_time is None:
            self.pink_pos_time = self.get_clock().now().nanoseconds * 1e-9
        if len(self.pink_tracks) == 0:
            self.pink_tracks = self.remove_placed_tracks([Track(pose, "Right") for pose in posemsg.poses]) 
            pass

        new_pink_tracks = self.remove_placed_tracks([Track(pose, "Right") for pose in posemsg.poses])
        if len(self.pink_tracks) > 0 and len(new_pink_tracks) > 0:
            self.track_filtering(self.pink_tracks, new_pink_tracks, self.pink_pos_time, self.pink_angle_time)

    def recvtracks_blue(self, posemsg):
        #Initalizes self.blue_tracks when program first starts
        if self.blue_angle_time is None:
            self.blue_angle_time = self.get_clock().now().nanoseconds * 1e-9
        if self.blue_pos_time is None:
            self.blue_pos_time = self.get_clock().now().nanoseconds * 1e-9
        if len(self.blue_tracks) == 0:
            self.blue_tracks = self.remove_placed_tracks([Track(pose, "Straight") for pose in posemsg.poses]) 
            pass

        new_blue_tracks = self.remove_placed_tracks([Track(pose, "Straight") for pose in posemsg.poses])
        if len(self.blue_tracks) > 0 and len(new_blue_tracks) > 0:
            self.track_filtering(self.blue_tracks, new_blue_tracks, self.blue_pos_time, self.blue_angle_time)

    def recvgreenrect(self, msg):
        if self.green_rect_time is None:
            self.green_rect_time = self.get_clock().now().nanoseconds * 1e-9

        if self.green_rect is None:
            self.green_rect = msg
            self.green_rect_filtered.publish(msg)
        else:
            green_polygon_msg = Polygon()
            for i in range(len(msg.points)):
                p = Point32()
                p.x, p.y = self.filter_position(self.green_rect.points[i].x,
                                                self.green_rect.points[i].y,
                                                msg.points[i].x,
                                                msg.points[i].y, self.green_rect_time)
                green_polygon_msg.points.append(p)
            self.green_rect_filtered.publish(green_polygon_msg)
            self.green_rect = green_polygon_msg

    def recvpurplecirc(self, msg):
        if self.purple_circ_time is None:
            self.purple_circ_time = self.get_clock().now().nanoseconds * 1e-9
        if self.purple_circ is None:
            self.purple_circ = msg
            self.purple_circ_filtered.publish(msg)
        else:
            point_msg = Point()
            point_msg.x, point_msg.y = self.filter_position(self.purple_circ.x,
                                                self.purple_circ.y,
                                                msg.x, msg.y, self.purple_circ_time)
            self.purple_circ_filtered.publish(point_msg)
            self.purple_circ = point_msg
            
            
        #(self.green_centroid, self.green_orientation) = self.green_rect_position(positions)

    def placed_track(self, posemsg):
        self.placed_tracks.append(Track(posemsg, self.important_tracks[0].track_type))
        self.important_tracks.pop(0)

    def cb_timer(self):
        """
        Each track has a pose message with x, y, and world angle.
        posemsg.orientation.x stores the track type as shown in self.STATES
        posemsg.orientaiton.y stores whether the track is the first track placed or not
            - 1.0 : first track placed
            - 0.0 : not first track placed
        Publishes a pose array of 2 poses
            Pose 1: Current pose of the track the arm is going to pick up
            Pose 2: If the track is the starting track, pose is hard coded in self.START_LOC.
                    Else, pose is the center and angle of the last track placed
        """
        # self.get_logger().info("Blue Tracks %r" % len(self.blue_tracks))
        # self.get_logger().info("Orange Tracks %r" % len(self.orange_tracks))
        # self.get_logger().info("Pink Tracks %r" % len(self.pink_tracks))

        #Intialize start times
        if self.important_angle_time is None:
            self.important_angle_time = self.get_clock().now().nanoseconds * 1e-9
        if self.important_pos_time is None:
            self.important_pos_time = self.get_clock().now().nanoseconds * 1e-9

        #Return if track detector cant see all the tracks described in self.important tracks
        if len(self.blue_tracks) <= 1 or len(self.orange_tracks) == 0 or len(self.pink_tracks) == 0:
            return
        
        # self.get_logger().info("Reached here")

        #Intialize important tracks
        if len(self.important_tracks) == 0:
            self.important_tracks = [self.blue_tracks[0], self.orange_tracks[0], 
                                     self.blue_tracks[1], self.pink_tracks[0]]
        #Filter tracks in important tracks
        else:
            new_important_tracks = [self.blue_tracks[0], self.orange_tracks[0], 
                                     self.blue_tracks[1], self.pink_tracks[0]]
                                     
            if len(self.placed_tracks) > 0:
                for _ in range(len(self.placed_tracks)):
                    new_important_tracks.pop(0)
            self.track_filtering(self.important_tracks, new_important_tracks, self.important_pos_time, self.important_angle_time)

        final_pose = PoseArray()
        posemsg_cur = self.important_tracks[0].pose
        #self.get_logger().info(f"{self.important_tracks[0]}") 
        #self.get_logger().info(f"{len(self.important_tracks)}") 

        #Store track type in orientation.x of the Pose messaage
        posemsg_cur.orientation.x = self.STATES[self.important_tracks[0].track_type]
        

        if len(self.placed_tracks) == 0:
            posemsg_dest = dh.get_rect_pose_msg(self.START_LOC, 0.0)
            posemsg_cur.orientation.y = 1.0
            posemsg_dest.orientation.y = 1.0
        else:
            prev_position = self.placed_tracks[len(self.placed_tracks) - 1].pose.position
            prev_orientation = self.placed_tracks[len(self.placed_tracks) - 1].pose.orientation
            posemsg_dest = Pose()
            posemsg_dest.position = prev_position
            posemsg_dest.orientation = prev_orientation
            posemsg_dest.orientation.x = self.STATES[self.important_tracks[0].track_type]

        final_pose.poses.append(posemsg_cur)
        final_pose.poses.append(posemsg_dest)
        self.sent_track_pub.publish(final_pose)

        
        


        

def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = GameState('gamestate')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
