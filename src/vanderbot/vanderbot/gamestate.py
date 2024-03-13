#!/usr/bin/env python3

import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion, Polygon, PoseArray, Point32
from std_msgs.msg               import Bool
import vanderbot.DetectHelpers as dh
from vanderbot.hexagonalplanner import GridNode, HexagonalGrid, Planner, Track

RATE = 100
THRESH = 0.1 #If the x or y coord of a track is at least THRESH away from a track in self.placed_track, ignore track
T_POS = 0.15 #Filtering coefficient for filtering position
T_ANGLE = 0.1 #Filtering coefficient for filtering angle
R = 0.1 #If a track is less than a distance R away from the destination, it will move that track away


class GameState(Node):
    STATES = {"Straight" : 0.0, "Right" : 1.0, "Left" : -1.0}
    START_LOC = (0.6, 0.4)
    GOAL_LOC = (-0.3, 0.3)
    # GOAL_LOC = (0.1, 0.1)
    DIST_THRESH = 1
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        ''' Detector Messages '''

        self.orange_track_sub = self.create_subscription(
            PoseArray, '/LeftTracksOrange', self.recvtracks_orange, 10)
        
        self.pink_track_sub = self.create_subscription(
            PoseArray, '/RightTracksPink', self.recvtracks_pink, 10)
        
        self.blue_track_sub = self.create_subscription(
            PoseArray, '/StraightTracksBlue', self.recvtracks_blue, 10)
        
        self.start_end_sub = self.create_subscription(
            Polygon, '/StartEndPoint', self.recvstart_end, 10)
        
        ''' Trajectory Code Messages '''

        #When a track has been placed
        self.placed_track_sub = self.create_subscription(PoseArray, '/PlacedTrack', self.placed_track, 10)

        #When a track has been grabbed
        self.grabbed_track_sub = self.create_subscription(Pose, '/GrabbedTrack', self.grabbed_track, 10)

        #When the gripper is hovering over the track it is about to grab
        self.grabbing_track_sub = self.create_subscription(Bool, '/GrabbingTrack', self.grabbing_track, 10)

        self.green_rect = self.create_subscription(
            Polygon, '/GreenRect', self.recvgreenrect, 10)
        
        # self.purple_circ = self.create_subscription(
        #     Point, '/PurpleCirc', self.recvpurplecirc, 10)
        
        self.orange_tracks = [] #List of tracks of type Track - only UNPLACED tracks
        self.pink_tracks = [] #List of tracks of type Track - only UNPLACED tracks
        self.blue_tracks = [] #List of tracks of type Track - only UNPLACED tracks
        self.green_rect = None #Last Polygon message of green rectangle sent from trackdetector
        self.purple_circ = None #Last point message of purple circle sent from trackdetector

        # self.orange_track_pub = self.create_publisher(PoseArray, '/OrangeTrack', 10)
        # self.pink_track_pub = self.create_publisher(PoseArray, '/PinkTrack', 10)
        # self.blue_track_pub = self.create_publisher(PoseArray, '/BlueTrack', 10)

        self.sent_track_pub = self.create_publisher(PoseArray, '/SentTrack', 10)
        self.green_rect_filtered = self.create_publisher(Polygon, "/GreenRectFiltered", 3) 
        self.purple_circ_filtered = self.create_publisher(Point, "/PurpleCircFiltered", 3)

        self.planner = Planner(self.START_LOC, self.GOAL_LOC)
        self.planned_tracks = self.planner.tracks #Tracks that should be used
        self.dest_track = self.planned_tracks[0] #Where the next track goes

        #Number of each track needed for the path
        self.num_blue_tracks = self.planner.num_blue_tracks
        self.num_pink_tracks = self.planner.num_pink_tracks
        self.num_orange_tracks = self.planner.num_orange_tracks

        self.important_tracks = [] #List of tracks that is in the path
        self.placed_tracks = [] #List of tracks that have been placed (of type Track)_

        self.timer = self.create_timer(1 / RATE, self.cb_timer)
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        #Intialize times for all of the filters
        #0 index: positional time (last time positional filter was used)
        #1 index: angluar time (last time angular filter was used)
        #2 index: removed time (last time a track is potentially removed)
        #3 index: added time (last time a track is potentially added)

        self.blue_times = [None, None, None, None]
        self.pink_times = [None, None, None, None]
        self.orange_times = [None, None, None, None]
        self.green_rect_time = None
        self.purple_circ_time = None

        self.grabbing = False #Check if the gripper has a track in its grasp
        self.blocking_tracks = []
        self.no_blocking_tracks = True

        ''' User interfacing global control '''
        self.placing = False # True when currently placing a path
        
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

        closest_dist_idx = np.argmin(distances)
        closest_angle_idx = np.argmin(angles)


        return closest_dist_idx, distances[closest_dist_idx], angles[closest_dist_idx]
        #return closest_angle_idx, distances[closest_angle_idx], angles[closest_angle_idx]
    

    def track_filtering(self, tracks, new_tracks, times, add = True):
        #Removes the appropriate track from tracks when a track is not visible for a certain time
        if len(new_tracks) < len(tracks):
            if self.grabbing == False:
                t_now = self.get_clock().now().nanoseconds * 1e-9
                if times[2] is None:
                    times[2] = t_now
                elif t_now - times[2] > 10.0:
                    self.get_logger().info(f"Removing {tracks[0].track_type} track")
                    distances = []
                    for i in range(len(tracks)):
                        _, dist, _ = self.find_closest_track(tracks[i], new_tracks)
                        distances.append(dist)
                    
                    max_dist_index = distances.index(max(distances))
                    tracks.pop(max_dist_index)

        #Adds the appropriate track from new_tracks to tracks when a track is added
        elif len(new_tracks) > len(tracks):
            if add:
                t_now = self.get_clock().now().nanoseconds * 1e-9
                if times[3] is None:
                    times[3] = t_now
                elif t_now - times[3] > 10.0:
                    self.get_logger().info(f"Adding {tracks[0].track_type} track")
                    distances = []
                    for i in range(len(tracks)):
                        _, dist, _ = self.find_closest_track(new_tracks[i], tracks)
                        distances.append(dist)
                    
                    max_dist_index = distances.index(max(distances))
                    tracks.append(new_tracks[max_dist_index])
        
        else:
            times[2] = None
            times[3] = None
            #Filters the positions and angles of the tracks in tracks
            for i in range(len(tracks)):
                idx, dist, _ = self.find_closest_track(tracks[i], new_tracks)
                #self.get_logger().info(f"{dist}")
                new_track = new_tracks[idx]
                x_new, y_new = self.filter_position(tracks[i].pose.position.x,
                                           tracks[i].pose.position.y,
                                           new_track.pose.position.x,
                                           new_track.pose.position.y, times[0])
                theta_new = self.filter_angle(2 * np.arcsin(tracks[i].pose.orientation.z),
                                           2 * np.arcsin(new_track.pose.orientation.z), times[1])
                tracks[i].pose.position.x = x_new
                tracks[i].pose.position.y = y_new
                tracks[i].pose.orientation.z = float(np.sin(theta_new / 2))
                tracks[i].pose.orientation.w = float(np.cos(theta_new / 2))
                
        #self.get_logger().info(f"{tracks[0]}") 
        #self.get_logger().info(f"{len(tracks)}")     

    def recvtracks_orange(self, posemsg):
        #Initalizes self.orange_tracks when program first starts
        if self.orange_times[0] is None:
            self.orange_times[0] = self.get_clock().now().nanoseconds * 1e-9
        if self.orange_times[1] is None:
            self.orange_times[1] = self.get_clock().now().nanoseconds * 1e-9
        if len(self.orange_tracks) == 0:
            self.orange_tracks = self.remove_placed_tracks([Track(pose, "Right") for pose in posemsg.poses]) 
            for track in self.orange_tracks:
                track.pose.orientation.x = 1.0
            pass

        new_orange_tracks = self.remove_placed_tracks([Track(pose, "Right") for pose in posemsg.poses])
        if len(self.orange_tracks) > 0 and len(new_orange_tracks) > 0:
            self.track_filtering(self.orange_tracks, new_orange_tracks, self.orange_times)
        test_pos = (round(self.orange_tracks[0].pose.position.x, 3), round(self.orange_tracks[0].pose.position.y, 3))
        test_angle = round(2 * np.arcsin(self.orange_tracks[0].pose.orientation.z) * 180 / np.pi, 3)
        #self.get_logger().info(f"{test_pos} @ {test_angle} deg.") 
        
    def recvtracks_pink(self, posemsg):
        #Initalizes self.pink_tracks when program first starts
        if self.pink_times[0] is None:
            self.pink_times[0] = self.get_clock().now().nanoseconds * 1e-9
        if self.pink_times[1] is None:
            self.pink_times[1] = self.get_clock().now().nanoseconds * 1e-9
        if len(self.pink_tracks) == 0:
            self.pink_tracks = self.remove_placed_tracks([Track(pose, "Left") for pose in posemsg.poses]) 
            for track in self.pink_tracks:
                track.pose.orientation.x = -1.0
            pass

        new_pink_tracks = self.remove_placed_tracks([Track(pose, "Left") for pose in posemsg.poses])
        if len(self.pink_tracks) > 0 and len(new_pink_tracks) > 0:
            self.track_filtering(self.pink_tracks, new_pink_tracks, self.pink_times)

    def recvtracks_blue(self, posemsg):
        #Initalizes self.blue_tracks when program first starts
        if self.blue_times[0] is None:
            self.blue_times[0] = self.get_clock().now().nanoseconds * 1e-9
        if self.blue_times[1] is None:
            self.blue_times[1] = self.get_clock().now().nanoseconds * 1e-9
        if len(self.blue_tracks) == 0:
            self.blue_tracks = self.remove_placed_tracks([Track(pose, "Straight") for pose in posemsg.poses])
            for track in self.blue_tracks:
                track.pose.orientation.x = 0.0
            pass

        new_blue_tracks = self.remove_placed_tracks([Track(pose, "Straight") for pose in posemsg.poses])
        if len(self.blue_tracks) > 0 and len(new_blue_tracks) > 0:
            self.track_filtering(self.blue_tracks, new_blue_tracks, self.blue_times)

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

    ''' Runs whenever we see the start and end point '''
    def recvstart_end(self, msg):
        [start, end] = msg.points
        start.x
        start.y
            


    def placed_track(self, posemsg):
        if self.no_blocking_tracks:
            track_type = self.dest_track.track_type
            x = posemsg.poses[0].position.x
            y = posemsg.poses[0].position.y
            if track_type == "Straight":
                self.get_logger().info(f"Placed blue track at {(round(x, 3), round(y, 3))}!")
            elif track_type == "Right":
                self.get_logger().info(f"Placed orange track at {(round(x, 3), round(y, 3))}!")
            else:
                self.get_logger().info(f"Placed pink track at {(round(x, 3), round(y, 3))}!")
            self.get_logger().info(f"Angle = {2*np.arcsin(posemsg.poses[0].orientation.z * 180 / np.pi)}")
            #self.get_logger().info(f"{track_type}")
            self.placed_tracks.append(Track(posemsg.poses[0], track_type))
            new_start = (posemsg.poses[1].position.x, posemsg.poses[1].position.y)
            self.get_logger().info(f"Purple Circle at {new_start}")

            for track in self.planned_tracks:
                track.pose.position.x += x - self.dest_track.pose.position.x
                track.pose.position.y += y - self.dest_track.pose.position.y

            if len(self.planned_tracks) != 0:
                self.dest_track = self.planned_tracks[0]
        else:
            if len(self.blocking_tracks) == 0:
                self.no_blocking_tracks = True
        self.grabbing = False
        self.important_tracks.pop(0)
        self.planned_tracks.pop(0)
        
    def grabbing_track(self, msg):
        self.grabbing = msg.data

    def grabbed_track(self, posemsg):
        if self.no_blocking_tracks == False:
            self.blocking_tracks.pop(0)
        else:
            track_type = self.important_tracks[0].track_type
            x = round(posemsg.position.x, 3)
            y = round(posemsg.position.y, 3)
            if track_type == "Straight":
                self.num_blue_tracks -= 1
                self.get_logger().info(f"Grabbed blue track at {(x, y)}!")
            elif track_type == "Right":
                self.num_orange_tracks -= 1
                self.get_logger().info(f"Grabbed orange track at {(x, y)}!")
            else:
                self.num_pink_tracks -= 1
                self.get_logger().info(f"Grabbed pink track at {(x, y)}!")
            
        
    def create_important_tracks(self):
        track_path = []
        cur_blue_idxs = []
        cur_orange_idxs = []
        cur_pink_idxs = []
        for track in self.planned_tracks:
            track_type = track.track_type
            if track_type == "Straight":
                closest_idx, _, _ = self.find_closest_track(track, self.blue_tracks)
                if closest_idx not in cur_blue_idxs:
                    track_path.append(self.blue_tracks[closest_idx])
                    cur_blue_idxs.append(closest_idx)
                else:
                    while True:
                        idx = np.random.randint(0, len(self.blue_tracks))
                        if idx not in cur_blue_idxs:
                            track_path.append(self.blue_tracks[idx])
                            cur_blue_idxs.append(idx)
                            break

            elif track_type == "Right":
                closest_idx, _, _ = self.find_closest_track(track, self.orange_tracks)
                if closest_idx not in cur_orange_idxs:
                    track_path.append(self.orange_tracks[closest_idx])
                    cur_orange_idxs.append(closest_idx)
                else:
                    while True:
                        idx = np.random.randint(0, len(self.orange_tracks))
                        if idx not in cur_orange_idxs:
                            track_path.append(self.orange_tracks[idx])
                            cur_orange_idxs.append(idx)
                            break
            else:
                closest_idx, _, _ = self.find_closest_track(track, self.pink_tracks)
                if closest_idx not in cur_pink_idxs:
                    track_path.append(self.pink_tracks[closest_idx])
                    cur_pink_idxs.append(closest_idx)
                else:
                    while True:
                        idx = np.random.randint(0, len(self.pink_tracks))
                        if idx not in cur_pink_idxs:
                            track_path.append(self.pink_tracks[idx])
                            cur_pink_idxs.append(idx)
                            break

        if len(track_path) != 0:
            self.get_logger().info(f"{[track.track_type for track in track_path]}")

        self.important_tracks = track_path

    def update_important_tracks(self):
        """
        Uses tracks from self.planned_tracks and adds them to
        self.important_tracks
        """

        track_path = []
        for track in self.important_tracks:
            if track.track_type == "Straight":
                idx, _, _ = self.find_closest_track(track, self.blue_tracks)
                track_path.append(self.blue_tracks[idx])
            elif track.track_type == "Right":
                idx, _, _ = self.find_closest_track(track, self.orange_tracks)
                track_path.append(self.orange_tracks[idx])
            else:
                idx, _, _ = self.find_closest_track(track, self.pink_tracks)
                track_path.append(self.pink_tracks[idx])

        self.important_tracks = track_path
    
    def find_distance(self, track1_pose, track2_pose):
        dx = track1_pose.position.x - track2_pose.position.x
        dy = track1_pose.position.y - track2_pose.position.y
        return np.sqrt(dx ** 2 + dy ** 2)

    def find_empty_space(self):
        all_tracks = self.blue_tracks + self.orange_tracks + self.pink_tracks
        while True:
            random_track_idx = np.random.randint(0, len(all_tracks))
            good_track = True
            for (i, track) in enumerate(all_tracks):
                if i != random_track_idx:
                    dist = self.find_distance(track.pose, all_tracks[random_track_idx].pose)
                    if dist < R:
                        good_track = False
            if good_track:
                break

        used_track = all_tracks[random_track_idx]
        center = (used_track.pose.position.x + 0.75 * R,
                  used_track.pose.position.y + 0.75 * R)
        angle = 2*np.arcsin(used_track.pose.orientation.z)
        pose_msg = dh.get_rect_pose_msg(center, angle)
        
        return pose_msg
        

    def find_blocking_tracks(self, curpose, destpose):
        if self.grabbing == False:
            all_tracks = [self.blue_tracks, self.orange_tracks, self.pink_tracks]
            for track_color in all_tracks:
                for (i, track) in enumerate(track_color):
                    dist = self.find_distance(track.pose, destpose)
                    if dist < R and self.find_distance(track.pose, curpose) > 2 * R:
                        self.blocking_tracks.append(track_color[i])
                        self.no_blocking_tracks = False

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
        self.get_logger().info(f"grabbing? {self.grabbing}")
        self.get_logger().info(f"placing?  {self.placing}")

        #Return if track detector cant see all the tracks described in self.important tracks
        if len(self.blue_tracks) < self.num_blue_tracks:
            self.get_logger().info("Not enough blue!")
            return
    
        if len(self.orange_tracks) < self.num_orange_tracks:
            self.get_logger().info("Not enough orange!")
            return
        
        if len(self.pink_tracks) < self.num_pink_tracks:
            self.get_logger().info("Not enough pink!")
            return

        #Intialize pose array sent to actuate.py
        final_pose = PoseArray()

        #Intialize self.important_tracks
        if len(self.important_tracks) == 0:
            if self.placing == True: # we have just fully placed a path
                self.get_logger().info("back here again")
                self.placing = False
                return
            else:
                self.get_logger().info("creating important tracks")
                self.create_important_tracks()
                self.placing = True # true when we are actively placing tracks
                test_Tracks = [str(track.track_type) for track in self.important_tracks]
                #posemsg_cur = self.dest_track.pose #Dummy pose to send to actuate, shouldn't affect anything
                #self.get_logger().info(f"{test_Tracks}")
        else:
            #Update important tracks after filtering
            self.update_important_tracks()
            

        #self.get_logger().info(f"{self.important_tracks[0].track_type} at {(self.important_tracks[0].pose.position.x, self.important_tracks[0].pose.position.y)} going to {(self.planned_tracks[0].pose.position.x, self.planned_tracks[0].pose.position.y)}") 
        # self.get_logger().info(f"{self.important_tracks}")
        # self.get_logger().info(f"{len(self.important_tracks)}") 
            
        posemsg_cur = self.important_tracks[0].pose
        posemsg_dest = self.dest_track.pose

        #self.find_blocking_tracks(posemsg_cur, posemsg_dest)
        #self.get_logger().info(f"{self.no_blocking_tracks}")

        if len(self.blocking_tracks) != 0:
            posemsg_cur = self.blocking_tracks[0].pose
            posemsg_cur.orientation.y = 1.0
            posemsg_dest = self.find_empty_space()
            posemsg_dest.orientation.y = 1.0
        else:
            if len(self.placed_tracks) == 0:
                posemsg_cur.orientation.y = 1.0
                posemsg_dest.orientation.y = 1.0

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
