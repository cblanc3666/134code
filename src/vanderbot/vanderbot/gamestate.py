#!/usr/bin/env python3

import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion, Polygon, PoseArray
import vanderbot.DetectHelpers as dh

RATE = 100
THRESH = 0.1 #If the x or y coord of a track is at least THRESH away from a track in self.placed_track, ignore track
             

class Track:
    def __init__(self, pose, track_type):
        """
        Inputs: Pose (Pose message)
                Track Type (String) (Either Left, Right, or Straight)
        """
        self.pose = pose
        self.track_type = track_type

class GameState(Node):
    STATES = {"Straight" : 0.0, "Right" : 1.0, "Left" : -1.0}
    START_LOC = (0.6, 0.2)
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
        
        self.orange_tracks = [] #List of tracks of type Track
        self.pink_tracks = [] #List of tracks of type Track
        self.blue_tracks = [] #List of tracks of type Track

        # self.orange_track_pub = self.create_publisher(PoseArray, '/OrangeTrack', 10)
        # self.pink_track_pub = self.create_publisher(PoseArray, '/PinkTrack', 10)
        # self.blue_track_pub = self.create_publisher(PoseArray, '/BlueTrack', 10)

        self.sent_track_pub = self.create_publisher(PoseArray, '/SentTrack', 10)
        self.important_tracks = []
        self.placed_tracks = [] #List of tracks of type Track

        self.timer = self.create_timer(1 / RATE, self.cb_timer)
        
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

    def recvtracks_orange(self, posemsg):
        self.orange_tracks = self.remove_placed_tracks([Track(pose, "Left") for pose in posemsg.poses])   
        # posemsg = PoseArray()
        # for i in range(2):
        #     posemsg.poses.append(self.orange_tracks[i])
        # self.orange_track_pub.publish(posemsg)

    def recvtracks_pink(self, posemsg):
        self.pink_tracks = self.remove_placed_tracks([Track(pose, "Right") for pose in posemsg.poses]) 
        # posemsg = PoseArray()
        # for i in range(3):
        #     posemsg.poses.append(self.pink_tracks[i])
        # self.pink_track_pub.publish(posemsg)

    def recvtracks_blue(self, posemsg):
        self.blue_tracks = self.remove_placed_tracks([Track(pose, "Straight") for pose in posemsg.poses]) 
        # posemsg = PoseArray()
        # for i in range(2):
        #     posemsg.poses.append(self.blue_tracks[i])
        # self.blue_track_pub.publish(posemsg)

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

        #TODO receive "placed" message from actuate instead of gamestate
        if len(self.blue_tracks) == 0 or len(self.orange_tracks) == 0 or len(self.pink_tracks) == 0:
            return
        
        # self.get_logger().info("Reached here")

        if len(self.important_tracks) == 0:
            self.important_tracks = [self.blue_tracks[0], self.blue_tracks[1],
                                     self.blue_tracks[2], self.blue_tracks[3]]

        final_pose = PoseArray()
        posemsg_cur = self.important_tracks[0].pose

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
            posemsg_dest.orientation.x = self.STATES[self.important_tracks[0].track_type]
            posemsg_dest.position = prev_position
            posemsg_dest.orientation = prev_orientation

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
