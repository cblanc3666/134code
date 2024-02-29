#!/usr/bin/env python3

import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion, Polygon, PoseArray

class GameState(Node):
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        self.orange_track_sub = self.create_subscription(
            PoseArray, '/LeftTracksOrange', self.recvtracks_orange, 10)
        
        self.pink_track_sub = self.create_subscription(
            PoseArray, '/RightTracksPink', self.recvtracks_pink, 10)
        
        self.blue_track_sub = self.create_subscription(
            PoseArray, '/StraightTracksBlue', self.recvtracks_blue, 10)
        
        self.orange_tracks = []
        self.pink_tracks = []
        self.blue_tracks = []

        self.orange_track_pub = self.create_publisher(PoseArray, '/OrangeTrack', 10)
        self.pink_track_pub = self.create_publisher(PoseArray, '/PinkTrack', 10)
        self.blue_track_pub = self.create_publisher(PoseArray, '/BlueTrack', 10)
        self.placed_tracks = []
        
    def recvtracks_orange(self, posemsg):
        self.orange_tracks = posemsg.poses
        posemsg = PoseArray()
        for i in range(2):
            posemsg.poses.append(self.orange_tracks[i])
        self.orange_track_pub.publish(posemsg)

    def recvtracks_pink(self, posemsg):
        self.pink_tracks = posemsg.poses
        posemsg = PoseArray()
        for i in range(3):
            posemsg.poses.append(self.pink_tracks[i])
        self.pink_track_pub.publish(posemsg)

    def recvtracks_blue(self, posemsg):
        self.blue_tracks = posemsg.poses
        posemsg = PoseArray()
        for i in range(2):
            posemsg.poses.append(self.blue_tracks[i])
        self.blue_track_pub.publish(posemsg)
        

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
