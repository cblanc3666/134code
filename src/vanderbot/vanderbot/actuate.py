#!/usr/bin/env python3
#   
#   This node interfaces with the HEBIs and is responsible for all the
#   low level motion control. To determine its waypoint locations, it
#   receives messages from the GameState node. It also receives
#   messages from the Detector node in the form of arm camera on-screen
#   locations of objects.

#   Use this to lie down the arm
#     ros2 topic pub -1 /LieDown std_msgs/msg/String "{data: 'LieDown'}"

import numpy as np
import rclpy
import vanderbot.DetectHelpers as dh

from enum import Enum
from vanderbot.SegmentQueueDelayed      import SegmentQueue

from rclpy.node                         import Node
from sensor_msgs.msg                    import JointState
from geometry_msgs.msg                  import Point, Pose, Polygon, PoseArray

from vanderbot.KinematicChain           import KinematicChain
from vanderbot.TransformHelpers         import *
from vanderbot.Segments                 import Goto5, QuinticSpline
from std_msgs.msg                       import String, Float32, Bool



''' -------------------- Constant Definitions --------------------- '''

RATE = 100.0                            # transmit rate, in Hertz


''' Joint Positions '''
IDLE_POS = np.array([-1.3,1.7,2.2,0.5,0.])       # Holding position over table
DOWN_POS = np.array([-1.4,0.,0.55,1.5,0.])       # Position lying straight out over table

OPEN_GRIP   = -0.2
CLOSED_GRIP = -0.8

ZERO_QDOT = np.zeros(5)                 # Zero joint velocity
ZERO_VEL  = np.zeros(5)                 # Zero task velocity

ALPHA_J = np.array([0.,1.,1.,1.,0.])    # Coeffs of motor angles for gripper world pitch
BETA_J  = np.array([1.,0.,0.,0.,1.])    # Coeffs of motor angles for gripper world yaw


''' Positions and Offsets '''
TRACK_DEPTH  = 0.012                    # Thickness of the trackW
TRACK_TURN_RADIUS = 0.1299              # Radius of the centerline of the curved tracks

TRACK_HEIGHT = 0.128                    # Gripper height to grab the track at
CHECK_HEIGHT = 0.05                     # Gripper height offset when looking for the nub
FIRST_ALIGN_HEIGHT = 0.03               # Arm height when it does first alignment check
HOVER_HEIGHT = 0.07                     # Gripper height offset when placing down the track

SEENUB_OFFSET = 0.015                   # increment of distance to move back along the track in order to see the nub when picking up
MAX_NUB_ATTEMPTS = 5
TRACK_OFFSET = 0.030                    # Distance offset when given the location of a placed track
                                        # to connect to
NUB_OFFSET_RIGHT_START = 0.004               # Oversizing the nub distance when aligning
NUB_OFFSET_LEFT_START = -0.002
NUB_IDEAL_THETA = 0.612                 # Theoretical nub theta when track center is grabbed

WIGGLE_ANGLE = 0.07                     # Angle in radians with which to wiggle back and forth 

''' Gravity and Torque '''
GRAV_ELBOW = -6.8                       # Nm
GRAV_ELBOW_OFFSET = 0.01                # Phase offset (radians)

GRAV_SHOULDER = 12.7                    # Nm
GRAV_SHOULDER_OFFSET = -0.11            # Phase offset (radians)

TAU_GRIP = -9.0                         # Closed gripper torque

def wrap(angle, fullrange):
    ''' Takes in an angle and restricts it to +- fullrange/2 '''
    return angle - fullrange * round(angle/fullrange)


class ArmState(Enum):
    '''
    Represents the states the arm can be in, as well as their
    associated spline durations.
    '''

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
    
    def __init__(self, duration, segments):
        self.duration = duration
        self.segments = segments

    START = 6.0, []                     # Initial state
    RETURN = 4.0, []                    # Moving back to IDLE_POS; always uses a joint spline
    IDLE = None, []                     # Nothing commanded; arm is at IDLE_POS
    GOTO_PICKUP  = 5.0, []              # Moving to a commanded point, either from IDLE_POS or a
                                        # previously commanded point
    CHECK_GRIP = 1.0, []                # Arm remains above track to detect the purple nub
    CHECK_FORNUB = 2.0, []              # Check if the nub is still visible after lowering
    BACKUP_FORNUB = 0.5, []             # Backup arm to bring purple nub more into view
    SPIN_180 = 3.0, []                  # Twist motor spins to check other track side for nub
    LOWER = 4.0, []                     # Lowering arm to pickup track
    GRAB = 3.0, []                      # Grab track
    RAISE_PICKUP = 2.0, []              # Raise track after pickup
    GOTO_PLACE = 6.0, []                # Moves to a desired placement location
    CHECK_TRACK = 1.0, []               # Arm remain above to verify correct track orientation
    CHECK_ALIGN = 1.0, []               # Arm remains above track to find purple and green ends
    RETURN_TRACK = 5.0, []              # Arm returns bad track and returns to idle to retry
    ALIGN = 5.0, []                     # Aligns the purple nub with the green track end
    PLACE = 6.0, []                     # Moves arm downwards to connect track
    WIGGLE = 1.0, []                    # Wiggles track to link into main track
    RELEASE = 2.0, []                   # Opens gripper to release track
    CHECK_CONNECTION = 2.0, []
    RAISE_RELEASE = 2.0, []             # Raise clear of track before moving away
    CLOSE_FOR_PUSH = 2.0, []            # Close gripper to push track into place
    PUSH_INTO_PLACE = 2.0, []           # Push track into place!!
    RAISE_FROM_PUSH = 0.5, []           # Lift up from pushing track into place

    LIE_DOWN = 1.0, []                  # Tells arm to go to DOWN_POS regardless of current action


class VanderNode(Node):
    ''' Vanderbot actuate node class '''
    ''' Joints are ordered ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper'] '''
    
    start_time = 0                      # Time of node initialization

    position =  None                    # Real joint positions  (does not include gripper)
    qdot =      None                    # Real joint velocities (does not include gripper)
    effort =    None                    # Real joint efforts    (does not include gripper)

    grip_position = None                # Real gripper position
    grip_qdot = None                    # Real gripper velocity
    grip_effort = None                  # Real gripper effort

    qg = None                           # Combined desired joint/gripper joint angles
    qgdot = None                        # Combined desired joint/gripper joint velocities

    arm_state = ArmState.START # initialize state machine
    arm_killed = False # true when someone wants the arm to die
    track_type = None
    current_track = None
    placed_pose = None # poses to hold the position of the track just placed, and its nub
    
    check_attempts = 0 # counts number of times we've looked for a purple nub
    nub_backup_attempts = 0 # counts number of times we've backed up (by SEENUB_OFFSET) looking for purple nub
    wiggle_counter = 0 # counts the number of wiggles :)

    align_position = None

    skip_align = False
    
    # Indicates whether the arm camera can see the purple nub on the gripped track
    purple_visible = False
    (purple_u, purple_v) = (None, None)
    nub_r = None
    nub_theta = None

    green_centroid = None
    green_orientation = None

    grav_elbow = 0 # start elbow torque constant at 0
    grav_shoulder = 0 # start shoulder torque constant at 0

    pickup_pt = None # where to pick up the track, in xyz
    pickup_pt_joint = None # joint space of pickup point
    desired_pt = None # Destination for placement - either existing track or location for starting track, in xyz

    wiggle_dir = 1.0  # 1.0 for right-left, -1.0 for left-right

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create a temporary subscriber to grab the initial position.
        fbk = self.grabfbk()
        self.position0 = np.array(fbk)

        # Set the initial desired position to initial position so robot stays put
        self.qg = self.position0
        self.qgdot = np.append(ZERO_QDOT, 0.0)
        
        self.get_logger().info("Initial positions: %r" % self.position0)

        # create kinematic chains
        self.chain = KinematicChain('world', 'tip', self.jointnames())
        self.cam_chain = KinematicChain('world', 'cam', self.jointnames())

        # Start the clock
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # initialize segment queue with the desired fkin function
        self.SQ = SegmentQueue(self.chain.fkin)
        self.SQ.update(self.start_time, self.qg, self.qgdot)
        
        # Set up first spline (only shoulder is moving)
        self.SQ.enqueue_joint(np.array([self.position0[0], IDLE_POS[1], self.position0[2], self.position0[3], self.position0[4]]), 
                              ZERO_QDOT,
                              OPEN_GRIP,
                              ArmState.START.duration)
        
        # Spline gravity in gradually TODO handle this using segmentqueue
        ArmState.START.segments.append(Goto5(np.array([self.grav_shoulder, self.grav_elbow]),
                                             np.array([GRAV_SHOULDER, GRAV_ELBOW]),
                                             ArmState.START.duration,
                                             space='Task'))

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        
        self.pointsub = self.create_subscription(
            Point, '/point', self.recvpoint, 10)
        
        self.sent_track = self.create_subscription(
            PoseArray, '/SentTrack', self.recvtrack, 10)
        
        self.green_rect = self.create_subscription(
            Polygon, '/GreenRect', self.recvgreenrect, 10)
        
        self.purple_circ = self.create_subscription(
            Point, '/PurpleCirc', self.recvpurplecirc, 10)
        
        self.start_end_sub = self.create_subscription(
            Polygon, '/StartEndPoint', self.recvstart_end, 10)
        
        self.start_on_left = None #Spin tracks by 180 deg if the start is on the left of the goal (True)
        
        self.placed_track_pub = self.create_publisher(PoseArray, '/PlacedTrack', 10) #Lets gamestate know when track is placed
        self.grabbing_track_pub = self.create_publisher(Bool, '/GrabbingTrack', 10) #Lets gamestate know when gripper is hovering over track
        self.grabbed_track_pub = self.create_publisher(Pose, '/GrabbedTrack', 10) #Lets gamestate know when gripper grabbed the track
        self.connected_track_pub = self.create_publisher(Bool, '/ConnectedTrack', 10) 

        # publisher to tell the arm to lie down so we can kill it safely
        self.liedown = self.create_subscription(
            String, '/LieDown', self.recvStr, 10)
        
        # Report.
        self.get_logger().info("Running %s" % name)
        

    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['base', 'shoulder', 'elbow', 'wrist', 'twist']

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()

    def arm_pixel_to_position(self, x, y):
        (pcam, Rcam, _, _) = self.cam_chain.fkin(np.reshape(self.position, (-1, 1)))
        # self.get_logger().info(f"pcam x {pcam[0][0]} y {pcam[1][0]}")
        z = (pcam[2][0] - TRACK_DEPTH)
        lamb = -z / np.dot([0, 0, 1], Rcam @ np.array([[x], [y], [1]]))

        pobj = pcam + Rcam @ np.array([[x], [y], [1]]) * lamb
        return pobj

    # Grab a single feedback - do not call this repeatedly. For all 6 motors.
    def grabfbk(self):
        # Create a temporary handler to grab the position.
        def cb(fbkmsg):
            self.grabpos   = list(fbkmsg.position)
            self.grabready = True

        # Temporarily subscribe to get just one message.
        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        # Return the values.
        return self.grabpos

    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        position = np.array(list(fbkmsg.position)) # trim off gripper and save
        qdot = np.array(list(fbkmsg.velocity)) 
        effort = np.array(list(fbkmsg.effort)) 

        self.position = position[:5]
        self.qdot = qdot[:5]
        self.effort = effort[:5]

        self.grip_position = position[5]
        self.grip_qdot = qdot[5]
        self.grip_effort = effort[5]

    def recvStr(self, strmsg):
        if strmsg.data == "LieDown":
            self.arm_state = ArmState.LIE_DOWN
            self.arm_killed = True

    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z

        self.gotopoint(x, y, z)
    
    def recvtrack(self, posemsg):
        """
        Message is a PoseArray of 2 poses
            Pose 1: Current pose of the track the arm is going to pick up
            Pose 2: If the track is the starting track, pose is hard coded in self.START_LOC.
                    Else, pose is the center and angle of the last track placed
        {"Straight" : 0.0, "Right" : 1.0, "Left" : -1.0}
        """
        if self.start_on_left == None:
            return
        
        curr_pose = posemsg.poses[0]
        desired_pose = posemsg.poses[1]
        x = curr_pose.position.x
        y = curr_pose.position.y
        z = TRACK_HEIGHT + CHECK_HEIGHT # hover for checking tracks

        self.track_type = curr_pose.orientation.x
        beta = 2*np.arcsin(desired_pose.orientation.z) #- self.track_type * np.pi/6 # angle offset for track type

        if self.start_on_left:
            beta = wrap(beta + np.pi, 2 * np.pi)

        angle = 2 * np.arcsin(curr_pose.orientation.z)

        self.pickup_pt = np.array([x, y, z, 0.0, angle])
        self.desired_pt = np.array([desired_pose.position.x, desired_pose.position.y, TRACK_HEIGHT, 0.0, beta])
        
        # self.get_logger().info("Found track at (%r, %r) with angle %r" % (x, y, angle))
        self.skip_align = (desired_pose.orientation.y == 1.0)

        if not self.skip_align:
            self.desired_pt[0] -= np.cos(beta) * TRACK_OFFSET
            self.desired_pt[1] -= np.sin(beta) * TRACK_OFFSET
        
        self.gotopoint(x,y,z, beta=angle)
    
    def recvstart_end(self, msg):
        [start, end] = msg.points
        if start.x < end.x:
            self.start_on_left = True
        else:
            self.start_on_left = False


    def green_rect_position(self, points):
        side1 = np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])
        side2 = np.linalg.norm(points[2] - points[1]) + np.linalg.norm(points[3] - points[0])
        
        if side1 < side2:
            short_side = ((points[0] - points[1]) + (points[3] - points[2]))/2
        else:
            short_side = ((points[1] - points[2]) + (points[0] - points[3]))/2
        
        centroid = np.mean(points, 0)
        direction_vec = short_side / np.linalg.norm(short_side)

        return (centroid, direction_vec)
    
    def recvgreenrect(self, msg):
        # self.get_logger().info(f"Time_diff {self.get_clock().now().nanoseconds * 1e-9 - self.prev_time}")
        self.prev_time = self.get_clock().now().nanoseconds * 1e-9
        positions = []
        self.green_px_centr = [0.0, 0.0]
        for corner in msg.points:
            self.green_px_centr[0] += corner.x/4
            self.green_px_centr[1] += corner.y/4
            positions.append(self.arm_pixel_to_position(corner.x, corner.y))
            
        
        (self.green_centroid, self.green_orientation) = self.green_rect_position(positions)

        # #self.get_logger().info(f"Centroid of green rect x {centroid[0][0]} y {centroid[1][0]}")
        # direction_90 = np.array([[-direction[1][0]], [direction[0][0]], [direction[2][0]]])
        
        # align_point = centroid + direction * TRACK_DISPLACEMENT_FORWARDS
        # align_point += direction_90 * TRACK_DISPLACEMENT_SIDE
        
        # self.align_position = align_point
        # self.align_position[2][0] = TRACK_HEIGHT + HOVER_HEIGHT
        # self.align_position = np.array(self.align_position.flatten())

    def recvpurplecirc(self, msg):
        self.purple_visible = True
        self.purple_u = msg.x
        self.purple_v = msg.y
        # self.get_logger().info(f"Purple circle u {u} and v {v}")

    # Used to go to pickup
    def gotopoint(self, x, y, z, beta=0):
        # Don't goto if not idle, or if we want arm to die
        if self.arm_state != ArmState.IDLE or self.arm_killed == True: 
            # self.get_logger().info("Already commanded!")
            return

        self.arm_state = ArmState.GOTO_PICKUP
        
        
        # set alpha desired to zero - want to be facing down on table
        pf = np.array([x, y, z, 0, beta])

        # hold current grip position
        self.SQ.enqueue_polar(pf, ZERO_VEL, self.qg[5], ArmState.GOTO_PICKUP.duration)

        # Report.
        #self.get_logger().info("Going to point %r, %r, %r" % (x,y,z))
    
    '''
    sets the next state
    enqueues a task space spline to the xyz (or r theta z) coordinates at
    the given offset from current position
    if r is none, we have cartesian offset
    if r is not none, then x and y offsets can be none
    theta is the offset desired in beta
    spline type determines whether the commanded spline is polar or cartesian
    if duration is none, we use the duration of next_state
    '''
    def goto_offset(self, qfgrip, next_state, x_offset, y_offset, z_offset, r=None, theta=None, spline_type="cartesian", duration=None):
        self.arm_state = next_state
        
        pgoal, _, _, _ = self.chain.fkin(np.reshape(self.position, (-1, 1)))
        pgoal = pgoal.flatten()
        
        alpha = self.qg[1] - self.qg[2] + self.qg[3]
        beta = self.qg[0] - self.qg[4]

        if (theta != None):
            beta += theta

        if (r != None):
            pgoal[0] += r * np.cos(beta)
            pgoal[1] += r * np.sin(beta)
        else:
            pgoal[0] += x_offset
            pgoal[1] += y_offset
            
        pgoal[2] += z_offset

        pgoal = np.append(pgoal, [alpha, beta])

        if duration is None:
            duration = next_state.duration

        if spline_type == "polar":
            self.SQ.enqueue_polar(pgoal, ZERO_VEL, qfgrip, duration)
        elif spline_type == "cartesian":
            self.SQ.enqueue_task(pgoal, ZERO_VEL, qfgrip, duration)
        else:
            self.get_logger().info("Unknown spline type, not gonna go to offset")


    '''
    Does this by getting the world space coordinates of the purple nub.
    This assumes that the track is on the table when it is called.
    Run right after gripping track, when its still on the table
    Finds the r and theta of the nub
    '''
    def get_nub_location(self):
        ptip, _, _, _  = self.chain.fkin(np.reshape(self.position, (-1, 1)))
        ptip = ptip.flatten()

        (x_ptip, y_ptip, _) = ptip

        nub_u = np.copy(self.purple_u)
        nub_v = np.copy(self.purple_v)
        
        # self.get_logger().info(f"Nub (u, v) {nub_u}, {nub_v}")

        (x_pnub, y_pnub, _) = self.arm_pixel_to_position(nub_u, nub_v).flatten()

        self.nub_r = np.sqrt((x_ptip - x_pnub)**2 + (y_ptip - y_pnub)**2 )
        self.nub_theta = self.track_type * np.arccos(1 - self.nub_r**2 / (2 * TRACK_TURN_RADIUS**2))
        
        self.nub_pos = (x_pnub, y_pnub)

    '''
    Get offsets for the end effector to align grabbed track with green rectangle
    * assuming slot is centroid of contour
    '''
    def align_calc(self):
        centroid = np.copy(self.green_centroid)
        cnt_x = centroid[0][0]
        cnt_y = centroid[1][0]
        direction_vec = np.linalg.norm(np.copy(self.green_orientation))
        # self.get_logger().info(f"Direction {direction_vec} cnt_x {centroid[0][0]} cnt_y {centroid[1][0]}")
        # self.get_logger().info(f"Nub theta {self.nub_theta}")

        ptip, _, _, _  = self.chain.fkin(np.reshape(self.position, (-1, 1)))
        ptip = ptip.flatten()
        (x_ptip, y_ptip, _) = ptip

        # get offset of green contour from end effector
        # self.get_logger().info(f"x_ptip {x_ptip}")
        # self.get_logger().info(f"y_ptip {y_ptip}")
        green_dx = cnt_x - x_ptip
        green_dy = cnt_y - y_ptip

        beta = self.qg[0] - self.qg[4]
        gamma = self.nub_theta / 2
        self.get_logger().info(f"Current track {self.current_track}")
        

        # offsets for the end effector
        d_theta = self.nub_theta - self.current_track * NUB_IDEAL_THETA # need to align to the bottom of the green contour
        dx = green_dx
        dy = green_dy

        # In possession of a curved track
        if self.current_track != 0:     # Curved track
            beta - self.current_track * np.pi/6
        
        # self.get_logger().info(f"Beta {beta}")
        # self.get_logger().info(f"Gamma {gamma}")
        # self.get_logger().info(f"Nub r dx {self.nub_r * np.cos(beta + gamma)}")
        # self.get_logger().info(f"Nub r dy {self.nub_r * np.sin(beta + gamma)}")

    
        if self.start_on_left:
            dx = green_dx - (self.nub_r + NUB_OFFSET_LEFT_START) * np.cos(beta + d_theta - gamma)
            dy = green_dy - (self.nub_r + NUB_OFFSET_LEFT_START) * np.sin(beta + d_theta - gamma)
        else:
            dx = green_dx - (self.nub_r + NUB_OFFSET_RIGHT_START) * np.cos(beta + d_theta - gamma)
            dy = green_dy - (self.nub_r + NUB_OFFSET_RIGHT_START) * np.sin(beta + d_theta - gamma)

        return dx, dy, d_theta 

        



        
    
    # Send a command - called repeatedly by the timer.
    def sendcmd(self):        
        # self.get_logger().info("Desired point %r" % self.desired_pt)
        # self.get_logger().info(f"skip align {self.skip_align}")
        # self.get_logger().info("Current state %r" % self.arm_state) # watermelon
        # Time since start
        time = self.get_clock().now().nanoseconds * 1e-9

        if self.position0 is None or \
            self.position is None or \
            self.qg is None: # no start position or desired position yet
            return

        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']
        
        # gravity compensation
        # cosine of shoulder angle minus (because they are oriented opposite) elbow angle
        tau_elbow = self.grav_elbow * np.cos(self.position[1] - self.position[2] + GRAV_ELBOW_OFFSET)
        tau_shoulder = -tau_elbow + self.grav_shoulder * np.cos(self.position[1] + GRAV_SHOULDER_OFFSET)
        tau_grip = TAU_GRIP
        self.cmdmsg.effort       = list([0.0, tau_shoulder, tau_elbow, 0.0, 0.0, tau_grip])

        # self.get_logger().info(f"effort {tau_elbow}, {tau_shoulder}")

        # update splines!
        self.SQ.update(time, self.qg, self.qgdot)

        qg = None
        qgdot = None

        if self.arm_state == ArmState.LIE_DOWN:
            # stop all motion
            self.SQ.clear()

            # note - if this lie down command happens during the start while
            # gravity is being splined in, gravity will continue to spline in
            # gradually 
            self.arm_state = ArmState.RETURN

            # go down
            self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)
            self.SQ.enqueue_joint(DOWN_POS, ZERO_QDOT, OPEN_GRIP, 2*ArmState.RETURN.duration)
        elif self.arm_state == ArmState.START:
            # evaluate gravity constants too
            self.grav_shoulder = ArmState.START.segments[0].evaluate(time-self.start_time)[0][0] # first index takes "position" from spline
            self.grav_elbow = ArmState.START.segments[0].evaluate(time-self.start_time)[0][1] # TODO figure out a way to get rid of this

            if self.SQ.isEmpty(): # once done, moving to IDLE_POS
                self.arm_state = ArmState.RETURN
                
                # set gravity constants to their final values
                self.grav_elbow = GRAV_ELBOW
                self.grav_shoulder = GRAV_SHOULDER

                # Sets up next spline since base and elbow joints start at arbitrary positions
                self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)
            
        elif self.arm_state == ArmState.RETURN:
            if self.SQ.isEmpty():
                # need to set qg and qgdot to stay at what they are right now
                qg = self.qg
                qgdot = self.qgdot
                self.arm_state = ArmState.IDLE
        
        elif self.arm_state == ArmState.IDLE:
            # Hold current desired position
            qg = self.qg
            qgdot = self.qgdot

        elif self.arm_state == ArmState.GOTO_PICKUP:
            if self.SQ.isEmpty():                
                # get pickup point in case we ever need to come back here
                self.pickup_pt_joint = np.copy(self.qg[0:5])
                self.arm_state = ArmState.CHECK_GRIP
                self.purple_visible = False
                # stay put while checking grip
                self.SQ.enqueue_hold(ArmState.CHECK_GRIP.duration)
                self.check_attempts = 0
                    
        elif self.arm_state == ArmState.CHECK_GRIP:
            if self.SQ.isEmpty():
                #Tell gamestate that gripper is about to grab a track
                grabbing = Bool()
                grabbing.data = True
                

                if self.purple_visible: # we see purple, so lower
                    self.goto_offset(qfgrip=OPEN_GRIP, 
                                     next_state=ArmState.LOWER, 
                                     x_offset=0, 
                                     y_offset=0, 
                                     z_offset=-(CHECK_HEIGHT), 
                                     r=None, 
                                     theta=None,
                                     spline_type="cartesian")
                    # give up & go home

                    if self.check_attempts >= 2:
                        self.check_attempts = 0
                        self.arm_state = ArmState.RETURN
                        self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)
                        grabbing.data = False

                    self.check_attempts += 1 # TODO I think this is wrong but need to check
                else:
                    # self.get_logger().info(f"check attempts {self.check_attempts}")
                    self.check_attempts = self.check_attempts+1
                    if self.check_attempts == 1:
                        self.arm_state = ArmState.SPIN_180
                        self.SQ.enqueue_joint(np.append(self.qg[0:4], wrap(self.qg[4]+np.pi, 2*np.pi)), ZERO_QDOT, OPEN_GRIP, ArmState.SPIN_180.duration)
                    elif self.check_attempts == 2:
                        # move back
                        self.goto_offset(qfgrip=OPEN_GRIP, 
                                         next_state=ArmState.CHECK_GRIP,
                                         x_offset=0, 
                                         y_offset=0, 
                                         z_offset=0, 
                                         r=-3*SEENUB_OFFSET, 
                                         theta=None,
                                         spline_type="cartesian")
                        # then check to see if purple dot visible
                        self.purple_visible = False
                        self.SQ.enqueue_hold(ArmState.CHECK_GRIP.duration)
                    elif self.check_attempts == 3:
                        # spin around and check if purple dot visible
                        self.arm_state = ArmState.SPIN_180
                        self.SQ.enqueue_joint(np.append(self.qg[0:4], wrap(self.qg[4]+np.pi, 2*np.pi)), ZERO_QDOT, OPEN_GRIP, ArmState.SPIN_180.duration)
                    elif self.check_attempts == 4:
                        # last effort. Move back and check if purple dot visible
                        self.goto_offset(qfgrip=OPEN_GRIP, 
                                         next_state=ArmState.CHECK_GRIP,
                                         x_offset=0, 
                                         y_offset=0, 
                                         z_offset=0, 
                                         r=-6*SEENUB_OFFSET, 
                                         theta=None,
                                         spline_type="cartesian")
                        self.purple_visible = False
                        self.SQ.enqueue_hold(ArmState.CHECK_GRIP.duration)
                    else:
                        # give up & go home
                        self.check_attempts = 0
                        self.arm_state = ArmState.RETURN
                        self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)
                        grabbing.data = False

                self.grabbing_track_pub.publish(grabbing)

        elif self.arm_state == ArmState.SPIN_180:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.CHECK_GRIP
                self.purple_visible = False
                # stay put while checking grip
                self.SQ.enqueue_hold(ArmState.CHECK_GRIP.duration)

        
        elif self.arm_state == ArmState.LOWER:
            if self.SQ.isEmpty():
                self.purple_visible = False
                self.arm_state = ArmState.CHECK_FORNUB
                # stay put while checking for nub
                self.purple_visible = False
                self.SQ.enqueue_hold(ArmState.CHECK_FORNUB.duration)
                
        elif self.arm_state == ArmState.CHECK_FORNUB:
            if self.SQ.isEmpty():
                if not self.purple_visible: # back up a bit so we can see purple!
                    self.goto_offset(qfgrip=OPEN_GRIP, 
                                    next_state=ArmState.BACKUP_FORNUB,
                                    x_offset=0, 
                                    y_offset=0, 
                                    z_offset=0, 
                                    r=-SEENUB_OFFSET, 
                                    theta=None,
                                    spline_type="cartesian")
                    # initialize counter to make sure we don't check for nub forever
                    self.nub_backup_attempts = 0 
                else:
                    self.arm_state = ArmState.GRAB
                    self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, CLOSED_GRIP, ArmState.GRAB.duration)

        elif self.arm_state == ArmState.BACKUP_FORNUB:
            if self.SQ.isEmpty():
                if self.purple_visible: # if we see, grab
                    self.arm_state = ArmState.GRAB
                    self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, CLOSED_GRIP, ArmState.GRAB.duration)
                else: # if don't see
                    if self.nub_backup_attempts < MAX_NUB_ATTEMPTS: # if we still have attempts left
                        # self.get_logger().info(f"backup attempts {self.nub_backup_attempts}")
                        # keep backing up
                        self.goto_offset(qfgrip=OPEN_GRIP, 
                                            next_state=ArmState.BACKUP_FORNUB,
                                            x_offset=0, 
                                            y_offset=0, 
                                            z_offset=0, 
                                            r=-SEENUB_OFFSET, 
                                            theta=None,
                                            spline_type="cartesian")
                        self.nub_backup_attempts += 1
                    else:
                        # if this happens, we're probably oriented the wrong way
                        # and saw the purple from another track originally
                        self.nub_backup_attempts = 0

                        # so let's go back to where we were, 
                        self.arm_state = ArmState.CHECK_GRIP
                        self.SQ.enqueue_joint(self.pickup_pt_joint, ZERO_VEL, OPEN_GRIP, 2.0)

                        # spin around,
                        self.SQ.enqueue_joint(np.append(self.pickup_pt_joint[0:4], wrap(self.pickup_pt_joint[4]+np.pi, 2*np.pi)), ZERO_QDOT, OPEN_GRIP, ArmState.SPIN_180.duration)

                        # and check again because we would probably have seen 
                        # the right nub if we were facing the other way originally
                        self.purple_visible = False
                        self.SQ.enqueue_hold(ArmState.CHECK_GRIP.duration) 

        elif self.arm_state == ArmState.GRAB:
            self.current_track = self.track_type
            if self.SQ.isEmpty():
                # don't get nub location until we've grabbed (since grabbing can change the position of the nub)
                self.get_nub_location()
                # self.get_logger().info("Nub r %r, nub theta %r" % (self.nub_r, self.nub_theta))
                
                #Tell gamestate that gripper grabbed the track
                ptip, _, _, _  = self.chain.fkin(np.reshape(self.position, (-1, 1)))
                ptip = ptip.flatten()
                posemsg = dh.get_rect_pose_msg((ptip[0], ptip[1]), self.qg[0] - self.qg[4])
                self.grabbed_track_pub.publish(posemsg)
                # Move upwards
                self.goto_offset(qfgrip=CLOSED_GRIP, 
                                next_state=ArmState.RAISE_PICKUP,
                                x_offset=0, 
                                y_offset=0, 
                                z_offset=HOVER_HEIGHT, 
                                r=None, 
                                theta=None,
                                spline_type="cartesian")

        elif self.arm_state == ArmState.RAISE_PICKUP:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.GOTO_PLACE
                
                goal_pt = np.copy(self.desired_pt)

                goal_pt[2] += FIRST_ALIGN_HEIGHT
                self.SQ.enqueue_polar(goal_pt, ZERO_VEL, CLOSED_GRIP, ArmState.GOTO_PLACE.duration)
        
        elif self.arm_state == ArmState.GOTO_PLACE:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.CHECK_TRACK
                self.purple_visible = False
                self.SQ.enqueue_hold(ArmState.CHECK_TRACK.duration)
    
        elif self.arm_state == ArmState.CHECK_TRACK:
            if self.SQ.isEmpty():
                if self.purple_visible:
                    if self.skip_align:
                        self.goto_offset(qfgrip=CLOSED_GRIP, 
                                        next_state=ArmState.PLACE,
                                        x_offset=0, 
                                        y_offset=0, 
                                        z_offset=-(FIRST_ALIGN_HEIGHT), 
                                        r=None, 
                                        theta=None,
                                        spline_type="cartesian")
                    else:
                        self.arm_state = ArmState.CHECK_ALIGN    
                        self.SQ.enqueue_hold(ArmState.CHECK_ALIGN.duration)
                else:
                    self.get_logger().info("Not getting track correctly! Going back to x %r, y %r" % (self.pickup_pt[0], self.pickup_pt[1]))
                    
                    self.SQ.enqueue_joint(np.append(self.pickup_pt_joint[0:4], wrap(self.pickup_pt_joint[4]+np.pi, 2*np.pi)), 
                                            ZERO_QDOT, CLOSED_GRIP, ArmState.GOTO_PLACE.duration)
                    
                    self.arm_state = ArmState.RETURN_TRACK 

        elif self.arm_state == ArmState.RETURN_TRACK:
            if self.SQ.isEmpty():
                self.goto_offset(qfgrip=OPEN_GRIP, 
                            next_state=ArmState.RETURN,
                            x_offset=0, 
                            y_offset=0, 
                            z_offset=-(CHECK_HEIGHT), 
                            r=None, 
                            theta=None,
                            spline_type="cartesian")

        elif self.arm_state == ArmState.CHECK_ALIGN:
            if self.SQ.isEmpty():
                dx, dy, dtheta = self.align_calc()
                self.get_logger().info("dx %r, dy %r, dtheta %r" % (dx, dy, dtheta))

                self.goto_offset(qfgrip=CLOSED_GRIP, 
                                next_state=ArmState.ALIGN,
                                x_offset=dx, 
                                y_offset=dy, 
                                z_offset=0, 
                                r=None, 
                                theta=dtheta,
                                spline_type="cartesian")
    
        elif self.arm_state == ArmState.ALIGN:
            if self.SQ.isEmpty():
                # float a bit as we wiggle
                self.goto_offset(qfgrip=CLOSED_GRIP, 
                                next_state=ArmState.PLACE,
                                x_offset=0, 
                                y_offset=0, 
                                z_offset=-(FIRST_ALIGN_HEIGHT)+TRACK_DEPTH, 
                                r=None, 
                                theta=None,
                                spline_type="cartesian")

        elif self.arm_state == ArmState.PLACE:
            if self.SQ.isEmpty():
                if self.skip_align: # no need to wiggle
                    self.arm_state = ArmState.RELEASE
                    self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, OPEN_GRIP, ArmState.RELEASE.duration)
                else:
                    # try a lil wiggle
                    if self.purple_u < self.green_px_centr[0]:
                        self.wiggle_dir = 1.0
                    else:
                        self.wiggle_dir = -1.0
                    self.wiggle_counter = 0
                    self.goto_offset(qfgrip=CLOSED_GRIP,
                                    next_state=ArmState.WIGGLE,
                                    x_offset=0,
                                    y_offset=0,
                                    z_offset=0,
                                    r=0,
                                    theta=-WIGGLE_ANGLE * self.wiggle_dir,
                                    spline_type="polar")

        elif self.arm_state == ArmState.WIGGLE:
            if self.SQ.isEmpty():
                if self.wiggle_counter == 0:
                    self.goto_offset(qfgrip=CLOSED_GRIP,
                                 next_state=ArmState.WIGGLE,
                                 x_offset=0,
                                 y_offset=0,
                                 z_offset=-TRACK_DEPTH, # finish descending
                                 r=0,
                                 theta=WIGGLE_ANGLE * self.wiggle_dir,
                                 spline_type="polar")
                    self.wiggle_counter += 1
                else:
                    self.arm_state = ArmState.RELEASE
                    self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, OPEN_GRIP, ArmState.RELEASE.duration)

        elif self.arm_state == ArmState.RELEASE:
            if self.SQ.isEmpty(): # report and then go up
                # get position of center of track
                ptip, _, _, _  = self.chain.fkin(np.reshape(self.position, (-1, 1)))
                ptip = ptip.flatten()
                center = dh.get_rect_pose_msg((ptip[0], ptip[1]), self.qg[0] - self.qg[4])
                self.get_logger().info(f"angle {center.orientation.z}")

                self.get_nub_location()
                # angle doesn't matter, just need pos
                nub = dh.get_rect_pose_msg(self.nub_pos, 0.0) 

                self.placed_pose = PoseArray() # tell gamestate...
                self.placed_pose.poses.append(center)  # where we placed track
                self.placed_pose.poses.append(nub)     # where its purple nub was

                # don't publish til we've pushed into place so we don't get a new track yet
                self.goto_offset(qfgrip=OPEN_GRIP, 
                                    next_state=ArmState.CHECK_CONNECTION,
                                    x_offset=0, 
                                    y_offset=0, 
                                    z_offset=3*TRACK_DEPTH, # go high enough that we can close and push track down 
                                    r=None, 
                                    theta=None,
                                    spline_type="cartesian")
                self.SQ.enqueue_hold(ArmState.CHECK_CONNECTION.duration)   
                
        elif self.arm_state == ArmState.CHECK_CONNECTION:  
            if self.SQ.isEmpty():
                connected = Bool()
                connected.data = True

                self.get_nub_location
                (x_pnub, y_pnub) = self.nub_pos
                centroid = np.copy(self.green_centroid)
                cnt_x = centroid[0][0]
                cnt_y = centroid[1][0]

                if self.skip_align:
                    self.get_logger().info("Placed first track correctly!")
                    self.connected_track_pub.publish(connected)
                    self.arm_state = ArmState.RETURN
                    self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)
                    self.placed_track_pub.publish(self.placed_pose)

                elif abs(x_pnub - cnt_x) <= 0.008 and abs(y_pnub - cnt_y) <= 0.008:
                    self.get_logger().info("Placed track correctly! Offset is x %r, y %r" % (abs(x_pnub - cnt_x), abs(y_pnub - cnt_y)))
                    self.goto_offset(qfgrip=CLOSED_GRIP, 
                                    next_state=ArmState.CLOSE_FOR_PUSH,
                                    x_offset=0, 
                                    y_offset=0, 
                                    z_offset=0, 
                                    r=None, 
                                    theta=None,
                                    spline_type="cartesian")
                    
                    self.connected_track_pub.publish(connected)
                    self.placed_track_pub.publish(self.placed_pose)
                    
                else:
                    connected.data = False
                    self.connected_track_pub.publish(connected)

                    self.get_logger().info("Did not place track correctly! Going back to x %r, y %r" % (self.pickup_pt[0], self.pickup_pt[1]))

                    self.goto_offset(qfgrip=OPEN_GRIP, 
                        next_state=ArmState.RETURN_TRACK,
                        x_offset=0, 
                        y_offset=0, 
                        z_offset=-3*TRACK_DEPTH, 
                        r=None, 
                        theta=None,
                        spline_type="cartesian")

                    self.SQ.enqueue_joint(self.position, ZERO_QDOT, CLOSED_GRIP, ArmState.RETURN_TRACK.duration)
                    
                    self.SQ.enqueue_joint(np.append(self.pickup_pt_joint[0:4], wrap(self.pickup_pt_joint[4]+np.pi, 2*np.pi)), 
                                            ZERO_QDOT, CLOSED_GRIP, ArmState.SPIN_180.duration)
        
        # NOT USED ANYMORE
        # elif self.arm_state == ArmState.RAISE_RELEASE:
        #     if self.SQ.isEmpty():
        #         if self.skip_align: # no need for push, go home
        #             self.arm_state = ArmState.RETURN
        #             self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)
        #         else:
        #             self.get_logger().info("closing for push")
        #             self.goto_offset(qfgrip=CLOSED_GRIP, 
        #                             next_state=ArmState.CLOSE_FOR_PUSH,
        #                             x_offset=0, 
        #                             y_offset=0, 
        #                             z_offset=0, 
        #                             r=None, 
        #                             theta=None,
        #                             spline_type="cartesian")

        #         # and make sure to publish!
        #         self.placed_track_pub.publish(self.placed_pose)
            
        elif self.arm_state == ArmState.CLOSE_FOR_PUSH:
            if self.SQ.isEmpty():
                self.goto_offset(qfgrip=CLOSED_GRIP, 
                                    next_state=ArmState.PUSH_INTO_PLACE,
                                    x_offset=0, 
                                    y_offset=0, 
                                    z_offset=-3.5*TRACK_DEPTH, 
                                    r=None, 
                                    theta=None,
                                    spline_type="cartesian")

        elif self.arm_state == ArmState.PUSH_INTO_PLACE:
            if self.SQ.isEmpty():
                self.goto_offset(qfgrip=CLOSED_GRIP, 
                                    next_state=ArmState.RAISE_FROM_PUSH,
                                    x_offset=0, 
                                    y_offset=0, 
                                    z_offset=5*TRACK_DEPTH, # raise just enough to get out of the way 
                                    r=None, 
                                    theta=None,
                                    spline_type="cartesian")

        elif self.arm_state == ArmState.RAISE_FROM_PUSH:
            if self.SQ.isEmpty(): # go home
                self.arm_state = ArmState.RETURN
                self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)

        else:
            self.get_logger().info("Arm in unknown state")
            self.arm_state = ArmState.RETURN
            self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)

        if self.arm_state != ArmState.IDLE:
            qg, qgdot = self.SQ.evaluate()
            # self.get_logger().info("queue length %r" % self.SQ.queueLength())

        # if self.arm_state == ArmState.HOLD and abs(self.qg[5] - self.grip_position) < 0.1:
        # if self.arm_state == ArmState.HOLD and abs(self.qg[5] - self.grip_position) < 0.1:
        #     self.get_logger().info("DROPPED OR MISSED TRACK")

        self.qg = qg
        self.qgdot = qgdot

        self.cmdmsg.position = list(qg)
        self.cmdmsg.velocity = list(qgdot)


        track_error = self.chain.fkin(self.qg[:5])[0][2]-self.chain.fkin(self.position)[0][2]
        # self.get_logger().info("z tracking error %r" % track_error)

        # self.get_logger().info("z height %r" % self.chain.fkin(self.position)[0][2])

        cam_diff = self.cam_chain.fkin(self.position)[0] - self.chain.fkin(self.position)[0]
        # self.get_logger().info("cam pos diff %r" % cam_diff)

        joint_err = self.qg[1:3] - self.position[1:3]
        # self.get_logger().info("joint tracking error %r" % joint_err)

        # self.get_logger().info("current gripper qg %r" % self.qg[5])
        # self.get_logger().info("current gripper pos %r" % self.grip_position)

        # self.get_logger().info("current state %r" % self.arm_state)

        # Publish commands, makes robot move
        
        # nan = float("nan")
        # self.cmdmsg.position = (nan, nan, nan, nan, nan, nan)
        # self.cmdmsg.velocity = (nan, nan, nan, nan, nan, nan)


        # self.get_logger().info("current gripper qg %r" % self.qg[5])

        self.cmdpub.publish(self.cmdmsg)

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = VanderNode('demo')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
