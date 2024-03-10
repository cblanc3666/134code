#!/usr/bin/env python3
#   
#   This node interfaces with the HEBIs and is responsible for all the
#   low level motion control. To determine its waypoint locations, it
#   receives messages from the GameState node. It also receives
#   messages from the Detector node in the form of arm camera on-screen
#   locations of objects.

import numpy as np
import rclpy
import vanderbot.DetectHelpers as dh

from enum import Enum
from vanderbot.SegmentQueueDelayed     import SegmentQueue

from rclpy.node                        import Node
from sensor_msgs.msg                   import JointState
from geometry_msgs.msg                 import Point, Pose, Polygon, PoseArray

from vanderbot.KinematicChain          import KinematicChain
from vanderbot.TransformHelpers        import *
from vanderbot.Segments                import Goto5, QuinticSpline
from std_msgs.msg                      import String, Float32

''' Constant Definitions '''
RATE = 100.0            # transmit rate, in Hertz

# Holding position over the table
IDLE_POS = np.array([0, 1.4, 1.4, 0.0, 0.0])    # Holding position over table

# Position lying straight out over table
DOWN_POS = np.array([0, 0.0, 0.55, 0.0, 0.0])

OPEN_GRIP = -0.3
CLOSED_GRIP = -0.8
IDLE_ALPHA = 0.0
IDLE_BETA = 0.0

SEENUB_OFFSET = 0.0#15 # move back along the track in order to see the nub when picking up
TRACK_OFFSET = 0.17 # TODO - this is currently hard-coded for a curved track. Should depend on both track holding and track seen

# Initial joint velocity (should be zero)
ZERO_QDOT = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# zero velocity in x, y, z, alpha, beta directions
ZERO_VEL = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Gripper initial Q and Qdot
GRIP_ZERO_QDOT = 0.0

# magnitude of the joint space divergence (||real - des||) that constitutes a 
# collision
Q_COLLISION_THRESHOLD = 0.07 # TODO
QDOT_COLLISION_THRESHOLD = 0.5 # TODO

# track height to grab at
TRACK_DEPTH = 0.125

# final gravity torque values (to attain as gravity is splined in)
GRAV_ELBOW = -6.8 #-6.5
GRAV_ELBOW_OFFSET = 0.01  # Radians
GRAV_SHOULDER = 12.7 #12.1
GRAV_SHOULDER_OFFSET = -0.11 # Raidans

# gripper closed hand torque value
TAU_GRIP = -6.0

# arrays indicating which angles (out of base, shoulder, elbow, wrist, twist)
# contribute to alpha and beta, respectively
ALPHA_J = np.array([0, 1, 1, 1, 0])
BETA_J = np.array([1, 0, 0, 0, 1])

TRACK_DISPLACEMENT_FORWARDS = -0.08
TRACK_DISPLACEMENT_SIDE = 0.0 #0.025

TRACK_TURN_RADIUS = 0.1299 # radius of a circle

HOVER_HEIGHT = 0.07
CHECK_HEIGHT = 0.05


def wrap(angle, fullrange):
    return angle - fullrange * round(angle/fullrange)

#   States the arm can be in
#   Duration concerns length of spline in each state
class ArmState(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
    
    def __init__(self, duration, segments):
        self.duration = duration
        self.segments = segments

    START = 6.0, []  # initial state
    RETURN = 4.0, [] # moving back to IDLE_POS, either because no other point has 
                    # been commanded or because the next point is too far from
                    # the previous point
                    # Returning ALWAYS uses a joint space spline
    IDLE = None, []  # nothing commanded, arm is at IDLE_POS
    
    GOTO_PICKUP  = 5.0, []  # moving to a commanded point, either from IDLE_POS or a 
                    # previously commanded point
    CHECK_GRIP = 1.0, []    # state allowing detector to check if we are about to
                            # grip track correctly
    SPIN_180 = 3.0, []
    LOWER = 6.0, []
    BACKUP_FORNUB = 2.0, [] 
    GRAB = 2.0, []  # grab onto track
    RAISE_PICKUP = 2.0, []
    GOTO_PLACE = 6.0, []
    CHECK_ALIGN = 1.0, []
    ALIGN = 5.0, []
    PLACE = 6.0, []
    RELEASE = 2.0, [] # release grip on track
    LIE_DOWN = 1.0, [] # tells arm to go to lying down position, no matter what it is doing



#   Track colors
#   Stores an angle offset used to align track with arm camera
#   Angle offset is angle of track relative to green rectangle pose
class TrackColor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
    
    def __init__(self, angle_offset):
        self.angle_offset = angle_offset

    BLUE = 0.0  # straight track has no angle offset from green rectangle
    PINK = np.pi/6.0 # left turning track needs to be rotated by 30 degrees right
    ORANGE = -np.pi/6.0


#
#   Vanderbot Node Class
#
class VanderNode(Node):
    # joints are in form ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']
    
    position =  None # real joint positions (does not include gripper)
    qdot =      None # real joint velocities (does not include gripper)
    effort =    None # real joint efforts (does not include gripper)

    grip_position = None # real gripper position
    grip_qdot = None # real gripper joint velocity
    grip_effort = None # real gripper effort

    qg = None # combined desired joint/gripper joint angles
    qgdot = None # combined desired joint/gripper angular velocities

    start_time = 0 # time of initialization

    arm_state = ArmState.START # initialize state machine
    arm_killed = False # true when someone wants the arm to die
    track_type = None
    

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

    desired_pt = None # Destination for placement - either existing track or location for starting track

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create a temporary subscriber to grab the initial position.
        fbk = self.grabfbk()
        self.position0 = np.array(fbk)

        # Set the initial desired position to initial position so robot stays put
        self.qg = self.position0
        self.qgdot = np.append(ZERO_QDOT, GRIP_ZERO_QDOT)
        
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
        
        self.placed_track_pub = self.create_publisher(Pose, '/PlacedTrack', 10)

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
        z = pcam[2][0]
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

        self.position = position[:5] # TODO maybe don't trim this
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
        # self.track_color = TODO
        curr_pose = posemsg.poses[0]
        desired_pose = posemsg.poses[1]
        x = curr_pose.position.x
        y = curr_pose.position.y
        z = TRACK_DEPTH + CHECK_HEIGHT # hover for checking tracks

        self.track_type = curr_pose.orientation.x
        beta = 2*np.arcsin(desired_pose.orientation.z) - self.track_type * np.pi/6 # angle offset for track type

        self.desired_pt = np.array([desired_pose.position.x, desired_pose.position.y, TRACK_DEPTH, 0.0, beta])
        angle = 2 * np.arcsin(curr_pose.orientation.z)
        # self.get_logger().info("Found track at (%r, %r) with angle %r" % (x, y, angle))
        self.skip_align = (desired_pose.orientation.y == 1)

        if not self.skip_align:
            self.desired_pt[0] -= np.cos(beta) * TRACK_OFFSET
            self.desired_pt[1] -= np.sin(beta) * TRACK_OFFSET
        
        self.gotopoint(x,y,z, beta=angle)

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
        positions = []
        for corner in msg.points:
            positions.append(self.arm_pixel_to_position(corner.x, corner.y))
            
        
        (self.green_centroid, self.green_orientation) = self.green_rect_position(positions)

        # #self.get_logger().info(f"Centroid of green rect x {centroid[0][0]} y {centroid[1][0]}")
        # direction_90 = np.array([[-direction[1][0]], [direction[0][0]], [direction[2][0]]])
        
        # align_point = centroid + direction * TRACK_DISPLACEMENT_FORWARDS
        # align_point += direction_90 * TRACK_DISPLACEMENT_SIDE
        
        # self.align_position = align_point
        # self.align_position[2][0] = TRACK_DEPTH + HOVER_HEIGHT
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
    '''
    def goto_offset(self, qfgrip, next_state, x_offset, y_offset, z_offset, r=None, theta=None):
        self.arm_state = next_state
        
        pgoal, _, _, _ = self.chain.fkin(np.reshape(self.position, (-1, 1)))
        pgoal = pgoal.flatten()
        
        alpha = self.qg[1] - self.qg[2] + self.qg[3]
        beta = self.qg[0] - self.qg[4]

        if (theta != None):
            beta -= theta

        if (r != None):
            pgoal[0] += r * np.cos(beta)
            pgoal[1] += r * np.sin(beta)
        else:
            pgoal[0] += x_offset
            pgoal[1] += y_offset
            
        pgoal[2] += z_offset

        pgoal = np.append(pgoal, [alpha, beta])

        self.SQ.enqueue_polar(pgoal, ZERO_VEL, qfgrip, next_state.duration)

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

        (x_pnub, y_pnub, _) = self.arm_pixel_to_position(nub_u, nub_v).flatten()

        self.nub_r = np.sqrt((x_ptip - x_pnub)**2 + (y_ptip - y_pnub)**2 )
        # TODO track type
        self.nub_theta = self.track_type * np.arccos(1 - self.nub_r**2 / (2 * TRACK_TURN_RADIUS**2))

    '''
    Get offsets for the end effector to align grabbed track with green rectangle
    * assuming slot is centroid of contour
    '''
    def align_calc(self):
        centroid = np.copy(self.green_centroid)
        cnt_x = centroid[0][0]
        cnt_y = centroid[1][0]
        direction_vec = np.linalg.norm(np.copy(self.green_orientation))
        self.get_logger().info(f"Direction {direction_vec} x{centroid[0][0]} y {centroid[1][0]}")
        self.get_logger().info(f"Nub theta {self.nub_theta}")

        ptip, _, _, _  = self.chain.fkin(np.reshape(self.position, (-1, 1)))
        ptip = ptip.flatten()
        (x_ptip, y_ptip, _) = ptip

        # get offset of green contour from end effector
        green_dx = cnt_x - x_ptip
        green_dy = cnt_y - y_ptip

        beta = self.qg[0] - self.qg[4]
        gamma = self.nub_theta / 2

        # offsets for the end effector
        d_theta = direction_vec - self.nub_theta # need to align to the bottom of the green contour
        dx = green_dx - self.nub_r * np.cos(beta + gamma)
        dy = green_dy - self.nub_r * np.sin(beta + gamma)

        return dx, dy, d_theta 

        



        
    
    # Send a command - called repeatedly by the timer.
    def sendcmd(self):        
        # self.get_logger().info("Desired point %r" % self.desired_pt)
        # self.get_logger().info("Current state %r" % self.arm_state) # TODO turn back on; watermelon
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

        # # collision checking TODO
        # if np.linalg.norm(ep(np.array(self.q_des), self.position)) > Q_COLLISION_THRESHOLD or \
        #    np.linalg.norm(ep(np.array(self.qdot_des), self.qdot)) > QDOT_COLLISION_THRESHOLD:
            
        #     if self.arm_state == ArmState.RETURN or self.arm_state == ArmState.GOTO: # only detect collisions while moving
        #         (ptip, _, _, _) = self.chain.fkin(np.reshape(self.position, (-1, 1)))
                
        #         alpha = self.position[1]-self.position[2]+self.position[3] 
        #         beta = self.position[0]-self.position[4] # base minus twist
        #         ptip = np.vstack((ptip, alpha, beta))
        #         # stay put, then try to go home
        #         ArmState.HOLD.segments.append(Hold(ptip, 
        #                                            ArmState.HOLD.duration,
        #                                            space='Joint'))
                
        #         ArmState.RETURN.segments = [] # clear it out just in case
        #         ArmState.GOTO.segments = []
        #         ArmState.RETURN.segments.append(Goto5(np.array(self.position), 
        #                                              np.array(IDLE_POS), 
        #                                              ArmState.RETURN.duration,
        #                                              space='Joint'))

        #         self.arm_state = ArmState.HOLD
        #         self.collided = False

        #         self.get_logger().info("COLLISION DETECTED")
            
        #     else:
        #         pass # do nothing if collision detected on hold or idle or start

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
                self.arm_state = ArmState.CHECK_GRIP
                self.purple_visible = False
                # stay put while checking grip
                self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, OPEN_GRIP, ArmState.CHECK_GRIP.duration)
                    
        elif self.arm_state == ArmState.CHECK_GRIP:
            if self.SQ.isEmpty():
                if self.purple_visible: # we see purple, so back up a bit and close hand
                    self.goto_offset(OPEN_GRIP, ArmState.BACKUP_FORNUB, 0, 0, 0, -SEENUB_OFFSET, None)
                else:
                    self.arm_state = ArmState.SPIN_180
                    self.SQ.enqueue_joint(np.append(self.qg[0:4], wrap(self.qg[4]+np.pi, 2*np.pi)), ZERO_QDOT, OPEN_GRIP, ArmState.SPIN_180.duration)

        elif self.arm_state == ArmState.SPIN_180:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.CHECK_GRIP
                self.purple_visible = False
                # stay put while checking grip
                self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, OPEN_GRIP, ArmState.CHECK_GRIP.duration)

        elif self.arm_state == ArmState.BACKUP_FORNUB:
            if self.SQ.isEmpty():
                self.goto_offset(OPEN_GRIP, ArmState.LOWER, 0, 0, -(CHECK_HEIGHT), None, None)


        elif self.arm_state == ArmState.LOWER:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.GRAB
                self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, CLOSED_GRIP, ArmState.GRAB.duration)
                self.get_nub_location()
                self.get_logger().info("Nub r %r, nub theta %r" % (self.nub_r, self.nub_theta))

        elif self.arm_state == ArmState.GRAB:
            if self.SQ.isEmpty():
                # Move upwards
                self.goto_offset(CLOSED_GRIP, ArmState.RAISE_PICKUP, 0, 0, HOVER_HEIGHT, None, None)

        elif self.arm_state == ArmState.RAISE_PICKUP:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.GOTO_PLACE
                
                goal_pt = np.copy(self.desired_pt)

                goal_pt[2] += HOVER_HEIGHT
                self.SQ.enqueue_polar(goal_pt, ZERO_VEL, CLOSED_GRIP, ArmState.GOTO_PLACE.duration)

        elif self.arm_state == ArmState.GOTO_PLACE:
            if self.SQ.isEmpty():
                if self.skip_align:
                    self.goto_offset(CLOSED_GRIP, ArmState.PLACE, 0, 0, -(HOVER_HEIGHT), None, None)
                else:
                    self.arm_state = ArmState.ALIGN

                    dx, dy, dtheta = self.align_calc()
                    self.get_logger().info("dx %r, dy %r, dtheta %r" % (dx, dy, dtheta))

                    self.goto_offset(CLOSED_GRIP, ArmState.ALIGN, dx, dy, 0, None, dtheta)

                    # alpha = self.qg[1] - self.qg[2] + self.qg[3]
                    # beta = d_theta

                    # align_goal = np.copy(self.align_position)
                    # alpha = self.qg[1] - self.qg[2] + self.qg[3]
                    # beta = self.qg[0] - self.qg[4]

                    # align_goal = np.append(align_goal, [alpha, beta])

                    # # keep gripper closed
                    # self.SQ.enqueue_polar(align_goal, ZERO_VEL, CLOSED_GRIP, ArmState.ALIGN.duration)

        elif self.arm_state == ArmState.ALIGN:
            if self.SQ.isEmpty():
                self.goto_offset(CLOSED_GRIP, ArmState.PLACE, 0, 0, -(HOVER_HEIGHT), None, None)

        elif self.arm_state == ArmState.PLACE:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.RELEASE
                ptip, _, _, _  = self.chain.fkin(np.reshape(self.position, (-1, 1)))
                ptip = ptip.flatten()
                posemsg = dh.get_rect_pose_msg((ptip[0], ptip[1]), self.qg[0] - self.qg[4])
                self.placed_track_pub.publish(posemsg)
                self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, OPEN_GRIP, ArmState.RELEASE.duration)

        elif self.arm_state == ArmState.RELEASE:
            if self.SQ.isEmpty(): # go home
                self.arm_state = ArmState.RETURN
                self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)
        else:
            self.get_logger().info("Arm in unknown state")
            self.arm_state = ArmState.RETURN
            self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, OPEN_GRIP, ArmState.RETURN.duration)

        # TODO wrap everyting in self.SQ.isEmpty()
        # TODO make it a case statement
        # TODO always enqueue align and down together

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

        joint_err = self.qg[1:3] - self.position[1:3]
        # self.get_logger().info("joint tracking error %r" % joint_err)

        # self.get_logger().info("current gripper qg %r" % self.qg[5])
        # self.get_logger().info("current gripper pos %r" % self.grip_position)

        # self.get_logger().info("current state %r" % self.arm_state)

        # Publish commands, makes robot move
        
        # nan = float("nan")
        # self.cmdmsg.position = (nan, nan, nan, nan, nan, nan)
        # self.cmdmsg.velocity = (nan, nan, nan, nan, nan, nan)

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