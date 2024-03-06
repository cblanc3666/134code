#!/usr/bin/env python3
#
#   actuate.py
#
#   node to interact with the HEBIs, runs looping code to move motors based on
#   inputs received
#
import numpy as np
import rclpy

from enum import Enum
from vanderbot.SegmentQueueDelayed import spline, SegmentQueue, JointSpline, star

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion, Polygon, PoseArray

from vanderbot.Segments          import Hold, Stay, Goto5, QuinticSpline
from vanderbot.KinematicChain    import KinematicChain
from vanderbot.TransformHelpers  import *
from std_msgs.msg               import Float32

# ros2 topic pub -1 /point geometry_msgs/msg/Point "{x: 0.2, y: 0.3, z: 0.1}"

#
#   Definitions
#
RATE = 100.0            # transmit rate, in Hertz


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

    START = 7.0, []  # initial state
    GOTO  = 7.0, []  # moving to a commanded point, either from IDLE_POS or a 
                    # previously commanded point
    CHECK_GRIP = 2.0, []    # state allowing detector to check if we are about to
                            # grip track correctly
    RETURN = 7.0, [] # moving back to IDLE_POS, either because no other point has 
                    # been commanded or because the next point is too far from
                    # the previous point
                    # Returning ALWAYS uses a joint space spline
    IDLE = None, []  # nothing commanded, arm is at IDLE_POS
    HOLD = 2.0, []   # holding at a commanded point, preparing to return or move to
                    # next point
    
    ALIGN = 4.0, [] # align track to track below
    DOWN = 8.0, []  # lower track into place
    GRAB = 2.0, []  # grab onto track
    RELEASE = 2.0, [] # release grip on track

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
   


# Holding position over the table
IDLE_POS = np.array([0, 1.4, 1.4, 0.0, 0.0])
OPEN_GRIP = 0.0
CLOSED_GRIP = -0.8 
IDLE_ALPHA = 0.0
IDLE_BETA = 0.0

TRACK_OFFSET = 0.0 #0.17 # TODO - this is currently hard-coded for a curved track. Should depend on both track holding and track seen

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
TRACK_DEPTH = 0.11

# final gravity torque values (to attain as gravity is splined in)
GRAV_ELBOW = -6.5
GRAV_SHOULDER = 12.5

# gripper closed hand torque value
TAU_GRIP = -6.0

# arrays indicating which angles (out of base, shoulder, elbow, wrist, twist)
# contribute to alpha and beta, respectively
ALPHA_J = np.array([0, 1, 1, 1, 0])
BETA_J = np.array([1, 0, 0, 0, 1])

TRACK_DISPLACEMENT_FORWARDS = 0.06
TRACK_DISPLACEMENT_SIDE = -0.025

HOVER_HEIGHT = 0.07
CHECK_HEIGHT = 0.1

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
    track_color = None

    align_position = None
    
    # Indicates whether the arm camera can see the purple nub on the gripped track
    purple_visible = False

    grav_elbow = 0 # start elbow torque constant at 0
    grav_shoulder = 0 # start shoulder torque constant at 0

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
        
        # Spline gravity in gradually TODO handle this
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
        
        self.fbksub = self.create_subscription(
            Point, '/point', self.recvpoint, 10)
        
        self.sent_track = self.create_subscription(
            PoseArray, '/SentTrack', self.recvtrack, 10)
        
        self.green_rect = self.create_subscription(
            Polygon, '/GreenRect', self.recvgreenrect, 10)
        
        self.purple_circ = self.create_subscription(
            Point, '/PurpleCirc', self.recvpurplecirc, 10)
        
        self.placed_track_pub = self.create_publisher(Pose, '/PlacedTrack', 10)
        
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
        # pass
        (pcam, Rcam, _, _) = self.cam_chain.fkin(np.reshape(self.position, (-1, 1)))
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


    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z

        self.gotopoint(x, y, z)
    
    def recvtrack(self, posemsg):
        # self.track_color = TODO
        curr_pose = posemsg.poses[0]
        desired_pose = posemsg.poses[1]
        x = curr_pose.position.x
        y = curr_pose.position.y
        z = TRACK_DEPTH
        self.desired_pt = np.append(desired_pose.position.x, desired_pose.position.y, TRACK_DEPTH, beta=2*np.arcsin(desired_pose.orientation.z))
        angle = 2 * np.arcsin(curr_pose.orientation.z)
        # self.get_logger().info("Found track at (%r, %r) with angle %r" % (x, y, angle))
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
        
        (centroid, direction) = self.green_rect_position(positions)
        direction_90 = np.array([[-direction[1][0]], [direction[0][0]], [direction[2][0]]])
        
        align_point = centroid + direction * TRACK_DISPLACEMENT_FORWARDS
        align_point += direction_90 * TRACK_DISPLACEMENT_SIDE
        
        self.align_position = align_point
        self.align_position[2][0] = TRACK_DEPTH + HOVER_HEIGHT
        self.align_position = np.array(self.align_position.flatten())

    def recvpurplecirc(self, msg):
        self.purple_visible = True

    def gotopoint(self, x, y, z, beta=0):
        if self.arm_state != ArmState.IDLE: 
            # self.get_logger().info("Already commanded!")
            return

        self.arm_state = ArmState.GOTO

        # Arm Closed - TODO make this smarter so it knows when it has a track, and knows how to offset given which track we have
        # self.get_logger().info("ARM CLOSED??? %r" % abs(self.qg[5] - OPEN_GRIP))
        if abs(self.qg[5] - OPEN_GRIP) > 0.1:
            # self.get_logger().info("Current state %r" % self.arm_state)
            self.get_logger().info("CLOSE, GOTO POINT %r" % self.qg[5])

            x -= np.sin(beta) * TRACK_OFFSET
            y -= np.cos(beta) * TRACK_OFFSET
            z += HOVER_HEIGHT
            # z += 0.0
        else:
            # Hover for checking tracks if gripper is open
            z += CHECK_HEIGHT
        
        # set alpha desired to zero - want to be facing down on table
        pf = np.array([x, y, z, 0, beta])

        # hold current grip position
        self.SQ.enqueue_task(pf, ZERO_VEL, self.qg[5], ArmState.GOTO.duration)

        # Report.
        #self.get_logger().info("Going to point %r, %r, %r" % (x,y,z))

        
    
    # Send a command - called repeatedly by the timer.
    def sendcmd(self):        
        # self.get_logger().info("Current state %r" % self.arm_state) # TODO turn back on
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
        tau_elbow = self.grav_elbow * np.cos(self.position[1] - self.position[2])
        tau_shoulder = -tau_elbow + self.grav_shoulder * np.cos(self.position[1])
        tau_grip = TAU_GRIP
        self.cmdmsg.effort       = list([0.0, tau_shoulder, tau_elbow, 0.0, 0.0, tau_grip])

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

        if self.arm_state == ArmState.START:
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
                self.get_logger().info("gripper at START EMPTY %r" % self.qg[5])
            
        elif self.arm_state == ArmState.RETURN:
            if self.SQ.isEmpty():
                # TODO we're better than this
                qg = np.append(IDLE_POS, self.qg[5])
                qgdot = np.append(ZERO_QDOT, GRIP_ZERO_QDOT)
                self.arm_state = ArmState.IDLE
                self.get_logger().info("gripper at RETURN %r" % self.qg[5])

        elif self.arm_state == ArmState.HOLD: 
            if self.SQ.isEmpty():
                self.arm_state = ArmState.RETURN
                self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, self.qg[5], ArmState.RETURN.duration) 
                self.get_logger().info("gripper at HOLD %r" % self.qg[5])            

        elif self.arm_state == ArmState.GOTO:
                       
            # Moving to commanded point
            if self.SQ.isEmpty():                
                # gripper needs to close
                if abs(self.qg[5] - OPEN_GRIP) < 0.1:
                    # check if we're gonna grab track the right way
                    self.arm_state = ArmState.CHECK_GRIP

                    # set the purple Boolean to False. We need to see purple 
                    # nub during the CHECK_GRIP state in order to proceed with grabbing
                    self.purple_visible = False
                    
                    # stay put while checking grip
                    self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, OPEN_GRIP, ArmState.CHECK_GRIP.duration)
                    
                else:
                    self.arm_state = ArmState.ALIGN

                    # self.collided = False # successfully finished a goto, reset collision boolean TODO do we need this

                    (ptip, _, _, _) = self.chain.fkin(self.qg[0:5])

                    align_goal = self.align_position
                    alpha = self.qg[1]-self.qg[2]+self.qg[3]
                    beta = self.qg[0]-self.qg[4]
                    self.get_logger().info("ALPHA%r" % alpha)
                    self.get_logger().info("BETA%r" % type(beta))
                    self.get_logger().info("ALIGN GOAL%r" % type(align_goal))


                    align_goal = np.append(align_goal, [alpha, beta])

                    down_goal = np.copy(align_goal)
                    down_goal[2] -= (HOVER_HEIGHT + 0.01)

                    self.get_logger().info("gripper at GOTO ALIGN %r" % qg)

                    # keep gripper closed
                    self.SQ.enqueue_task(align_goal, ZERO_VEL, self.qg[5], ArmState.ALIGN.duration)
                    self.SQ.enqueue_task(down_goal, ZERO_VEL, self.qg[5], 2*ArmState.ALIGN.duration) # TODO need to make durations more dynamic 

        elif self.arm_state == ArmState.CHECK_GRIP:
            if self.SQ.isEmpty():
                if self.purple_visible: # close hand, we see purple
                    self.arm_state = ArmState.GRAB

                    # self.collided = False # successfully finished a goto, reset collision boolean TODO do we need this

                    # stay put during grab
                    ptip, _, _, _ = self.chain.fkin(self.qg[0:5])
                    ptip = ptip.flatten()

                    alpha = self.qg[1] - self.qg[2] + self.qg[3]
                    beta = self.qg[0] - self.qg[4]

                    # Modify z height
                    ptip[2] -= CHECK_HEIGHT
                    ptip = np.append(ptip, [alpha, beta])

                    self.SQ.enqueue_task(ptip, ZERO_VEL, OPEN_GRIP, ArmState.CHECK_GRIP.duration)
                    self.SQ.enqueue_task(ptip, ZERO_VEL, CLOSED_GRIP, ArmState.GRAB.duration)
                    self.get_logger().info("gripper at GOTO W ARM OPEN%r" % self.qg[5])
                else:
                    # spin twist joint around 180 degrees. Wrap it properly
                    self.SQ.enqueue_joint(np.append(self.qg[0:4], wrap(self.qg[4]+np.pi, 2*np.pi)), ZERO_QDOT, self.qg[5], ArmState.CHECK_GRIP.duration)
                    self.SQ.enqueue_joint(np.append(self.qg[0:4], wrap(self.qg[4]+np.pi, 2*np.pi)), ZERO_QDOT, OPEN_GRIP, ArmState.CHECK_GRIP.duration)

        elif self.arm_state == ArmState.ALIGN:
            if self.SQ.isEmpty():
                self.arm_state = ArmState.RELEASE
                pointmsg = Point()
                pointmsg.x = ptip[0][0]
                pointmsg.y = ptip[0][1]
                self.placed_track_pub.publish(pointmsg)
                # stay put during hold
                self.SQ.enqueue_joint(self.qg[0:5], ZERO_QDOT, OPEN_GRIP, ArmState.RELEASE.duration)
        
        elif self.arm_state == ArmState.GRAB:
            if self.SQ.isEmpty(): # go home
                self.arm_state = ArmState.RETURN
                self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, self.qg[5], ArmState.RETURN.duration)

        elif self.arm_state == ArmState.RELEASE:
            if self.SQ.isEmpty(): # go home
                self.arm_state = ArmState.RETURN
                self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, self.qg[5], ArmState.RETURN.duration)

        elif self.arm_state == ArmState.IDLE:
            qg = np.append(IDLE_POS, self.qg[5])
            qgdot = np.append(ZERO_QDOT, GRIP_ZERO_QDOT) # TODO figure out a better way to handle idle

        else:
            self.get_logger().info("Arm in unknown state")
            self.arm_state = ArmState.RETURN
            self.SQ.enqueue_joint(IDLE_POS, ZERO_QDOT, self.qg[5], ArmState.RETURN.duration)

        # TODO wrap everyting in self.SQ.isEmpty()
        # TODO make it a case statement
        # TODO always enqueue align and down together

        # if self.arm_state != ArmState.IDLE and self.arm_state != ArmState.RETURN:
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

        # self.get_logger().info("current gripper qg %r" % self.qg[5])
        # self.get_logger().info("current gripper pos %r" % self.grip_position)

        # self.get_logger().info("current state %r" % self.arm_state)


        # Publish commands, makes robot move
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