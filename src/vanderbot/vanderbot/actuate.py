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
from SegmentQueueDelayed import SegmentQueue, JointSpline, star

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion, Polygon

from vanderbot.Segments          import Hold, Stay, Goto5, QuinticSpline
from vanderbot.KinematicChain    import KinematicChain
from vanderbot.TransformHelpers  import *
from std_msgs.msg               import Float32

# ros2 topic pub -1 /point geometry_msgs/msg/Point "{x: 0.2, y: 0.3, z: 0.1}"

#
#   Definitions
#
RATE = 100.0            # transmit rate, in Hertz
gamma = 0.1


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

    START = 8.0, []  # initial state
    GOTO  = 8.0, []  # moving to a commanded point, either from IDLE_POS or a 
                    # previously commanded point
    RETURN = 15.0, [] # moving back to IDLE_POS, either because no other point has 
                    # been commanded or because the next point is too far from
                    # the previous point
                    # Returning ALWAYS uses a joint space spline
    IDLE = None, []  # nothing commanded, arm is at IDLE_POS
    HOLD = 2.0, []   # holding at a commanded point, preparing to return or move to
                    # next point
    
    ALIGN = 4.0, []
    DOWN = 8.0, []
    RELEASE = 2.0, []


# Holding position over the table
IDLE_POS = [0.0, 1.4, 1.4, 0.0, 0.0]
IDLE_GRIP = 0.0
CLOSED_GRIP = -0.8 
IDLE_ALPHA = 0.0
IDLE_BETA = 0.0

TRACK_OFFSET = 0.17

# Initial joint velocity (should be zero)
QDOT_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]

# Gripper initial Q and Qdot
GRIP_QDOT_INIT = 0.0

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

TRACK_DISPLACEMENT_FORWARDS = 0.08
TRACK_DISPLACEMENT_SIDE = -0.025

HOVER_HEIGHT = 0.07

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

    q_des =     None # desired joint positions (does not include gripper)
    qdot_des =  None # desired joint velocities (does not include gripper)

    grip_q_des =  None # desired gripper position
    grip_qdot_des = None # desired gripper joint velocity

    start_time = 0 # time of initialization
    seg_start_time = 0  # logs the time (relative to start_time) that last segment started
                        # or, if state is IDLE, time that it has been idle
    arm_state = ArmState.START # initialize state machine

    align_position = None


    grav_elbow = 0 # start elbow torque constant at 0
    grav_shoulder = 0 # start shoulder torque constant at 0

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create a temporary subscriber to grab the initial position.
        fbk = self.grabfbk()
        self.position0 = fbk[:5] # trim off gripper position to save separately
        self.grip_position0 = fbk[5]

        # Set the initial desired position to initial position so robot stays put
        self.q_des = self.position0
        self.qdot_des = QDOT_INIT
        self.grip_q_des = self.grip_position0
        self.grip_qdot_des = GRIP_QDOT_INIT
        
        self.get_logger().info("Initial positions: %r" % self.position0)

        # create kinematic chains
        self.chain = KinematicChain('world', 'tip', self.jointnames())
        self.cam_chain = KinematicChain('world', 'cam', self.jointnames())

        # Start the clock
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # initialize segment queue with the desired fkin function
        self.SQ = SegmentQueue(self.chain.fkin)
        self.SQ.update(self.start_time, self.q_des, self.qdot_des)
        
        # Set up first spline (only shoulder is moving)
        self.SQ.enqueue(JointSpline())
        ArmState.START.segments.append(Goto5(np.array(self.position0), 
                                     np.array([self.position0[0], IDLE_POS[1], self.position0[2], self.position0[3], self.position0[4]]),
                                     ArmState.START.duration,
                                     space='Joint'))
        
        ArmState.START.segments.append(Goto5(np.array(self.grip_position0),
                                        IDLE_GRIP,
                                        ArmState.START.duration,
                                        space='Joint'))
        
        # Spline gravity in gradually
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
        
        self.orange_track = self.create_subscription(
            Pose, '/StraightTrackOrange', self.recvtrack_orange, 10)
        
        self.pink_track = self.create_subscription(
            Pose, '/StraightTrackPink', self.recvtrack_pink, 10)
        
        self.green_rect = self.create_subscription(
            Polygon, '/GreenRect', self.recvgreenrect, 10)
        
        # Report.
        self.get_logger().info("Running %s" % name)
        
        # Pick the convergence bandwidth.
        self.lam = 20.0

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

        self.position = position[:5]
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
    
    def recvtrack_orange(self, posemsg):
        x = posemsg.position.x
        y = posemsg.position.y
        z = TRACK_DEPTH
        angle = 2 * np.arcsin(posemsg.orientation.z)
        self.get_logger().info("Found track at (%r, %r) with angle %r" % (x, y, angle))
        self.gotopoint(x,y,z, beta=angle)

    def recvtrack_pink(self, posemsg):
        x = posemsg.position.x
        y = posemsg.position.y
        z = TRACK_DEPTH
        angle = 2 * np.arcsin(posemsg.orientation.z)
        # self.get_logger().info("Found track at (%r, %r) with angle %r" % (x, y, angle))
        self.gotopoint(x,y,z, beta=angle)

    def green_rect_position(self, points):
        side1 = np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])
        side2 = np.linalg.norm(points[2] - points[1]) + np.linalg.norm(points[3] - points[0])
        
        if side1 > side2:
            long_side = ((points[0] - points[1]) + (points[3] - points[2]))/2
        else:
            long_side = ((points[2] - points[1]) + (points[3] - points[0]))/2
        
        centroid = np.mean(points, 0)
        direction_vec = long_side / np.linalg.norm(long_side)


        return (centroid, direction_vec)
    
    def recvgreenrect(self, msg):
        # pass
        positions = []
        for corner in msg.points:
            positions.append(self.arm_pixel_to_position(corner.x, corner.y))
        
        (centroid, direction) = self.green_rect_position(positions)
        direction_90 = np.array([[-direction[1][0]], [direction[0][0]], [direction[2][0]]])
        

        align_point = centroid + direction * TRACK_DISPLACEMENT_FORWARDS
        align_point += direction_90 * TRACK_DISPLACEMENT_SIDE
        
        self.align_position = align_point
        self.align_position[2][0] = TRACK_DEPTH + HOVER_HEIGHT

        (ptip, _, _, _) = self.chain.fkin(self.position)
        # self.get_logger().info("Tip position, Target position: (%r, %r), (%r, %r)"
        #                        % (ptip[0][0], ptip[1][0], align_point[0][0], align_point[1][0]))
        
        
        # self.green_rect_pose()

    def gotopoint(self, x, y, z, beta=0):
        if self.arm_state != ArmState.IDLE: 
            # self.get_logger().info("Already commanded!")
            return

        self.seg_start_time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        self.arm_state = ArmState.GOTO
        
        # Go to command position JOINT SPACE
        (idle_pos, _, _, _) = self.chain.fkin(np.reshape(IDLE_POS, (-1, 1)))
        
        idle_pos = np.vstack((idle_pos, IDLE_ALPHA, IDLE_BETA))

        # Arm Closed
        if abs(self.grip_q_des - IDLE_GRIP) > 0.1:
            x -= np.sin(beta) * TRACK_OFFSET
            y -= np.cos(beta) * TRACK_OFFSET
            z += HOVER_HEIGHT
            # z += 0.0

        # insert at position zero because sometimes we already have splines
        # set alpha desired to zero - want to be facing down on table
        ArmState.GOTO.segments.insert(0, Goto5(np.reshape(idle_pos, (-1, 1)), 
                                               np.reshape([x, y, z, 0, beta], (-1, 1)), 
                                               ArmState.GOTO.duration,
                                               space='Task'))

        # Report.
        #self.get_logger().info("Going to point %r, %r, %r" % (x,y,z))

        
    
    # Send a command - called repeatedly by the timer.
    def sendcmd(self):        
        # self.get_logger().info("Current state %r" % self.arm_state)
        # Time since start
        time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        if self.position0 is None or \
            self.position is None or \
            self.q_des is None: # no start position or desired position yet
            return

        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']
        self.cmdmsg.velocity     = list(np.append(QDOT_INIT, GRIP_QDOT_INIT))
        
        # gravity compensation
        # cosine of shoulder angle minus (because they are oriented opposite) elbow angle
        tau_elbow = self.grav_elbow * np.cos(self.position[1] - self.position[2])
        tau_shoulder = -tau_elbow + self.grav_shoulder * np.cos(self.position[1])
        tau_grip = TAU_GRIP
        self.cmdmsg.effort       = list([0.0, tau_shoulder, tau_elbow, 0.0, 0.0, tau_grip])

        # Code for turning off effort to test gravity
        # nan = float("nan")
        # self.cmdmsg.position = (nan, nan, nan, nan, nan, nan)
        # self.cmdmsg.velocity = (nan, nan, nan, nan, nan, nan)
        # self.cmdpub.publish(self.cmdmsg)
        # return

        # self.get_logger().info("position: %r" % self.position)
        # self.get_logger().info("qdes: %r" % self.q_des)

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
        #         self.seg_start_time = time
        #         self.collided = False

        #         self.get_logger().info("COLLISION DETECTED")
            
        #     else:
        #         pass # do nothing if collision detected on hold or idle or start

        pd = None
        vd = None

        if self.arm_state == ArmState.START:
            # Evaluate p and v at time using the first cubic spline
            (q, qdot) = ArmState.START.segments[0].evaluate(time)
            (qgrip, qdotgrip) = ArmState.START.segments[1].evaluate(time)

            self.q_des = list(q)
            self.qdot_des = list(qdot)

            self.grip_q_des = qgrip
            self.grip_qdot_des = qdotgrip

            # evaluate gravity constants too
            self.grav_shoulder = ArmState.START.segments[2].evaluate(time)[0][0] # first index takes "position" from spline
            self.grav_elbow = ArmState.START.segments[2].evaluate(time)[0][1]

            if time >= ArmState.START.duration: # once dozne, moving to IDLE_POS
                self.arm_state = ArmState.RETURN
                self.seg_start_time = time
                
                ArmState.START.segments.pop(0) # remove the segment since we're done
                ArmState.START.segments.pop(0) # remove joint spline
                ArmState.START.segments.pop(0) # remove the gravity spline

                # set gravity constants to their final values
                self.grav_elbow = GRAV_ELBOW
                self.grav_shoulder = GRAV_SHOULDER

                # Sets up next spline since base and elbow joints start at arbitrary positions
                ArmState.RETURN.segments.append(Goto5(np.array(self.q_des), 
                                                      np.array(IDLE_POS), 
                                                      ArmState.RETURN.duration,
                                                      space='Joint'))
            
        elif self.arm_state == ArmState.RETURN:
            # Moving the base and elbow to waiting position
            (q, qdot) = ArmState.RETURN.segments[0].evaluate(time - self.seg_start_time)
            
            self.q_des = list(q)
            self.qdot_des = list(qdot)

            if time - self.seg_start_time >= ArmState.RETURN.duration:
                self.arm_state = ArmState.IDLE
                self.seg_start_time = time

                ArmState.RETURN.segments.pop(0) # remove the segment since we're done

        elif self.arm_state == ArmState.HOLD: 
            # Waiting at commanded point - end of previous spline
            (pd, vd) = ArmState.HOLD.segments[0].evaluate(time - self.seg_start_time)

            (qgrip, qdotgrip) = ArmState.HOLD.segments[1].evaluate(time - self.seg_start_time)
            self.grip_q_des = qgrip
            self.grip_qdot_des = qdotgrip

            if time - self.seg_start_time >= ArmState.HOLD.duration:
                # TEMPORARY CHANGE TO TEST ARM CAMERA. DELETE BELOW ONCE DONE
                # (pd, vd) = ArmState.HOLD.segments[0].evaluate(ArmState.HOLD.duration) # HOLD POSITION WITH ARM CLOSED
                # self.grip_q_des = CLOSED_GRIP
                # self.grip_qdot_des = GRIP_QDOT_INIT # KEEP GRIPPER CLOSED

                # TEMPORARY CHANGE TO TEST ARM CAMERA. UNCOMMENT BELOW ONCE DONE
                self.seg_start_time = time
                ArmState.HOLD.segments.pop(0) # remove the segment since we're done
                ArmState.HOLD.segments.pop(0) # remove the gripper segment since we're done
                
                if len(ArmState.GOTO.segments) > 0: # more places to go
                    self.arm_state = ArmState.GOTO
                else:
                    self.arm_state = ArmState.RETURN
                    ArmState.RETURN.segments.append(Goto5(np.array(self.q_des), 
                                                          np.array(IDLE_POS), 
                                                          ArmState.RETURN.duration,
                                                          space='Joint'))

#             

        elif self.arm_state == ArmState.GOTO:
            # Moving to commanded point
            (pd, vd) = ArmState.GOTO.segments[0].evaluate(time - self.seg_start_time)
            if time - self.seg_start_time >= ArmState.GOTO.duration:
                ## NON-TRACK ALIGN CODE

                # self.arm_state = ArmState.HOLD
                # self.seg_start_time = time

                # ArmState.GOTO.segments.pop(0) # remove the segment since we're done
                # self.collided = False # successfully finished a goto, reset collision boolean

                # # stay put during hold
                # ArmState.HOLD.segments.append(Hold(pd, 
                #                                    ArmState.HOLD.duration,
                #                                    space='Joint'))
                # # gripper closes
                # if abs(self.grip_q_des - IDLE_GRIP) < 0.1:
                #     ArmState.HOLD.segments.append(Goto5(np.array(self.grip_q_des),
                #                             CLOSED_GRIP,
                #                             ArmState.HOLD.duration,
                #                             space='Joint'))
                # else:
                #     ArmState.HOLD.segments.append(Goto5(np.array(self.grip_q_des),
                #                             IDLE_GRIP,
                #                             ArmState.HOLD.duration,
                #                             space='Joint'))

                '''NEW TRACK ALIGN CODE'''
                # gripper closes
                if abs(self.grip_q_des - IDLE_GRIP) < 0.1:
                    self.arm_state = ArmState.HOLD
                    self.seg_start_time = time

                    ArmState.GOTO.segments.pop(0) # remove the segment since we're done
                    self.collided = False # successfully finished a goto, reset collision boolean

                    # stay put during hold
                    ArmState.HOLD.segments.append(Hold(pd, 
                                                    ArmState.HOLD.duration,
                                                    space='Joint'))
                    ArmState.HOLD.segments.append(Goto5(np.array(self.grip_q_des),
                                            CLOSED_GRIP,
                                            ArmState.HOLD.duration,
                                            space='Joint'))
                else:
                    self.arm_state = ArmState.ALIGN
                    self.seg_start_time = time

                    ArmState.GOTO.segments.pop(0) # remove the segment since we're done
                    self.collided = False # successfully finished a goto, reset collision boolean

                    (ptip, _, _, _) = self.chain.fkin(self.q_des)

                    align_goal = self.align_position
                    align_goal = np.vstack((align_goal, pd[3][0], pd[4][0]))

                    ArmState.ALIGN.segments.append(Goto5(pd, align_goal,
                                                    ArmState.ALIGN.duration,
                                                    space='Task'))

                    ArmState.ALIGN.segments.append(Goto5(np.array(self.grip_q_des),
                                            CLOSED_GRIP, # TODO CHANGE TO IDLE TO OPEN
                                            ArmState.ALIGN.duration,
                                            space='Joint'))



        elif self.arm_state == ArmState.ALIGN:
            (pd, vd) = ArmState.ALIGN.segments[0].evaluate(time - self.seg_start_time)

            (qgrip, qdotgrip) = ArmState.ALIGN.segments[1].evaluate(time - self.seg_start_time)
            self.grip_q_des = qgrip
            self.grip_qdot_des = qdotgrip

            if time - self.seg_start_time >= ArmState.ALIGN.duration:
                self.arm_state = ArmState.DOWN
                self.seg_start_time = time

                ArmState.ALIGN.segments.pop(0) # remove the segment since we're done
                self.collided = False # successfully finished a goto, reset collision boolean

                down_goal = np.copy(pd)
                down_goal[2][0] -= (HOVER_HEIGHT + 0.01)

                ArmState.DOWN.segments.append(Goto5(pd, down_goal,
                                                ArmState.DOWN.duration,
                                                space='Task'))

                ArmState.DOWN.segments.append(Goto5(np.array(self.grip_q_des),
                                        CLOSED_GRIP,
                                        ArmState.ALIGN.duration,
                                        space='Joint'))
        
        elif self.arm_state == ArmState.DOWN:
            (pd, vd) = ArmState.DOWN.segments[0].evaluate(time - self.seg_start_time)

            (qgrip, qdotgrip) = ArmState.DOWN.segments[1].evaluate(time - self.seg_start_time)
            self.grip_q_des = qgrip
            self.grip_qdot_des = qdotgrip

            if time - self.seg_start_time >= ArmState.DOWN.duration:
                self.arm_state = ArmState.RELEASE
                self.seg_start_time = time
                ArmState.DOWN.segments.pop(0) # remove the segment since we're done

                ArmState.RELEASE.segments.append(Hold(pd, 
                                                ArmState.RELEASE.duration,
                                                space='Joint'))
                ArmState.RELEASE.segments.append(Goto5(np.array(self.grip_q_des),
                                        IDLE_GRIP,
                                        ArmState.RELEASE.duration,
                                        space='Joint'))
        
        elif self.arm_state == ArmState.RELEASE:
            # Waiting at commanded point - end of previous spline
            (pd, vd) = ArmState.RELEASE.segments[0].evaluate(time - self.seg_start_time)

            (qgrip, qdotgrip) = ArmState.RELEASE.segments[1].evaluate(time - self.seg_start_time)
            self.grip_q_des = qgrip
            self.grip_qdot_des = qdotgrip

            if time - self.seg_start_time >= ArmState.RELEASE.duration:
                # TEMPORARY CHANGE TO TEST ARM CAMERA. DELETE BELOW ONCE DONE
                (pd, vd) = ArmState.RELEASE.segments[0].evaluate(ArmState.HOLD.duration) # HOLD POSITION WITH ARM CLOSED
                self.grip_q_des = IDLE_GRIP
                self.grip_qdot_des = GRIP_QDOT_INIT # KEEP GRIPPER CLOSED

                # TEMPORARY CHANGE TO TEST ARM CAMERA. UNCOMMENT BELOW ONCE DONE

        elif self.arm_state == ArmState.IDLE:
            q = IDLE_POS
            qdot = QDOT_INIT
            self.q_des = list(q)
            self.qdot_des = list(qdot)

        else:
            self.get_logger().info("Arm in unknown state")
            self.arm_state = ArmState.RETURN
            self.seg_start_time = time

            # reset return just in case
            ArmState.RETURN.segments = [Goto5(np.array(self.q_des), 
                                              np.array(IDLE_POS), 
                                              ArmState.RETURN.duration,
                                              space='Joint')]


        if pd is not None: # we are using task space spline
            # run fkin on previous qdes
            (ptip_des, _, Jv, _) = self.chain.fkin(np.reshape(self.q_des, (-1, 1)))
            
            # extract alpha and beta directly from joint space
            alpha = self.q_des[1]-self.q_des[2]+self.q_des[3] 
            beta = self.q_des[0]-self.q_des[4] # base minus twist

            ptip_des = np.vstack((ptip_des, alpha, beta))

            vr   = vd + self.lam * ep(pd, ptip_des)

            # TODO hard-code these arrays in as constants somewhere since we use them to calculate alpha and beta in fkin
            Jv_mod = np.vstack((Jv, [0, 1, -1, 1, 0], [1, 0, 0, 0, -1]))

            Jinv = Jv_mod.T @ np.linalg.pinv(Jv_mod @ Jv_mod.T + gamma**2 * np.eye(5))
            qdot = Jinv @ vr # ikin result

            # new version that makes q depend solely on qdot instead of 
            # current position as well
            q = np.reshape(self.q_des, (-1, 1)) + qdot / RATE

            self.q_des = list(q.flatten())
            self.qdot_des = list(qdot.flatten())

            # self.get_logger().info("cmdpos: %r" % self.cmdmsg.position)
            # self.get_logger().info("cmdvel: %r" % self.cmdmsg.velocity)
            # self.get_logger().info("desired vel: %r" % qdot)
            # self.get_logger().info("Error: %r" % ep(pd, ptip_des))

        # print(np.reshape(self.position, (-1, 1)), "\n")

        # print(qdot.flatten(), "\n")
        # print(self.cmdmsg.position)

        self.cmdmsg.position = list(np.append(self.q_des, self.grip_q_des))
        self.cmdmsg.velocity = list(np.append(self.qdot_des, self.grip_qdot_des))

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
