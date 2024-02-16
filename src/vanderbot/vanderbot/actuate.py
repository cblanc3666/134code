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

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion

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

    # TODO remove unused states
    START = 5.0, []  # initial state
    GOTO  = 3.0, []  # moving to a commanded point, either from IDLE_POS or a 
                    # previously commanded point
    RETURN = 3.0, [] # moving back to IDLE_POS, either because no other point has 
                    # been commanded or because the next point is too far from
                    # the previous point
                    # Returning ALWAYS uses a joint space spline
    IDLE = None, []  # nothing commanded, arm is at IDLE_POS
    HOLD = 3, []   # holding at a commanded point, preparing to return or move to
                    # next point


# Holding position over the table
IDLE_POS = [0.0, np.pi/2, 1.8, 0.0, 0.0, 0.0] # TODO tweak this

# Initial joint velocity (should be zero)
QDOT_INIT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# magnitude of the joint space divergence (||real - des||) that constitutes a 
# collision
Q_COLLISION_THRESHOLD = 0.07 # TODO
QDOT_COLLISION_THRESHOLD = 0.5 # TODO

#
#   Vanderbot Node Class
#
class VanderNode(Node):
    # joints are in form ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']
    
    position =  None # real joint positions
    qdot =      None # real joint velocities
    effort =    None # real joint efforts

    q_des =     None # desired joint positions 
    qdot_des =  None # desired joint velocities
    start_time = 0 # time of initialization
    seg_start_time = 0  # logs the time (relative to start_time) that last segment started
                        # or, if state is IDLE, time that it has been idle
    arm_state = ArmState.START # initialize state machine


    grav_elbow = -6 # elbow torque constant
    grav_shoulder = 12.5 # shoulder torque constant

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()

        # Set the initial desired position to initial position so robot stay put
        self.q_des = self.position0
        self.qdot_des = QDOT_INIT
        
        self.get_logger().info("Initial positions: %r" % self.position0)

        # Start the clock
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # Set up first spline (only shoulder is moving)
        ArmState.START.segments.append(Goto5(np.array(self.position0), 
                                     np.array([self.position0[0], IDLE_POS[1], self.position0[2], self.position0[3], self.position0[4], self.position0[5]]),
                                     ArmState.START.duration,
                                     space='Joint'))

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

        # TODO subscribe to track messages
        
        self.fbksub = self.create_subscription(
            Point, '/point', self.recvpoint, 10)
        
        # #Create subscriber to circle message
        # self.circle_pos = None
        # self.fbkcirc = self.create_subscription(Point, '/Circle', self.recvCircle, 10)

        # #Create subscriber to rectangle message
        # self.rect_pos = None
        # self.fbkrect = self.create_subscription(Pose, '/Rectangle', self.recvRect, 10)


        # Report.
        self.get_logger().info("Running %s" % name)
        self.chain = KinematicChain('world', 'tip', self.jointnames()) # TODO add tip to URDF

        
        # Pick the convergence bandwidth.
        self.lam = 20.0

    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['base', 'shoulder', 'elbow', 'wrist', 'twist']

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Grab a single feedback - do not call this repeatedly.
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
        self.position = np.array(list(fbkmsg.position))
        self.qdot = np.array(list(fbkmsg.velocity))
        self.effort = np.array(list(fbkmsg.effort))

    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        
        self.gotopoint(x, y, z)

    def gotopoint(self, x, y, z):
        if self.arm_state != ArmState.IDLE: 
            # self.get_logger().info("Already commanded!")
            return

        self.seg_start_time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        self.arm_state = ArmState.GOTO
        
        # Go to command position JOINT SPACE
        (idle_pos, _, _, _) = self.chain.fkin(np.reshape(IDLE_POS, (-1, 1)))
        
        # insert at position zero because sometimes we already have splines
        ArmState.GOTO.segments.insert(0, Goto5(idle_pos, 
                                               np.reshape([x, y, z], (-1, 1)), 
                                               ArmState.GOTO.duration,
                                               space='Task'))

        # Report.
        #self.get_logger().info("Going to point %r, %r, %r" % (x,y,z))
    
    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        self.get_logger().info(str(self.arm_state))
        
        # self.get_logger().info("Current state %r" % self.arm_state)
        # Time since start
        time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        if self.position0 is None or \
            self.position is None or \
            self.q_des is None: # no start position or desired position yet
            return

        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']
        self.cmdmsg.velocity     = QDOT_INIT
        
        # gravity compensation
        # cosine of shoulder angle minus (because they are oriented opposite) elbow angle
        tau_elbow = self.grav_elbow * np.cos(self.position[1] - self.position[2])
        tau_shoulder = -tau_elbow + self.grav_shoulder * np.cos(self.position[1])
        self.cmdmsg.effort       = [0.0, tau_shoulder, tau_elbow, 0.0, 0.0, 0.0]

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
                
        #         # stay put, then try to go home
        #         ArmState.HOLD.segments.append(Hold(ptip, 
        #                                            ArmState.HOLD.duration,
        #                                            space='Joint'))
                
        #         ArmState.RETURN.segments = [] # clear it out just in case
        #         ArmState.GOTO.segments = []
        #         ArmState.RETURN.segments.append(Goto5(np.array(self.position), 
        #                                               np.array(IDLE_POS), 
        #                                               ArmState.RETURN.duration,
        #                                               space='Joint'))

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

            self.q_des = list(q)
            self.qdot_des = list(qdot)

            if time >= ArmState.START.duration: # once done, moving to IDLE_POS
                self.arm_state = ArmState.RETURN
                self.seg_start_time = time
                
                ArmState.START.segments.pop(0) # remove the segment since we're done

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
            
            if time - self.seg_start_time >= ArmState.HOLD.duration:
                self.seg_start_time = time
                ArmState.HOLD.segments.pop(0) # remove the segment since we're done
                
                if len(ArmState.GOTO.segments) > 0: # more places to go
                    self.arm_state = ArmState.GOTO
                else:
                    self.arm_state = ArmState.RETURN
                    ArmState.RETURN.segments.append(Goto5(np.array(self.q_des), 
                                                          np.array(IDLE_POS), 
                                                          ArmState.RETURN.duration,
                                                          space='Joint'))

        elif self.arm_state == ArmState.GOTO:
            # Moving to commanded point
            (pd, vd) = ArmState.GOTO.segments[0].evaluate(time - self.seg_start_time)
            if time - self.seg_start_time >= ArmState.GOTO.duration:
                self.arm_state = ArmState.HOLD
                self.seg_start_time = time

                ArmState.GOTO.segments.pop(0) # remove the segment since we're done
                self.collided = False # successfully finished a goto, reset collision boolean

                # stay put during hold
                ArmState.HOLD.segments.append(Hold(pd, 
                                                   ArmState.HOLD.duration,
                                                   space='Joint'))

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
            
            vr   = vd + self.lam * ep(pd, ptip_des)

            Jinv = Jv.T @ np.linalg.pinv(Jv @ Jv.T + gamma**2 * np.eye(3))
            qdot = Jinv @ vr ## ikin result TODO 

            # old version that mixed ikin feedback loop with motor fdbk loop
            # q = np.reshape(self.position, (-1, 1)) + qdot / RATE 
            
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

        self.cmdmsg.position = self.q_des
        self.cmdmsg.velocity = self.qdot_des

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
