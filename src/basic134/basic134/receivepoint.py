#!/usr/bin/env python3
#
#   receivepoint.py
#
#   Demonstration node to interact with the HEBIs, made to handle multiple 
#   points from camera.
#
import numpy as np
import rclpy

from enum import Enum

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Pose, Quaternion

from basic134.Segments          import GotoCubic
from basic134.KinematicChain    import KinematicChain
from basic134.TransformHelpers  import *
from std_msgs.msg               import Float32

# ros2 topic pub -1 /point geometry_msgs/msg/Point "{x: 0.2, y: 0.3, z: 0.1}"

#
#   Definitions
#
RATE = 100.0            # transmit rate, in Hertz
gamma = 0.1

#   States the arm can be in
class ArmState(Enum):
    START = 1   # initial state
    GOTO  = 2   # moving to a commanded point, either from IDLE_POS or a 
                # previously commanded point
    RETURN = 3  # moving back to IDLE_POS, either because no other point has 
                # been commanded or because the next point is too far from the
                # previous point
                # Returning ALWAYS uses a joint space spline
    IDLE = 4    # nothing commanded, arm is at IDLE_POS
    HOLD = 5    # holding at a commanded point, preparing to return or move to
                # next point


# Holding position over the table
IDLE_POS = [0.0, 0.0, -np.pi/2]

# Initial joint velocity (should be zero)
QDOT_INIT = [0.0, 0.0, 0.0]

# Duration for each spline segment
# DURATIONS[3] = Hold time at commanded point
DURATIONS = [5.0, 3.0, 6.0, 3.0, 6.0] # TODO refactor this

#
#   DEMO Node Class
#
class DemoNode(Node):
    position = None
    q_des = None # desired joint positions 
    qdot_des = None # desired joint velocities
    start_time = 0 # time of initialization
    seg_start_time = 0  # logs the time (relative to start_time) that last segment started
                        # or, if state is IDLE, time that it has been idle
    arm_state = ArmState.START
    
    # first segment splines from initial position to zero shoulder
    # second segment splines from any point to hold position
    # third segment splines from any point to any point
    # fourth segment splines from any point to hold position TODO refactor this out
    segments = [None, None, None, None]

    grav = 1.65

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
        self.segments[0] = GotoCubic(np.array(self.position0), 
                                     np.array([self.position0[0], IDLE_POS[1], self.position0[2]]),
                                     DURATIONS[0])

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
        
        #Create subscriber to circle message
        self.circle_pos = None
        self.fbkcirc = self.create_subscription(Point, '/Circle', self.recvCircle, 10)

        #Create subscriber to rectangle message
        self.rect_pos = None
        self.fbkrect = self.create_subscription(Pose, '/Rectangle', self.recvRect, 10)


        # Report.
        self.get_logger().info("Running %s" % name)
        self.chain = KinematicChain('world', 'tip', self.jointnames())

        self.numbersub = self.create_subscription(Float32, '/number', self.cb_number, 1)

        # Pick the convergence bandwidth.
        self.lam = 20.0

    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['base', 'shoulder', 'elbow']

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
    
    def recvRect(self, rec_Rect):
        x = rec_Rect.position.x
        y = rec_Rect.position.y
        z = rec_Rect.position.z

        theta = 2 * np.arcsin(rec_Rect.orientation.z)

        self.rect_pos = [x, y, z, theta]
        if x is not None:
            self.get_logger().info("Rectangle" + str(self.rect_pos))

    def recvCircle(self, rec_Circle):
        x = rec_Circle.x
        y = rec_Circle.y
        z = rec_Circle.z

        self.circle_pos = [x, y, z]
        if x is not None:
            self.get_logger().info("Circle" + str(self.circle_pos))


    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        # Just print the position (for now).
        self.position = np.array(list(fbkmsg.position))

    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z

        if self.arm_state != ArmState.IDLE: # TODO allow commands to be sent while currently running
            self.get_logger().info("Already commanded!")
            return

        self.seg_start_time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        self.arm_state = ArmState.GOTO
        
        # Go to command position JOINT SPACE TODO refactor this
        (idle_pos, _, _, _) = self.chain.fkin(np.reshape(IDLE_POS, (-1, 1)))
        self.segments[2] = GotoCubic(idle_pos, np.reshape([x, y, z], (-1, 1)), DURATIONS[2])
        # Return back
        self.segments[3] = GotoCubic(np.reshape([x, y, z], (-1, 1)), idle_pos, DURATIONS[4])

        # Report.
        self.get_logger().info("Going to point %r, %r, %r" % (x,y,z))
    
    def cb_number(self, msg):
        self.grav = msg.data
        self.get_logger().info("Received: %r gravity" % msg.data)

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Time since start
        time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        
        self.get_logger().info("Time: %r" % time)

        if self.position0 is None or \
            self.position is None or \
            self.q_des is None: # no start position or desired position yet
            return

        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow']
        self.cmdmsg.velocity     = [0.0, 0.0, 0.0]
        self.cmdmsg.effort       = [0.0, self.grav * -np.sin(self.position[1]), 0.0]

        # Code for turning off effort to test gravity
        # nan = float("nan")
        # self.cmdmsg.position = (nan, nan, nan)
        # self.cmdmsg.velocity = (nan, nan, nan)
        # self.cmdpub.publish(self.cmdmsg)
        # return

        pd = None
        vd = None

        if self.arm_state == ArmState.START:
            # Evaluate p and v at time using the first cubic spline
            (q, qdot) = self.segments[0].evaluate(time)

            self.q_des = list(q)
            self.qdot_des = list(qdot)

            if time >= DURATIONS[0]: # once done, moving to IDLE_POS
                self.arm_state = ArmState.RETURN
                self.seg_start_time = time
                
                # Sets up next spline since base and elbow joints start at arbitrary positions
                self.segments[1] = GotoCubic(np.array(self.q_des), np.array(IDLE_POS), DURATIONS[1])
            
        elif self.arm_state == ArmState.RETURN:
            # Moving the base and elbow to waiting position
            (q, qdot) = self.segments[1].evaluate(time - self.seg_start_time)
            
            self.q_des = list(q)
            self.qdot_des = list(qdot)

            if time - self.seg_start_time >= DURATIONS[1]:
                self.arm_state = ArmState.IDLE
                self.seg_start_time = time

        elif self.arm_state == ArmState.HOLD: 
            # Waiting at commanded point - end of previous spline
            (pd, vd) = self.segments[2].evaluate(DURATIONS[2])
            
            if time - self.seg_start_time >= DURATIONS[3]:
                self.arm_state = ArmState.RETURN
                self.seg_start_time = time
                self.segments[1] = GotoCubic(np.array(self.q_des), np.array(IDLE_POS), DURATIONS[1])

        elif self.arm_state == ArmState.GOTO:
            # Moving to commanded point
            (pd, vd) = self.segments[2].evaluate(time - self.seg_start_time)

            if time - self.seg_start_time >= DURATIONS[2]:
                self.arm_state = ArmState.HOLD
                self.seg_start_time = time

        elif self.arm_state == ArmState.IDLE:
            q = IDLE_POS
            qdot = [0.0, 0.0, 0.0]
            self.q_des = list(q)
            self.qdot_des = list(qdot)

        else:
            self.get_logger().info("Arm in unknown state")
            self.arm_state = ArmState.RETURN
            self.seg_start_time = time
            self.segments[1] = GotoCubic(np.array(self.q_des), np.array(IDLE_POS), DURATIONS[1])


        if pd is not None: # we are using task space spline
            # run fkin on previous qdes
            (ptip_des, _, Jv, _) = self.chain.fkin(np.reshape(self.q_des, (-1, 1)))
            
            vr   = vd + self.lam * ep(pd, ptip_des)

            Jinv = Jv.T @ np.linalg.pinv(Jv @ Jv.T + gamma**2 * np.eye(3))
            qdot = Jinv @ vr ## ikin result

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
            self.get_logger().info("Error: %r" % ep(pd, ptip_des))

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
    node = DemoNode('demo')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
