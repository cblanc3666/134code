#!/usr/bin/env python3
#
#   demo134.py
#
#   Demonstration node to interact with the HEBIs.
#
import numpy as np
import rclpy

from rclpy.node                 import Node
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point

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

# Target position for each spline segment
END_POS = [[0.0, 0.0, 0.0],
           [0.0, 0.0, -np.pi/2]]

# Initial joint velocity (should be zero)
QDOT_INIT = [0.0, 0.0, 0.0]

# Duration for each spline segment
# DURATIONS[3] = Hold time at commanded point
DURATIONS = [5.0, 3.0, 6.0, 3.0, 6.0]

#
#   DEMO Node Class
#
class DemoNode(Node):
    position = None
    q_des = None # desired joint positions 
    qdot_des = None # desired joint velocities
    start_time = 0
    msg_time = -sum(DURATIONS)
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
                                     np.array([self.position0[0], END_POS[0][1], self.position0[2]]),
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


    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        # Just print the position (for now).
        self.position = np.array(list(fbkmsg.position))

    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z

        time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        if (time - self.msg_time < sum(DURATIONS[2:5])):
            self.get_logger().info("Already commanded!")
            return
        
        self.msg_time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        
        # Go to command position JOINT SPACE
        (task_wait_pos, _, _, _) = self.chain.fkin(np.reshape(END_POS[1], (-1, 1)))
        self.segments[2] = GotoCubic(task_wait_pos, np.reshape([x, y, z], (-1, 1)), DURATIONS[2])
        # Return back
        self.segments[3] = GotoCubic(np.reshape([x, y, z], (-1, 1)), task_wait_pos, DURATIONS[4])

        # Report.
        self.get_logger().info("Going to point %r, %r, %r" % (x,y,z))
    
    def cb_number(self, msg):
        self.grav = msg.data
        self.get_logger().info("Received: %r gravity" % msg.data)

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Time since start
        time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        
        if self.position0 is None or \
            self.position is None or \
            self.q_des is None: # no start position or desired position yet
            return

        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow']
        self.cmdmsg.velocity     = [0.0, 0.0, 0.0]
        self.cmdmsg.effort       = [0.0, self.grav * -np.sin(self.position[1]), 0.0]

        # nan = float("nan")
        # self.cmdmsg.position = (nan, nan, nan)
        # self.cmdmsg.velocity = (nan, nan, nan)
        # self.cmdpub.publish(self.cmdmsg)
        # return

        if time < DURATIONS[0]:
            # Evaluate p and v at time using the first cubic spline
            (p, v) = self.segments[0].evaluate(time)

            self.q_des = list(p)
            self.qdot_des = list(v)

            # Sets up next spline since base and elbow joints start at arbitrary positions
            self.segments[1] = GotoCubic(np.array(p), np.array(END_POS[1]), DURATIONS[1])
            
        elif time < sum(DURATIONS[0:2]):
            #Segement 2: Moving the base and elbow to waiting position
            (p, v) = self.segments[1].evaluate(time - DURATIONS[0])
            
            self.q_des = list(p)
            self.qdot_des = list(v)

            # indicates that 
            self.msg_time = time - sum(DURATIONS)

            (wait_pos, _, _, _) = self.chain.fkin(np.reshape(END_POS[1], (-1, 1)))
            ending_pos = wait_pos + np.reshape([0.0, -0.5, 0.2], (-1, 1))

        else:
            # run fkin on previous qdes
            (ptip_des, _, Jv, _) = self.chain.fkin(np.reshape(self.q_des, (-1, 1)))

            if time - self.msg_time < DURATIONS[2]:
                (pd, vd) = self.segments[2].evaluate(time - self.msg_time)
            elif time - self.msg_time < sum(DURATIONS[2:4]):
                (pd, vd) = self.segments[2].evaluate(DURATIONS[2])
            elif time - self.msg_time < sum(DURATIONS[2:5]):
                (pd, vd) = self.segments[3].evaluate(time - self.msg_time - sum(DURATIONS[2:4]))
            else:
                p = END_POS[1]
                v = [0.0, 0.0, 0.0]
                self.q_des = list(p)
                self.qdot_des = list(v)
                
                (pd, vd) = (None, None)
            
            if pd is not None: # we are using spline
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
        # print(time - self.msg_time)

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
