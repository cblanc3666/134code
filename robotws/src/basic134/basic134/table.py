#!/usr/bin/env python3
#
#   demo134.py
#
#   Demonstration node to interact with the HEBIs.
#
import numpy as np
from basic134.Segments  import GotoCubic
import rclpy

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from geometry_msgs.msg  import Point


#
#   Definitions
#
RATE = 100.0            # Hertz

# Target position for each spline segment
END_POS = [[0.0, 0.0, 0.0],
           [0.0, 0.0, -np.pi/2],
           [0.38, -0.91, -1.86]]

# Duration for each spline segment
DURATIONS = [4.0, 3.0, 5.0, 4.0]

#
#   DEMO Node Class
#
class DemoNode(Node):
    position = None
    start_pos = None
    start_time = 0
    segments = [None, None, None, None]

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()
        self.get_logger().info("Initial positions: %r" % self.position0)

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

        #Initialize segements for pointing at tabe and returning
        self.segments[2] = GotoCubic(np.array(END_POS[1]), np.array(END_POS[2]), DURATIONS[2])
        self.segments[3] = GotoCubic(np.array(END_POS[2]), np.array(END_POS[1]), DURATIONS[3])

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

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        #Intialize self.start_pos when robot gets very first position
        if self.position is not None and self.start_pos is None:
            self.start_pos = self.position
            self.start_time = self.get_clock().now().nanoseconds * 1e-9

            #Set up first spline (only shoulder is moving)
            self.segments[0] = GotoCubic(np.array(self.start_pos), 
                                          np.array([self.start_pos[0], END_POS[0][1], self.start_pos[2]]),
                                          DURATIONS[0])

        #Time since start
        time = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        
        if self.start_pos is None:
            return

        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow']
        self.cmdmsg.velocity     = [0.0, 0.0, 0.0]
        self.cmdmsg.effort       = [0.0, 0.0, 0.0]

        if time < DURATIONS[0]:
            # Evaluate p and v at time using the first cubic spline
            (p, v) = self.segments[0].evaluate(time)

            # Update position and velocity ROS message
            self.cmdmsg.position = list(p)
            self.cmdmsg.velocity = list(v)

            # Sets up next spline since base and elbow joints start at arbitrary positions
            self.segments[1] = GotoCubic(np.array(p), np.array(END_POS[1]), DURATIONS[1])
            
            # Publish commands, makes robot move
            self.cmdpub.publish(self.cmdmsg)
        elif time < sum(DURATIONS[0:2]):
            #Segement 2: Moving the base and elbow to waiting position
            (p, v) = self.segments[1].evaluate(time - DURATIONS[0])
            self.cmdmsg.position = list(p)
            self.cmdmsg.velocity = list(v)

            self.cmdpub.publish(self.cmdmsg)
        else:
            # Cyclic timer for pointing motion
            # time_table = 0 at waiting position
            # time_table = DURATIONS[2] when the tip hits point
            time_table = (time - sum(DURATIONS[0:2])) % sum(DURATIONS[2:4])
            if time_table < DURATIONS[2]:
                (p, v) = self.segments[2].evaluate(time_table)
            else:
                (p, v) = self.segments[3].evaluate(time_table - DURATIONS[2])
            self.cmdmsg.position = list(p)
            self.cmdmsg.velocity = list(v)
            self.cmdpub.publish(self.cmdmsg)

        # print(self.cmdmsg.position)




#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = DemoNode('table')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
