#!/usr/bin/env python3
#
#   hebinode.py
#
#   Create the HEBI node.  This sets up the HEBI connection, then
#   passes then HEBI feedback to the /joint_states topic, while
#   transfering the /joint_commands topic to the HEBIs.
#
#   It maps the HEBI actuator names to joint names according to the
#   ROS parameters:
#
#     family    Name of the HEBI actuator family ('robotlab')
#     motors    List of motor names (['9.3', '9.6', ...])
#     joints    List of joint names (['base', 'joint2', 'eblow', ...])
#     rate      HEBI Feedback rate in Hertz
#     lifetime  HEBI command lifetime in milliseconds
#
try:
    import hebi
except:
    print("Unable to import HEBI!!!")

import numpy as np
import rclpy

from time      import sleep
from traceback import print_exc

from rclpy.node         import Node
from rclpy.parameter    import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType

from sensor_msgs.msg    import JointState


#
#   Definitions
#
FAMILY   = 'robotlab'
RATE     = 100.0        # Hertz
LIFETIME =  50.0        # ms

WATCHDOGDT = 0.2        # Watchdog Time step (seconds)


#
#   HEBI Node Class
#
class HebiNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

    # Startup
    def startup(self):
        # Check the testing mode.
        self.readtestmode()
        
        # Locate any HEBIs. Then grab the ROS parameters to confirm
        # which actuators to use for which joints.  Connect to them.
        # And finally set the feedback rate and command lifetime.
        if self.online:
            self.hebilookup()
        self.readconfiguration()
        if self.online:
            self.hebiconnect()
            self.hebirate(self.rate)
            self.hebilifetime(self.lifetime)

        # Feedback: Create a feedback message, a publisher to send the
        # joint states.
        self.fbktime     = self.get_clock().now()
        self.fbkmsg      = JointState()
        self.fbkmsg.name = self.joints   # Set to the joint names

        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Feedback Source: If online, create a HEBI feeedback handler
        # to drive this.  Else use a timer.
        if self.online:
            self.group.add_feedback_handler(self.feedbackCB)
        else:
            self.fbktimer = self.create_timer(1.0/self.rate, self.fakefbkCB)

        # Command: If online, create a HEBI command structure and pass
        # joint command messages there.  Else just store them.
        if self.online:
            self.cmd  = hebi.GroupCommand(self.group.size)
            self.nans = np.full(self.N, np.nan)
            self.sub  = self.create_subscription(
                JointState, '/joint_commands', self.commandCB, 10)
        else:
            self.cmdmsg = JointState()
            self.fbkmsg.name     = self.joints
            self.cmdmsg.position = [0.0] * self.N
            self.cmdmsg.velocity = [0.0] * self.N
            self.cmdmsg.effort   = [0.0] * self.N
            self.sub  = self.create_subscription(
                JointState, '/joint_commands', self.fakecmdCB, 10)

        # Finally create a watchdog timer to check the HEBI connection.
        self.watchdog = self.create_timer(WATCHDOGDT, self.watchdogCB)

        # raise Exception("End") # Test: end without spinning 

    # Shutdown
    def shutdown(self):
        # If online, clear the HEBI feedback handler.  Else destroy
        # the feedback timer.  Then kill the watchdog timer and node.
        if self.online:
            self.group.clear_feedback_handlers()
        else:
            self.destroy_timer(self.fbktimer)
        self.destroy_timer(self.watchdog)
        self.destroy_node()


    # Grab the ROS parameters.
    def getstring(self, name, default):
        # Declare the parameter, then get/return the value.
        type       = ParameterType.PARAMETER_STRING
        descriptor = ParameterDescriptor(type=type)
        self.declare_parameter(name, descriptor=descriptor, value=default)
        parameter  = self.get_parameter(name)
        return parameter.get_parameter_value().string_value
    
    def getstringarray(self, name):
        # Declare the parameter, then get/return the value.
        type       = ParameterType.PARAMETER_STRING_ARRAY
        descriptor = ParameterDescriptor(type=type)
        self.declare_parameter(name, descriptor=descriptor)
        parameter  = self.get_parameter(name)
        return parameter.get_parameter_value().string_array_value
    
    def getdouble(self, name, default):
        # Declare the parameter, then get/return the value.
        type       = ParameterType.PARAMETER_DOUBLE
        descriptor = ParameterDescriptor(type=type)
        self.declare_parameter(name, descriptor=descriptor, value=default)
        parameter  = self.get_parameter(name)
        return parameter.get_parameter_value().double_value
    
    def readtestmode(self):
        # Get the parameter and set the flags.
        self.testmode = self.getstring('testmode', 'off')
        self.online   = (self.testmode == 'off')
        self.tracking = (self.testmode == 'track')
        if not self.online:
            self.get_logger().info("RUNNING OFFLINE '%s'" % self.testmode)

    def readconfiguration(self):
        # Get the parameters.
        self.family   = self.getstring('family', FAMILY)
        self.motors   = self.getstringarray('motors')
        self.joints   = self.getstringarray('joints')
        self.rate     = self.getdouble('rate', RATE)
        self.lifetime = self.getdouble('lifetime', LIFETIME)
        self.N        = len(self.joints)

        # Check the parameters for consistency.
        if len(self.motors) == 0:
            self.get_logger().error("No motors specified!")
            raise Exception("Inconsistent ROS parameters")
        if len(self.joints) == 0:
            self.get_logger().error("No joints specified!")
            raise Exception("Inconsistent ROS parameters")
        if len(self.motors) != len(self.joints):
            self.get_logger().error("Unequal number of joints/motors specified!")
            raise Exception("Inconsistent ROS parameters")

        # Report the parameters.
        self.get_logger().info("Selecting...")
        self.get_logger().info("HEBI family  '%s'" % self.family)
        for i in range(self.N):
            self.get_logger().info("HEBI motor %ld '%s' = joint '%s'" %
                (i, self.motors[i], self.joints[i]))
        self.get_logger().info("HEBI update rate %3.0fHz" % self.rate)
        self.get_logger().info("HEBI command lifetime %3.0fms" % self.lifetime)


    # Locate the HEBI actuators
    def hebilookup(self):
        # Locate HEBI actuators on the network, waiting 1s for discovery.
        self.get_logger().info("Locating HEBIs...")
        self.lookup = hebi.Lookup()
        sleep(1)

        # Report the connected HEBIs.
        if sum(1 for _ in self.lookup.entrylist) == 0:
            self.get_logger().info("No HEBIs located.")
        else:
            for entry in self.lookup.entrylist:
                self.get_logger().info(
                    "Located family '%s' name '%s' at address %s" %
                    (entry.family, entry.name, entry.mac_address))

    # Connect to the HEBI group to send commands/receive feedback.
    def hebiconnect(self):
        # Use the HEBI lookup to create the actuator group.
        self.group = self.lookup.get_group_from_names([self.family], self.motors)
        if self.group is None:
            self.get_logger().error("Unable to connect to selected motors!")
            self.get_logger().error("Make sure motors are powered/connected.")
            self.get_logger().error("(for example using the Scope application)")
            raise Exception("No Motor Connection")
        else:
            self.get_logger().info("Connected to HEBIs.")

    # Set the HEBI command lifetime (in milliseconds).
    def hebilifetime(self, lifetime):
        if lifetime > 0.0:
            self.group.command_lifetime = lifetime

    # Set the HEBI feedback frequency (in Hertz).
    def hebirate(self, rate):
        if rate > 0.0:
            self.group.feedback_frequency = rate


    # HEBI feedback callback - send feedback on ROS message.
    def feedback(self, position, velocity, effort):
        # Grab the current time.
        self.fbktime = self.get_clock().now()

        # Build up the message and publish.  The joint names are preset.
        self.fbkmsg.header.stamp = self.fbktime.to_msg()
        self.fbkmsg.position     = position
        self.fbkmsg.velocity     = velocity
        self.fbkmsg.effort       = effort
        self.pub.publish(self.fbkmsg)

        # Reset the watchdog.
        self.watchdog.reset()

    def feedbackCB(self, fbk):
        # Send the HEBI feedback info.
        self.feedback(fbk.position.tolist(),
                      fbk.velocity.tolist(),
                      fbk.effort.tolist())

    def fakefbkCB(self):
        # Echo the last command message back.
        self.feedback(self.cmdmsg.position,
                      self.cmdmsg.velocity,
                      self.cmdmsg.effort)

    # JointState message callback - send commands to HEBI or save.
    def cmdmsgerror(self, cmdmsg):
        # Check the message names matching the joint names.
        if not (cmdmsg.name == self.joints):
            self.get_logger().warn("Joint commands not matching joint names!")
            return True

        # Check the command dimensions.
        l  = self.N
        lp = len(cmdmsg.position)
        lv = len(cmdmsg.velocity)
        le = len(cmdmsg.effort)
        if not (lp==0 or lp==l) or not (lv==0 or lv==l) or not (le==0 or le==l):
            self.get_logger().warn("Illegal length of pos/vel/eff commands!")
            return True

        # Else is ok.
        return False
    
    def commandCB(self, cmdmsg):
        # Check the command message.
        if self.cmdmsgerror(cmdmsg):
            return

        # Copy the commands.
        lp = len(cmdmsg.position)
        lv = len(cmdmsg.velocity)
        le = len(cmdmsg.effort)
        self.cmd.position = self.nans if (lp==0) else np.array(cmdmsg.position)
        self.cmd.velocity = self.nans if (lv==0) else np.array(cmdmsg.velocity)
        self.cmd.effort   = self.nans if (le==0) else np.array(cmdmsg.effort)

        # And send.
        self.group.send_command(self.cmd)

    def fakecmdCB(self, cmdmsg):
        # Check the command message.
        if self.cmdmsgerror(cmdmsg):
            return

        # Save if tracking, save.
        if self.tracking:
            self.cmdmsg = cmdmsg

    # Watchdog callback
    def watchdogCB(self):
        self.get_logger().warn("Not getting HEBI feedback - check connection")



#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the HEBI node.
    node = HebiNode('hebinode')

    # Run the startup and spin the node until interrupted.
    try:
        node.startup()
        rclpy.spin(node)
    except BaseException as ex:
        print("Ending due to exception: %s" % repr(ex))
        print_exc()

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
