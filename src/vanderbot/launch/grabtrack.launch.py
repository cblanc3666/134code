"""Send commands to motors and receive data from them

   This should start
     1) RVIZ, ready to view the robot
     2) The robot_state_publisher (listening to /joint_commands)
     3) The HEBI node to communicate with the motors
     4) Scripts to issue commands

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('vanderbot'), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir('vanderbot'), 'urdf/vanderbot.urdf') # TODO UPDATE
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher_ACTUAL = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    node_robot_state_publisher_COMMAND = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}],
        remappings = [('/joint_states', '/joint_commands')])

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    # Configure a node for the hebi interface.  Note the 200ms timeout
    # is useful as the GUI only runs at 10Hz.
    node_hebi_SLOW = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']},
                      {'joints':   ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']},
                      {'lifetime': 200.0}],
        on_exit    = Shutdown())

    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']},
                      {'joints':   ['base', 'shoulder', 'elbow', 'wrist', 'twist', 'gripper']}],
        on_exit    = Shutdown())

    # Runs code to move actuators and receive feedback from them
    node_actuate = Node(
        name       = 'actuate', 
        package    = 'vanderbot',
        executable = 'actuate',
        output     = 'screen')

    # Configure a node for the GUI to command the robot.
    node_gui = Node(
        name       = 'gui', 
        package    = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        output     = 'screen',
        remappings = [('/joint_states', '/joint_commands')],
        on_exit    = Shutdown())
    
    node_usbcam = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        # param_path=Path(USB_CAM_DIR, 'config', 'params_1.yaml'),
        parameters = [{'camera_name':         'logitech'},
                      {'video_device':        '/dev/video0'},
                      {'pixel_format':        'yuyv2rgb'},
                      {'image_width':         640},
                      {'image_height':        480},
                      {'framerate':           15.0},
                      {'brightness':          175}, # -1
                      {'contrast':            150}, # -1
                      {'saturation':          128}, # -1
                      {'sharpness':           200}, # -1
                      {'gain':                1}, # -1
                      {'auto_white_balance':  False},
                      {'white_balance':       4000},
                      {'autoexposure':        False},
                      {'exposure':            100},
                      {'autofocus':           True},
                      {'focus':               -1}]
    )

    # Configure the demo detector node
    node_trackdetector = Node(
        name       = 'trackdetector', 
        package    = 'vanderbot',
        executable = 'trackdetector',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Run the robot AND watch the COMMANDS.
        node_robot_state_publisher_COMMAND,
        node_rviz,
        node_hebi,
        
        node_usbcam,
        node_trackdetector,

        node_actuate,
    ])
