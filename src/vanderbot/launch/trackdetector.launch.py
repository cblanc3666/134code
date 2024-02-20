"""Launch the USB camera node and track detector.


"""

import os
import yaml
import xacro
from pathlib import Path

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node

# from camera_config import CameraConfig, USB_CAM_DIR
USB_CAM_DIR = pkgdir('usb_cam')

with open(Path(USB_CAM_DIR, 'config', 'cam_params.yaml'), 'r') as stream:
    params =  yaml.safe_load(stream)
    ceil_cam_params = params['ceil_cam_params']
    arm_cam_params = params['arm_cam_params']

#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure the ceiling camera node
    node_ceilcam = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'ceilcam',
        output     = 'screen',
        parameters = ceil_cam_params,
    )

    # Configure the arm camera node
    node_armcam = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'armcam',
        output     = 'screen',
        parameters = arm_cam_params,
    )

    # Configure the demo detector node
    node_trackdetector = Node(
        name       = 'trackdetector', 
        package    = 'vanderbot',
        executable = 'trackdetector',
        output     = 'screen',
        remappings = [('/ceil_image_raw', '/ceilcam/image_raw'),
                      ('/arm_image_raw', '/armcam/image_raw')])


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        node_ceilcam,
        node_armcam,
        node_trackdetector,
        # TODO need to add node for end effector camera
    ])
