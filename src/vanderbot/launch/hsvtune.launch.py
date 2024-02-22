"""Launch the USB camera node and HSV tuning utility.



"""

import os
import yaml
from pathlib import Path
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node

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

    # Configure the USB camera node
    node_usbcam = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        parameters = arm_cam_params,
                      )

    # Configure the HSV tuning utility node
    node_hsvtune = Node(
        name       = 'hsvtuner', 
        package    = 'vanderbot',
        executable = 'hsvtune',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        node_usbcam,
        node_hsvtune,
    ])
