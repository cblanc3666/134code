"""Launch the USB camera node and track detector.


"""

import os
import xacro
from pathlib import Path

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node

#from camera_config import CameraConfig, USB_CAM_DIR
USB_CAM_DIR = pkgdir('usb_cam')

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
                      {'focus':               -1}],
        
    )

    # Configure the demo detector node
    node_trackdetector = Node(
        name       = 'trackdetector', 
        package    = 'detectors',
        executable = 'trackdetector',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        node_usbcam,
        node_trackdetector,
        # TODO need to add node for end effector camera
    ])
