"""Launch the gamestate node.


"""

import os
import yaml
import xacro
from pathlib import Path

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node

def generate_launch_description():
    node_gamestate = Node(
        name       = 'gamestate', 
        package    = 'vanderbot',
        executable = 'gamestate',
        output     = 'screen')
    
    return LaunchDescription([
        # Start the nodes.
        node_gamestate
    ])