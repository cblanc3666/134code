import bisect
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg  import Point, Pose2D, Pose, Quaternion, Polygon, Point32, PoseArray

class Track:
    def __init__(self, pose, track_type):
        """
        Inputs: Pose (Pose message)
                Track Type (String) (Either Left, Right, or Straight)
        """
        self.pose = pose
        self.track_type = track_type

    def __str__(self):
        string_1 = f"Pos: {(self.pose.position.x, self.pose.position.y)}, "
        string_2 = f"Angle: {self.pose.orientation.z, self.pose.orientation.w}"
        return string_1
    
    def __eq__(self, other): 
        if not isinstance(other, Track):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.pose.position.x == other.pose.position.x and self.pose.position.y == other.pose.position.y

class GridNode:
    WALL      = -1      # Not a legal state - just to indicate the wall
    UNKNOWN   =  0      # "Air"
    ONDECK    =  1      # "Leaf"  
    PROCESSED =  2      # "Trunk"
    PATH      =  3      # Processed and later marked as on path to goal

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.status = self.UNKNOWN
        self.creach = 0.0
        self.cost = 0.0
        self.parent = None
        self.neighbors = []
        self.angle = None #either -60, 0, or 60 degrees. represents the angle of the line the node is on

    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other): 
        if not isinstance(other, GridNode):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.x == other.x and self.y == other.y

    # Define the Euclidean distance.
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def midpoint(self, other):
        return (0.5 * (self.x + other.x), 0.5 * (self.y + other.y))
    
    def __str__(self):
        return f"({self.x} , {self.y})"
    
class HexagonalGrid:

    X_MIN = -0.762
    X_MAX = 0.762
    Y_MIN = 0
    Y_MAX = 0.762
    TRACK_LENGTH = 0.15

    def __init__(self, start, goal):
        """
        Intialize hexagonal grid.
        Inputs: 
            start: (x, y) tuple that represents the starting position
            goal: (x, y) tuple that represents the ideal goal position

        Parameters:
            self.start_node: self.start as a Node type
            self.nodes: a list of lists filled with Node types
            self.num_rows: number of rows in hexagonal grid
            self.num_cols: number of cols in hexagonal grid
            self.goal_node: closest node to self.goal that is in self.nodes
        """
        self.start = start
        self.goal = goal

        self.points = self.generate_points()
        self.segments = self.get_segments()
        
        self.num_rows = len(self.points)
        self.num_cols = len(self.points[0])

        self.nodes = self.generate_nodes()
        self.add_neighbors()

        self.start_node, _, _ = self.closest_node(start)
        self.goal_node, _, _ = self.closest_node(goal)

    def generate_points(self):
        """
        Generate all points of hexagonal grid based on starting position
        """
        #Initialize final list (will eventually be self.nodes)
        all_points = []

        #Intialize above and below points lists
        above_points = []
        below_points = []

        #Create starting corner point
        start_x = self.start[0] + self.TRACK_LENGTH / (4 * np.sqrt(3))
        start_y = self.start[1] + self.TRACK_LENGTH / 4

        m = 1
        if start_x > self.X_MAX and start_y > self.Y_MAX:
            start_x = self.start[0] - self.TRACK_LENGTH / (4 * np.sqrt(3))
            start_y = self.start[1] + self.TRACK_LENGTH / 4
        elif start_x > self.X_MAX:
            start_x -= 2 * self.TRACK_LENGTH / np.sqrt(3)
            m += 1
        elif start_y > self.Y_MAX:
            start_y -= self.TRACK_LENGTH

        start_pt = (start_x, start_y)

        #Create starting above nodes
        while True:
            x = start_pt[0] + 0.5 * self.TRACK_LENGTH * ((len(above_points) + 1) % 2) / np.sqrt(3) * (-1) ** m
            y = start_pt[1] + 0.5 * (len(above_points) + 1) * self.TRACK_LENGTH
            if y > self.Y_MAX:
                break
            above_points.append((x, y))
        
        #Create starting below nodes
        while True:
            x = start_pt[0] + 0.5 * self.TRACK_LENGTH * ((len(below_points) + 1) % 2) / np.sqrt(3) * (-1) ** m
            y = start_pt[1] - 0.5 * (len(below_points) + 1) * self.TRACK_LENGTH
            if y < self.Y_MIN:
                break
            below_points.append((x, y))
        
        #Concatenate above and below nodes to a single list with starting node
        starting_nodes = above_points[::-1] + [start_pt] + below_points

        for starting_pt in starting_nodes:
            #Starting x & y position for all nodes in iteration
            cur_x = starting_pt[0]
            y = starting_pt[1]

            #Initialize right and left nodes lists
            right_pts = []
            left_pts = []

            #Determine whether x_coord of first node is 2 * TRACK_LENGTH or TRACK_LENGTH away
            start_pos = 0
            if cur_x == start_x and m != 1:
                start_pos = 1
            elif cur_x != start_x and m == 1:
                start_pos = 1

            #Generate right nodes
            while True:
                if len(right_pts) == 0:
                    next_x = cur_x + self.TRACK_LENGTH * (1 + start_pos) / np.sqrt(3)
                else:
                    next_x = right_pts[-1][0] + self.TRACK_LENGTH * (1 + ((len(right_pts) + start_pos) % 2)) / np.sqrt(3)
                if next_x > self.X_MAX:
                    break
                right_pts.append((next_x, y))


            #Generate left nodes
            while True:
                if len(left_pts) == 0:
                    next_x = cur_x - self.TRACK_LENGTH * (1 + ((start_pos + 1) % 2)) / np.sqrt(3)
                else:
                    next_x = left_pts[-1][0] - self.TRACK_LENGTH * (1 + ((len(left_pts) + start_pos + 1) % 2)) / np.sqrt(3)
                if next_x < self.X_MIN:
                    break
                left_pts.append((next_x, y))

            #Concatenate lists (with starting node) and add to all_nodes list
            all_points.append(left_pts[::-1] + [starting_pt] + right_pts)

        return all_points
    
    def get_segments(self):
        segs = []
        for i in range(len(self.points)):
            for j in range(len(self.points[i])):
                pt1 = self.points[i][j]
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        try:
                            pt2 = self.points[i + di][j + dj]
                            dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                            if abs(dist - self.TRACK_LENGTH / np.sqrt(3)) < 0.01:
                                seg = ([pt1[0], pt2[0]], [pt1[1], pt2[1]])
                                seg_rev = ([pt2[0], pt1[0]], [pt2[1], pt1[1]])
                                if seg_rev not in segs:
                                    segs.append(seg)  
                        except IndexError:
                            continue                       
        
        return segs
    
    def add_neighbors(self):
        def add(i, j, di, dj):
            try:
                if i + di >= 0 and j + dj >= 0:
                    self.nodes[i][j].neighbors.append(self.nodes[i + di][j + dj])
            except IndexError:
                pass

        dist = self.TRACK_LENGTH * np.sqrt(3) / 2
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                if abs(self.nodes[i][j].angle) < 0.001:
                    for di in [-4, 4]:
                        add(i, j, di, 0)
                
                else:
                    for di, dj in [(-2, int(-np.sign(self.nodes[i][j].angle))), (2, int(np.sign(self.nodes[i][j].angle)))]:
                        add(i, j, di, dj)
                    for di, dj in [(0, -1), (0, 1)]:
                        add(i, j, di, dj)
                for di in [-3, 3]:
                    try:
                        if i + di >= 0:
                            for j_prime in range(len(self.nodes[i + di])):
                                if abs(self.nodes[i + di][j_prime].distance(self.nodes[i][j]) - dist) < 0.01:
                                    self.nodes[i][j].neighbors.append(self.nodes[i + di][j_prime])
                    except IndexError:
                        continue
               

    def generate_nodes(self):
        if len(self.segments) == 0:
            pass
        midpts = [(0.5 * sum(seg[0]), 0.5 * sum(seg[1])) for seg in self.segments]
        angles = [np.arctan((seg[1][1] - seg[1][0]) / (seg[0][1] - seg[0][0])) for seg in self.segments]   
        node_list = [GridNode(pt[0], pt[1]) for pt in midpts]
        nodes_ordered = []

        for (i, node) in enumerate(node_list):
            node.angle = angles[i]

        def sort_x(node):
            return node.x

        def sort_y(node):
            return node.y
        
        node_list.sort(key = sort_y, reverse = True)

        nodes_dict = {}
        for node in node_list:
            if node.y not in nodes_dict.keys():
                nodes_dict[node.y] = [node]
            else:
                nodes_dict[node.y].append(node)
        
        for nodes_y_list in nodes_dict.values():
            sorted_list = nodes_y_list
            sorted_list.sort(key = sort_x)
            nodes_ordered.append(sorted_list)

        return nodes_ordered
        
    
    def closest_node(self, loc):
        """
        Only used to find start node.
        """
        if len(self.nodes) == 0:
            pass
        temp_node = GridNode(loc[0], loc[1])
        closest_dist = None
        iter = 0
        idx = 0
        idy = 0
        for (i, node_list) in enumerate(self.nodes):
            for (j, node) in enumerate(node_list):
                if iter == 0:
                    closest_dist = temp_node.distance(node)
                    iter += 1
                else:
                    dist = temp_node.distance(node)
                    if dist < closest_dist:
                        closest_dist = dist
                        idx = i
                        idy = j

        return self.nodes[idx][idy], idx, idy
    
    def display(self, show = False):
        """
        Plots all nodes. Start is red. Goal is blue.
        """
        plt.figure(figsize = (14, 7))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim([self.X_MIN, self.X_MAX])
        plt.ylim([self.Y_MIN, self.Y_MAX])

        for seg in self.segments:
            plt.plot(seg[0], seg[1], color = "#808080", lw = 0.5)

        for node_list in self.nodes:
            for node in node_list:
                plt.scatter(node.x, node.y, s= 3, c = "k")
            
        plt.scatter(self.start_node.x, self.start_node.y, s = 20, c = "r")    
        plt.scatter(self.goal_node.x, self.goal_node.y, s = 20, c = "b")
        if show:
            plt.show()

class Planner:
    DISTANCE_WEIGHT = 0.1
    NO_THIRTY_DEG_TURNS = 1000  #Try to elimnate any 30-degree turns
    NEIGHBORS_WEIGHT = 0.01 #Don't go to nodes with fewer neighbors
    ANGLE_DIFF_WEIGHT = 5 
    TRACK_COLOR = {"Straight" : "#00CED1", "Left" : "#FF1493", "Right" : "#FFA500"}
    STATES = {"Straight" : 0.0, "Right" : 1.0, "Left" : -1.0}
    D = 0.0174

    def __init__(self, start, goal):
        self.num_blue_tracks = 0
        self.num_pink_tracks = 0
        self.num_orange_tracks = 0
        self.grid = HexagonalGrid(start, goal)
        self.final_nodes = self.astar()
        self.track_locations = None
        self.track_angles = None
        self.track_types = None
        self.get_track_info()
        self.tracks = self.create_tracks()

    def costtoreach(self, node):
        cost = node.parent.creach + 1
        cost += self.NEIGHBORS_WEIGHT * (6 - len(node.neighbors))
        #cost += self.ANGLE_DIFF_WEIGHT * (self.)

        return cost

    def costtogo(self, node):
        return self.DISTANCE_WEIGHT * node.distance(self.grid.goal_node) 

    def plot_path(self):
        if len(self.final_nodes) == 0:
            pass
        
        for i in range(len(self.final_nodes) - 1):
            xvals = [self.final_nodes[i].x, self.final_nodes[i + 1].x]
            yvals = [self.final_nodes[i].y, self.final_nodes[i + 1].y]
            plt.plot(xvals, yvals, color = self.TRACK_COLOR[self.track_types[i]])
            plt.scatter(self.final_nodes[i].x, self.final_nodes[i].y, s = 7, c = "k")
            plt.text(self.track_locations[i][0], self.track_locations[i][1], f"{round(self.track_angles[i], 2)}")
   
        plt.scatter(self.final_nodes[-1].x, self.final_nodes[-1].y, s = 7, c = "g")
        
    def get_orientation(self, node1, node2):
        mag_a = np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        mag_b = np.sqrt((node1.x - node1.parent.x) ** 2 + (node1.y - node1.parent.y) ** 2)
        a = (node1.x - node2.x) * (node1.x - node1.parent.x)
        b = (node1.y - node2.y) * (node1.y - node1.parent.y)

        val = (a + b) / mag_a / mag_b
        if val < -1.0:
            val = -1.0
        
        if val > 1.0:
            val = 1.0

        return np.arccos(val)
        

          
    def astar(self):
        # Prepare the still empty *sorted* on-deck queue.
        onDeck = []

        # Setup the start state/cost to initialize the algorithm.
        self.grid.start_node.status = GridNode.ONDECK
        self.grid.start_node.creach = 0.0
        self.grid.start_node.cost   = self.costtogo(self.grid.start_node)
        self.grid.start_node.parent = None
        bisect.insort(onDeck, self.grid.start_node)

        # Continually expand/build the search tree.

        while True:
            #Remove any neighbors that are 30 degrees away since those are impossible turns
            if onDeck[0].parent is not None:
                for (k, j) in enumerate(onDeck[0].neighbors):
                    orientation = self.get_orientation(onDeck[0], j)
                    if abs(abs(orientation) - np.pi / 6) < 0.001:
                        onDeck[0].neighbors.pop(k)

            for j in onDeck[0].neighbors:
                if j.status == GridNode.UNKNOWN:
                    j.status = GridNode.ONDECK
                    j.parent = onDeck[0]
                    j.creach = self.costtoreach(j)
                    j.cost = self.costtogo(j) + j.creach
                    bisect.insort(onDeck, j, 1)
            onDeck[0].status = GridNode.PROCESSED
            if self.grid.goal_node.status == GridNode.PROCESSED:
                break
            else:
                onDeck.remove(onDeck[0])
            #############

        # Create the path to the goal (backwards) and show.
        current = self.grid.goal_node
        current.status = GridNode.PATH
        final_nodes = []
        while current != self.grid.start_node:
            current.parent.status = GridNode.PATH
            final_nodes.append(current)
            current = current.parent
        
        final_nodes.append(self.grid.start_node)
        if self.grid.start_node.x > self.grid.goal_node.x:
            return final_nodes
        return final_nodes[::-1]
    
    def get_track_info(self):
        if len(self.final_nodes) == 0:
            pass

        track_types = []
        angles = []
        midpts = []
        for i in range(len(self.final_nodes) - 1):
            angle_diff = self.final_nodes[i + 1].angle - self.final_nodes[i].angle
            angle = np.arctan2(self.final_nodes[i + 1].y - self.final_nodes[i].y,
                            self.final_nodes[i + 1].x - self.final_nodes[i].x )
            if abs(angle_diff) <= 0.001:
                track_types.append("Straight")
            elif abs(angle_diff + np.pi / 3) < 0.001:
                if self.grid.start_node.x > self.grid.goal_node.x:
                    track_types.append("Right")
                else:
                    track_types.append("Left")
            elif abs(angle_diff - np.pi / 3) < 0.001:
                if self.grid.start_node.x > self.grid.goal_node.x:
                    track_types.append("Left")
                else:
                    track_types.append("Right")
            elif abs(angle_diff + 2 * np.pi / 3) < 0.001:
                if self.grid.start_node.x > self.grid.goal_node.x:
                    track_types.append("Left")
                else:
                    track_types.append("Right")
            else:
                if self.grid.start_node.x > self.grid.goal_node.x:
                    track_types.append("Right")
                else:
                    track_types.append("Left")

            midpts.append(self.final_nodes[i + 1].midpoint(self.final_nodes[i]))
            angles.append(angle)

        self.track_locations = midpts
        self.track_angles = angles
        self.track_types = track_types

    def create_posemsg(self, position, angle, track_type):
        posemsg = Pose()
        position_msg = Point()
        orientation_msg = Quaternion()
        position_msg.x = float(position[0])
        position_msg.y = float(position[1])
        position_msg.z = 0.0
        orientation_msg.x = self.STATES[track_type]
        orientation_msg.y = 0.0
        orientation_msg.z = float(np.sin(angle / 2))
        orientation_msg.w = float(np.cos(angle / 2))
        posemsg.position = position_msg
        posemsg.orientation = orientation_msg

        return posemsg


    def create_tracks(self):
        tracks = []
        for i in range(len(self.track_angles)):
            dx = 0
            dy = 0
            if self.grid.start_node.x > self.grid.goal_node.x:
                if self.track_types[i] == "Left":
                    dx += self.D * np.sin(self.track_angles[i])
                    dy += -self.D * np.cos(self.track_angles[i]) 
                elif self.track_types[i] == "Right":
                    dx += -self.D * np.sin(self.track_angles[i])
                    dy += self.D * np.cos(self.track_angles[i])
            else:
                if self.track_types[i] == "Right":
                    dx += self.D * np.sin(self.track_angles[i])
                    dy += -self.D * np.cos(self.track_angles[i]) 
                elif self.track_types[i] == "Left":
                    dx += -self.D * np.sin(self.track_angles[i])
                    dy += self.D * np.cos(self.track_angles[i])

            new_location = (self.track_locations[i][0] + dx, self.track_locations[i][1] + dy) 
            posemsg = self.create_posemsg(new_location,
                                        self.track_angles[i], self.track_types[i])
            new_track = Track(posemsg, self.track_types[i])
            tracks.append(new_track)
            if self.track_types[i] == "Straight":
                self.num_blue_tracks += 1
            elif self.track_types[i] == "Right":
                self.num_orange_tracks += 1
            else:
                self.num_pink_tracks += 1

        if self.grid.start_node.x > self.grid.goal_node.x:
            return tracks[::-1]
        return tracks



    
# if __name__ == "__main__":
#     start = (10, 23.4)
#     goal = (5.3, 10)

#     planner = Planner(start, goal)
#     planner.grid.display()
#     planner.plot_path()

#     plt.show()

    


        
