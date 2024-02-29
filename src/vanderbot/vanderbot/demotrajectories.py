import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools 
'''
CHOOSE CORRECT SEGMENT QUEUE IMPORT
'''
# from SegmentQueue import SegmentQueue, Spline
from SegmentQueueDelayed import SegmentQueue, JointSpline, star

waypointValues = np.array([[5, 5], [-7, 4], [2, -9]]) 

def spline(t, T, p0, pf, v0, vf):
    # Compute the parameters.
    a = p0
    b = v0
    c =   3*(pf-p0)/T**2 - vf/T    - 2*v0/T
    d = - 2*(pf-p0)/T**3 + vf/T**2 +   v0/T**2
    # Compute the current (p,v).
    p = a + b * t +   c * t**2 +   d * t**3
    v =     b     + 2*c * t    + 3*d * t**2
    return (p,v)


# REGULAR SEGMENT QUEUE
# SQ = SegmentQueue()
# SQ.enqueue(Spline(np.array([0,0]), np.array([5,5]), np.array([0, 0]), np.array([3, 3]), 3, 'Joint'))
# SQ.enqueue(Spline(np.array([5,5]), np.array([-7,4]), np.array([3, 3]), np.array([-12/4, 1/4]), 4, 'Joint'))
# SQ.enqueue(Spline(np.array([-7,4]), np.array([2,-9]), np.array([-12/4, 1/4]), np.array([9/3, -13/3]), 3, 'Joint'))

#SEGMENT QUEUE DELAYED 

Q_INIT = np.array([0, 0])
Q_DOT_INIT= np.array([0, 0])
T_INIT = 0
SQ = SegmentQueue(None) # Fkin none since we are only working in joint space
SQ.update(T_INIT, Q_INIT, Q_DOT_INIT)

q = Q_INIT
qdot = Q_DOT_INIT

# SQ.enqueue(JointSpline(np.array([5, 5]), np.array([3, 3]), 3))
# SQ.enqueue(JointSpline(np.array([-7, 4]), np.array([-12/4, 1/4]), 4))
# SQ.enqueue(JointSpline(np.array([2, -9]), np.array([9/3, -13/3]), 3))

# SEGMENT QUEUE DELAYED WITH MOTION PLANNER
SQ.enqueueList(star(np.array([1, 3]), 2))
SQ.enqueueList(star(np.array([5, 7]), 3))


# Define the trajectory function
def trajectory(t):
    
    # BASIC MODE 
    
    # if t < 3:
    #     p, v = spline(t, 3, np.array([0,0]), np.array([5,5]), np.array([0, 0]), np.array([3, 3]))
    
    # elif t < 7:
    #     p, v = spline(t - 3, 4, np.array([5,5]), np.array([-7,4]), np.array([3, 3]), np.array([-12/4, 1/4]))

    # elif t < 10:
    #     p, v = spline(t - 7, 3, np.array([-7,4]), np.array([2,-9]), np.array([-12/4, 1/4]), np.array([9/3, -13/3]))

    # else:
    #     p = (2, -9)

    # x = p[0]
    # y = p[1]

    # return x, y
         
    
    # Segment Queue   

    # SQ.update(t)
    # (x, y), v = SQ.evaluate()
    # return x, y
        
    # Segment Queue Delayed

    global q, qdot 

    SQ.update(t, q, qdot)
    q, qdot = SQ.evaluate()
    return q[0], q[1]

# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Initialize the point and trail
point, = ax.plot([], [], marker='o', color='r')
trail, = ax.plot([], [], linestyle='-', color='b', alpha=0.5)

# Initialize Waypoints
waypoints = ax.scatter(waypointValues[:, 0], waypointValues[:, 1], marker='*', color='y', s=200)

# Number of points in the trail
trail_length = 500
trail_x = np.zeros(trail_length)
trail_y = np.zeros(trail_length)

# Initialization function: plot the background of each frame
def init():
    point.set_data([], [])
    trail.set_data([], [])
    return point, trail

# Animation function: update the plot for each frame
def update(frame):
    global waypointValues

    x, y = trajectory(frame / 20)

    # Update point position
    point.set_data(x, y)
    
    # Update trail
    trail_x[1:] = trail_x[:-1]  # Shift x values
    trail_y[1:] = trail_y[:-1]  # Shift y values
    trail_x[0] = x
    trail_y[0] = y
    trail.set_data(trail_x, trail_y)

    if (len(waypointValues) > 0):
        distances = np.sqrt((x - waypointValues[:, 0])**2 + (y - waypointValues[:, 1])**2)
        ind, minDst = np.argmin(distances), min(distances)

        if minDst < 0.05:
            waypointValues = np.delete(waypointValues, ind, 0)
            # Update the scatter plot with the new data
            waypoints.set_offsets(waypointValues)  

    return point, trail, waypoints

# Create animation
ani = FuncAnimation(fig, update, frames=itertools.count(0),
                    init_func=init, blit=False, interval=20)

plt.title('Moving Point Along a Trajectory with Trail')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()