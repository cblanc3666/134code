import copy 
import numpy as np 

def crossmat(e):
    e = e.flatten()
    return np.array([[  0.0, -e[2],  e[1]],
                     [ e[2],   0.0, -e[0]],
                     [-e[1],  e[0],  0.0]])

def Rote(e, alpha):
    ex = crossmat(e)
    return np.eye(3) + np.sin(alpha) * ex + (1.0-np.cos(alpha)) * ex @ ex

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

class JointSpline():
    '''    
    Joint Spline Class 

    Takes q, qdot from last known q, qdot before the spline runs
    '''

    def __init__(self, qf, qdotf, T) -> None:
        self.q0 = None 
        self.qf = qf 
        self.qdot0 = None 
        self.qdotf = qdotf 
        self.T = T 

        self.space = 'Joint' 

    def getSpace(self):
        '''
        Return if joint or task space
        '''
        return self.space 
    
    def evaluate(self, fkin, t):
        '''
        Compute the q, qdot of the given spline at time t.
        
        Inputs:
        t - time in seconds since the start of the current spline 

        Outputs:
        q, qdot - the position and the velocity 
        '''
        q, qdot = spline(t, self.T, self.q0, self.qf, self.qdot0, self.qdotf)          
        return q, qdot   
    
    def completed(self, t):
        '''
        Returns true if the spline is completed, false otherwise 
        '''
        return t > self.T

    def calculateParameters(self, q, qdot, fkin): 
        '''
        Fills up the q0 & qdot0 parameters. fkin is useless, but is passed in 
        since the tip spline needs it. 
        '''
        self.q0 = q.copy() 
        self.qdot0 = qdot.copy()

class TaskSpline():
    '''
    A task space spline. The inputs are pf, vf, Rf and T.
    '''
    def __init__(self, pf, vf, Rf, T) -> None:
        self.p0 = None 
        self.pf = pf 
        self.v0 = None 
        self.vf = vf 
        self.T = T 

        self.space = 'Tip' 

    def getSpace(self):
        '''
        Returns the space of the spline
        '''
        return self.space 
    
    def evaluate(self, fkin, t): # TODO fix this
        '''
        Compute the p, v of the given spline.
        
        Inputs:
        t - time in seconds since the start of the current spline 

        Outputs:
        q, qdot - the position and velocity
        '''

        p, v = spline(t, self.T, self.p0, self.pf, self.v0, self.vf)

        (ptip_des, _, Jv, _) = fkin(np.reshape(self.q_des, (-1, 1)))

        return (p, v)

    
    def completed(self, t):
        '''
        Returns true if the spline is completed, false otherwise 
        '''
        return t > self.T

    def calculateParameters(self, q, qdot, fkin): # TODO fix this
        '''
        Fills in all the parameters - R0, p0, v0, R, theta, e.
        '''
        p0, R0, Jv, _ = fkin(q)

        self.R0 = R0
        self.p0 = p0 
        self.v0 = Jv @ qdot 

        self.R = self.R0.T @ self.Rf
        self.theta = np.arccos((np.trace(self.R) - 1)/2)

        if self.theta == 0:
            self.noR = True
        
        else:
            self.e = np.array([self.R[2,1] - self.R[1,2], self.R[0,2] - self.R[2,0], self.R[1,0] - self.R[0,1]]).T / (2 * np.sin(self.theta))

class SegmentQueue():
    '''
    Attributes: 

    self.queue: The queue which holds the queue of splines that are enqueued
    self.q: The joint space configuration of the robot currently 
    self.t: The current time 
    self.t0: The time the curret spline was initiated. This is useless if there 
    is no current spline 

    '''
    def __init__(self, fkin, rate, lam=20.0) -> None:
        '''
        q: The initial joint space configuration of the robot.     
        qdot: The initial joint space velocity of the robot.     

        fkin: The fkin function to find the current tip space position, for 
        delayed inputs
        '''
        self.queue = []
        self.t0 = 0
        self.t = 0 
        self.q = None 
        self.qdot = None
        self.fkin = fkin
        self.rate = rate # TODO


    def enqueue(self, segment):
        '''
        Function to enqueue a segment
        segment: A "Spline" object. Can be a JointSpline or a TaskSpline
        '''
        if len(self.queue) == 0:
            # If empty queue, set current segment's start time as the current time, and update its inputs
            self.t0 = self.t 
            self.queue.append(copy.deepcopy(segment))
            self.calculateParameters()
            
        else:
            self.queue.append(copy.deepcopy(segment))

    def enqueueList(self, segments):
        for segment in segments:
            self.enqueue(segment)
        

    def clear(self):
        '''
        Clears the queue, stopping all motion. 
        '''
        self.queue = []

    def update(self, t, q, qdot):
        '''
        Update function. Every single tick, this function must be run, so that 
        the segment queue can keep track of the current time and q.

        t: The current time
        q: The current joint space configuration of the robot
        qdot: The current joint space velocity of the robot
        
        '''

        self.t = t 
        self.q = q
        self.qdot = qdot

        if len(self.queue) == 0:
            # If nothing in queue, just update the time
            return 
        
        if self.queue[0].completed(self.t - self.t0):
            # If the current segment is completed, discard it
            self.queue.pop(0)

            if len(self.queue) == 0:
                return 
            else:
                # If there is a new segment, set the start time of that segment to the current time
                self.t0 = t 
                # Update that splines inputs
                self.calculateParameters()


    def evaluate(self):
        '''
        Evaluates the current object, and gives you either the q, qdot if it 
        is a joint spline, or p, v, R, w if it is a task spline
        '''
        if len(self.queue) == 0: 
            raise Exception("No Segment to Run")

        else: 
            return self.queue[0].evaluate(self.t - self.t0)


    def getCurrentSpace(self):
        '''
        Returns the space of the current spline, if there is any spline 
        '''

        if len(self.queue) == 0: 
            return 'No Spline'

        else: 
            return self.queue[0].getSpace()

    def calculateParameters(self): 
        '''
        Calculates the parameters of the latest spline 
        '''

        if len(self.queue) != 0:
            self.queue[0].calculateParameters(self.q, self.qdot, self.fkin)


def star(center, radius = 3):
    angles = np.linspace(0, 2*np.pi, 6)[:-1]  # Angles for the vertices
    inner_radius = radius / 2 # Inner radius of the star

    outer_x = radius * np.cos(angles)  # x-coordinates of outer vertices
    outer_y = radius * np.sin(angles)  # y-coordinates of outer vertices

    inner_x = inner_radius * np.cos(angles + np.pi/5)  # x-coordinates of inner vertices
    inner_y = inner_radius * np.sin(angles + np.pi/5)  # y-coordinates of inner vertices

    waypoints = []
    for i in range(len(outer_x)):
        waypoints.append(np.array([outer_x[i], outer_y[i]]))
        waypoints.append(np.array([inner_x[i], inner_y[i]]))

    return [JointSpline(p + center, np.array([0, 0]), 1) for p in waypoints] + [JointSpline(waypoints[0] + center, np.array([0, 0]), 1)] 





