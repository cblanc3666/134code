import copy 
import numpy as np 
from vanderbot.TransformHelpers  import *

# ALL INPUTS AND OUTPUTS ARE FLAT

# compensation for z error, which gets higher (grabbing too high) proportional
# to radius from the origin of the space
# currently not using this
Z_COMPENSATION = 0.0

def crossmat(e):
    e = e.flatten()
    return np.array([[  0.0, -e[2],  e[1]],
                     [ e[2],   0.0, -e[0]],
                     [-e[1],  e[0],  0.0]])

def Rote(e, alpha):
    ex = crossmat(e)
    return np.eye(3) + np.sin(alpha) * ex + (1.0-np.cos(alpha)) * ex @ ex

# quintic spline with 0 init and final velocity
def spline(t, T, p0, pf, v0, vf):
    a0 = 0.0
    af = 0.0
    
    # Compute the parameters.
    a = p0
    b = v0
    c = a0
    d = -10*p0/T**3 - 6*v0/T**2 - 3*a0/T + 10*pf/T**3 - 4*vf/T**2 + 0.5*af/T
    e = 15*p0/T**4 + 8*v0/T**3 + 3*a0/T**2 - 15*pf/T**4 + 7*vf/T**3 - 1*af/T**2
    f = -6*p0/T**5 - 3*v0/T**4 - 1*a0/T**3 + 6*pf/T**5 - 3*vf/T**4 + 0.5*af/T**3

    # Compute and return the position and velocity.
    p = a + b * t + c * t**2 + d * t**3 + e * t**4 + f * t**5
    v = b + 2*c * t + 3*d * t**2 + 4*e * t**3 + 5*f * t**4
    return (p,v)

class JointSpline():
    '''    
    Joint Spline Class 

    Takes q, qdot from last known q, qdot before the spline runs
    '''

    def __init__(self, qf, qdotf, qgripf, T) -> None:
        self.q0 = None 
        self.qf = qf 
        self.qdot0 = None 
        self.qdotf = qdotf 
        self.qgrip0 = None
        self.qgripf = qgripf
        self.qdotgrip0 = None
        self.qdotgripf = 0.0 # gripper always still at end of spline
        self.T = T 

        self.space = 'Joint' 

    def getSpace(self):
        '''
        Return if joint or task space
        '''
        return self.space 
    
    def evaluate(self, fkin, q_prev, t):
        '''
        Compute the q, qdot of the given spline at time t.
        
        Inputs:
        fkin - not used.
        q_prev - Last commanded position. Not used.
        t - time in seconds since the start of the current spline 

        Outputs:
        q, qdot - the position and the velocity INCLUDING the gripper 
        '''
        q, qdot = spline(t, self.T, self.q0, self.qf, self.qdot0, self.qdotf)          
        qgrip, qdotgrip = spline(t, self.T, self.qgrip0, self.qgripf, self.qdotgrip0, self.qdotgripf)
        qg = np.append(q, qgrip)
        qgdot = np.append(qdot, qdotgrip)

        return qg, qgdot   
        
    
    def completed(self, t):
        '''
        Returns true if the spline is completed, false otherwise 
        '''
        return t > self.T

    def calculateParameters(self, q, qdot, qgrip, fkin): 
        '''
        Fills up the q0 & qdot0 parameters. fkin is useless, but is passed in 
        since the tip spline needs it. 
        '''

        self.qgrip0 = qgrip
        self.qdotgrip0 = 0.0 # gripper always still at end of spline

        self.q0 = q.copy() 
        self.qdot0 = qdot.copy()

class TaskSpline():
    '''
    A task space spline. The inputs are pf, vf, Rf and T.
    '''
    def __init__(self, pf, vf, qgripf, T, lam, gamma, rate) -> None:
        self.p0 = None 
        self.pf = pf 
        self.v0 = None 
        self.vf = vf 
        self.qgrip0 = None
        self.qgripf = qgripf
        self.qdotgrip0 = None
        self.qdotgripf = 0.0 # gripper always still at end of spline
        self.T = T 

        self.space = 'Tip' 

        # convergence bandwidth for ikin
        self.lam = lam
        self.gamma = gamma
        self.rate = rate

    def getSpace(self):
        '''
        Returns the space of the spline
        '''
        return self.space 
    
    def evaluate(self, fkin, q_prev, t):
        '''
        Compute the q, qdot of the given task spline using ikin.
        
        Inputs:
        fkin - allows us to transform q_prev to task space
        q_prev - Last commanded position. Not used.
        t - time in seconds since the start of the current spline 

        Outputs:
        q, qdot - the position and velocity
        '''
        pd, vd = spline(t, self.T, self.p0, self.pf, self.v0, self.vf)

        qgrip, qdotgrip = spline(t, self.T, self.qgrip0, self.qgripf, self.qdotgrip0, self.qdotgripf)

        (p_prev, _, Jv, _) = fkin(np.reshape(q_prev, (-1, 1)))

        # extract alpha and beta directly from joint space
        alpha = q_prev[1]-q_prev[2]+q_prev[3] 
        beta = q_prev[0]-q_prev[4] # base minus twist

        p_prev = np.vstack((p_prev, alpha, beta))

        vr   = vd + self.lam * ep(pd, p_prev)

        # TODO hard-code these arrays in as constants somewhere since we use them to calculate alpha and beta in fkin
        Jv_mod = np.vstack((Jv, [0, 1, -1, 1, 0], [1, 0, 0, 0, -1]))

        Jinv = Jv_mod.T @ np.linalg.pinv(Jv_mod @ Jv_mod.T + self.gamma**2 * np.eye(5))
        qdot = Jinv @ vr # ikin result

        q = np.reshape(q_prev, (-1, 1)) + qdot / self.rate

        q = q.flatten()
        qdot = qdot.flatten()

        qg = np.append(q, qgrip)
        qgdot = np.append(qdot, qdotgrip)

        return qg, qgdot

    
    def completed(self, t):
        '''
        Returns true if the spline is completed, false otherwise 
        '''
        return t > self.T

    def calculateParameters(self, q, qdot, qgrip, fkin):
        '''
        Fills in all the parameters - p0 and v0.
        '''
        p0, _, Jv, _ = fkin(np.reshape(q, (-1, 1)))
 
        v0 = Jv @ np.reshape(qdot, (-1, 1)) 

        # extract alpha and beta directly from joint space
        alpha = q[1]-q[2]+q[3] 
        beta = q[0]-q[4] # base minus twist

        alphadot = qdot[1]-qdot[2]+qdot[3]
        betadot = qdot[0]-qdot[4]

        self.p0 = np.vstack((p0, alpha, beta))
        self.v0 = np.vstack((v0, alphadot, betadot))

        self.qgrip0 = qgrip
        self.qdotgrip0 = 0.0 # gripper always still at end of spline

class PolarSpline():
    '''
    A task space spline. The inputs are pf, vf, Rf and T.
    '''
    def __init__(self, pf, vf, qgripf, T, lam, gamma, rate) -> None:
        self.p0 = None 
        self.pf = pf 
        self.v0 = None 
        self.vf = vf 
        self.qgrip0 = None
        self.qgripf = qgripf
        self.qdotgrip0 = None
        self.qdotgripf = 0.0 # gripper always still at end of spline
        self.T = T 

        self.space = 'Polar' 

        # convergence bandwidth for ikin
        self.lam = lam
        self.gamma = gamma
        self.rate = rate

    def toPolar(self, cartesian):
        '''
        Converts from Cartesian to Polar space

        Takes in   cartesian = 5 element 1D np array: [x, y, z, alpha, beta]
        returns        polar = 5 element 1D np array: [r, theta, z, alpha, beta]
        '''
        cartesian = cartesian.flatten()
        polar = np.copy(cartesian)
        polar[0] = np.linalg.norm(cartesian[0:2])

        polar[1] = np.arctan2(cartesian[1], cartesian[0])
        # Wrap between -90 and 270 to avoid wrapping along table centerline
        polar[1] = np.fmod(polar[1] + (5/2) * np.pi, 2 * np.pi) - np.pi/2

        return np.reshape(polar, (-1, 1))
    
    def toPolarVel(self, cart_p, cart_v):
        cart_p = cart_p.flatten()
        cart_v = cart_v.flatten()

        polar_v = np.copy(cart_v)

        r = np.linalg.norm(cart_p[0:2])
        polar_v[0] = np.dot(cart_p[0:2], cart_v[0:2]) / r

        polar_v[1] = (cart_p[0] * cart_v[1] - cart_p[1] * cart_v[0]) / (cart_p[1] **2)

        return np.reshape(polar_v, (-1, 1))

    def toCartesian(self, polar):
        '''
        Converts from Polar to Cartesian space

        Takes in       polar = 5 element 1D np array: [r, theta, z, alpha, beta]
        returns        cartesian = 5 element 1D np array: [x, y, z, alpha, beta]
        '''
        polar = polar.flatten()
        cartesian = np.copy(polar)
        cartesian[0] = polar[0] * np.cos(polar[1])
        cartesian[1] = polar[0] * np.sin(polar[1])

        return np.reshape(cartesian, (-1, 1))

    def toCartesianVel(self, polar_p, polar_v):
        polar_p = polar_p.flatten()
        polar_v = polar_v.flatten()

        cartesian_v = np.copy(polar_v)
        
        cartesian_v[0] = polar_v[0] * np.cos(polar_p[1]) - polar_v[1] * polar_p[0] * np.sin(polar_p[1])
        cartesian_v[1] = polar_v[0] * np.sin(polar_p[1]) + polar_v[1] * polar_p[0] * np.cos(polar_p[1])

        return np.reshape(cartesian_v, (-1, 1))

    def getSpace(self):
        '''
        Returns the space of the spline
        '''
        return self.space 
    
    def evaluate(self, fkin, q_prev, t):
        '''
        Compute the q, qdot of the given task spline using ikin.
        
        Inputs:
        fkin - allows us to transform q_prev to task space
        q_prev - Last commanded position. Not used.
        t - time in seconds since the start of the current spline 

        Outputs:
        q, qdot - the position and velocity
        '''
        polar_pd, polar_vd = spline(t, self.T, self.toPolar(self.p0),    self.toPolar(self.pf), 
                                    self.toPolarVel(self.p0, self.v0), self.toPolarVel(self.pf, self.vf))
        
        pd = self.toCartesian(polar_pd)
        vd = self.toCartesianVel(polar_pd, polar_vd)

        qgrip, qdotgrip = spline(t, self.T, self.qgrip0, self.qgripf, self.qdotgrip0, self.qdotgripf)

        (p_prev, _, Jv, _) = fkin(np.reshape(q_prev, (-1, 1)))

        # extract alpha and beta directly from joint space
        alpha = q_prev[1]-q_prev[2]+q_prev[3] 
        beta = q_prev[0]-q_prev[4] # base minus twist

        p_prev = np.vstack((p_prev, alpha, beta))

        vr   = vd + self.lam * ep(pd, p_prev)

        # TODO hard-code these arrays in as constants somewhere since we use them to calculate alpha and beta in fkin
        Jv_mod = np.vstack((Jv, [0, 1, -1, 1, 0], [1, 0, 0, 0, -1]))

        Jinv = Jv_mod.T @ np.linalg.pinv(Jv_mod @ Jv_mod.T + self.gamma**2 * np.eye(5))
        qdot = Jinv @ vr # ikin result

        q = np.reshape(q_prev, (-1, 1)) + qdot / self.rate

        q = q.flatten()
        qdot = qdot.flatten()

        qg = np.append(q, qgrip)
        qgdot = np.append(qdot, qdotgrip)

        return qg, qgdot

    
    def completed(self, t):
        '''
        Returns true if the spline is completed, false otherwise 
        '''
        return t > self.T

    def calculateParameters(self, q, qdot, qgrip, fkin):
        '''
        Fills in all the parameters - p0 and v0.
        '''
        p0, _, Jv, _ = fkin(np.reshape(q, (-1, 1)))
 
        v0 = Jv @ np.reshape(qdot, (-1, 1)) 

        # extract alpha and beta directly from joint space
        alpha = q[1]-q[2]+q[3] 
        beta = q[0]-q[4] # base minus twist

        alphadot = qdot[1]-qdot[2]+qdot[3]
        betadot = qdot[0]-qdot[4]

        self.p0 = np.vstack((p0, alpha, beta))
        self.v0 = np.vstack((v0, alphadot, betadot))

        self.qgrip0 = qgrip
        self.qdotgrip0 = 0.0 # gripper always still at end of spline


class SegmentQueue():
    '''
    Attributes: 

    self.queue: The queue which holds the queue of splines that are enqueued
    self.q: The joint space configuration of the robot currently 
    self.t: The current time 
    self.t0: The time the curret spline was initiated. This is useless if there 
    is no current spline 

    '''
    def __init__(self, fkin, rate=100.0, lam=20.0, gamma=0.4) -> None:
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
        self.qgrip = None
        self.fkin = fkin
        self.rate = rate
        self.lam = lam
        self.gamma = gamma

    def enqueue_task(self, pf, vf, qgrip_f, T):
        pf[2] -= Z_COMPENSATION * np.sqrt(pf[0]**2 + pf[1]**2)
        segment = TaskSpline(np.reshape(pf, (-1, 1)), np.reshape(vf, (-1, 1)), qgrip_f, T, self.lam, self.gamma, self.rate)
        self.enqueue(segment)

    def enqueue_polar(self, pf, vf, qgrip_f, T):
        pf[2] -= Z_COMPENSATION * np.sqrt(pf[0]**2 + pf[1]**2)
        segment = PolarSpline(np.reshape(pf, (-1, 1)), np.reshape(vf, (-1, 1)), qgrip_f, T, self.lam, self.gamma, self.rate)
        self.enqueue(segment)

    def enqueue_joint(self, qf, qdotf, qgrip_f, T):
        segment = JointSpline(qf, qdotf, qgrip_f, T)
        self.enqueue(segment)

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

    def update(self, t, qg, qgdot):
        '''
        Update function. Every single tick, this function must be run, so that 
        the segment queue can keep track of the current time and q.

        t: The current time
        q: The current joint space configuration of the robot
        qdot: The current joint space velocity of the robot
        '''

        self.t = t 
        self.q = qg[0:5]
        self.qdot = qgdot[0:5]
        self.qgrip = qg[5]

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
                
        return False

    def isEmpty(self):
        return len(self.queue) == 0
    
    def queueLength(self):
        return len(self.queue)

    def evaluate(self):
        '''
        Evaluates the current object, and gives you the q, qdot. Runs ikin if 
        it is a task spline
        '''
        if len(self.queue) == 0: 
            raise Warning("No Segment to Run")
            return self.qg

        else: 
            return self.queue[0].evaluate(self.fkin, self.q, self.t - self.t0)


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
            self.queue[0].calculateParameters(self.q, self.qdot, self.qgrip, self.fkin)


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





