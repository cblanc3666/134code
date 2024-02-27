'''DetectHelpers.py

   These are helper functions for detection to keep the detection file clean.
   
   The Helper functions are:

   Cross Product:   cross(e1,e2)    Cross product of two 3x1 vectors
                    crossmat(e)     Cross product matrix

   Position         pzero()         Zero position vector
                    pxyz(x,y,z)     Position vector

   Axis Vectors     ex()            Unit x-axis
                    ey()            Unit y-axis
                    ez()            Unit z-axis
                    exyz(x,y,z)     Unit vector

   Rotation Matrix  Reye()          Identity rotation matrix
                    Rotx(alpha)     Rotation matrix about x-axis
                    Roty(alpha)     Rotation matrix about y-axis
                    Rotz(alpha)     Rotation matrix about z-axis
                    Rote(e, alpha)  Rotation matrix about unit vector e

   Error Vectors    ep(pd, p)       Translational error vector
                    eR(Rd, R)       Rotational error vector

   Transforms       T_from_Rp(R,p)  Compose T matrix
                    p_from_T(T)     Extract position vector from T
                    R_from_T(T)     Extract rotation matrix from T

   Quaternions      R_from_quat(quat)   Convert quaternion to R
                    quat_from_R(R)      Convert R to quaternion

   URDF Elements    T_from_URDF_origin(origin)   Construct transform
                    e_from_URDF_axis(axis)       Construct axis vector

'''