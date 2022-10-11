#!/usr/bin/env python3

'''
Euler angle order needs to be ZXY for consistent stability 
Follows Right hand convention. X is forward Z is right Y is up in drone frame
TODO: Possible issues with Optitrack environment alignment on initialization. May need a calibration protocol
'''

import rospy
from crazyflie_ros.msg import Attitude_Setpoint, Attitude_Error, Pid_Gains, Drone_Pid_Settings
from acsi_observer.msg import Drone_States
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseStamped, Quaternion
import numpy as np
from math import sin, cos, pi
from copy import deepcopy
import threading

class PID:

    gains = Drone_Pid_Settings()

    integral_accumulation = Drone_States()
    current_local_error   = Drone_States()
    current_global_error  = Drone_States()
    current_global_state  = Drone_States()
    desired_global_state  = Drone_States()
    
    global_error_hist = []
    local_error_hist = []

    control_output = Attitude_Setpoint()

    def __init__(self, controller_settings):
        self.gains = controller_settings
        self.lock = threading.RLock()
        return
        
    def spin_controller(self,desired_state):
        self.desired_global_state = desired_state
        if self.current_global_state == Drone_States():
            print('current state not recieved')
        else:
            #print(self.current_global_state)
            self.error_update()
            self.altitude_controller()
            self.yaw_controller()
            self.position_controller()
            #print(self.current_local_error)
        
        return self.control_output

    def error_update(self):

        if len(self.global_error_hist) < 10:
            self.local_error_hist.insert(0, Drone_States())
            self.global_error_hist.insert(0, Drone_States())

            self.error_calc()
        else:
            self.global_error_hist.insert(0, Drone_States())
            self.local_error_hist.insert(0, Drone_States())
            self.error_calc()
            self.global_error_hist.pop()
            self.local_error_hist.pop()
   
    def error_calc(self):
        
        self.current_global_error.time = rospy.Time.now().nsecs*1e-9+rospy.Time.now().secs

        self.current_global_error.x    = self.desired_global_state.x   - self.current_global_state.x
        self.current_global_error.y    = self.desired_global_state.y   - self.current_global_state.y
        self.current_global_error.z    = self.desired_global_state.z   - self.current_global_state.z
        self.current_global_error.yaw  = self.desired_global_state.yaw - self.current_global_state.yaw

        self.current_local_error.time = self.current_global_error.time
        self.current_local_error.yaw  = self.current_global_error.yaw
        self.current_local_error.y    = self.current_global_error.y
        
        rot             = self.current_global_state.yaw
        rot_matrix      = np.matrix([[cos(rot), -sin(rot)],[sin(rot), cos(rot)]])
        error_vector = np.matrix([[self.current_global_error.x],[self.current_global_error.z]])

        local_xz_err = np.dot(rot_matrix,error_vector)

        self.current_local_error.x = local_xz_err[0][0]
        self.current_local_error.z = local_xz_err[1][0]

        self.global_error_hist[0].time = self.current_global_error.time
        self.global_error_hist[0].x    = self.current_global_error.x
        self.global_error_hist[0].y    = self.current_global_error.y
        self.global_error_hist[0].z    = self.current_global_error.z
        self.global_error_hist[0].yaw  = self.current_global_error.yaw

        self.local_error_hist[0].time  = self.current_local_error.time
        self.local_error_hist[0].x     = self.current_local_error.x
        self.local_error_hist[0].y     = self.current_local_error.y
        self.local_error_hist[0].z     = self.current_local_error.z
        self.local_error_hist[0].yaw   = self.current_local_error.yaw


        if len(self.global_error_hist) >= 10:
            
            Ts = (self.global_error_hist[0].time-self.global_error_hist[4].time)/(4.0) #making sample time the average in the last five samples converted to seconds

            self.current_global_error.dx = -(-self.global_error_hist[0].x + 8.0*self.global_error_hist[1].x - 8.0*self.global_error_hist[3].x + self.global_error_hist[4].x)/(12.0*Ts)
            self.current_global_error.dy = -(-self.global_error_hist[0].y + 8.0*self.global_error_hist[1].y - 8.0*self.global_error_hist[3].y + self.global_error_hist[4].y)/(12.0*Ts)
            self.current_global_error.dz = -(-self.global_error_hist[0].z + 8.0*self.global_error_hist[1].z - 8.0*self.global_error_hist[3].z + self.global_error_hist[4].z)/(12.0*Ts)
            self.current_global_error.dyaw = -(-self.global_error_hist[0].yaw + 8.0*self.global_error_hist[1].yaw - 8.0*self.global_error_hist[3].yaw + self.global_error_hist[4].yaw)/(12.0*Ts)

            self.current_local_error.dx = -(-self.local_error_hist[0].x + 8.0*self.local_error_hist[1].x - 8.0*self.local_error_hist[3].x + self.local_error_hist[4].x)/(12.0*Ts)
            self.current_local_error.dy = -(-self.local_error_hist[0].y + 8.0*self.local_error_hist[1].y - 8.0*self.local_error_hist[3].y + self.local_error_hist[4].y)/(12.0*Ts)
            self.current_local_error.dz = -(-self.local_error_hist[0].z + 8.0*self.local_error_hist[1].z - 8.0*self.local_error_hist[3].z + self.local_error_hist[4].z)/(12.0*Ts)
            self.current_local_error.dyaw = -(-self.local_error_hist[0].yaw + 8.0*self.local_error_hist[1].yaw - 8.0*self.local_error_hist[3].yaw + self.local_error_hist[4].yaw)/(12.0*Ts)


        self.global_error_hist[0] = deepcopy(self.current_global_error)
        self.local_error_hist[0] = deepcopy(self.current_local_error)

    def altitude_controller(self):#no integral control atm

        thrust = 47000 + self.gains.thrust.p * self.current_local_error.y - self.gains.thrust.d * self.current_local_error.dy 

        if thrust >= 65535:
            thrust = 65534
        if thrust < 0:
            thrust = 0
        
        self.control_output.thrust = thrust

    def yaw_controller(self):#no integral control atm
        
        self.control_output.yaw_rate = self.gains.yaw.p * self.current_local_error.yaw - self.gains.yaw.d * self.current_local_error.dyaw

    def position_controller(self):#no integral control atm

        self.control_output.pitch = self.gains.pitch.p * self.current_local_error.z - self.gains.pitch.d * self.current_local_error.dz
        self.control_output.roll = -(self.gains.roll.p * self.current_local_error.x - self.gains.roll.d * self.current_local_error.dx)

        if self.control_output.pitch > .5:
            self.control_output.pitch = .5
        elif self. control_output.pitch < -.5:
            self.control_output.pitch = -.5

        if self.control_output.roll > .5:
            self.control_output.roll = .5
        elif self. control_output.roll < -.5:
            self.control_output.roll = -.5
