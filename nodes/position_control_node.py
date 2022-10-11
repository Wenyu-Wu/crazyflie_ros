#!/usr/bin/env python3

'''
Euler angle order needs to be ZXY for consistent stability 
Follows Right hand convention. X is forward Z is right Y is up in drone frame
TODO: Possible issues with Optitrack environment alignment on initialization. May need a calibration protocol
'''

import rospy
from acsi_controller.msg import Attitude_Setpoint, Attitude_Error, Drone_Pid_Settings
from acsi_observer.msg import Drone_States, Drone_States_Array
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseStamped
from std_msgs.msg import String
from math import sin, cos, pi, sqrt
import numpy as np
from copy import deepcopy
import os
from scipy.io import loadmat
from acsi_mpc.mpc import MPC
from acsi_pid import pid as PID
from acsi_trajectory.srv import Trajectory_Service

trajectory = PoseArray()
recieved_state = False

class Manager:
    
    def __init__(self,gains,fname):
        self.pid = PID.PID(gains)
        self.current_global_state = Drone_States()
        self.load_mpc(fname)
        self.command = Attitude_Setpoint()
        self.X = np.zeros((self.A.shape[1], 1))
        self.current_trajectory = Drone_States_Array()
        self.current_setpoint = Drone_States()
        self.run_state = self.wait_for_drone
        self.status = 'initiating'
        self.observer_states = Drone_States()
        self.take_off_goal = Drone_States()
        self.hover_height = 1.5
        self.drone_speed = .5
        self.control_type = 'mpc'

    def observer_handle(self,states_message):
        self.pid.current_global_state = states_message
        self.correct_states(states_message)
        self.observer_states = states_message

    def wait_for_drone(self):
        self.status = 'waiting for drone state'
        if self.observer_states!=Drone_States():
            self.run_state = self.take_off_protocol
            self.take_off_goal = self.observer_states
            self.take_off_goal.y = self.hover_height

    def run_trajectory(self):
        if self.control_type == 'mpc':
            self.current_setpoint = self.current_trajectory.state_array[0]
            self.spin_mpc()
        elif self.control_type == 'pid':
            self.current_setpoint = self.current_trajectory.state_array[0]
            self.spin_pid()
            if(len(self.current_trajectory.state_array)>1):
                self.current_trajectory.state_array.pop(0)
        
        if len(self.current_trajectory.state_array) == 1:
            self.status = 'trajectory finished'
        else:
            self.status = 'executing trajectory'

    def rise(self):
        self.status = 'rising'
        self.current_setpoint.y = 1.5
        self.spin_pid()
        if self.error_distance() < .3:
            self.status = 'stabalizing'
        else:
            print(self.error_distance())
            

    def take_off_protocol(self):
        self.status = 'in take off protocol'
        self.current_setpoint = self.take_off_goal
        self.current_trajectory.state_array = [self.take_off_goal]
        self.spin_pid()
        if self.error_distance() < .2:
            self.run_state = self.command_mode
        else:
            print(self.error_distance())

    def command_mode(self):
        self.status = 'locked and loaded'
        self.spin_pid()

    def return_home(self):
        self.status = 'returning home'
        self.current_setpoint = self.take_off_goal
        self.spin_pid()
        if self.error_distance() < .3:
            self.run_state = self.landing_protocol
        else:
            print(self.error_distance())
    
    def landing_protocol(self):
        self.status = 'landing'
        self.current_setpoint = self.take_off_goal
        self.current_setpoint.y = .5
        if self.error_distance() < .3:
            self.run_state = self.kill_motors
        else:
            print(self.error_distance())

    def kill_motors(self):
        self.status = 'killing motors'
        self.command.roll     = 0
        self.command.pitch    = 0
        self.command.yaw_rate = 0
        self.command.thrust   = 0

    def error_distance(self):
        
        return sqrt((self.observer_states.x-self.current_setpoint.x)**2 + (self.observer_states.y-self.current_setpoint.y)**2 + (self.observer_states.z-self.current_setpoint.z)**2)


    def load_mpc(self,fname):

        dirname = os.path.dirname(__file__)
        full_path = os.path.join(dirname, '../models/'+fname)

        matfile = loadmat(full_path)

        self.A = matfile['A']
        self.B = matfile['B']
        self.C = matfile['C']
        self.Q = np.diag(np.array([1000., 10000., 1000., 1000.]))
        self.R = np.diag(np.array([1., 1., 1., 1e-8]))
        self.RD = np.diag(np.array([1000, 1000, 10, 1e-5]))
        self.U = np.zeros((self.B.shape[1], 1))
        self.X = np.zeros((self.A.shape[1], 1))

        self.umin = np.array([-.4, -.4, -.4, -18000.])[:, np.newaxis]
        self.umax = np.array([.4, .4, .4, 18000.])[:, np.newaxis]

        self.N = 30
        self.mpc = MPC(self.A, self.B, self.C, self.Q, self.R, self.RD, self.umin, self.umax, self.N)

    def spin_mpc(self):
        mpc_traj = []
        for i in self.current_trajectory.state_array:
            if mpc_traj == []:
                mpc_traj = np.matrix([[i.x],[i.y],[i.z],[i.yaw]])
            else:
                mpc_traj = np.column_stack((mpc_traj,np.matrix([[i.x],[i.y],[i.z],[i.yaw]])))
        
        self.U = self.mpc.get_control_input(self.X,self.U,mpc_traj)
        self.command.roll = self.U[0,0]
        self.command.pitch = self.U[1,0]
        self.command.yaw_rate = self.U[2,0]
        self.command.thrust = self.U[3,0] + 47000
        self.update_states()
        if(len(self.current_trajectory.state_array)>1):
            self.current_trajectory.state_array.pop(0)
        
    
    def spin_pid(self):
        self.pid.current_global_state = self.observer_states
        self.command = self.pid.spin_controller(self.current_setpoint)

    def update_states(self):
        self.X = self.A @ self.X + self.B @ self.U
    
    def correct_states(self, observation):
        self.X[0,0] = observation.z
        self.X[1,0] = observation.x
        self.X[2,0] = observation.y
        self.X[3,0] = observation.yaw

def observer_callback(observer_states,manager):
    manager.observer_handle(observer_states)

def set_gains(gains): #Could be a param server pull at some point
    gains.pitch.p  = .6
    gains.pitch.i  = 0.0
    gains.pitch.d  = .3

    gains.roll.p   = .6
    gains.roll.i   = 0.0
    gains.roll.d   = .3

    gains.yaw.p    = 0.0
    gains.yaw.i    = 0.0
    gains.yaw.d    = 0.0

    gains.thrust.p = 25000/2
    gains.thrust.i = 0.0
    gains.thrust.d = 25000/2

if __name__ == '__main__':
    rospy.init_node('position_controller_node')

    model_fname = 'Crazyflie_Model.mat'

    gains = Drone_Pid_Settings() #bring down from the param server eventually?
    set_gains(gains)

    status_pub = rospy.Publisher('pid_controller/status',String,queue_size=2)

    manager = Manager(gains, model_fname)

    goal_pub = rospy.Publisher('controller/goal',Drone_States,queue_size=2)
    setpoint_pub = rospy.Publisher('controller/ypr',Attitude_Setpoint,queue_size=2)
    rospy.Subscriber('/observer/states',Drone_States,observer_callback,callback_args=manager)
    state_pub = rospy.Publisher('controller/state',Drone_States,queue_size=2)

    get_trajectory = rospy.ServiceProxy('generate_trajectory',Trajectory_Service)

    hover_state = Drone_States()
    hover_state.x   = 0
    hover_state.y   = 1.5
    hover_state.z   = 0
    hover_state.yaw = 0

    freq = 1/.04
    r = rospy.Rate(freq)

    start_time = rospy.Time.now()
    command_setpoint = Attitude_Setpoint()

    print("Control node")

    start_time = rospy.Time.now().secs + rospy.Time.now().nsecs*1e-9
    got_trajectory = False
    i = 0
    current_goal = Drone_States()
    current_goal = hover_state

    start_time = rospy.Time.now().secs + rospy.Time.now().nsecs*1e-9
    timer = 0
    stage = 1
    while not rospy.is_shutdown():
        manager.run_state()
        if manager.run_state == manager.command_mode:
            if stage == 1:
                timer = rospy.Time.now().secs + rospy.Time.now().nsecs*1e-9
                stage = 2
            elif rospy.Time.now().secs + rospy.Time.now().nsecs*1e-9 - timer >= 3 and stage == 2:
                traj = get_trajectory(freq,manager.drone_speed)
                manager.current_trajectory = traj.trajectory_states
                stage = 3
            elif stage == 3:
                manager.control_type = 'pid'
                manager.run_state = manager.run_trajectory
        if manager.status == 'trajectory finished':
            manager.run_state = manager.rise
        
        print(manager.status)

        manager.current_setpoint.time = rospy.Time.now().secs + rospy.Time.now().nsecs*1e-9 - start_time
        manager.observer_states.time = rospy.Time.now().secs + rospy.Time.now().nsecs*1e-9 - start_time

        goal_pub.publish(manager.current_setpoint)
        status_pub.publish(manager.status)
        state_pub.publish(manager.observer_states)
        setpoint_pub.publish(manager.command)
            
        r.sleep()
