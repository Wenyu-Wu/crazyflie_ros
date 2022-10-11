#!/usr/bin/env python3

"""
Simple example that connects to the first Crazyflie found, ramps up/down
the motors and disconnects.
"""

#TODO: Export the uneccessary stuff for the ros things into an src folder

import rospy
import logging
import time
import threading
from math import pi
import cflib
from cflib.crazyflie import Crazyflie
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from crazyflie_ros.msg import Attitude_Setpoint

command_goal = Attitude_Setpoint()
command_lock = threading.Lock()
logging.basicConfig(level=logging.ERROR)

old_t = time.time()

def command_callback(command):
    global command_goal
    command_goal = radians_to_degrees(command)

def keyboard_callback(command):
    global command_goal
    command_lock.acquire()
    command_goal.pitch = command.linear.x
    command_goal.roll = command.linear.y
    command_goal.yaw_rate = command.angular.z
    command_goal.thrust = int(33000 + command.linear.z * 20000)
    command_lock.release()
    #print(command_goal)

def setpoint_manager(drone):
    global command_goal, old_t
    r = rospy.Rate(200) #100 Hz is the recommended communication baudrate
    drone._cf.commander.send_setpoint(0, 0, 0, 0)
    while not rospy.is_shutdown():
        command_lock.acquire()
        drone._cf.commander.send_setpoint(command_goal.roll, command_goal.pitch, command_goal.yaw_rate, int(command_goal.thrust))
        command_lock.release()
        print("*************FPS:***************", 1/(time.time() - old_t))
        old_t = time.time()
        r.sleep()

def radians_to_degrees(command):

    new_command = Attitude_Setpoint()
    new_command.roll     = command.roll/pi*180.0
    new_command.pitch    = command.pitch/pi*180.0
    new_command.yaw_rate = command.yaw_rate/pi*180.0
    new_command.thrust = command.thrust

    return new_command


class CrazyflieComm:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""
    
    start_thread = False

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self._cf.open_link(link_uri)

        print('Connecting to %s' % link_uri)

    def _connected(self, link_uri):
        """ This callback is called from the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor control.
        # Anything done in here will hijack the external thread 
        self.start_thread = True

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)


if __name__ == '__main__':
    rospy.init_node('crazyflie_comms_node')
    status_pub = rospy.Publisher('crazyflie_comms/status',String,queue_size=10)
    tries = 0
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)
    # Scan for Crazyflies and use the first one found
    print('Scanning interfaces for Crazyflies:\n')
    r = rospy.Rate(.2)

    while not rospy.is_shutdown():
        tries = tries + 1
        available = cflib.crtp.scan_interfaces()
        print(available)
        found = False
        if len(available) > 0:
            print('Crazyflies found:')
            for i in available:
                print(i[0])
                # if(i[0]=='radio://0/70/2M'):
                #     le = CrazyflieComm(i[0])
                #     found = True
                if(i[0]=='radio://0/80/2M'):
                    le = CrazyflieComm(i[0])
                    found = True
                else:
                    print('looking for radio://0/80/2M')
                    
        elif found == False :
            print('\rAttempt ' + str(tries) + ' failed, no correct Crazyflie found         ', end =" ")
        if found == True:
            break
        r.sleep()

    rospy.Subscriber('controller/ypr',Attitude_Setpoint,command_callback) 
    # rospy.Subscriber('cmd_vel',Twist,keyboard_callback)

    r = rospy.Rate(200)
    crazy_thread = threading.Thread(target=setpoint_manager,daemon=True,args=[le])

    while not rospy.is_shutdown():
        print(command_goal)

        if le.start_thread:

            print("Successfully connected to Crazyflie")
            crazy_thread.start()
            le.start_thread = False

        r.sleep()

