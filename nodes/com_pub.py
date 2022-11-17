#!/usr/bin/env python3
# license removed for brevity
from crazyflie_ros.msg import Attitude_Setpoint
import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('controller/ypr', Attitude_Setpoint, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(200) # 10hz
    ll = Attitude_Setpoint()
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        ll.roll = 0
        ll.pitch = 0
        ll.yaw_rate = 0
        ll.thrust = 1000
        pub.publish(ll)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
