#!/usr/bin/env python

import rospy, math
from std_msgs.msg import Int32MultiArray

fl,fr=0,0

def callback(msg):
	global fl,fr
	fl,_,fr,_,_,_,_,_ = msg.data
	#print(msg.data)

rospy.init_node('guide')
motor_pub = rospy.Publisher('xycar_motor_msg', Int32MultiArray, queue_size=1)
ultra_sub = rospy.Subscriber('ultrasonic', Int32MultiArray, callback)

xycar_msg = Int32MultiArray()


while not rospy.is_shutdown():
	dx = fr/math.sqrt(2) - fl/math.sqrt(2)
	dy = (fr/2 + fl/2) / 2
	angle = 0
	try: angle = math.degrees(math.atan(dx/dy))
	except: 0
	xycar_msg.data = [angle, 50]
	motor_pub.publish(xycar_msg)
