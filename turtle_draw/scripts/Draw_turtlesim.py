#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist



def move():
    # Starts a new node
    rospy.init_node('Lens_drawer', anonymous=True)
    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=100)
    vel_msg = Twist()
    flag = 0
    vel_msg.linear.x = 0
    rospy.sleep(1)

    while not rospy.is_shutdown():
        # vel_msg.angular.z = 15
        # vel_msg.linear.x = 15
        # vel_msg.linear.z = 0
        # vel_msg.angular.x = 0
        # vel_msg.angular.y = 0
        
        velocity_publisher.publish(vel_msg)

        if flag < 2 :
            vel_msg.angular.z =10
            vel_msg.linear.x = 10
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            
            velocity_publisher.publish(vel_msg)

        if 5 < flag < 7:
            vel_msg.linear.x = 2
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0
            velocity_publisher.publish(vel_msg)


        if 7 <flag < 9:
            vel_msg.linear.x = 5
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 5
            velocity_publisher.publish(vel_msg)

        print("Publishing")
        

        flag = flag + 1
        rospy.sleep(1)
        

if __name__ == '__main__':
    try:
        #Testing our function
        move()
    except rospy.ROSInterruptException: pass
