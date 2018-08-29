#!/usr/bin/env python  
import roslib
roslib.load_manifest('learning_tf')
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv

def PID(val_error, kp):
    error = val_error *kp
    return error

if __name__ == '__main__':
    rospy.init_node('turtle_tf_listener')

    listener = tf.TransformListener()

    rospy.wait_for_service('spawn')
    spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
    spawner(4, 2, 0, 'turtle2')

    turtle_vel = rospy.Publisher('turtle2/cmd_vel', geometry_msgs.msg.Twist,queue_size=1)


    x_past = 0
    y_past = 0
    ang_last = 0
    t= 0
    angle_rotated = 0
    t_taken  = 0
    rospy.sleep(1)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/turtle2', '/turtle1', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Something is wrong")
            continue

        euler = tf.transformations.euler_from_quaternion(rot)        
        
        x = trans[0] - 1.5444445610046387
        y = trans[1] - 3.5444445610046387
        distance = trans[0]**2 + trans[1]**2
        rospy.loginfo(distance)
        rospy.loginfo(trans)
        # if x != 0 and y !=0 :
        # print(x,y)
        ang_now = euler[2]
        # angular1 = (ang_now-ang_last)/10
        if trans[0] < 0 and trans[1] <0 and (14.455 > distance or distance > 14.52):
            linear =  -PID((distance-14.5),1)
        elif trans[0] < 0 and trans[1] > 0 and (14.455 > distance or distance > 14.52):
            linear =  -PID((distance-14.5),1)
        elif (14.455 > distance or distance > 14.52) and trans[0]:
            linear =  PID((distance-14.5),1)
        # elif x > 0 or y < 0 and int(distance) != 14:
        #     linear =  PID((distance-14.5),0.5)
        # elif x < 0 or y > 0 and int(distance) != 14:
        #     linear =  PID((distance-14.5),0.5)
        # elif x < 0 or y < 0 and int(distance) != 14:
        #     linear =  PID((distance-14.5),0.5)
        else:
            linear = 0
        rospy.loginfo(linear)   
 
        if  ang_now > 0.05 :
            angular = 2            
            t = PID(euler[2], 0.05)
            angle_rotated += 2.016*abs(t)
            t_taken += abs(t)
            rospy.sleep(abs(t))
        elif  ang_now < -0.05:
            angular = -2
            t = PID(euler[2],0.04)
            # angle_rotated += -2.016*abs(t)*3.14
            rospy.sleep(abs(t))
        else:
            angular = 0


        # rospy.loginfo(ang_now)
        # rospy.loginfo(rot)

        cmd = geometry_msgs.msg.Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        turtle_vel.publish(cmd)
        rate.sleep()

        


        
