#!/usr/bin/env python

import rospy
from face_recognition import FaceRecognition

rospy.init_node('face_recognition_node')

def main():
    fr = FaceRecognition()
    rospy.spin()


if __name__=='__main__':
    main()
