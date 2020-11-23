


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'some system configuration will be defined as below'

__author__ = 'Yifan'

num_of_cameras = 4  #only support 2/3/4 rn
base_Cam_Index = 'A'    #A/B/C/D
video_resolution = (1080,1920) #specified resized video size # decide from video
SAVE_FOLDER = 'output/'

Len_of_frame = 50 #how many frames you want to reconstruct 3d #whole video option
start_frame = 0

#save video option



#=============================================USER INPUT DATA================================

dataFile = 'C:/Users/chris/NewTest'

RawVideo = ''

SourceVideoFolder = dataFile +'/SourceVideos' 

DLCConfigpath = 'C:Users/'

openPoseFolderPath = 'C:/Users/chris/openpose'

include_OpenPoseHands = False
include_OpenPoseFace = False

Source_video_List =  [['Cam1.mp4','CamA'],['Cam2.mp4','CamB'],['Cam3.mp4','CamC'],['Cam4.mp4','CamD']]

rotation = 0 #Need for openpose if recorded with a rotation 

Pixel_coord_FIlE_PATH = dataFile+'/OpenPoseData'

