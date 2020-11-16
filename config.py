


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'some system configuration will be defined as below'

__author__ = 'Yifan'

num_of_cameras = 4  #only support 2/3/4 rn
base_Cam_Index = 'A'    #A/B/C/D
video_resolution = (1080,1920) #specified resized video size # decide from video
SAVE_FOLDER = 'output/'

Len_of_frame = 5000 #how many frames you want to reconstruct 3d #whole video option
start_frame = 0

#save video option



#=============================================USER INPUT DATA================================

dataFile = 'C:/Users/chris/Test'

SourceVideoFolder = dataFile +'/SourceVideos' 

DLCConfigpath = 'C:Users/'

openPoseFolderPath = 'C:/Users/chris'


Source_video_List =  [['Cam1.avi','CamA'],['Cam2.avi','CamB'],['Cam3.avi','CamC'],['Cam4.avi','CamD']]

rotation = 0 #Need for openpose if recorded with a rotation 

Pixel_coord_FIlE_PATH = dataFile+'OpenPose'

