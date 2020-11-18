import os
import re
import h5py
import subprocess
import json
import numpy as np
import pandas as pd
import ffmpeg
import cv2
import deeplabcut
import glob
from config import dataFile, num_of_cameras, DLCConfigpath, openPoseFolderPath, include_OpenPoseFace, include_OpenPoseHands


class Exceptions(Exception):
    pass

##create sub-folders for each camera under the calibration folder, store image in each folder
def create_calibration():
    for i in range(num_of_cameras):
        path = 'calibration' + str(i+1)
        if not os.path.exists(path):
            os.mkdir(path)
    
    video_path = os.listdir(VideoFolder)
    
    if len(video_path) != num_of_cameras:
        raise Exceptions('number of videos not match number of cameras')

    video_path.sort(key=lambda f: int(re.sub('\D', '', f)))
    for i in range(len(video_path)):
        vidcap = cv2.VideoCapture(os.path.join(video_path,path[i]))
        success,image = vidcap.read()
        count = 0

        while success:
            success,image = vidcap.read()
            if success:
                height , width , layers =  image.shape
                #resize = cv2.resize(image, video_resolution) 
                if count < 2:
                    cv2.imwrite("Calibration/"+str(i+1)+"/frame%d.jpg" % count, image)     # save frame as JPEG file
                else:
                    break
                count += 1          
            else:
                break



#Concat Videos
############This function needs work, I need to find a way to place this in a for loop so amount of cameras doesnt matter
def concatVideos(Source_video_folder):
    '''Functions input is filepath is path to raw video folder
    If the videos in the folder are multiple parts the function uses ffmpeg to concat the video parts together
    It saves the concated video to an output folder 
    '''
    if not os.path.exists(dataFile+'/Synced'): 
        os.mkdir(dataFile+'/Synced')
    syncPath = dataFile+'/Synced' 
    videoList = os.listdir(Source_video_folder)
    if len(videoList) > num_of_cameras: 
        multipleParts = True
        #Create a txt file for names of video parts
        txtFileList = []
        camNameList = []
        for ii in range(num_of_cameras): 
            #camNameTxt = open(Source_video_folder+'/cam'+str(ii+1)+'vids.txt','a')
            #txtFileList.append(camNameTxt)
            camName = 'Cam'+str(ii+1)
            camNameList.append(camName)
        numOfParts = len(videoList)/num_of_cameras
        k = 0
        for video in os.listdir(Source_video_folder):  #for loop parses through the video folder 
            if video[:4] in camNameList:
                x = open(Source_video_folder+'/'+video[:4]+'vids.txt','a')
                #idx = camNameList.index(video[:4])
                x.write('file'+" '" +'\\'+video+"'")
                x.write('\n')
            k+=1
        #Use ffmpeg to join all parts of the video together
        in_file= ffmpeg.input
        for jj in range(num_of_cameras):
            (ffmpeg
            .input(Source_video_folder+'/Cam'+str(jj+1)+'vids.txt', format='concat', safe=0)
            .output(syncPath+'/'+camNameList[jj][:4]+'.mp4', c='copy')
            .run()
            )
    else:
        multipleParts = False
    return multipleParts
   
''' 
#################### Undistortion #########################
#ALSO NEEDS WORK
def undistortVideos(Inputfilepath,CameraParamsFilePath,Outputfilepath):
   Function input is raw distorted videos filepath and the filepath to save the videos to  
    Uses ffmpeg and camera intrinsics to undistort the video
    Outputs the undistorted video to the specified file path
   
    if not os.path.exists(dataFile+'/Synced'): 
        os.mkdir(dataFile+'/Synced')
    syncPath = dataFile+'/Synced' 
    for jj in range(num_of_cameras):
        dist = np.load(CameraParamsFilePath +'/'+cam_names[jj]+'/Calibration_dist.npy')
        
        #Uses subprocess for a command line prompt to use ffmpeg to undistort video based on intrinsics 
        if boolRotateVid ==True:
            subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+cam_names[jj]+'.mp4', '-vf', "lenscorrection=cx=0.5:cy=0.5:k1="+dist[4]+":k2="+dist[3]+",transpose="+str(rotateVid), Outputfilepath+'/'+cam_names[jj]+'.mp4'])
        if boolRotateVid ==False:
            subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+cam_names[jj]+'.mp4', '-vf', "lenscorrection=cx=0.5:cy=0.5:k1="+dist[4]+":k2="+dist[3]+, Outputfilepath+'/'+cam_names[jj]+'.mp4'])
'''

def trimVideos(Inputfilepath, multipleParts):
    '''Function input is the filepath for undistorted videos and a filepath for the desired output path
    The function finds the frame at the beginning and end of the video where a light flash occurs 
    The video is then trimmed based on those frame numbers
    Outputs the trimmed video to specified filepath
    '''    
    if not os.path.exists(dataFile+'/Synced'): 
        os.mkdir(dataFile+'/Synced')
    if multipleParts:
        Source_video_folder = dataFile+'/Synced'
    else:
        Source_video_folder = Inputfilepath
    syncPath = dataFile+'/Synced' 
    videoList = os.listdir(Source_video_folder)    
    for ii in range(num_of_cameras):
        vidcap = cv2.VideoCapture(Source_video_folder+'/'+videoList[ii])#Open video
        vidWidth  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) #Get video height
        vidHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) #Get video width
        video_resolution = (int(vidWidth),int(vidHeight)) #Create variable for video resolution
        vidLength = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        vidfps = vidcap.get(cv2.CAP_PROP_FPS)
        success,image = vidcap.read() #read a frame
        maxfirstGray = 0 #Intialize the variable for the threshold of the max brightness of beginning of video
        maxsecondGray = 0 #Intialize the variable for the threshold of the max brightness of end of video
        
        for jj in range(int(vidLength)):#For each frame in the video
            
            success,image = vidcap.read() #read a frame
            if success: #If frame is correctly read
                if jj < int(vidLength/3): #If the frame is in the first third of video
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert image to greyscale
                    if np.average(gray) > maxfirstGray:#If the average brightness is greater than the threshold
                        maxfirstGray = np.average(gray)#That average brightness becomes the threshold
                        firstFlashFrame = jj#Get the frame number of the brightest frame
                if jj > int((2*vidLength)/3):
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert image to greyscale
                    if np.average(gray) > maxsecondGray:#If the average brightness is greater than the threshold
                        maxsecondGray = np.average(gray)#That average brightness becomes the threshold
                        secondFlashFrame = jj #Get the frame number of the brightest frame
            else:#If the frame is not correctly read
                continue#Continue
        input1 = ffmpeg.input(Source_video_folder+'/'+videoList[ii])#input for ffmpeg

        node1_1 = input1.trim(start_frame=firstFlashFrame,end_frame=secondFlashFrame).setpts('PTS-STARTPTS')#Trim video based on the frame numbers
        node1_1.output(syncPath+'/Synced'+videoList[ii]).run()#Save to output folder
        
def runDeepLabCut(Inputfilepath):
    '''Function inputs are filepath to videos to be tracked by DLC and the folder to save the output to
    Videos are copied to output folder, than processed in DLC based on the dlc config path 
    DLC output is saved in outputfilepath and the output is also converted to npy and saved as well
    '''
    if not os.path.exists(dataFile+'/DLCData'): 
        os.mkdir(dataFile+'/DLCData')
    DLCPath = dataFile+'/DLCData' 
    syncPath = dataFile+'/Synced'
    #####################Copy Videos to DLC Folder############
    #for video in os.listdir(dir):#Iterates through each video in folder
        #ffmpeg call to copy videos to dlc folder
    #    subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+video,  OutputFilepath+'/'+video])


    #################### DeepLabCut ############################
    videoList =os.listdir(syncPath)
    for video in videoList:
            #Analyze the videos through deeplabcut
            deeplabcut.analyze_videos(baseProjectPath+'/'+DLCConfigpath, DLCPath+videoList[video], save_as_csv=True)
            deeplabcut.plot_trajectories(baseProjectPath+'/'+DLCConfigpath, DLCPath + videoList[video])
 
    #Load all dlc csv output files  
    csvfiles = os.listdir(DLCPath)
    #For loop gets csv data from all cameras
    j=0
    for data in csvfiles:     
        datapoints = pd.read_csv(data) # read in the csv data            

        parsedDlcData = datapoints.iloc[3:,7:10].values#the last element in the array is the P value
        vidName,_ = os.path.splitext(videoList[j])
        np.save(DLCPath+'/'+vidName+'.npy',parsedDlcData)#Save data
        j+=1
    
    
           

def runOpenPose(Inputfilepath,rotation):
    '''Function inputs are the undistorted video filepath, the filepath to save the video output, and the filepath to save the data output
    The function takes the undistorted video and processes the videos in openpose
    The output is openpose overlayed videos and raw openpose data
    '''
    if not os.path.exists(dataFile+'/OpenPoseDataRaw'): 
        os.mkdir(dataFile+'/OpenPoseDataRaw')
    OpenPosePath = dataFile+'/OpenPoseDataRaw' 
    syncPath = dataFile+'/Synced'
    vidList = os.listdir(syncPath)
    ###################### OpenPose ######################################
    os.chdir(openPoseFolderPath) # change the directory to openpose
    for jj in range(num_of_cameras):
        vidName,_ = os.path.splitext(vidList[jj])
        subprocess.call(['bin/OpenPoseDemo.exe', '--video', syncPath+'/'+vidList[jj]])#, '--frame_rotate='+str(rotation) ,'--hand','--face', '--write_json', OpenPosePath+'/'+vidList[jj]])
        
       
    print('LoopThroughOpenpose')


def Parse_Openpose(Inputfilepath):
    '''Function inputs is the filepath to rawopenpose data and the filepath to where to save the parsed openpose data
    Function takes the raw openpose data and organizes in a h5 file, that h5 file is then opened and the data is saved as an npy file
    Outputs one h5 file and an npy file for each camera and returns the amount of points in the frame
    '''
    
    OpenPoseRaw = dataFile+'/OpenPoseDataRaw'
    if not os.path.exists(dataFile+'/OpenPoseData'): 
        os.mkdir(dataFile+'/OpenPoseData')
    OpenPosePath = dataFile+'/OpenPoseData' 
    '''
    #Establish how many points are being used from the user input
    if include_OpenPoseFace:
        points_from_face = 70
    else:
        points_from_face = 0

    if include_OpenPoseHands:
        points_from_Hands = 42
    else:
        points_from_Hands = 0 

    if include_OpenPoseSkeleton:
        points_from_skeleton = 25
    else:
        points_from_skeleton = 0

    points_inFrame = points_from_skeleton + points_from_Hands + points_from_face'''
    j = 0 #Counter variable

    with  h5py.File(OpenPosePath + '/AllOpenPoseData.hdf5', 'w') as f:
        cams = f.create_group('Cameras')
        for cam in os.listdir(OpenPoseRaw):# Loops through each camera
            
            cameraGroup = cams.create_group(cam)
            k=0
            for files in os.listdir(OpenPoseRaw+'/'+cam): #loops through each json file   
                fileGroup = cameraGroup.create_group('Frame'+str(k))
                inputFile = open(OpenPoseRaw+'/'+cam+'/'+files) #open json file
                data = json.load(inputFile) #load json content
                inputFile.close() #close the input file
                k+=1
                ii =0
                for people in data['people']:
                    skeleton = np.array(people['pose_keypoints_2d']).reshape((-1,3))
                    hand_left = np.array(people['hand_left_keypoints_2d']).reshape((-1,3))
                    hand_right = np.array(people['hand_right_keypoints_2d']).reshape((-1,3))
                    face = np.array(people['face_keypoints_2d']).reshape((-1,3))  #Get skeleton points

                    persongroup = fileGroup.create_group('Person'+str(ii))
                    skeletondata = persongroup.create_dataset('Skeleton', data =skeleton)
                    rightHanddata = persongroup.create_dataset('RightHand', data =hand_right) 
                    leftHanddata = persongroup.create_dataset('LeftHand', data =hand_left)
                    facedata = persongroup.create_dataset('Face', data =face)                                       
                    ii+=1
   

    #Create a list variable to store all frame numbers where there is no person in frame
    noPersonInFrame =[]

    k = 0#Initialize counter
 
    
    cam_names = os.listdir(OpenPoseRaw)
    with h5py.File(OpenPosePath + '/AllOpenPoseData.hdf5', 'r') as f:
        allCameras = f.get('Cameras')
        j=0
        for camera in range(len(allCameras)):
            ret = []#intialize an array to store each json file
            target_skeleton = f.get('Cameras/'+str(cam_names[camera])+'/Frame0/Person0/Skeleton')
            target_skeleton = target_skeleton[()]
            allFrames = f.get('Cameras/'+str(cam_names[camera]))
            framesOfPeople = []
            for frame in range(len(allFrames)):
                peopleInFrame = 0
                allPeople = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame))
                if len(allPeople) == 0:
                    noPersonInFrame.append(frame) 
                    empty = (points_inFrame,3)
                    a = np.zeros(empty)
                    ret.append(a)
                    continue
                
                else:
                    c = 0
                    res = 0
                    for person in range(len(allPeople)):
                        zeroPoint =[]
                        peopleInFrame+=1
                        #========================Load body point data
                        
                        skeleton  = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Skeleton')  
                        skeleton = skeleton[()]
                        if include_OpenPoseHands: #If you include hands
                            hand_left = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/LeftHand')
                            hand_left = hand_left[()]
                            hand_right = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/RightHand')
                            hand_right = hand_right[()]
                        if include_OpenPoseFace:#If you include face
                            face = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Face')
                            face = face[()]
                    
                        #============================Find correct skeleton
                        #distance = sum(sum(abs(target_skeleton-skeleton))) #Calculate the distance of the person in this frame compared to the target person from last frame
                        pval = skeleton[:,2]
                        avgPval = sum(pval)/len(pval)
                        #for jj in range(len(skeleton)):
                        #    if skeleton[jj,0] > .001:
                        #        zeroPoint.append(jj)


                        #if distance < c: #If the distance is less than the threshold than this person is the target skeleton
                        if avgPval > c:
                            c = avgPval
                            #c = distance #the distance becomes threshold
                            #c = len(zeroPoint)
                            if include_OpenPoseHands:
                                if include_OpenPoseFace:
                                    HL = hand_left
                                    HR = hand_right
                                    newFace = face     
                                    res = skeleton
                                    fullPoints = np.concatenate((res,HL,HR,newFace),axis = 0)
                                else:
                                    HL = hand_left
                                    HR = hand_right
                                    res = skeleton     
                                    fullPoints = np.concatenate((res,HL,HR),axis = 0)
                            else:   
                                if include_OpenPoseFace:
                                    newFace = face
                                    res = skeleton     
                                    fullPoints = np.concatenate((res,newFace),axis = 0)
                                else:
                                    res = skeleton        
                                    fullPoints =  res
                        
                framesOfPeople.append(peopleInFrame)
                ret.append(fullPoints)
            ret = np.array(ret)
            print(ret.shape)
            np.save(OpenPosePath+'/OP_'+cam_names[j]+'.npy',ret)
            j+=1
            #np.savetxt(OutputFilepath+'/OP_'+cam_names[k]+'.txt',ret[:,8,0])
            

    return noPersonInFrame

